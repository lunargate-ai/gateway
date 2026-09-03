package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/lunargate-ai/gateway/pkg/models"
	"github.com/rs/zerolog/log"
)

// openAIStreamTranslator keeps Responses API snapshot events from being
// exposed as duplicate Chat Completions deltas. A translator is created for
// each upstream stream, so all state below is request-local.
type openAIStreamTranslator struct {
	base *OpenAITranslator

	id      string
	model   string
	created int64

	textParts      map[openAIStreamPartKey]string
	reasoningParts map[openAIStreamPartKey]string

	toolAliases        map[string]*openAIStreamToolState
	toolsByOutputIndex map[int]*openAIStreamToolState
	nextToolIndex      int
}

type openAIStreamPartKey struct {
	outputIndex  int
	contentIndex int
	kind         string
}

type openAIStreamToolState struct {
	index       int
	outputIndex int
	itemID      string
	callID      string
	stableID    string
	name        string
	arguments   string
	announced   bool
	emittedName string
}

func NewOpenAIStreamTranslator(base *OpenAITranslator) models.ProviderTranslator {
	return &openAIStreamTranslator{
		base:               base,
		textParts:          make(map[openAIStreamPartKey]string, 4),
		reasoningParts:     make(map[openAIStreamPartKey]string, 4),
		toolAliases:        make(map[string]*openAIStreamToolState, 8),
		toolsByOutputIndex: make(map[int]*openAIStreamToolState, 4),
	}
}

func (t *openAIStreamTranslator) Name() string {
	return t.base.Name()
}

func (t *openAIStreamTranslator) DefaultModel() string {
	return t.base.DefaultModel()
}

func (t *openAIStreamTranslator) BaseURL() string {
	return t.base.BaseURL()
}

func (t *openAIStreamTranslator) TranslateRequest(ctx context.Context, req *models.UnifiedRequest) (*http.Request, error) {
	return t.base.TranslateRequest(ctx, req)
}

func (t *openAIStreamTranslator) ParseResponse(resp *http.Response) (*models.UnifiedResponse, error) {
	return t.base.ParseResponse(resp)
}

func (t *openAIStreamTranslator) ParseStreamChunk(data []byte) (*models.StreamChunk, error) {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 || string(trimmed) == "[DONE]" {
		return t.base.ParseStreamChunk(trimmed)
	}

	var raw map[string]interface{}
	if err := json.Unmarshal(trimmed, &raw); err != nil {
		return t.base.ParseStreamChunk(trimmed)
	}
	typeName := strings.TrimSpace(interfaceToString(raw["type"]))
	if !strings.HasPrefix(typeName, "response.") {
		return t.base.ParseStreamChunk(trimmed)
	}

	t.updateResponseMetadata(raw)

	switch typeName {
	case "response.output_text.delta":
		return t.textDeltaChunk(raw, interfaceToString(raw["delta"])), nil
	case "response.output_text.done":
		return t.textSnapshotChunk(raw, interfaceToString(raw["text"])), nil
	case "response.content_part.added", "response.content_part.done":
		return t.contentPartSnapshotChunk(raw), nil
	case "response.reasoning_summary_text.delta":
		return t.reasoningDeltaChunk(raw, "reasoning_summary", interfaceToString(raw["delta"])), nil
	case "response.reasoning_summary_text.done":
		return t.reasoningSnapshotChunk(raw, "reasoning_summary", interfaceToString(raw["text"])), nil
	case "response.reasoning_text.delta":
		return t.reasoningDeltaChunk(raw, "reasoning_text", interfaceToString(raw["delta"])), nil
	case "response.reasoning_text.done":
		return t.reasoningSnapshotChunk(raw, "reasoning_text", interfaceToString(raw["text"])), nil
	case "response.reasoning_summary_part.added", "response.reasoning_summary_part.done":
		return t.reasoningPartSnapshotChunk(raw), nil
	case "response.function_call_arguments.delta":
		return t.functionArgumentsDeltaChunk(raw), nil
	case "response.function_call_arguments.done":
		return t.functionArgumentsSnapshotChunk(raw, nil, false), nil
	case "response.output_item.added":
		return t.outputItemChunk(raw, true), nil
	case "response.output_item.done":
		return t.outputItemChunk(raw, false), nil
	default:
		chunk, err := t.base.ParseStreamChunk(trimmed)
		return t.decorateChunk(chunk), err
	}
}

func (t *openAIStreamTranslator) SupportsStreaming() bool {
	return t.base.SupportsStreaming()
}

func (t *openAIStreamTranslator) Models() []models.ModelInfo {
	return t.base.Models()
}

func (t *openAIStreamTranslator) updateResponseMetadata(raw map[string]interface{}) {
	if responseID := responsesEventResponseID(raw); responseID != "" {
		t.id = responseID
	}
	model, created := responsesEventResponseMeta(raw)
	if model != "" {
		t.model = model
	}
	if created != 0 {
		t.created = created
	}
}

func (t *openAIStreamTranslator) decorateChunk(chunk *models.StreamChunk) *models.StreamChunk {
	if chunk == nil {
		return nil
	}
	if chunk.ID == "" {
		chunk.ID = t.id
	}
	if chunk.Model == "" {
		chunk.Model = t.model
	}
	if chunk.Created == 0 {
		chunk.Created = t.created
	}
	return chunk
}

func (t *openAIStreamTranslator) messageChunk(message *models.Message) *models.StreamChunk {
	if message == nil {
		return nil
	}
	return &models.StreamChunk{
		ID:      t.id,
		Object:  "chat.completion.chunk",
		Created: t.created,
		Model:   t.model,
		Choices: []models.Choice{{
			Index: 0,
			Delta: message,
		}},
	}
}

func openAIStreamPartPosition(raw map[string]interface{}, kind string) openAIStreamPartKey {
	contentIndex := intFromAny(raw["content_index"])
	if strings.Contains(kind, "summary") {
		contentIndex = intFromAny(raw["summary_index"])
	}
	return openAIStreamPartKey{
		outputIndex:  intFromAny(raw["output_index"]),
		contentIndex: contentIndex,
		kind:         kind,
	}
}

func (t *openAIStreamTranslator) textDeltaChunk(raw map[string]interface{}, delta string) *models.StreamChunk {
	if delta == "" {
		return nil
	}
	key := openAIStreamPartPosition(raw, "output_text")
	t.textParts[key] += delta
	return t.messageChunk(&models.Message{Content: delta})
}

func (t *openAIStreamTranslator) textSnapshotChunk(raw map[string]interface{}, snapshot string) *models.StreamChunk {
	key := openAIStreamPartPosition(raw, "output_text")
	tail := openAIStreamSnapshotTail(t.textParts, key, snapshot)
	if tail == "" {
		return nil
	}
	return t.messageChunk(&models.Message{Content: tail})
}

func (t *openAIStreamTranslator) reasoningDeltaChunk(raw map[string]interface{}, kind, delta string) *models.StreamChunk {
	if delta == "" {
		return nil
	}
	key := openAIStreamPartPosition(raw, kind)
	t.reasoningParts[key] += delta
	return t.messageChunk(&models.Message{ReasoningContent: delta})
}

func (t *openAIStreamTranslator) reasoningSnapshotChunk(raw map[string]interface{}, kind, snapshot string) *models.StreamChunk {
	key := openAIStreamPartPosition(raw, kind)
	tail := openAIStreamSnapshotTail(t.reasoningParts, key, snapshot)
	if tail == "" {
		return nil
	}
	return t.messageChunk(&models.Message{ReasoningContent: tail})
}

func openAIStreamSnapshotTail(parts map[openAIStreamPartKey]string, key openAIStreamPartKey, snapshot string) string {
	if snapshot == "" {
		return ""
	}
	current := parts[key]
	if current == "" {
		parts[key] = snapshot
		return snapshot
	}
	if strings.HasPrefix(snapshot, current) {
		tail := strings.TrimPrefix(snapshot, current)
		parts[key] = snapshot
		return tail
	}

	log.Debug().
		Str("responses_part_kind", key.kind).
		Int("output_index", key.outputIndex).
		Int("content_index", key.contentIndex).
		Int("streamed_len", len(current)).
		Int("snapshot_len", len(snapshot)).
		Msg("responses stream snapshot diverged from emitted deltas")
	return ""
}

func (t *openAIStreamTranslator) contentPartSnapshotChunk(raw map[string]interface{}) *models.StreamChunk {
	part, _ := raw["part"].(map[string]interface{})
	if part == nil {
		return nil
	}
	partType := strings.TrimSpace(interfaceToString(part["type"]))
	text := interfaceToString(part["text"])
	switch partType {
	case "output_text", "text":
		return t.textSnapshotChunk(raw, text)
	case "reasoning_summary_text", "summary_text":
		return t.reasoningSnapshotChunk(raw, "reasoning_summary", text)
	case "reasoning", "reasoning_text":
		return t.reasoningSnapshotChunk(raw, "reasoning_text", text)
	default:
		return nil
	}
}

func (t *openAIStreamTranslator) reasoningPartSnapshotChunk(raw map[string]interface{}) *models.StreamChunk {
	part, _ := raw["part"].(map[string]interface{})
	if part == nil {
		return nil
	}
	return t.reasoningSnapshotChunk(raw, "reasoning_summary", interfaceToString(part["text"]))
}

func (t *openAIStreamTranslator) outputItemChunk(raw map[string]interface{}, added bool) *models.StreamChunk {
	item, _ := raw["item"].(map[string]interface{})
	if item == nil {
		return nil
	}

	switch strings.TrimSpace(interfaceToString(item["type"])) {
	case "message":
		return t.messageOutputItemSnapshotChunk(raw, item)
	case "reasoning":
		return t.reasoningOutputItemSnapshotChunk(raw, item)
	case "function_call":
		return t.functionArgumentsSnapshotChunk(raw, item, added)
	default:
		return nil
	}
}

func (t *openAIStreamTranslator) messageOutputItemSnapshotChunk(raw, item map[string]interface{}) *models.StreamChunk {
	content, ok := item["content"].([]interface{})
	if !ok {
		text := openaiMessageContentToString(item["content"])
		return t.textSnapshotChunk(raw, text)
	}

	var tails strings.Builder
	for contentIndex, rawPart := range content {
		part, _ := rawPart.(map[string]interface{})
		if part == nil {
			continue
		}
		partType := strings.TrimSpace(interfaceToString(part["type"]))
		if partType != "output_text" && partType != "text" {
			continue
		}
		position := cloneOpenAIStreamEventPosition(raw, "content_index", contentIndex)
		key := openAIStreamPartPosition(position, "output_text")
		tails.WriteString(openAIStreamSnapshotTail(t.textParts, key, interfaceToString(part["text"])))
	}
	if tails.Len() == 0 {
		return nil
	}
	return t.messageChunk(&models.Message{Content: tails.String()})
}

func (t *openAIStreamTranslator) reasoningOutputItemSnapshotChunk(raw, item map[string]interface{}) *models.StreamChunk {
	var tails strings.Builder
	if summary, ok := item["summary"].([]interface{}); ok {
		for summaryIndex, rawPart := range summary {
			part, _ := rawPart.(map[string]interface{})
			if part == nil {
				continue
			}
			position := cloneOpenAIStreamEventPosition(raw, "summary_index", summaryIndex)
			key := openAIStreamPartPosition(position, "reasoning_summary")
			tails.WriteString(openAIStreamSnapshotTail(t.reasoningParts, key, interfaceToString(part["text"])))
		}
	}
	if content, ok := item["content"].([]interface{}); ok {
		for contentIndex, rawPart := range content {
			part, _ := rawPart.(map[string]interface{})
			if part == nil {
				continue
			}
			partType := strings.TrimSpace(interfaceToString(part["type"]))
			if partType != "reasoning" && partType != "reasoning_text" {
				continue
			}
			position := cloneOpenAIStreamEventPosition(raw, "content_index", contentIndex)
			key := openAIStreamPartPosition(position, "reasoning_text")
			tails.WriteString(openAIStreamSnapshotTail(t.reasoningParts, key, interfaceToString(part["text"])))
		}
	}
	if tails.Len() == 0 {
		return nil
	}
	return t.messageChunk(&models.Message{ReasoningContent: tails.String()})
}

func cloneOpenAIStreamEventPosition(raw map[string]interface{}, key string, value int) map[string]interface{} {
	position := map[string]interface{}{
		"output_index": raw["output_index"],
		key:            value,
	}
	return position
}

func (t *openAIStreamTranslator) functionArgumentsDeltaChunk(raw map[string]interface{}) *models.StreamChunk {
	state := t.resolveToolState(raw, nil)
	delta := interfaceToString(raw["delta"])
	if delta == "" {
		return nil
	}
	state.arguments += delta
	return t.toolCallChunk(state, delta)
}

func (t *openAIStreamTranslator) functionArgumentsSnapshotChunk(raw, item map[string]interface{}, forceAnnouncement bool) *models.StreamChunk {
	state := t.resolveToolState(raw, item)
	if name := openAIStreamToolName(raw, item); name != "" {
		state.name = name
	}

	snapshot := interfaceToString(raw["arguments"])
	if item != nil {
		snapshot = interfaceToString(item["arguments"])
	}
	tail := openAIStreamStringSnapshotTail(&state.arguments, snapshot, "function_arguments", state.outputIndex)
	metadataChanged := state.name != "" && state.name != state.emittedName
	if tail == "" && state.announced && !metadataChanged && !forceAnnouncement {
		return nil
	}
	return t.toolCallChunk(state, tail)
}

func (t *openAIStreamTranslator) resolveToolState(raw, item map[string]interface{}) *openAIStreamToolState {
	aliases := openAIStreamToolAliases(raw, item)
	for _, alias := range aliases {
		if state := t.toolAliases[alias]; state != nil {
			t.updateToolState(state, raw, item, aliases)
			return state
		}
	}

	outputIndex, hasOutputIndex := openAIStreamOutputIndex(raw)
	if hasOutputIndex {
		if state := t.toolsByOutputIndex[outputIndex]; state != nil {
			t.updateToolState(state, raw, item, aliases)
			return state
		}
	}

	state := &openAIStreamToolState{
		index:       t.nextToolIndex,
		outputIndex: outputIndex,
	}
	t.nextToolIndex++
	t.updateToolState(state, raw, item, aliases)
	return state
}

func (t *openAIStreamTranslator) updateToolState(state *openAIStreamToolState, raw, item map[string]interface{}, aliases []string) {
	if outputIndex, ok := openAIStreamOutputIndex(raw); ok {
		state.outputIndex = outputIndex
		t.toolsByOutputIndex[outputIndex] = state
	}

	if itemID := strings.TrimSpace(interfaceToString(raw["item_id"])); itemID != "" {
		state.itemID = itemID
	}
	if item != nil {
		if itemID := strings.TrimSpace(interfaceToString(item["id"])); itemID != "" {
			state.itemID = itemID
		}
		if callID := strings.TrimSpace(interfaceToString(item["call_id"])); callID != "" {
			state.callID = callID
		}
	}
	if name := openAIStreamToolName(raw, item); name != "" {
		state.name = name
	}

	if !state.announced {
		if state.callID != "" {
			state.stableID = state.callID
		} else if state.itemID != "" {
			state.stableID = openAIStreamCallIDFromItemID(state.itemID)
		}
	}
	if state.stableID == "" {
		state.stableID = fmt.Sprintf("call_%s_%d", t.id, state.index)
	}

	for _, alias := range aliases {
		if alias != "" {
			t.toolAliases[alias] = state
		}
	}
	if state.itemID != "" {
		t.toolAliases[state.itemID] = state
	}
	if state.callID != "" {
		t.toolAliases[state.callID] = state
	}
}

func (t *openAIStreamTranslator) toolCallChunk(state *openAIStreamToolState, arguments string) *models.StreamChunk {
	idx := state.index
	state.announced = true
	state.emittedName = state.name
	return t.messageChunk(&models.Message{ToolCalls: []models.ToolCall{{
		Index: &idx,
		ID:    state.stableID,
		Type:  "function",
		Function: models.ToolCallFunction{
			Name:      state.name,
			Arguments: arguments,
		},
	}}})
}

func openAIStreamToolAliases(raw, item map[string]interface{}) []string {
	aliases := make([]string, 0, 3)
	if itemID := strings.TrimSpace(interfaceToString(raw["item_id"])); itemID != "" {
		aliases = append(aliases, itemID)
	}
	if item != nil {
		if itemID := strings.TrimSpace(interfaceToString(item["id"])); itemID != "" {
			aliases = append(aliases, itemID)
		}
		if callID := strings.TrimSpace(interfaceToString(item["call_id"])); callID != "" {
			aliases = append(aliases, callID)
		}
	}
	return aliases
}

func openAIStreamToolName(raw, item map[string]interface{}) string {
	if name := strings.TrimSpace(interfaceToString(raw["name"])); name != "" {
		return name
	}
	if item != nil {
		return strings.TrimSpace(interfaceToString(item["name"]))
	}
	return ""
}

func openAIStreamOutputIndex(raw map[string]interface{}) (int, bool) {
	if raw == nil {
		return 0, false
	}
	value, ok := raw["output_index"]
	if !ok || value == nil {
		return 0, false
	}
	return intFromAny(value), true
}

func openAIStreamCallIDFromItemID(itemID string) string {
	itemID = strings.TrimSpace(itemID)
	if strings.HasPrefix(itemID, "fc_") {
		return "call_" + strings.TrimPrefix(itemID, "fc_")
	}
	return itemID
}

func openAIStreamStringSnapshotTail(current *string, snapshot, kind string, outputIndex int) string {
	if current == nil || snapshot == "" {
		return ""
	}
	if *current == "" {
		*current = snapshot
		return snapshot
	}
	if strings.HasPrefix(snapshot, *current) {
		tail := strings.TrimPrefix(snapshot, *current)
		*current = snapshot
		return tail
	}

	log.Debug().
		Str("responses_part_kind", kind).
		Int("output_index", outputIndex).
		Int("streamed_len", len(*current)).
		Int("snapshot_len", len(snapshot)).
		Msg("responses stream snapshot diverged from emitted deltas")
	return ""
}
