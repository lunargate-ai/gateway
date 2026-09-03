package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/lunargate-ai/gateway/pkg/models"
	"github.com/rs/zerolog/log"
)

const (
	openAIStreamStateMaxBytes     = 16 << 20
	openAIStreamMaxParts          = 128
	openAIStreamMaxTools          = 128
	openAIStreamMaxAliasesPerTool = 4
	openAIStreamMaxAliases        = openAIStreamMaxTools * openAIStreamMaxAliasesPerTool
)

var (
	errOpenAIStreamStateTooLarge  = errors.New("OpenAI Responses stream translation state exceeds 16 MiB limit")
	errOpenAIStreamTooManyParts   = errors.New("OpenAI Responses stream exceeds 128 content parts")
	errOpenAIStreamTooManyTools   = errors.New("OpenAI Responses stream exceeds 128 tool calls")
	errOpenAIStreamTooManyAliases = errors.New("OpenAI Responses stream tool aliases exceed limit")
	errOpenAIStreamAliasConflict  = errors.New("OpenAI Responses stream tool alias is ambiguous")
)

// openAIStreamTranslator keeps Responses API snapshot events from being
// exposed as duplicate Chat Completions deltas. A translator is created for
// each upstream stream, so all state below is request-local.
type openAIStreamTranslator struct {
	base *OpenAITranslator

	id      string
	model   string
	created int64

	textParts      map[openAIStreamPartKey]*openAIStreamPartState
	reasoningParts map[openAIStreamPartKey]*openAIStreamPartState

	toolAliases        map[string]*openAIStreamToolState
	toolsByOutputIndex map[int]*openAIStreamToolState
	nextToolIndex      int
	partCount          int
	toolCount          int
	stateBytes         int
}

type openAIStreamPartKey struct {
	outputIndex  int
	contentIndex int
	kind         string
}

type openAIStreamPartState struct {
	content strings.Builder
}

type openAIStreamToolState struct {
	index       int
	outputIndex int
	itemID      string
	callID      string
	stableID    string
	name        string
	arguments   strings.Builder
	announced   bool
	emittedName string
	aliases     map[string]struct{}
}

func NewOpenAIStreamTranslator(base *OpenAITranslator) models.ProviderTranslator {
	return &openAIStreamTranslator{
		base:               base,
		textParts:          make(map[openAIStreamPartKey]*openAIStreamPartState, 4),
		reasoningParts:     make(map[openAIStreamPartKey]*openAIStreamPartState, 4),
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

	if err := t.updateResponseMetadata(raw); err != nil {
		return nil, err
	}

	switch typeName {
	case "response.output_text.delta":
		return t.textDeltaChunk(raw, interfaceToString(raw["delta"]))
	case "response.output_text.done":
		return t.textSnapshotChunk(raw, interfaceToString(raw["text"]))
	case "response.content_part.added", "response.content_part.done":
		return t.contentPartSnapshotChunk(raw)
	case "response.reasoning_summary_text.delta":
		return t.reasoningDeltaChunk(raw, "reasoning_summary", interfaceToString(raw["delta"]))
	case "response.reasoning_summary_text.done":
		return t.reasoningSnapshotChunk(raw, "reasoning_summary", interfaceToString(raw["text"]))
	case "response.reasoning_text.delta":
		return t.reasoningDeltaChunk(raw, "reasoning_text", interfaceToString(raw["delta"]))
	case "response.reasoning_text.done":
		return t.reasoningSnapshotChunk(raw, "reasoning_text", interfaceToString(raw["text"]))
	case "response.reasoning_summary_part.added", "response.reasoning_summary_part.done":
		return t.reasoningPartSnapshotChunk(raw)
	case "response.function_call_arguments.delta":
		return t.functionArgumentsDeltaChunk(raw)
	case "response.function_call_arguments.done":
		return t.functionArgumentsSnapshotChunk(raw, nil, false)
	case "response.output_item.added":
		return t.outputItemChunk(raw, true)
	case "response.output_item.done":
		return t.outputItemChunk(raw, false)
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

func (t *openAIStreamTranslator) updateResponseMetadata(raw map[string]interface{}) error {
	if responseID := responsesEventResponseID(raw); responseID != "" {
		if err := t.replaceStateString(&t.id, responseID); err != nil {
			return err
		}
	}
	model, created := responsesEventResponseMeta(raw)
	if model != "" {
		if err := t.replaceStateString(&t.model, model); err != nil {
			return err
		}
	}
	if created != 0 {
		t.created = created
	}
	return nil
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

func (t *openAIStreamTranslator) textDeltaChunk(raw map[string]interface{}, delta string) (*models.StreamChunk, error) {
	if delta == "" {
		return nil, nil
	}
	key := openAIStreamPartPosition(raw, "output_text")
	state, err := t.resolvePartState(t.textParts, key)
	if err != nil {
		return nil, err
	}
	if err := t.appendStateString(&state.content, delta); err != nil {
		return nil, err
	}
	return t.messageChunk(&models.Message{Content: delta}), nil
}

func (t *openAIStreamTranslator) textSnapshotChunk(raw map[string]interface{}, snapshot string) (*models.StreamChunk, error) {
	if snapshot == "" {
		return nil, nil
	}
	key := openAIStreamPartPosition(raw, "output_text")
	tail, err := t.snapshotTail(t.textParts, key, snapshot)
	if err != nil {
		return nil, err
	}
	if tail == "" {
		return nil, nil
	}
	return t.messageChunk(&models.Message{Content: tail}), nil
}

func (t *openAIStreamTranslator) reasoningDeltaChunk(raw map[string]interface{}, kind, delta string) (*models.StreamChunk, error) {
	if delta == "" {
		return nil, nil
	}
	key := openAIStreamPartPosition(raw, kind)
	state, err := t.resolvePartState(t.reasoningParts, key)
	if err != nil {
		return nil, err
	}
	if err := t.appendStateString(&state.content, delta); err != nil {
		return nil, err
	}
	return t.messageChunk(&models.Message{ReasoningContent: delta}), nil
}

func (t *openAIStreamTranslator) reasoningSnapshotChunk(raw map[string]interface{}, kind, snapshot string) (*models.StreamChunk, error) {
	if snapshot == "" {
		return nil, nil
	}
	key := openAIStreamPartPosition(raw, kind)
	tail, err := t.snapshotTail(t.reasoningParts, key, snapshot)
	if err != nil {
		return nil, err
	}
	if tail == "" {
		return nil, nil
	}
	return t.messageChunk(&models.Message{ReasoningContent: tail}), nil
}

func (t *openAIStreamTranslator) snapshotTail(parts map[openAIStreamPartKey]*openAIStreamPartState, key openAIStreamPartKey, snapshot string) (string, error) {
	state, err := t.resolvePartState(parts, key)
	if err != nil {
		return "", err
	}
	current := state.content.String()
	if current == "" {
		if err := t.appendStateString(&state.content, snapshot); err != nil {
			return "", err
		}
		return snapshot, nil
	}
	if strings.HasPrefix(snapshot, current) {
		tail := snapshot[len(current):]
		if err := t.appendStateString(&state.content, tail); err != nil {
			return "", err
		}
		return tail, nil
	}

	log.Debug().
		Str("responses_part_kind", key.kind).
		Int("output_index", key.outputIndex).
		Int("content_index", key.contentIndex).
		Int("streamed_len", len(current)).
		Int("snapshot_len", len(snapshot)).
		Msg("responses stream snapshot diverged from emitted deltas")
	return "", nil
}

func (t *openAIStreamTranslator) resolvePartState(parts map[openAIStreamPartKey]*openAIStreamPartState, key openAIStreamPartKey) (*openAIStreamPartState, error) {
	if state := parts[key]; state != nil {
		return state, nil
	}
	if t.partCount >= openAIStreamMaxParts {
		return nil, errOpenAIStreamTooManyParts
	}
	state := &openAIStreamPartState{}
	parts[key] = state
	t.partCount++
	return state, nil
}

func (t *openAIStreamTranslator) reserveStateBytes(size int) error {
	if size <= 0 {
		return nil
	}
	if size > openAIStreamStateMaxBytes-t.stateBytes {
		return errOpenAIStreamStateTooLarge
	}
	t.stateBytes += size
	return nil
}

func (t *openAIStreamTranslator) appendStateString(dst *strings.Builder, value string) error {
	if dst == nil || value == "" {
		return nil
	}
	if err := t.reserveStateBytes(len(value)); err != nil {
		return err
	}
	_, _ = dst.WriteString(value)
	return nil
}

func (t *openAIStreamTranslator) replaceStateString(dst *string, value string) error {
	if dst == nil || *dst == value {
		return nil
	}
	delta := len(value) - len(*dst)
	if delta > 0 {
		if err := t.reserveStateBytes(delta); err != nil {
			return err
		}
	} else {
		t.stateBytes += delta
	}
	*dst = value
	return nil
}

func (t *openAIStreamTranslator) contentPartSnapshotChunk(raw map[string]interface{}) (*models.StreamChunk, error) {
	part, _ := raw["part"].(map[string]interface{})
	if part == nil {
		return nil, nil
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
		return nil, nil
	}
}

func (t *openAIStreamTranslator) reasoningPartSnapshotChunk(raw map[string]interface{}) (*models.StreamChunk, error) {
	part, _ := raw["part"].(map[string]interface{})
	if part == nil {
		return nil, nil
	}
	return t.reasoningSnapshotChunk(raw, "reasoning_summary", interfaceToString(part["text"]))
}

func (t *openAIStreamTranslator) outputItemChunk(raw map[string]interface{}, added bool) (*models.StreamChunk, error) {
	item, _ := raw["item"].(map[string]interface{})
	if item == nil {
		return nil, nil
	}

	switch strings.TrimSpace(interfaceToString(item["type"])) {
	case "message":
		return t.messageOutputItemSnapshotChunk(raw, item)
	case "reasoning":
		return t.reasoningOutputItemSnapshotChunk(raw, item)
	case "function_call":
		return t.functionArgumentsSnapshotChunk(raw, item, added)
	default:
		return nil, nil
	}
}

func (t *openAIStreamTranslator) messageOutputItemSnapshotChunk(raw, item map[string]interface{}) (*models.StreamChunk, error) {
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
		tail, err := t.snapshotTail(t.textParts, key, interfaceToString(part["text"]))
		if err != nil {
			return nil, err
		}
		tails.WriteString(tail)
	}
	if tails.Len() == 0 {
		return nil, nil
	}
	return t.messageChunk(&models.Message{Content: tails.String()}), nil
}

func (t *openAIStreamTranslator) reasoningOutputItemSnapshotChunk(raw, item map[string]interface{}) (*models.StreamChunk, error) {
	var tails strings.Builder
	if summary, ok := item["summary"].([]interface{}); ok {
		for summaryIndex, rawPart := range summary {
			part, _ := rawPart.(map[string]interface{})
			if part == nil {
				continue
			}
			position := cloneOpenAIStreamEventPosition(raw, "summary_index", summaryIndex)
			key := openAIStreamPartPosition(position, "reasoning_summary")
			tail, err := t.snapshotTail(t.reasoningParts, key, interfaceToString(part["text"]))
			if err != nil {
				return nil, err
			}
			tails.WriteString(tail)
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
			tail, err := t.snapshotTail(t.reasoningParts, key, interfaceToString(part["text"]))
			if err != nil {
				return nil, err
			}
			tails.WriteString(tail)
		}
	}
	if tails.Len() == 0 {
		return nil, nil
	}
	return t.messageChunk(&models.Message{ReasoningContent: tails.String()}), nil
}

func cloneOpenAIStreamEventPosition(raw map[string]interface{}, key string, value int) map[string]interface{} {
	position := map[string]interface{}{
		"output_index": raw["output_index"],
		key:            value,
	}
	return position
}

func (t *openAIStreamTranslator) functionArgumentsDeltaChunk(raw map[string]interface{}) (*models.StreamChunk, error) {
	state, err := t.resolveToolState(raw, nil)
	if err != nil {
		return nil, err
	}
	delta := interfaceToString(raw["delta"])
	if delta == "" {
		return nil, nil
	}
	if err := t.appendStateString(&state.arguments, delta); err != nil {
		return nil, err
	}
	return t.toolCallChunk(state, delta), nil
}

func (t *openAIStreamTranslator) functionArgumentsSnapshotChunk(raw, item map[string]interface{}, forceAnnouncement bool) (*models.StreamChunk, error) {
	state, err := t.resolveToolState(raw, item)
	if err != nil {
		return nil, err
	}

	snapshot := interfaceToString(raw["arguments"])
	if item != nil {
		snapshot = interfaceToString(item["arguments"])
	}
	tail, err := t.toolArgumentsSnapshotTail(state, snapshot)
	if err != nil {
		return nil, err
	}
	metadataChanged := state.name != "" && state.name != state.emittedName
	if tail == "" && state.announced && !metadataChanged && !forceAnnouncement {
		return nil, nil
	}
	return t.toolCallChunk(state, tail), nil
}

func (t *openAIStreamTranslator) resolveToolState(raw, item map[string]interface{}) (*openAIStreamToolState, error) {
	aliases := openAIStreamToolAliases(raw, item)
	var resolved *openAIStreamToolState
	for _, alias := range aliases {
		if state := t.toolAliases[alias]; state != nil {
			if resolved != nil && resolved != state {
				return nil, fmt.Errorf("%w: aliases resolve to different tool calls", errOpenAIStreamAliasConflict)
			}
			resolved = state
		}
	}

	outputIndex, hasOutputIndex := openAIStreamOutputIndex(raw)
	if hasOutputIndex {
		if state := t.toolsByOutputIndex[outputIndex]; state != nil {
			if resolved != nil && resolved != state {
				return nil, fmt.Errorf("%w: output_index %d conflicts with an alias", errOpenAIStreamAliasConflict, outputIndex)
			}
			resolved = state
		}
	}
	if resolved != nil {
		if err := t.updateToolState(resolved, raw, item, aliases); err != nil {
			return nil, err
		}
		return resolved, nil
	}
	if t.toolCount >= openAIStreamMaxTools {
		return nil, errOpenAIStreamTooManyTools
	}

	state := &openAIStreamToolState{
		index:       t.nextToolIndex,
		outputIndex: outputIndex,
		aliases:     make(map[string]struct{}, 3),
	}
	t.nextToolIndex++
	t.toolCount++
	if err := t.updateToolState(state, raw, item, aliases); err != nil {
		return nil, err
	}
	return state, nil
}

func (t *openAIStreamTranslator) updateToolState(state *openAIStreamToolState, raw, item map[string]interface{}, aliases []string) error {
	if outputIndex, ok := openAIStreamOutputIndex(raw); ok {
		if existing := t.toolsByOutputIndex[outputIndex]; existing != nil && existing != state {
			return fmt.Errorf("%w: output_index %d identifies multiple tool calls", errOpenAIStreamAliasConflict, outputIndex)
		}
		state.outputIndex = outputIndex
		t.toolsByOutputIndex[outputIndex] = state
	}

	if itemID := strings.TrimSpace(interfaceToString(raw["item_id"])); itemID != "" {
		if err := t.replaceStateString(&state.itemID, itemID); err != nil {
			return err
		}
	}
	if item != nil {
		if itemID := strings.TrimSpace(interfaceToString(item["id"])); itemID != "" {
			if err := t.replaceStateString(&state.itemID, itemID); err != nil {
				return err
			}
		}
		if callID := strings.TrimSpace(interfaceToString(item["call_id"])); callID != "" {
			if err := t.replaceStateString(&state.callID, callID); err != nil {
				return err
			}
		}
	}
	if name := openAIStreamToolName(raw, item); name != "" {
		if err := t.replaceStateString(&state.name, name); err != nil {
			return err
		}
	}

	if !state.announced {
		if state.callID != "" {
			if err := t.replaceStateString(&state.stableID, state.callID); err != nil {
				return err
			}
		} else if state.itemID != "" {
			if err := t.replaceStateString(&state.stableID, openAIStreamCallIDFromItemID(state.itemID)); err != nil {
				return err
			}
		}
	}
	if state.stableID == "" {
		if err := t.replaceStateString(&state.stableID, fmt.Sprintf("call_%s_%d", t.id, state.index)); err != nil {
			return err
		}
	}

	for _, alias := range aliases {
		if err := t.addToolAlias(state, alias); err != nil {
			return err
		}
	}
	if err := t.addToolAlias(state, state.itemID); err != nil {
		return err
	}
	if err := t.addToolAlias(state, state.callID); err != nil {
		return err
	}
	return nil
}

func (t *openAIStreamTranslator) addToolAlias(state *openAIStreamToolState, alias string) error {
	alias = strings.TrimSpace(alias)
	if alias == "" {
		return nil
	}
	if existing := t.toolAliases[alias]; existing != nil {
		if existing != state {
			return fmt.Errorf("%w: alias %q identifies multiple tool calls", errOpenAIStreamAliasConflict, alias)
		}
		return nil
	}
	if len(state.aliases) >= openAIStreamMaxAliasesPerTool || len(t.toolAliases) >= openAIStreamMaxAliases {
		return errOpenAIStreamTooManyAliases
	}
	if err := t.reserveStateBytes(len(alias)); err != nil {
		return err
	}
	state.aliases[alias] = struct{}{}
	t.toolAliases[alias] = state
	return nil
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

func (t *openAIStreamTranslator) toolArgumentsSnapshotTail(state *openAIStreamToolState, snapshot string) (string, error) {
	if state == nil || snapshot == "" {
		return "", nil
	}
	current := state.arguments.String()
	if current == "" {
		if err := t.appendStateString(&state.arguments, snapshot); err != nil {
			return "", err
		}
		return snapshot, nil
	}
	if strings.HasPrefix(snapshot, current) {
		tail := snapshot[len(current):]
		if err := t.appendStateString(&state.arguments, tail); err != nil {
			return "", err
		}
		return tail, nil
	}

	log.Debug().
		Str("responses_part_kind", "function_arguments").
		Int("output_index", state.outputIndex).
		Int("streamed_len", len(current)).
		Int("snapshot_len", len(snapshot)).
		Msg("responses stream snapshot diverged from emitted deltas")
	return "", nil
}
