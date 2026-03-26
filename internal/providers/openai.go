package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/pkg/models"
	"github.com/rs/zerolog/log"
)

func splitThinkTags(s string) (reasoning string, content string, changed bool) {
	startTag := "<think>"
	endTag := "</think>"

	content = s
	var r strings.Builder
	for {
		start := strings.Index(content, startTag)
		if start < 0 {
			break
		}
		end := strings.Index(content[start+len(startTag):], endTag)
		if end < 0 {
			break
		}
		end = start + len(startTag) + end

		inner := content[start+len(startTag) : end]
		inner = strings.TrimSpace(inner)
		if inner != "" {
			if r.Len() > 0 {
				r.WriteString("\n")
			}
			if inner != "" {
				r.WriteString(inner)
			}
		}

		content = content[:start] + content[end+len(endTag):]
		changed = true
	}

	if changed {
		reasoning = strings.TrimSpace(r.String())
		content = strings.TrimSpace(content)
	}

	return reasoning, content, changed
}

// OpenAITranslator handles translation for the OpenAI API.
// Since our unified format IS the OpenAI format, this is mostly pass-through.
type OpenAITranslator struct {
	cfg config.ProviderConfig
}

func NewOpenAITranslator(cfg config.ProviderConfig) *OpenAITranslator {
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.openai.com/v1"
	}
	if cfg.DefaultModel == "" {
		cfg.DefaultModel = "gpt-4-turbo"
	}
	return &OpenAITranslator{cfg: cfg}
}

func (t *OpenAITranslator) Name() string {
	return "openai"
}

func (t *OpenAITranslator) DefaultModel() string {
	return t.cfg.DefaultModel
}

func (t *OpenAITranslator) BaseURL() string {
	return strings.TrimRight(strings.TrimSpace(t.cfg.BaseURL), "/")
}

func (t *OpenAITranslator) TranslateRequest(ctx context.Context, req *models.UnifiedRequest) (*http.Request, error) {
	reqCopy := *req
	if reqCopy.Stream {
		if reqCopy.StreamOptions == nil {
			reqCopy.StreamOptions = &models.StreamOptions{}
		}
		reqCopy.StreamOptions.IncludeUsage = true
	}

	upstreamRequestType := strings.TrimSpace(UpstreamRequestTypeFromContext(ctx))
	bodyPayload := interface{}(&reqCopy)
	endpoint := fmt.Sprintf("%s/chat/completions", t.cfg.BaseURL)
	if strings.EqualFold(upstreamRequestType, "responses") {
		endpoint = fmt.Sprintf("%s/responses", t.cfg.BaseURL)
		bodyPayload = unifiedToResponsesPayload(&reqCopy)
	}

	body, err := json.Marshal(bodyPayload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal openai request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create openai http request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+t.cfg.APIKey)
	if t.cfg.Organization != "" {
		httpReq.Header.Set("OpenAI-Organization", t.cfg.Organization)
	}

	return httpReq, nil
}

func (t *OpenAITranslator) TranslateEmbeddingsRequest(ctx context.Context, req *models.EmbeddingsRequest) (*http.Request, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal openai embeddings request: %w", err)
	}

	endpoint := fmt.Sprintf("%s/embeddings", t.cfg.BaseURL)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create openai embeddings http request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+t.cfg.APIKey)
	if t.cfg.Organization != "" {
		httpReq.Header.Set("OpenAI-Organization", t.cfg.Organization)
	}

	return httpReq, nil
}

func (t *OpenAITranslator) ParseResponse(resp *http.Response) (*models.UnifiedResponse, error) {
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read openai response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		var errResp models.ErrorResponse
		if jsonErr := json.Unmarshal(body, &errResp); jsonErr == nil {
			return nil, &ProviderError{
				StatusCode: resp.StatusCode,
				Message:    errResp.Error.Message,
				Type:       errResp.Error.Type,
				Provider:   "openai",
			}
		}
		return nil, &ProviderError{
			StatusCode: resp.StatusCode,
			Message:    string(body),
			Provider:   "openai",
		}
	}

	var envelope struct {
		Object string `json:"object"`
	}
	_ = json.Unmarshal(body, &envelope)
	if strings.EqualFold(strings.TrimSpace(envelope.Object), "response") {
		var responsesResp models.ResponsesResponse
		if err := json.Unmarshal(body, &responsesResp); err != nil {
			return nil, fmt.Errorf("failed to unmarshal openai responses object: %w", err)
		}
		return responsesResponseToUnified(&responsesResp), nil
	}

	var result models.UnifiedResponse
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal openai response: %w", err)
	}

	for i := range result.Choices {
		c := &result.Choices[i]
		if c.Message == nil {
			continue
		}
		contentStr, ok := c.Message.Content.(string)
		if !ok || strings.Index(contentStr, "<think>") < 0 {
			continue
		}
		reasoning, cleaned, changed := splitThinkTags(contentStr)
		if !changed {
			continue
		}
		if reasoning != "" {
			if strings.TrimSpace(c.Message.ReasoningContent) == "" {
				c.Message.ReasoningContent = reasoning
			} else {
				c.Message.ReasoningContent = strings.TrimSpace(c.Message.ReasoningContent) + "\n" + reasoning
			}
		}
		c.Message.Content = cleaned
	}

	return &result, nil
}

func (t *OpenAITranslator) ParseEmbeddingsResponse(resp *http.Response) (*models.EmbeddingsResponse, error) {
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read openai embeddings response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		var errResp models.ErrorResponse
		if jsonErr := json.Unmarshal(body, &errResp); jsonErr == nil {
			return nil, &ProviderError{
				StatusCode: resp.StatusCode,
				Message:    errResp.Error.Message,
				Type:       errResp.Error.Type,
				Provider:   "openai",
			}
		}
		return nil, &ProviderError{
			StatusCode: resp.StatusCode,
			Message:    string(body),
			Provider:   "openai",
		}
	}

	var result models.EmbeddingsResponse
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal openai embeddings response: %w", err)
	}

	return &result, nil
}

func (t *OpenAITranslator) ParseStreamChunk(data []byte) (*models.StreamChunk, error) {
	trimmed := bytes.TrimSpace(data)

	if len(trimmed) == 0 {
		return nil, nil
	}

	if string(trimmed) == "[DONE]" {
		return nil, ErrStreamDone
	}

	var eventEnvelope struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(trimmed, &eventEnvelope); err == nil && strings.HasPrefix(strings.TrimSpace(eventEnvelope.Type), "response.") {
		return responsesEventToStreamChunk(trimmed)
	}

	var chunk models.StreamChunk
	if err := json.Unmarshal(trimmed, &chunk); err != nil {
		return nil, fmt.Errorf("failed to unmarshal openai stream chunk: %w", err)
	}

	for i := range chunk.Choices {
		c := &chunk.Choices[i]
		if c.Delta == nil {
			continue
		}
		contentStr, ok := c.Delta.Content.(string)
		if !ok {
			continue
		}
		if strings.Index(contentStr, "<think>") < 0 || strings.Index(contentStr, "</think>") < 0 {
			continue
		}
		reasoning, cleaned, changed := splitThinkTags(contentStr)
		if !changed {
			continue
		}
		if reasoning != "" {
			if strings.TrimSpace(c.Delta.ReasoningContent) == "" {
				c.Delta.ReasoningContent = reasoning
			} else {
				c.Delta.ReasoningContent = strings.TrimSpace(c.Delta.ReasoningContent) + "\n" + reasoning
			}
		}
		c.Delta.Content = cleaned
	}

	return &chunk, nil
}

func unifiedToResponsesPayload(req *models.UnifiedRequest) *models.ResponsesRequest {
	input := make([]interface{}, 0, len(req.Messages))
	instructions := make([]string, 0, 1)

	for i := range req.Messages {
		msg := req.Messages[i]
		if strings.EqualFold(strings.TrimSpace(msg.Role), "system") {
			if s := strings.TrimSpace(openaiMessageContentToString(msg.Content)); s != "" {
				instructions = append(instructions, s)
			}
			continue
		}

		if strings.EqualFold(strings.TrimSpace(msg.Role), "assistant") && len(msg.ToolCalls) > 0 {
			for _, tc := range msg.ToolCalls {
				callID := strings.TrimSpace(tc.ID)
				if callID == "" {
					callID = "call_" + strings.TrimSpace(tc.Function.Name)
				}
				if callID == "call_" {
					callID = "call_lunargate"
				}
				itemID := responsesFunctionItemID(callID)
				input = append(input, map[string]interface{}{
					"type":      "function_call",
					"id":        itemID,
					"call_id":   callID,
					"name":      strings.TrimSpace(tc.Function.Name),
					"arguments": tc.Function.Arguments,
				})
			}
		}

		if strings.EqualFold(strings.TrimSpace(msg.Role), "tool") {
			callID := strings.TrimSpace(msg.ToolCallID)
			if callID == "" {
				callID = strings.TrimSpace(msg.Name)
			}
			if callID == "" {
				callID = "tool_call"
			}
			input = append(input, map[string]interface{}{
				"type":    "function_call_output",
				"call_id": callID,
				"output":  msg.Content,
			})
			continue
		}

		if msg.Content == nil {
			continue
		}
		normalizedContent, ok := normalizeResponsesMessageContent(msg.Role, msg.Content)
		if !ok {
			continue
		}
		input = append(input, map[string]interface{}{
			"role":    msg.Role,
			"content": normalizedContent,
		})
	}

	out := &models.ResponsesRequest{
		Model:       req.Model,
		Input:       input,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Tools:       make([]models.ResponsesTool, 0, len(req.Tools)),
		ToolChoice:  normalizeResponsesToolChoiceForUpstream(req.ToolChoice),
		Stream:      req.Stream,
		User:        req.User,
	}
	if len(instructions) > 0 {
		out.Instructions = strings.Join(instructions, "\n")
	}
	if req.MaxTokens != nil {
		out.MaxOutputTokens = req.MaxTokens
	}
	for _, tool := range req.Tools {
		fn := tool.Function
		out.Tools = append(out.Tools, models.ResponsesTool{
			Type:        "function",
			Name:        fn.Name,
			Description: fn.Description,
			Parameters:  fn.Parameters,
		})
	}
	if len(out.Tools) == 0 {
		out.Tools = nil
	}

	return out
}

func responsesResponseToUnified(resp *models.ResponsesResponse) *models.UnifiedResponse {
	if resp == nil {
		return nil
	}

	message := &models.Message{Role: "assistant"}
	if strings.TrimSpace(resp.OutputText) != "" {
		message.Content = resp.OutputText
	}

	toolCalls := make([]models.ToolCall, 0)
	for i := range resp.Output {
		item := resp.Output[i]
		if item.Type != "function_call" {
			continue
		}
		idx := i
		callID := strings.TrimSpace(item.CallID)
		if callID == "" {
			callID = strings.TrimSpace(item.ID)
		}
		toolCalls = append(toolCalls, models.ToolCall{
			Index: &idx,
			ID:    callID,
			Type:  "function",
			Function: models.ToolCallFunction{
				Name:      item.Name,
				Arguments: item.Arguments,
			},
		})
	}
	if len(toolCalls) > 0 {
		message.ToolCalls = toolCalls
		if message.Content == nil {
			message.Content = ""
		}
	}

	out := &models.UnifiedResponse{
		ID:      resp.ID,
		Object:  "chat.completion",
		Created: resp.CreatedAt,
		Model:   resp.Model,
		Choices: []models.Choice{{
			Index:   0,
			Message: message,
		}},
	}
	if resp.Usage != nil {
		out.Usage = &models.Usage{
			PromptTokens:     resp.Usage.InputTokens,
			CompletionTokens: resp.Usage.OutputTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		}
	}

	return out
}

func responsesEventToStreamChunk(data []byte) (*models.StreamChunk, error) {
	var raw map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("failed to unmarshal responses stream event: %w", err)
	}

	responseID := responsesEventResponseID(raw)
	responseModel, responseCreated := responsesEventResponseMeta(raw)
	typeName := strings.TrimSpace(interfaceToString(raw["type"]))
	switch typeName {
	case "response.created", "response.in_progress":
		if responseID == "" {
			logIgnoredResponsesEvent(typeName, raw)
			return nil, nil
		}
		return &models.StreamChunk{
			ID:      responseID,
			Object:  "chat.completion.chunk",
			Created: responseCreated,
			Model:   responseModel,
			Choices: []models.Choice{},
		}, nil
	case "response.completed", "response.done":
		log.Debug().
			Str("provider", "openai").
			Str("responses_event_type", typeName).
			Msg("responses stream completed event")
		return nil, ErrStreamDone
	case "response.output_text.delta":
		delta := interfaceToString(raw["delta"])
		if delta == "" {
			return nil, nil
		}
		return &models.StreamChunk{
			ID:     responseID,
			Object: "chat.completion.chunk",
			Choices: []models.Choice{{
				Index: 0,
				Delta: &models.Message{Content: delta},
			}},
		}, nil
	case "response.output_text.done":
		// Some responses streams emit only *.done text without prior deltas.
		text := interfaceToString(raw["text"])
		if text == "" {
			return nil, nil
		}
		return &models.StreamChunk{
			ID:     responseID,
			Object: "chat.completion.chunk",
			Choices: []models.Choice{{
				Index: 0,
				Delta: &models.Message{Content: text},
			}},
		}, nil
	case "response.content_part.done":
		part, _ := raw["part"].(map[string]interface{})
		if part == nil {
			return nil, nil
		}
		partType := strings.TrimSpace(interfaceToString(part["type"]))
		partText := interfaceToString(part["text"])
		if partText == "" {
			return nil, nil
		}
		switch partType {
		case "output_text", "text":
			return &models.StreamChunk{
				ID:     responseID,
				Object: "chat.completion.chunk",
				Choices: []models.Choice{{
					Index: 0,
					Delta: &models.Message{Content: partText},
				}},
			}, nil
		case "reasoning", "reasoning_text", "reasoning_summary_text":
			return &models.StreamChunk{
				ID:     responseID,
				Object: "chat.completion.chunk",
				Choices: []models.Choice{{
					Index: 0,
					Delta: &models.Message{ReasoningContent: partText},
				}},
			}, nil
		default:
			return nil, nil
		}
	case "response.reasoning_summary_text.delta", "response.reasoning_summary_text.done":
		text := interfaceToString(raw["delta"])
		if text == "" {
			text = interfaceToString(raw["text"])
		}
		if text == "" {
			return nil, nil
		}
		return &models.StreamChunk{
			ID:     responseID,
			Object: "chat.completion.chunk",
			Choices: []models.Choice{{
				Index: 0,
				Delta: &models.Message{ReasoningContent: text},
			}},
		}, nil
	case "response.function_call_arguments.delta":
		delta := interfaceToString(raw["delta"])
		if delta == "" {
			return nil, nil
		}
		id := strings.TrimSpace(interfaceToString(raw["item_id"]))
		idx := intFromAny(raw["output_index"])
		log.Debug().
			Str("provider", "openai").
			Str("responses_event_type", typeName).
			Str("item_id", id).
			Int("output_index", idx).
			Int("delta_len", len(delta)).
			Msg("responses stream function arguments delta")
		return &models.StreamChunk{
			ID:     responseID,
			Object: "chat.completion.chunk",
			Choices: []models.Choice{{
				Index: 0,
				Delta: &models.Message{ToolCalls: []models.ToolCall{{
					Index: &idx,
					ID:    id,
					Type:  "function",
					Function: models.ToolCallFunction{
						Arguments: delta,
					},
				}}},
			}},
		}, nil
	case "response.output_item.added", "response.output_item.done":
		item, _ := raw["item"].(map[string]interface{})
		itemType := strings.TrimSpace(interfaceToString(func() interface{} {
			if item == nil {
				return nil
			}
			return item["type"]
		}()))
		if item != nil && itemType == "message" && typeName == "response.output_item.done" {
			// Fallback for streams that only deliver final assistant content via item.content.
			text := openaiMessageContentToString(item["content"])
			if text == "" {
				return nil, nil
			}
			return &models.StreamChunk{
				ID:     responseID,
				Object: "chat.completion.chunk",
				Choices: []models.Choice{{
					Index: 0,
					Delta: &models.Message{Content: text},
				}},
			}, nil
		}
		if item == nil || itemType != "function_call" {
			logIgnoredResponsesEvent(typeName, raw)
			return nil, nil
		}
		id := strings.TrimSpace(interfaceToString(item["call_id"]))
		if id == "" {
			id = strings.TrimSpace(interfaceToString(item["id"]))
		}
		name := strings.TrimSpace(interfaceToString(item["name"]))
		args := interfaceToString(item["arguments"])
		if typeName == "response.output_item.done" {
			// Arguments are streamed via response.function_call_arguments.delta/done.
			// Passing the full payload here would duplicate accumulated arguments.
			args = ""
		}
		idx := intFromAny(raw["output_index"])
		log.Debug().
			Str("provider", "openai").
			Str("responses_event_type", typeName).
			Str("item_id", id).
			Str("tool_name", name).
			Int("output_index", idx).
			Int("arguments_len", len(args)).
			Msg("responses stream function item event")
		return &models.StreamChunk{
			ID:     responseID,
			Object: "chat.completion.chunk",
			Choices: []models.Choice{{
				Index: 0,
				Delta: &models.Message{ToolCalls: []models.ToolCall{{
					Index: &idx,
					ID:    id,
					Type:  "function",
					Function: models.ToolCallFunction{
						Name:      name,
						Arguments: args,
					},
				}}},
			}},
		}, nil
	default:
		logIgnoredResponsesEvent(typeName, raw)
		return nil, nil
	}
}

func logIgnoredResponsesEvent(typeName string, raw map[string]interface{}) {
	item, _ := raw["item"].(map[string]interface{})
	part, _ := raw["part"].(map[string]interface{})
	outputIndex := intFromAny(raw["output_index"])
	contentIndex := intFromAny(raw["content_index"])

	itemID := ""
	itemType := ""
	itemCallID := ""
	if item != nil {
		itemID = strings.TrimSpace(interfaceToString(item["id"]))
		itemType = strings.TrimSpace(interfaceToString(item["type"]))
		itemCallID = strings.TrimSpace(interfaceToString(item["call_id"]))
	}

	partType := ""
	partTextLen := 0
	if part != nil {
		partType = strings.TrimSpace(interfaceToString(part["type"]))
		partTextLen = len(interfaceToString(part["text"]))
	}

	log.Debug().
		Str("provider", "openai").
		Str("responses_event_type", typeName).
		Str("response_id", strings.TrimSpace(interfaceToString(raw["response_id"]))).
		Str("item_id", strings.TrimSpace(interfaceToString(raw["item_id"]))).
		Int("output_index", outputIndex).
		Int("content_index", contentIndex).
		Str("item_type", itemType).
		Str("item_id_embedded", itemID).
		Str("item_call_id", itemCallID).
		Str("part_type", partType).
		Int("part_text_len", partTextLen).
		Strs("raw_keys", mapKeys(raw)).
		Msg("responses stream event ignored by translator")
}

func mapKeys(m map[string]interface{}) []string {
	if len(m) == 0 {
		return nil
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func responsesEventResponseID(raw map[string]interface{}) string {
	if id := strings.TrimSpace(interfaceToString(raw["response_id"])); id != "" {
		return id
	}
	resp, _ := raw["response"].(map[string]interface{})
	if resp != nil {
		return strings.TrimSpace(interfaceToString(resp["id"]))
	}
	return ""
}

func responsesEventResponseMeta(raw map[string]interface{}) (string, int64) {
	resp, _ := raw["response"].(map[string]interface{})
	if resp == nil {
		return "", 0
	}
	model := strings.TrimSpace(interfaceToString(resp["model"]))
	created := int64(intFromAny(resp["created_at"]))
	return model, created
}

func openaiMessageContentToString(content interface{}) string {
	switch v := content.(type) {
	case string:
		return v
	case []interface{}:
		var b strings.Builder
		for _, part := range v {
			m, ok := part.(map[string]interface{})
			if !ok {
				continue
			}
			t := strings.TrimSpace(interfaceToString(m["type"]))
			if t != "" && t != "text" && t != "input_text" && t != "output_text" {
				continue
			}
			txt := interfaceToString(m["text"])
			if strings.TrimSpace(txt) == "" {
				continue
			}
			b.WriteString(txt)
		}
		return b.String()
	default:
		return ""
	}
}

func normalizeResponsesMessageContent(role string, content interface{}) (interface{}, bool) {
	partType := responsesTextPartTypeForRole(role)
	if s, ok := content.(string); ok {
		s = strings.TrimSpace(s)
		if s == "" {
			return nil, false
		}
		return []map[string]interface{}{{"type": partType, "text": s}}, true
	}

	parts, ok := content.([]interface{})
	if !ok {
		return content, true
	}
	normalized := make([]map[string]interface{}, 0, len(parts))
	for _, part := range parts {
		obj, ok := part.(map[string]interface{})
		if !ok {
			continue
		}
		copyObj := make(map[string]interface{}, len(obj))
		for k, v := range obj {
			copyObj[k] = v
		}
		t := strings.TrimSpace(interfaceToString(copyObj["type"]))
		if t == "" || t == "text" || t == "input_text" || t == "output_text" {
			copyObj["type"] = partType
		}
		normalized = append(normalized, copyObj)
	}
	if len(normalized) == 0 {
		return nil, false
	}
	return normalized, true
}

func responsesTextPartTypeForRole(role string) string {
	switch strings.ToLower(strings.TrimSpace(role)) {
	case "assistant":
		return "output_text"
	default:
		return "input_text"
	}
}

func normalizeResponsesToolChoiceForUpstream(choice interface{}) interface{} {
	if choice == nil {
		return nil
	}
	switch v := choice.(type) {
	case string:
		return strings.TrimSpace(v)
	case map[string]interface{}:
		t := strings.TrimSpace(interfaceToString(v["type"]))
		if t != "function" {
			return v
		}
		if name := strings.TrimSpace(interfaceToString(v["name"])); name != "" {
			return map[string]interface{}{"type": "function", "name": name}
		}
		if fn, ok := v["function"].(map[string]interface{}); ok {
			if name := strings.TrimSpace(interfaceToString(fn["name"])); name != "" {
				return map[string]interface{}{"type": "function", "name": name}
			}
		}
		return v
	default:
		b, err := json.Marshal(choice)
		if err != nil {
			return choice
		}
		var obj map[string]interface{}
		if err := json.Unmarshal(b, &obj); err != nil {
			return choice
		}
		return normalizeResponsesToolChoiceForUpstream(obj)
	}
}

func intFromAny(v interface{}) int {
	switch n := v.(type) {
	case float64:
		return int(n)
	case int:
		return n
	case int64:
		return int(n)
	default:
		return 0
	}
}

func interfaceToString(v interface{}) string {
	s, _ := v.(string)
	return s
}

func responsesFunctionItemID(callID string) string {
	trimmed := strings.TrimSpace(callID)
	if trimmed == "" {
		return "fc_lunargate"
	}
	if strings.HasPrefix(trimmed, "fc") {
		return trimmed
	}
	if strings.HasPrefix(trimmed, "call_") {
		return "fc_" + strings.TrimPrefix(trimmed, "call_")
	}
	return "fc_" + trimmed
}

func (t *OpenAITranslator) SupportsStreaming() bool {
	return true
}

func (t *OpenAITranslator) Models() []models.ModelInfo {
	return []models.ModelInfo{
		{ID: "gpt-4-turbo", Object: "model", Created: time.Now().Unix(), OwnedBy: "openai"},
		{ID: "gpt-4", Object: "model", Created: time.Now().Unix(), OwnedBy: "openai"},
		{ID: "gpt-4o", Object: "model", Created: time.Now().Unix(), OwnedBy: "openai"},
		{ID: "gpt-4o-mini", Object: "model", Created: time.Now().Unix(), OwnedBy: "openai"},
		{ID: "gpt-3.5-turbo", Object: "model", Created: time.Now().Unix(), OwnedBy: "openai"},
		{ID: "text-embedding-3-small", Object: "model", Created: time.Now().Unix(), OwnedBy: "openai"},
		{ID: "text-embedding-3-large", Object: "model", Created: time.Now().Unix(), OwnedBy: "openai"},
		{ID: "text-embedding-ada-002", Object: "model", Created: time.Now().Unix(), OwnedBy: "openai"},
	}
}
