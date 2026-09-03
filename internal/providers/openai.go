package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/safeurl"
	"github.com/lunargate-ai/gateway/pkg/models"
	"github.com/rs/zerolog/log"
)

const openAIDefaultModel = "gpt-5.6-terra"

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
		if inner != "" {
			if r.Len() > 0 {
				r.WriteString("\n")
			}
			r.WriteString(inner)
		}

		content = content[:start] + content[end+len(endTag):]
		changed = true
	}

	if changed {
		reasoning = r.String()
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
		cfg.DefaultModel = openAIDefaultModel
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
	upstreamRequestType := strings.TrimSpace(UpstreamRequestTypeFromContext(ctx))
	if err := t.ValidateRequestCompatibilityForUpstream("openai", upstreamRequestType, req); err != nil {
		return nil, err
	}

	reqCopy := normalizeOpenAICompatibleRequestForProvider(*req, t.cfg)
	reqCopy.Reasoning = nil

	endpointPath := "chat/completions"
	if strings.EqualFold(upstreamRequestType, "responses") {
		endpointPath = "responses"
	}

	var body []byte
	var err error
	if strings.EqualFold(upstreamRequestType, "responses") {
		body, err = openAIResponsesRequestBody(&reqCopy)
	} else {
		body, err = openAIChatRequestBody(&reqCopy, t.cfg)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to marshal openai request: %w", err)
	}
	endpoint, err := safeurl.JoinHTTPPath(t.cfg.BaseURL, endpointPath)
	if err != nil {
		return nil, fmt.Errorf("failed to build openai endpoint: %w", err)
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
	applyUpstreamRequestHeaders(ctx, httpReq, "Idempotency-Key", "OpenAI-Beta")

	return httpReq, nil
}

func openAIChatRequestBody(req *models.UnifiedRequest, cfg config.ProviderConfig) ([]byte, error) {
	if req == nil {
		return nil, fmt.Errorf("request is required")
	}

	if len(bytes.TrimSpace(req.RawJSON)) == 0 || strings.EqualFold(strings.TrimSpace(req.SourceRequestType), "responses") {
		requestCopy := *req
		if requestCopy.Stream {
			if requestCopy.StreamOptions == nil {
				requestCopy.StreamOptions = &models.StreamOptions{}
			}
			requestCopy.StreamOptions.IncludeUsage = true
		}
		return json.Marshal(&requestCopy)
	}

	var payload map[string]json.RawMessage
	if err := json.Unmarshal(req.RawJSON, &payload); err != nil {
		return nil, fmt.Errorf("decode preserved chat request: %w", err)
	}
	if payload == nil {
		return nil, fmt.Errorf("chat request must be a JSON object")
	}

	setRawJSONValue(payload, "model", req.Model)
	setRawJSONPointerDefault(payload, "temperature", req.Temperature)
	setRawJSONPointerDefault(payload, "top_p", req.TopP)

	if req.Stream {
		setRawJSONValue(payload, "stream", true)
		streamOptions := make(map[string]json.RawMessage)
		if raw := bytes.TrimSpace(payload["stream_options"]); len(raw) > 0 && string(raw) != "null" {
			if err := json.Unmarshal(raw, &streamOptions); err != nil {
				return nil, fmt.Errorf("decode stream_options: %w", err)
			}
		}
		streamOptions["include_usage"] = json.RawMessage("true")
		setRawJSONValue(payload, "stream_options", streamOptions)
	}

	if rawMessages := bytes.TrimSpace(payload["messages"]); len(rawMessages) > 0 {
		var messages []map[string]interface{}
		if err := decodeJSONPreserveNumbers(rawMessages, &messages); err != nil {
			return nil, fmt.Errorf("decode messages: %w", err)
		}
		for _, message := range messages {
			if shouldNormalizeDeveloperRole(cfg) {
				if role, _ := message["role"].(string); strings.EqualFold(strings.TrimSpace(role), "developer") {
					message["role"] = "system"
				}
			}
			normalizeOpenAIChatContentParts(message)
		}
		setRawJSONValue(payload, "messages", messages)
	}

	return json.Marshal(payload)
}

func openAIResponsesRequestBody(req *models.UnifiedRequest) ([]byte, error) {
	if req == nil {
		return nil, fmt.Errorf("request is required")
	}
	if !strings.EqualFold(strings.TrimSpace(req.SourceRequestType), "responses") || len(bytes.TrimSpace(req.RawJSON)) == 0 {
		return json.Marshal(unifiedToResponsesPayload(req))
	}

	var payload map[string]json.RawMessage
	if err := json.Unmarshal(req.RawJSON, &payload); err != nil {
		return nil, fmt.Errorf("decode preserved responses request: %w", err)
	}
	if payload == nil {
		return nil, fmt.Errorf("responses request must be a JSON object")
	}

	setRawJSONValue(payload, "model", req.Model)
	setRawJSONPointerDefault(payload, "temperature", req.Temperature)
	setRawJSONPointerDefault(payload, "top_p", req.TopP)
	if req.Stream {
		setRawJSONValue(payload, "stream", true)
	}
	if req.Store != nil {
		setRawJSONValue(payload, "store", *req.Store)
	}

	return json.Marshal(payload)
}

func setRawJSONPointerDefault(payload map[string]json.RawMessage, key string, value *float64) {
	if value == nil {
		return
	}
	if _, exists := payload[key]; exists {
		return
	}
	setRawJSONValue(payload, key, *value)
}

func setRawJSONValue(payload map[string]json.RawMessage, key string, value interface{}) {
	raw, err := json.Marshal(value)
	if err != nil {
		return
	}
	payload[key] = raw
}

func normalizeOpenAIChatContentParts(message map[string]interface{}) {
	parts, ok := message["content"].([]interface{})
	if !ok {
		return
	}
	for _, rawPart := range parts {
		part, ok := rawPart.(map[string]interface{})
		if !ok {
			continue
		}
		partType, _ := part["type"].(string)
		switch strings.TrimSpace(partType) {
		case "input_text", "output_text":
			part["type"] = "text"
		}
	}
}

func (t *OpenAITranslator) TranslateEmbeddingsRequest(ctx context.Context, req *models.EmbeddingsRequest) (*http.Request, error) {
	body, err := openAIEmbeddingsRequestBody(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal openai embeddings request: %w", err)
	}

	endpoint, err := safeurl.JoinHTTPPath(t.cfg.BaseURL, "embeddings")
	if err != nil {
		return nil, fmt.Errorf("failed to build openai embeddings endpoint: %w", err)
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create openai embeddings http request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+t.cfg.APIKey)
	if t.cfg.Organization != "" {
		httpReq.Header.Set("OpenAI-Organization", t.cfg.Organization)
	}
	applyUpstreamRequestHeaders(ctx, httpReq, "Idempotency-Key", "OpenAI-Beta")

	return httpReq, nil
}

func openAIEmbeddingsRequestBody(req *models.EmbeddingsRequest) ([]byte, error) {
	if req == nil {
		return nil, fmt.Errorf("request is required")
	}
	if len(bytes.TrimSpace(req.RawJSON)) == 0 {
		return json.Marshal(req)
	}

	var payload map[string]json.RawMessage
	if err := json.Unmarshal(req.RawJSON, &payload); err != nil {
		return nil, fmt.Errorf("decode preserved embeddings request: %w", err)
	}
	if payload == nil {
		return nil, fmt.Errorf("embeddings request must be a JSON object")
	}
	setRawJSONValue(payload, "model", req.Model)
	return json.Marshal(payload)
}

func (t *OpenAITranslator) ParseResponse(resp *http.Response) (*models.UnifiedResponse, error) {
	body, err := readUpstreamResponseBody(resp, "openai")
	if err != nil {
		return nil, fmt.Errorf("failed to read openai response body: %w", err)
	}

	nativeResponsesRequest := resp.Request != nil &&
		strings.EqualFold(strings.TrimSpace(UpstreamRequestTypeFromContext(resp.Request.Context())), "responses") &&
		strings.EqualFold(strings.TrimSpace(SourceRequestTypeFromContext(resp.Request.Context())), "responses")
	nativeResponsesSuccess := nativeResponsesRequest &&
		resp.StatusCode >= http.StatusOK && resp.StatusCode < http.StatusMultipleChoices
	if resp.StatusCode != http.StatusOK && !nativeResponsesSuccess {
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
	if envelope.Object == "response" {
		responsesResp, terminalFailure, err := decodeOpenAIResponsesResponse(body)
		if err != nil {
			return nil, fmt.Errorf("failed to unmarshal openai responses object: %w", err)
		}
		result := responsesResponseToUnified(responsesResp)
		if nativeResponsesRequest {
			result.RawJSON = append(json.RawMessage(nil), body...)
			return result, nil
		}
		finishReason, err := openAIResponsesTerminalFinishReason(responsesResp, terminalFailure, "")
		if err != nil {
			return nil, err
		}
		result.Choices[0].FinishReason = finishReason
		return result, nil
	}
	if nativeResponsesRequest {
		return nil, errors.New("native Responses object requires exact object value response")
	}

	normalizedBody, normalizedEnvelope := normalizeOpenAIChatResponseEnvelope(body, t.DefaultModel())
	if normalizedEnvelope {
		body = normalizedBody
	}

	if t.cfg.ExtractReasoningTags {
		if normalizedBody, changed := normalizeOpenAIChatReasoningTags(body); changed {
			body = normalizedBody
		}
	}

	var result models.UnifiedResponse
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal openai response: %w", err)
	}
	result.RawJSON = append(json.RawMessage(nil), body...)

	return &result, nil
}

// normalizeOpenAIChatReasoningTags extracts the non-standard <think> wrapper
// only when a provider explicitly opts in. RawMessage containers keep unknown
// fields and large JSON numbers intact while the two affected message fields
// are updated.
func normalizeOpenAIChatReasoningTags(body []byte) ([]byte, bool) {
	var payload map[string]json.RawMessage
	if err := json.Unmarshal(body, &payload); err != nil || payload == nil {
		return body, false
	}

	var choices []json.RawMessage
	if err := json.Unmarshal(payload["choices"], &choices); err != nil {
		return body, false
	}

	changed := false
	for index, rawChoice := range choices {
		var choice map[string]json.RawMessage
		if err := json.Unmarshal(rawChoice, &choice); err != nil || choice == nil {
			continue
		}
		var message map[string]json.RawMessage
		if err := json.Unmarshal(choice["message"], &message); err != nil || message == nil {
			continue
		}

		var content string
		if err := json.Unmarshal(message["content"], &content); err != nil ||
			!strings.Contains(content, "<think>") || !strings.Contains(content, "</think>") {
			continue
		}
		reasoning, cleaned, extracted := splitThinkTags(content)
		if !extracted {
			continue
		}

		setRawJSONValue(message, "content", cleaned)
		if reasoning != "" {
			var existing string
			if rawExisting, ok := message["reasoning_content"]; ok &&
				len(bytes.TrimSpace(rawExisting)) > 0 && string(bytes.TrimSpace(rawExisting)) != "null" {
				if err := json.Unmarshal(rawExisting, &existing); err != nil {
					continue
				}
			}
			if existing != "" {
				reasoning = existing + "\n" + reasoning
			}
			setRawJSONValue(message, "reasoning_content", reasoning)
		}
		setRawJSONValue(choice, "message", message)
		normalizedChoice, err := json.Marshal(choice)
		if err != nil {
			continue
		}
		choices[index] = normalizedChoice
		changed = true
	}
	if !changed {
		return body, false
	}
	setRawJSONValue(payload, "choices", choices)
	normalized, err := json.Marshal(payload)
	if err != nil {
		return body, false
	}
	return normalized, true
}

func normalizeOpenAIChatResponseEnvelope(body []byte, defaultModel string) ([]byte, bool) {
	var payload map[string]json.RawMessage
	if err := json.Unmarshal(body, &payload); err != nil || payload == nil {
		return body, false
	}

	changed := false
	if parseOpenAIResponseString(payload["id"]) == "" {
		setRawJSONValue(payload, "id", "chatcmpl-"+strings.ReplaceAll(uuid.NewString(), "-", ""))
		changed = true
	}
	if parseOpenAIResponseString(payload["object"]) == "" {
		setRawJSONValue(payload, "object", "chat.completion")
		changed = true
	}
	if _, ok := payload["created"]; !ok {
		setRawJSONValue(payload, "created", time.Now().Unix())
		changed = true
	}
	if parseOpenAIResponseString(payload["model"]) == "" && strings.TrimSpace(defaultModel) != "" {
		setRawJSONValue(payload, "model", strings.TrimSpace(defaultModel))
		changed = true
	}

	if rawUsage := bytes.TrimSpace(payload["usage"]); len(rawUsage) > 0 && string(rawUsage) != "null" {
		var usage map[string]json.RawMessage
		if err := json.Unmarshal(rawUsage, &usage); err == nil && usage != nil {
			usageChanged := copyOpenAIUsageAlias(usage, "prompt_tokens", "input_tokens")
			usageChanged = copyOpenAIUsageAlias(usage, "completion_tokens", "output_tokens") || usageChanged
			if _, ok := usage["total_tokens"]; !ok {
				promptTokens, promptOK := parseOpenAIResponseInteger(usage["prompt_tokens"])
				completionTokens, completionOK := parseOpenAIResponseInteger(usage["completion_tokens"])
				if promptOK && completionOK {
					setRawJSONValue(usage, "total_tokens", models.SaturatingTokenSum(promptTokens, completionTokens))
					usageChanged = true
				}
			}
			if usageChanged {
				setRawJSONValue(payload, "usage", usage)
				changed = true
			}
		}
	}

	if !changed {
		return body, false
	}
	normalized, err := json.Marshal(payload)
	if err != nil {
		return body, false
	}
	return normalized, true
}

func copyOpenAIUsageAlias(usage map[string]json.RawMessage, target string, source string) bool {
	if _, ok := usage[target]; ok {
		return false
	}
	value, ok := usage[source]
	if !ok || len(bytes.TrimSpace(value)) == 0 {
		return false
	}
	usage[target] = append(json.RawMessage(nil), value...)
	return true
}

func parseOpenAIResponseString(raw json.RawMessage) string {
	var value string
	if err := json.Unmarshal(raw, &value); err != nil {
		return ""
	}
	return strings.TrimSpace(value)
}

func parseOpenAIResponseInteger(raw json.RawMessage) (int, bool) {
	var value int
	if err := json.Unmarshal(raw, &value); err != nil {
		return 0, false
	}
	return value, true
}

func (t *OpenAITranslator) ParseEmbeddingsResponse(resp *http.Response) (*models.EmbeddingsResponse, error) {
	body, err := readUpstreamResponseBody(resp, "openai")
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
	result.RawJSON = append(json.RawMessage(nil), body...)

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

	if streamErr, ok := parseOpenAIStreamError(trimmed); ok {
		return nil, streamErr
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

	if t.cfg.ExtractReasoningTags {
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
				if c.Delta.ReasoningContent == "" {
					c.Delta.ReasoningContent = reasoning
				} else {
					c.Delta.ReasoningContent += "\n" + reasoning
				}
			}
			c.Delta.Content = cleaned
		}
	}
	chunk.RawJSON = append(json.RawMessage(nil), trimmed...)

	return &chunk, nil
}

type openAIStreamErrorDetail struct {
	Message string          `json:"message"`
	Type    string          `json:"type"`
	Code    json.RawMessage `json:"code"`
}

func parseOpenAIStreamError(data []byte) (*ProviderError, bool) {
	var envelope map[string]json.RawMessage
	if err := json.Unmarshal(data, &envelope); err != nil || envelope == nil {
		return nil, false
	}

	var detail openAIStreamErrorDetail
	if rawError, ok := envelope["error"]; ok {
		trimmedError := bytes.TrimSpace(rawError)
		if len(trimmedError) > 0 && trimmedError[0] == '{' {
			_ = json.Unmarshal(trimmedError, &detail)
			return newOpenAIStreamProviderError(detail), true
		}
		if len(trimmedError) > 0 && trimmedError[0] == '"' {
			_ = json.Unmarshal(trimmedError, &detail.Message)
			return newOpenAIStreamProviderError(detail), true
		}
	}

	var eventType string
	if err := json.Unmarshal(envelope["type"], &eventType); err != nil || !strings.EqualFold(strings.TrimSpace(eventType), "error") {
		return nil, false
	}
	_ = json.Unmarshal(envelope["message"], &detail.Message)
	detail.Code = envelope["code"]
	return newOpenAIStreamProviderError(detail), true
}

func newOpenAIStreamProviderError(detail openAIStreamErrorDetail) *ProviderError {
	message := strings.TrimSpace(detail.Message)
	if message == "" {
		message = "openai stream error"
	}
	errorType := strings.TrimSpace(detail.Type)
	if errorType == "" {
		var code string
		if err := json.Unmarshal(detail.Code, &code); err == nil {
			errorType = strings.TrimSpace(code)
		}
	}
	if errorType == "" {
		errorType = "upstream_error"
	}
	return &ProviderError{
		StatusCode: http.StatusBadGateway,
		Message:    message,
		Type:       errorType,
		Provider:   "openai",
	}
}

func normalizeOpenAICompatibleRequestForProvider(req models.UnifiedRequest, cfg config.ProviderConfig) models.UnifiedRequest {
	if req.Temperature == nil && cfg.Temperature != nil {
		v := *cfg.Temperature
		req.Temperature = &v
	}
	if req.TopP == nil && cfg.TopP != nil {
		v := *cfg.TopP
		req.TopP = &v
	}
	// top_k is not part of OpenAI's Chat Completions or Responses payloads.
	// It remains available to translators for providers that support it.
	req.TopK = nil

	if !shouldNormalizeDeveloperRole(cfg) {
		return req
	}

	if len(req.Messages) == 0 {
		return req
	}

	req.Messages = append([]models.Message(nil), req.Messages...)
	for i := range req.Messages {
		if strings.EqualFold(strings.TrimSpace(req.Messages[i].Role), "developer") {
			req.Messages[i].Role = "system"
		}
	}

	return req
}

func shouldNormalizeDeveloperRole(cfg config.ProviderConfig) bool {
	if isDeepSeekCompatibilityProfile(cfg) {
		return true
	}

	if cfg.NormalizeDeveloperRole {
		return true
	}

	if enabled, ok := providerExtraBool(cfg, "normalize_developer_role"); ok {
		return enabled
	}

	return false
}

func isDeepSeekCompatibilityProfile(cfg config.ProviderConfig) bool {
	profile := strings.ToLower(strings.TrimSpace(cfg.CompatibilityProfile))
	if profile == "" {
		profile = strings.ToLower(strings.TrimSpace(providerExtraValue(cfg, "compatibility_profile")))
	}
	return profile == "deepseek"
}

func providerExtraValue(cfg config.ProviderConfig, key string) string {
	if cfg.Extra == nil {
		return ""
	}
	return cfg.Extra[key]
}

func providerExtraBool(cfg config.ProviderConfig, key string) (bool, bool) {
	raw := strings.TrimSpace(providerExtraValue(cfg, key))
	if raw == "" {
		return false, false
	}
	switch strings.ToLower(raw) {
	case "1", "true", "yes", "on":
		return true, true
	case "0", "false", "no", "off":
		return false, true
	default:
		return false, false
	}
}

func unifiedToResponsesPayload(req *models.UnifiedRequest) *models.ResponsesRequest {
	input := make([]interface{}, 0, len(req.Messages))
	store := false

	for i := range req.Messages {
		msg := req.Messages[i]
		if strings.EqualFold(strings.TrimSpace(msg.Role), "assistant") && len(msg.ToolCalls) > 0 {
			for _, tc := range msg.ToolCalls {
				callID := tc.ID
				input = append(input, map[string]interface{}{
					"type":      "function_call",
					"call_id":   callID,
					"name":      tc.Function.Name,
					"arguments": tc.Function.Arguments,
				})
			}
		}

		if strings.EqualFold(strings.TrimSpace(msg.Role), "tool") {
			callID := msg.ToolCallID
			input = append(input, map[string]interface{}{
				"type":    "function_call_output",
				"call_id": callID,
				"output":  normalizeResponsesToolOutput(msg.Content),
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
		Model:              req.Model,
		Input:              input,
		PreviousResponseID: req.PreviousResponseID,
		Temperature:        req.Temperature,
		TopP:               req.TopP,
		Tools:              make([]models.ResponsesTool, 0, len(req.Tools)),
		ToolChoice:         normalizeResponsesToolChoiceForUpstream(req.ToolChoice),
		Stream:             req.Stream,
		Store:              &store,
		User:               req.User,
	}
	if effort := strings.TrimSpace(req.ReasoningEffort); effort != "" {
		out.Reasoning = &models.Reasoning{Effort: effort}
	}
	if req.MaxTokens != nil {
		out.MaxOutputTokens = req.MaxTokens
	}
	for _, tool := range req.Tools {
		fn := tool.Function
		strict := false
		if fn.Strict != nil {
			strict = *fn.Strict
		}
		out.Tools = append(out.Tools, models.ResponsesTool{
			Type:        "function",
			Name:        fn.Name,
			Description: fn.Description,
			Parameters:  fn.Parameters,
			Strict:      &strict,
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
	if text := firstNonEmptyResponsesText(resp); strings.TrimSpace(text) != "" {
		message.Content = text
	}
	if reasoning := firstNonEmptyResponsesReasoning(resp); strings.TrimSpace(reasoning) != "" {
		message.ReasoningContent = reasoning
	}
	if refusal := firstNonEmptyResponsesRefusal(resp); refusal != "" {
		message.Refusal = refusal
	}

	toolCalls := make([]models.ToolCall, 0)
	for i := range resp.Output {
		item := resp.Output[i]
		if item.Type != "function_call" {
			continue
		}
		idx := i
		callID := item.CallID
		if callID == "" {
			callID = item.ID
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
			PromptTokens:        resp.Usage.InputTokens,
			CompletionTokens:    resp.Usage.OutputTokens,
			TotalTokens:         resp.Usage.TotalTokens,
			PromptTokensDetails: models.CloneInputTokensDetails(resp.Usage.InputTokensDetails),
		}
	}

	return out
}

func firstNonEmptyResponsesText(resp *models.ResponsesResponse) string {
	if resp == nil {
		return ""
	}
	if strings.TrimSpace(resp.OutputText) != "" {
		return resp.OutputText
	}

	parts := make([]string, 0, 2)
	for i := range resp.Output {
		item := resp.Output[i]
		if item.Type != "message" {
			continue
		}
		for j := range item.Content {
			part := item.Content[j]
			switch strings.TrimSpace(part.Type) {
			case "output_text", "text":
				if strings.TrimSpace(part.Text) != "" {
					parts = append(parts, part.Text)
				}
			}
		}
	}
	return strings.Join(parts, "\n")
}

func firstNonEmptyResponsesReasoning(resp *models.ResponsesResponse) string {
	if resp == nil {
		return ""
	}

	parts := make([]string, 0, 2)
	for i := range resp.Output {
		item := resp.Output[i]
		if item.Type != "reasoning" {
			continue
		}
		for _, summary := range item.Summary {
			if strings.TrimSpace(summary.Text) != "" {
				parts = append(parts, summary.Text)
			}
		}
		for _, content := range item.Content {
			partType := strings.TrimSpace(content.Type)
			if partType != "reasoning_text" && partType != "summary_text" && partType != "text" {
				continue
			}
			if strings.TrimSpace(content.Text) != "" {
				parts = append(parts, content.Text)
			}
		}
	}
	return strings.Join(parts, "\n")
}

func firstNonEmptyResponsesRefusal(resp *models.ResponsesResponse) string {
	if resp == nil {
		return ""
	}

	parts := make([]string, 0, 1)
	for i := range resp.Output {
		item := resp.Output[i]
		if item.Type != "message" {
			continue
		}
		for j := range item.Content {
			part := item.Content[j]
			if strings.TrimSpace(part.Type) != "refusal" || strings.TrimSpace(part.Refusal) == "" {
				continue
			}
			parts = append(parts, part.Refusal)
		}
	}
	return strings.Join(parts, "\n")
}

func responsesEventToStreamChunk(data []byte) (*models.StreamChunk, error) {
	var raw map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("failed to unmarshal responses stream event: %w", err)
	}

	responseID, err := responsesEventResponseID(raw)
	if err != nil {
		return nil, err
	}
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
	case "response.completed", "response.done", "response.incomplete", "response.failed", "response.cancelled", "response.canceled":
		log.Debug().
			Str("provider", "openai").
			Str("responses_event_type", typeName).
			Msg("responses stream terminal event")
		var event struct {
			Response json.RawMessage `json:"response"`
		}
		if err := json.Unmarshal(data, &event); err != nil {
			return nil, fmt.Errorf("failed to unmarshal responses terminal event: %w", err)
		}
		if len(bytes.TrimSpace(event.Response)) == 0 || bytes.Equal(bytes.TrimSpace(event.Response), []byte("null")) {
			return nil, openAIResponsesInvalidStatusError("terminal event is missing its response object")
		}
		terminalResponse, terminalFailure, err := decodeOpenAIResponsesResponse(event.Response)
		if err != nil {
			return nil, fmt.Errorf("failed to unmarshal responses terminal response: %w", err)
		}
		statusOverride := strings.TrimPrefix(typeName, "response.")
		if typeName == "response.done" {
			statusOverride = terminalResponse.Status
			if strings.TrimSpace(statusOverride) == "" {
				statusOverride = "completed"
			}
		}
		finishReason, err := openAIResponsesTerminalFinishReason(terminalResponse, terminalFailure, statusOverride)
		if err != nil {
			return nil, err
		}
		chunk := &models.StreamChunk{
			ID:      terminalResponse.ID,
			Object:  "chat.completion.chunk",
			Created: terminalResponse.CreatedAt,
			Model:   terminalResponse.Model,
			Choices: []models.Choice{{
				Index:        0,
				Delta:        &models.Message{},
				FinishReason: finishReason,
			}},
		}
		if terminalResponse.Usage != nil {
			chunk.Usage = &models.Usage{
				PromptTokens:        terminalResponse.Usage.InputTokens,
				CompletionTokens:    terminalResponse.Usage.OutputTokens,
				TotalTokens:         terminalResponse.Usage.TotalTokens,
				PromptTokensDetails: models.CloneInputTokensDetails(terminalResponse.Usage.InputTokensDetails),
			}
		}
		return chunk, ErrStreamDone
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
	case "response.refusal.delta":
		delta := interfaceToString(raw["delta"])
		if delta == "" {
			return nil, nil
		}
		return &models.StreamChunk{
			ID:     responseID,
			Object: "chat.completion.chunk",
			Choices: []models.Choice{{
				Index: 0,
				Delta: &models.Message{Refusal: delta},
			}},
		}, nil
	case "response.refusal.done":
		// Some Responses streams emit only the finalized refusal snapshot.
		refusal := interfaceToString(raw["refusal"])
		if refusal == "" {
			return nil, nil
		}
		return &models.StreamChunk{
			ID:     responseID,
			Object: "chat.completion.chunk",
			Choices: []models.Choice{{
				Index: 0,
				Delta: &models.Message{Refusal: refusal},
			}},
		}, nil
	case "response.content_part.done":
		part, _ := raw["part"].(map[string]interface{})
		if part == nil {
			return nil, nil
		}
		partType := strings.TrimSpace(interfaceToString(part["type"]))
		partText := interfaceToString(part["text"])
		switch partType {
		case "output_text", "text":
			if partText == "" {
				return nil, nil
			}
			return &models.StreamChunk{
				ID:     responseID,
				Object: "chat.completion.chunk",
				Choices: []models.Choice{{
					Index: 0,
					Delta: &models.Message{Content: partText},
				}},
			}, nil
		case "reasoning", "reasoning_text", "reasoning_summary_text":
			if partText == "" {
				return nil, nil
			}
			return &models.StreamChunk{
				ID:     responseID,
				Object: "chat.completion.chunk",
				Choices: []models.Choice{{
					Index: 0,
					Delta: &models.Message{ReasoningContent: partText},
				}},
			}, nil
		case "refusal":
			refusal := interfaceToString(part["refusal"])
			if refusal == "" {
				return nil, nil
			}
			return &models.StreamChunk{
				ID:     responseID,
				Object: "chat.completion.chunk",
				Choices: []models.Choice{{
					Index: 0,
					Delta: &models.Message{Refusal: refusal},
				}},
			}, nil
		default:
			return nil, nil
		}
	case "response.reasoning_summary_text.delta", "response.reasoning_summary_text.done",
		"response.reasoning_text.delta", "response.reasoning_text.done":
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
	case "response.reasoning_summary_part.added", "response.reasoning_summary_part.done":
		part, _ := raw["part"].(map[string]interface{})
		if part == nil {
			return nil, nil
		}
		text := interfaceToString(part["text"])
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
		id, err := openAIStreamIdentifier(raw, "item_id")
		if err != nil {
			return nil, err
		}
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
			refusal := openaiMessageRefusalToString(item["content"])
			if text == "" && refusal == "" {
				return nil, nil
			}
			message := &models.Message{Refusal: refusal}
			if text != "" {
				message.Content = text
			}
			return &models.StreamChunk{
				ID:     responseID,
				Object: "chat.completion.chunk",
				Choices: []models.Choice{{
					Index: 0,
					Delta: message,
				}},
			}, nil
		}
		if item == nil || itemType != "function_call" {
			logIgnoredResponsesEvent(typeName, raw)
			return nil, nil
		}
		id, err := openAIStreamIdentifier(item, "call_id")
		if err != nil {
			return nil, err
		}
		if id == "" {
			id, err = openAIStreamIdentifier(item, "id")
			if err != nil {
				return nil, err
			}
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

func responsesEventResponseID(raw map[string]interface{}) (string, error) {
	if _, present := raw["response_id"]; present {
		return openAIStreamIdentifier(raw, "response_id")
	}
	resp, _ := raw["response"].(map[string]interface{})
	if resp != nil {
		return openAIStreamIdentifier(resp, "id")
	}
	return "", nil
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

func openaiMessageRefusalToString(content interface{}) string {
	parts, ok := content.([]interface{})
	if !ok {
		return ""
	}

	refusals := make([]string, 0, 1)
	for _, rawPart := range parts {
		part, ok := rawPart.(map[string]interface{})
		if !ok || strings.TrimSpace(interfaceToString(part["type"])) != "refusal" {
			continue
		}
		refusal := interfaceToString(part["refusal"])
		if strings.TrimSpace(refusal) != "" {
			refusals = append(refusals, refusal)
		}
	}
	return strings.Join(refusals, "\n")
}

func normalizeResponsesMessageContent(role string, content interface{}) (interface{}, bool) {
	partType := responsesTextPartTypeForRole(role)
	if s, ok := content.(string); ok {
		return []map[string]interface{}{{"type": partType, "text": s}}, true
	}

	parts, ok := content.([]interface{})
	if !ok {
		encoded, err := json.Marshal(content)
		if err != nil || decodeJSONPreserveNumbers(encoded, &parts) != nil {
			return nil, false
		}
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
		if t == "image_url" {
			url, detail, _, ok := openAIChatImageReference(copyObj["image_url"])
			if ok {
				inputImage := map[string]interface{}{
					"type":      "input_image",
					"image_url": url,
				}
				if detail != "" {
					inputImage["detail"] = detail
				}
				normalized = append(normalized, inputImage)
				continue
			}
		}
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

func normalizeResponsesToolOutput(content interface{}) interface{} {
	if _, ok := content.(string); ok {
		return content
	}
	normalized, ok := normalizeResponsesMessageContent("tool", content)
	if !ok {
		return content
	}
	return normalized
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

func (t *OpenAITranslator) SupportsStreaming() bool {
	return true
}

func (t *OpenAITranslator) Models() []models.ModelInfo {
	return []models.ModelInfo{
		{ID: openAIDefaultModel, Object: "model", Created: time.Now().Unix(), OwnedBy: "openai"},
		{ID: "gpt-5.6-sol", Object: "model", Created: time.Now().Unix(), OwnedBy: "openai"},
		{ID: "gpt-5.6-luna", Object: "model", Created: time.Now().Unix(), OwnedBy: "openai"},
		{ID: "gpt-5.5", Object: "model", Created: time.Now().Unix(), OwnedBy: "openai"},
		{ID: "gpt-5.4", Object: "model", Created: time.Now().Unix(), OwnedBy: "openai"},
		{ID: "gpt-5.4-mini", Object: "model", Created: time.Now().Unix(), OwnedBy: "openai"},
		{ID: "gpt-5.2", Object: "model", Created: time.Now().Unix(), OwnedBy: "openai"},
		{ID: "gpt-4o", Object: "model", Created: time.Now().Unix(), OwnedBy: "openai"},
		{ID: "gpt-4o-mini", Object: "model", Created: time.Now().Unix(), OwnedBy: "openai"},
		{ID: "text-embedding-3-small", Object: "model", Created: time.Now().Unix(), OwnedBy: "openai"},
		{ID: "text-embedding-3-large", Object: "model", Created: time.Now().Unix(), OwnedBy: "openai"},
	}
}
