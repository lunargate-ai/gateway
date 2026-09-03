package providers

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/safeurl"
	"github.com/lunargate-ai/gateway/pkg/models"
	"github.com/rs/zerolog/log"
)

type OllamaTranslator struct {
	cfg config.ProviderConfig
}

func NewOllamaTranslator(cfg config.ProviderConfig) *OllamaTranslator {
	if cfg.BaseURL == "" {
		cfg.BaseURL = "http://localhost:11434"
	}
	if cfg.DefaultModel == "" {
		cfg.DefaultModel = "llama3.2"
	}
	return &OllamaTranslator{cfg: cfg}
}

func (t *OllamaTranslator) Name() string {
	return "ollama"
}

func (t *OllamaTranslator) DefaultModel() string {
	return t.cfg.DefaultModel
}

func (t *OllamaTranslator) BaseURL() string {
	return strings.TrimRight(strings.TrimSpace(t.cfg.BaseURL), "/")
}

type ollamaChatRequest struct {
	Model    string                 `json:"model"`
	Messages []ollamaMessage        `json:"messages"`
	Stream   bool                   `json:"stream"`
	Think    interface{}            `json:"think,omitempty"`
	Tools    []models.Tool          `json:"tools,omitempty"`
	Format   interface{}            `json:"format,omitempty"`
	Options  map[string]interface{} `json:"options,omitempty"`
}

type ollamaMessage struct {
	Role       string           `json:"role"`
	Content    string           `json:"content"`
	Thinking   string           `json:"thinking,omitempty"`
	Reasoning  string           `json:"reasoning,omitempty"`
	Images     [][]byte         `json:"images,omitempty"`
	ToolCalls  []ollamaToolCall `json:"tool_calls,omitempty"`
	ToolName   string           `json:"tool_name,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
}

type ollamaToolCall struct {
	ID       string             `json:"id,omitempty"`
	Function ollamaToolFunction `json:"function"`
}

type ollamaToolFunction struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
	Index     *int            `json:"index"`
}

type ollamaChatResponse struct {
	Model           string        `json:"model"`
	CreatedAt       string        `json:"created_at"`
	Message         ollamaMessage `json:"message"`
	Done            bool          `json:"done"`
	DoneReason      string        `json:"done_reason"`
	PromptEvalCount int           `json:"prompt_eval_count"`
	EvalCount       int           `json:"eval_count"`
	Error           string        `json:"error,omitempty"`
}

type ollamaEmbedRequest struct {
	Model string      `json:"model"`
	Input interface{} `json:"input,omitempty"`
}

type ollamaEmbedResponse struct {
	Model           string          `json:"model"`
	Embeddings      json.RawMessage `json:"embeddings,omitempty"`
	Embedding       []float64       `json:"embedding,omitempty"`
	PromptEvalCount int             `json:"prompt_eval_count,omitempty"`
	Error           string          `json:"error,omitempty"`
}

func (t *OllamaTranslator) TranslateRequest(ctx context.Context, req *models.UnifiedRequest) (*http.Request, error) {
	if err := t.validateRequestFields("ollama", req); err != nil {
		return nil, err
	}

	msgs, err := translateOllamaMessages(req.Messages, "ollama")
	if err != nil {
		return nil, err
	}

	selectedTools, toolChoiceMode, err := resolveOllamaToolChoice(req.Tools, req.ToolChoice, "ollama")
	if err != nil {
		return nil, err
	}
	selectedTools = ollamaToolsForUpstream(selectedTools)
	stop, err := resolveOllamaStop(req.Stop, "ollama")
	if err != nil {
		return nil, err
	}
	format, err := resolveOllamaResponseFormat(req, "ollama")
	if err != nil {
		return nil, err
	}

	options := make(map[string]interface{}, 9)
	if req.Temperature != nil {
		options["temperature"] = *req.Temperature
	} else if t.cfg.Temperature != nil {
		options["temperature"] = *t.cfg.Temperature
	}
	if req.TopP != nil {
		options["top_p"] = *req.TopP
	} else if t.cfg.TopP != nil {
		options["top_p"] = *t.cfg.TopP
	}
	if req.TopK != nil {
		options["top_k"] = *req.TopK
	} else if t.cfg.TopK != nil {
		options["top_k"] = *t.cfg.TopK
	}
	if req.MaxTokens != nil {
		options["num_predict"] = *req.MaxTokens
	}
	if req.PresencePenalty != nil {
		options["presence_penalty"] = *req.PresencePenalty
	}
	if req.FrequencyPenalty != nil {
		options["frequency_penalty"] = *req.FrequencyPenalty
	}
	if req.Seed != nil {
		options["seed"] = *req.Seed
	}
	if req.Stop != nil {
		options["stop"] = stop
	}
	if len(options) == 0 {
		options = nil
	}

	ollamaReq := ollamaChatRequest{
		Model:    req.Model,
		Messages: msgs,
		Stream:   req.Stream,
		Think:    resolveOllamaThink(req, t.cfg),
		Tools:    selectedTools,
		Format:   format,
		Options:  options,
	}

	body, err := json.Marshal(ollamaReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal ollama request: %w", err)
	}

	endpoint, err := safeurl.JoinHTTPPath(t.cfg.BaseURL, "api/chat")
	if err != nil {
		return nil, fmt.Errorf("failed to build ollama endpoint: %w", err)
	}
	log.Debug().
		Str("provider", "ollama").
		Str("model", ollamaReq.Model).
		Bool("stream", ollamaReq.Stream).
		Int("messages_count", len(ollamaReq.Messages)).
		Strs("message_roles", ollamaMessageRoles(ollamaReq.Messages)).
		Int("tools_count", len(ollamaReq.Tools)).
		Strs("tool_names", ollamaToolNames(ollamaReq.Tools)).
		Str("tool_choice_mode", toolChoiceMode).
		Bool("has_think", ollamaReq.Think != nil).
		Bool("has_format", ollamaReq.Format != nil).
		Msg("sending chat request to ollama")

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create ollama http request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	return httpReq, nil
}

// ValidateRequestCompatibility reports client-visible fields that cannot be
// represented faithfully by Ollama's native chat API.
func (t *OllamaTranslator) ValidateRequestCompatibility(providerID string, req *models.UnifiedRequest) error {
	if err := t.validateRequestFields(providerID, req); err != nil {
		return err
	}
	if req == nil {
		return nil
	}
	_, err := translateOllamaMessages(req.Messages, providerID)
	return err
}

func (t *OllamaTranslator) validateRequestFields(providerID string, req *models.UnifiedRequest) error {
	if req == nil {
		return nil
	}
	if err := validateTranslatedChatRawControls(providerID, req); err != nil {
		return err
	}
	if req.N != nil && *req.N != 1 {
		return ollamaCompatibilityError(providerID, "n", "Ollama returns exactly one choice")
	}
	if len(req.LogitBias) > 0 {
		return ollamaCompatibilityError(providerID, "logit_bias", "Ollama does not expose token-level logit bias")
	}
	if strings.TrimSpace(req.User) != "" {
		return ollamaCompatibilityError(providerID, "user", "Ollama has no equivalent end-user identifier field")
	}
	if req.Store != nil && *req.Store && !strings.EqualFold(strings.TrimSpace(req.SourceRequestType), "responses") {
		return ollamaCompatibilityError(providerID, "store", "Ollama cannot create a stored response")
	}
	if len(req.Functions) > 0 {
		return ollamaCompatibilityError(providerID, "functions", "normalize legacy functions into tools before using Ollama")
	}
	if req.FunctionCall != nil {
		return ollamaCompatibilityError(providerID, "function_call", "normalize legacy function_call into tool_choice before using Ollama")
	}
	if err := validateTranslatedChatTypedToolChoice(providerID, req.ToolChoice); err != nil {
		return err
	}
	if err := validateOllamaTools(providerID, req.Tools); err != nil {
		return err
	}
	if effort := requestedOllamaReasoningEffort(req); effort != "" {
		if _, ok := normalizeOllamaReasoningEffort(effort); !ok {
			return ollamaCompatibilityError(
				providerID,
				requestedOllamaReasoningEffortField(req),
				"Ollama supports none, low, medium, and high reasoning effort",
			)
		}
	}
	if _, err := resolveOllamaStop(req.Stop, providerID); err != nil {
		return err
	}
	if _, err := resolveOllamaResponseFormat(req, providerID); err != nil {
		return err
	}
	if _, _, err := resolveOllamaToolChoice(req.Tools, req.ToolChoice, providerID); err != nil {
		return err
	}
	return nil
}

func validateOllamaTools(providerID string, tools []models.Tool) error {
	for index := range tools {
		path := fmt.Sprintf("tools[%d]", index)
		tool := tools[index]
		toolType := strings.ToLower(strings.TrimSpace(tool.Type))
		if toolType != "" && toolType != "function" {
			return ollamaCompatibilityError(providerID, path+".type", "Ollama only supports function tools")
		}
		if strings.TrimSpace(tool.Function.Name) == "" {
			return ollamaCompatibilityError(providerID, path+".function.name", "Ollama requires a function name")
		}
		if tool.Function.Strict != nil && *tool.Function.Strict {
			return ollamaCompatibilityError(providerID, path+".function.strict", "Ollama /api/chat does not guarantee strict function arguments")
		}
		if tool.Function.Parameters != nil {
			encoded, err := json.Marshal(tool.Function.Parameters)
			if err != nil {
				return ollamaCompatibilityError(providerID, path+".function.parameters", "function parameters must be a JSON schema object")
			}
			var schema map[string]interface{}
			if err := json.Unmarshal(encoded, &schema); err != nil || schema == nil {
				return ollamaCompatibilityError(providerID, path+".function.parameters", "function parameters must be a JSON schema object")
			}
		}
	}
	return nil
}

func ollamaToolsForUpstream(tools []models.Tool) []models.Tool {
	if len(tools) == 0 {
		return nil
	}
	out := make([]models.Tool, len(tools))
	copy(out, tools)
	for index := range out {
		out[index].Function.Strict = nil
	}
	return out
}

func resolveOllamaStop(value interface{}, providerID string) ([]string, error) {
	switch stop := value.(type) {
	case nil:
		return nil, nil
	case string:
		return []string{stop}, nil
	case []string:
		return append([]string(nil), stop...), nil
	case []interface{}:
		out := make([]string, 0, len(stop))
		for _, value := range stop {
			sequence, ok := value.(string)
			if !ok {
				return nil, ollamaCompatibilityError(providerID, "stop", "Ollama requires a string or an array of strings")
			}
			out = append(out, sequence)
		}
		return out, nil
	default:
		return nil, ollamaCompatibilityError(providerID, "stop", "Ollama requires a string or an array of strings")
	}
}

func resolveOllamaResponseFormat(req *models.UnifiedRequest, providerID string) (interface{}, error) {
	if req == nil || req.ResponseFormat == nil {
		return nil, nil
	}

	switch strings.ToLower(strings.TrimSpace(req.ResponseFormat.Type)) {
	case "text":
		return nil, nil
	case "json", "json_object":
		return "json", nil
	case "json_schema":
		schema, err := translatedChatAnnotatedJSONSchema(providerID, req.ResponseFormat.JSONSchema)
		if err != nil {
			return nil, err
		}
		return schema, nil
	default:
		return nil, ollamaCompatibilityError(providerID, "response_format.type", "supported values are text, json_object, and json_schema")
	}
}

func ollamaCompatibilityError(providerID string, field string, reason string) *models.CompatibilityError {
	providerID = strings.TrimSpace(providerID)
	if providerID == "" {
		providerID = "ollama"
	}
	return &models.CompatibilityError{Field: field, Provider: providerID, Reason: reason}
}

func (t *OllamaTranslator) TranslateEmbeddingsRequest(ctx context.Context, req *models.EmbeddingsRequest) (*http.Request, error) {
	input, err := normalizeOllamaEmbeddingInput(req.Input)
	if err != nil {
		return nil, err
	}

	body, err := json.Marshal(ollamaEmbedRequest{
		Model: req.Model,
		Input: input,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to marshal ollama embeddings request: %w", err)
	}

	endpoint, err := safeurl.JoinHTTPPath(t.cfg.BaseURL, "api/embed")
	if err != nil {
		return nil, fmt.Errorf("failed to build ollama embeddings endpoint: %w", err)
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create ollama embeddings http request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	return httpReq, nil
}

func (t *OllamaTranslator) ParseResponse(resp *http.Response) (*models.UnifiedResponse, error) {
	body, err := readUpstreamResponseBody(resp, "ollama")
	if err != nil {
		return nil, fmt.Errorf("failed to read ollama response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		msg := strings.TrimSpace(string(body))
		var errResp struct {
			Error string `json:"error"`
		}
		if jsonErr := json.Unmarshal(body, &errResp); jsonErr == nil {
			if strings.TrimSpace(errResp.Error) != "" {
				msg = strings.TrimSpace(errResp.Error)
			}
		}
		if msg == "" {
			msg = http.StatusText(resp.StatusCode)
		}
		return nil, &ProviderError{StatusCode: resp.StatusCode, Message: msg, Type: "upstream_error", Provider: "ollama"}
	}

	var result ollamaChatResponse
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal ollama response: %w", err)
	}
	if strings.TrimSpace(result.Error) != "" {
		return nil, &ProviderError{StatusCode: http.StatusBadGateway, Message: strings.TrimSpace(result.Error), Type: "upstream_error", Provider: "ollama"}
	}

	finishReason := mapOllamaDoneReason(result.DoneReason)

	var usage *models.Usage
	if result.PromptEvalCount > 0 || result.EvalCount > 0 {
		usage = &models.Usage{
			PromptTokens:     result.PromptEvalCount,
			CompletionTokens: result.EvalCount,
			TotalTokens:      models.SaturatingTokenSum(result.PromptEvalCount, result.EvalCount),
		}
	}

	id := fmt.Sprintf("chatcmpl_ollama_%d", time.Now().UnixNano())
	created := time.Now().Unix()

	toolCalls := make([]models.ToolCall, 0, len(result.Message.ToolCalls))
	for i := range result.Message.ToolCalls {
		call := result.Message.ToolCalls[i]
		idx := i
		if call.Function.Index != nil {
			idx = *call.Function.Index
		}
		callID := call.ID
		if strings.TrimSpace(callID) == "" {
			callID = fmt.Sprintf("call_%s_%d", id, idx)
		}
		args := "{}"
		if len(call.Function.Arguments) > 0 {
			args = string(call.Function.Arguments)
		}
		toolCalls = append(toolCalls, models.ToolCall{
			Index: &idx,
			ID:    callID,
			Type:  "function",
			Function: models.ToolCallFunction{
				Name:      call.Function.Name,
				Arguments: args,
			},
		})
	}

	reasoning := ""
	if result.Message.Reasoning != "" {
		reasoning = result.Message.Reasoning
	} else if result.Message.Thinking != "" {
		reasoning = result.Message.Thinking
	}

	respMsg := &models.Message{Role: "assistant", Content: result.Message.Content, ReasoningContent: reasoning}
	if len(toolCalls) > 0 {
		respMsg.ToolCalls = toolCalls
	}

	return &models.UnifiedResponse{
		ID:      id,
		Object:  "chat.completion",
		Created: created,
		Model:   strings.TrimSpace(result.Model),
		Choices: []models.Choice{{
			Index:        0,
			Message:      respMsg,
			FinishReason: finishReason,
		}},
		Usage: usage,
	}, nil
}

func resolveOllamaThink(req *models.UnifiedRequest, cfg config.ProviderConfig) interface{} {
	if req != nil {
		if val, ok := normalizeOllamaReasoningEffort(requestedOllamaReasoningEffort(req)); ok {
			return val
		}
	}
	if cfg.Extra != nil {
		if raw := strings.TrimSpace(cfg.Extra["think"]); raw != "" {
			if val, ok := normalizeOllamaThinkValue(raw); ok {
				return val
			}
		}
	}
	return nil
}

func requestedOllamaReasoningEffort(req *models.UnifiedRequest) string {
	if req == nil {
		return ""
	}
	if effort := strings.TrimSpace(req.ReasoningEffort); effort != "" {
		return effort
	}
	if req.Reasoning != nil {
		return strings.TrimSpace(req.Reasoning.Effort)
	}
	return ""
}

func requestedOllamaReasoningEffortField(req *models.UnifiedRequest) string {
	if req == nil {
		return "reasoning_effort"
	}
	var payload map[string]json.RawMessage
	if json.Unmarshal(bytes.TrimSpace(req.RawJSON), &payload) == nil && payload != nil {
		var topLevel string
		if json.Unmarshal(payload["reasoning_effort"], &topLevel) == nil && strings.TrimSpace(topLevel) != "" {
			return "reasoning_effort"
		}
		var reasoning map[string]json.RawMessage
		if json.Unmarshal(payload["reasoning"], &reasoning) == nil && reasoning != nil {
			var nested string
			if json.Unmarshal(reasoning["effort"], &nested) == nil && strings.TrimSpace(nested) != "" {
				return "reasoning.effort"
			}
		}
	}
	if strings.TrimSpace(req.ReasoningEffort) != "" {
		return "reasoning_effort"
	}
	if req.Reasoning != nil && strings.TrimSpace(req.Reasoning.Effort) != "" {
		return "reasoning.effort"
	}
	return "reasoning_effort"
}

func normalizeOllamaReasoningEffort(raw string) (interface{}, bool) {
	switch v := strings.ToLower(strings.TrimSpace(raw)); v {
	case "none":
		return false, true
	case "low", "medium", "high":
		return v, true
	default:
		return nil, false
	}
}

func normalizeOllamaThinkValue(raw string) (interface{}, bool) {
	v := strings.ToLower(strings.TrimSpace(raw))
	switch v {
	case "":
		return nil, false
	case "true", "1", "yes", "on":
		return true, true
	case "false", "0", "no", "off", "none":
		return false, true
	case "low", "medium", "high":
		return v, true
	default:
		return nil, false
	}
}

func (t *OllamaTranslator) ParseEmbeddingsResponse(resp *http.Response) (*models.EmbeddingsResponse, error) {
	body, err := readUpstreamResponseBody(resp, "ollama")
	if err != nil {
		return nil, fmt.Errorf("failed to read ollama embeddings response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		msg := strings.TrimSpace(string(body))
		var errResp struct {
			Error string `json:"error"`
		}
		if jsonErr := json.Unmarshal(body, &errResp); jsonErr == nil {
			if strings.TrimSpace(errResp.Error) != "" {
				msg = strings.TrimSpace(errResp.Error)
			}
		}
		if msg == "" {
			msg = http.StatusText(resp.StatusCode)
		}
		return nil, &ProviderError{StatusCode: resp.StatusCode, Message: msg, Type: "upstream_error", Provider: "ollama"}
	}

	var result ollamaEmbedResponse
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal ollama embeddings response: %w", err)
	}
	if strings.TrimSpace(result.Error) != "" {
		return nil, &ProviderError{StatusCode: http.StatusBadGateway, Message: strings.TrimSpace(result.Error), Type: "upstream_error", Provider: "ollama"}
	}

	vectors := make([][]float64, 0, 1)
	if len(result.Embedding) > 0 {
		vectors = append(vectors, result.Embedding)
	}
	if len(result.Embeddings) > 0 {
		var batch [][]float64
		if err := json.Unmarshal(result.Embeddings, &batch); err == nil {
			vectors = batch
		} else {
			var single []float64
			if err := json.Unmarshal(result.Embeddings, &single); err != nil {
				return nil, fmt.Errorf("failed to decode ollama embeddings vectors: %w", err)
			}
			vectors = [][]float64{single}
		}
	}

	data := make([]models.EmbeddingData, 0, len(vectors))
	for i, vector := range vectors {
		data = append(data, models.EmbeddingData{Object: "embedding", Embedding: models.NewFloatEmbeddingValue(vector), Index: i})
	}

	var usage *models.EmbeddingUsage
	if result.PromptEvalCount > 0 {
		usage = &models.EmbeddingUsage{PromptTokens: result.PromptEvalCount, TotalTokens: result.PromptEvalCount}
	}

	return &models.EmbeddingsResponse{
		Object: "list",
		Data:   data,
		Model:  strings.TrimSpace(result.Model),
		Usage:  usage,
	}, nil
}

func (t *OllamaTranslator) ParseStreamChunk(data []byte) (*models.StreamChunk, error) {
	return nil, fmt.Errorf("ollama streaming requires a per-request stream translator")
}

func (t *OllamaTranslator) SupportsStreaming() bool {
	return true
}

func (t *OllamaTranslator) Models() []models.ModelInfo {
	id := strings.TrimSpace(t.cfg.DefaultModel)
	if id == "" {
		return nil
	}
	return []models.ModelInfo{{ID: id, Object: "model", Created: time.Now().Unix(), OwnedBy: "ollama"}}
}

func translateOllamaMessages(messages []models.Message, providerID string) ([]ollamaMessage, error) {
	out := make([]ollamaMessage, 0, len(messages))
	for i := range messages {
		messagePath := fmt.Sprintf("messages[%d]", i)
		message := messages[i]
		role := strings.ToLower(strings.TrimSpace(message.Role))
		switch role {
		case "system", "user", "assistant", "tool":
		default:
			return nil, ollamaCompatibilityError(providerID, messagePath+".role", "Ollama /api/chat supports system, user, assistant, and tool messages")
		}

		if message.FunctionCall != nil {
			return nil, ollamaCompatibilityError(providerID, messagePath+".function_call", "normalize legacy function_call into assistant.tool_calls before using Ollama")
		}
		if message.Refusal != "" {
			return nil, ollamaCompatibilityError(providerID, messagePath+".refusal", "Ollama /api/chat has no assistant refusal-history field")
		}
		if message.Name != "" && role != "tool" {
			return nil, ollamaCompatibilityError(providerID, messagePath+".name", "Ollama only represents a name as tool_name on tool-result messages")
		}
		if message.ToolCallID != "" && role != "tool" {
			return nil, ollamaCompatibilityError(providerID, messagePath+".tool_call_id", "Ollama only uses tool_call_id on tool-result messages")
		}
		if message.ReasoningContent != "" && role != "assistant" {
			return nil, ollamaCompatibilityError(providerID, messagePath+".reasoning_content", "Ollama thinking history belongs to assistant messages")
		}
		if len(message.ToolCalls) > 0 && role != "assistant" {
			return nil, ollamaCompatibilityError(providerID, messagePath+".tool_calls", "Ollama tool calls belong to assistant messages")
		}

		content, images, err := translateOllamaMessageContent(message.Content, messagePath+".content", providerID)
		if err != nil {
			return nil, err
		}
		toolCalls, err := translateOllamaToolCalls(message.ToolCalls, messagePath+".tool_calls", providerID)
		if err != nil {
			return nil, err
		}

		translated := ollamaMessage{
			Role:       role,
			Content:    content,
			Thinking:   message.ReasoningContent,
			Images:     images,
			ToolCalls:  toolCalls,
			ToolCallID: message.ToolCallID,
		}
		if role == "tool" {
			translated.ToolName = message.Name
		}
		out = append(out, translated)
	}
	return out, nil
}

func translateOllamaMessageContent(content interface{}, field, providerID string) (string, [][]byte, error) {
	switch value := content.(type) {
	case nil:
		return "", nil, nil
	case string:
		return value, nil, nil
	case []interface{}:
		return translateOllamaContentParts(value, field, providerID)
	case []map[string]interface{}:
		parts := make([]interface{}, len(value))
		for i := range value {
			parts[i] = value[i]
		}
		return translateOllamaContentParts(parts, field, providerID)
	default:
		return "", nil, ollamaCompatibilityError(providerID, field, "Ollama message content must be a string or an array of text and inline-image parts")
	}
}

func translateOllamaContentParts(parts []interface{}, field, providerID string) (string, [][]byte, error) {
	var text strings.Builder
	images := make([][]byte, 0)
	for i := range parts {
		partPath := fmt.Sprintf("%s[%d]", field, i)
		part, ok := parts[i].(map[string]interface{})
		if !ok {
			return "", nil, ollamaCompatibilityError(providerID, partPath, "Ollama content parts must be JSON objects")
		}
		partType, ok := part["type"].(string)
		if !ok || strings.TrimSpace(partType) == "" {
			return "", nil, ollamaCompatibilityError(providerID, partPath+".type", "content part type is required")
		}

		switch strings.ToLower(strings.TrimSpace(partType)) {
		case "text", "input_text", "output_text":
			if unsupported := firstUnsupportedOllamaField(part, "text", "type"); unsupported != "" {
				return "", nil, ollamaCompatibilityError(providerID, partPath+"."+unsupported, "Ollama text parts cannot represent this field")
			}
			value, ok := part["text"].(string)
			if !ok {
				return "", nil, ollamaCompatibilityError(providerID, partPath+".text", "Ollama text parts require string text")
			}
			text.WriteString(value)
		case "image_url", "input_image":
			if unsupported := firstUnsupportedOllamaField(part, "detail", "image_url", "type"); unsupported != "" {
				return "", nil, ollamaCompatibilityError(providerID, partPath+"."+unsupported, "Ollama images cannot represent this field")
			}
			if detail, exists := part["detail"]; exists {
				value, ok := detail.(string)
				if !ok || (strings.TrimSpace(value) != "" && !strings.EqualFold(strings.TrimSpace(value), "auto")) {
					return "", nil, ollamaCompatibilityError(providerID, partPath+".detail", "Ollama cannot enforce image detail settings other than the automatic default")
				}
			}
			image, err := translateOllamaImageReference(part["image_url"], partPath+".image_url", providerID)
			if err != nil {
				return "", nil, err
			}
			images = append(images, image)
		default:
			return "", nil, ollamaCompatibilityError(providerID, partPath+".type", "Ollama /api/chat cannot represent this content part type")
		}
	}
	return text.String(), images, nil
}

func translateOllamaImageReference(value interface{}, field, providerID string) ([]byte, error) {
	imageURL := ""
	valueField := field
	switch reference := value.(type) {
	case string:
		imageURL = reference
	case map[string]interface{}:
		if unsupported := firstUnsupportedOllamaField(reference, "detail", "url"); unsupported != "" {
			return nil, ollamaCompatibilityError(providerID, field+"."+unsupported, "Ollama images cannot represent this image URL option")
		}
		if detail, exists := reference["detail"]; exists {
			value, ok := detail.(string)
			if !ok || (strings.TrimSpace(value) != "" && !strings.EqualFold(strings.TrimSpace(value), "auto")) {
				return nil, ollamaCompatibilityError(providerID, field+".detail", "Ollama cannot enforce image detail settings other than the automatic default")
			}
		}
		url, ok := reference["url"].(string)
		if !ok {
			return nil, ollamaCompatibilityError(providerID, field+".url", "image URL must be a string")
		}
		imageURL = url
		valueField = field + ".url"
	default:
		return nil, ollamaCompatibilityError(providerID, field, "image_url must be a data URL string or an object containing one")
	}

	imageURL = strings.TrimSpace(imageURL)
	if !strings.HasPrefix(strings.ToLower(imageURL), "data:") {
		return nil, ollamaCompatibilityError(providerID, valueField, "Ollama requires inline base64 image data and LunarGate does not fetch remote image URLs")
	}
	comma := strings.IndexByte(imageURL, ',')
	if comma < 0 {
		return nil, ollamaCompatibilityError(providerID, valueField, "image data URL is missing its payload")
	}
	metadata := strings.Split(imageURL[len("data:"):comma], ";")
	if len(metadata) == 0 || !strings.HasPrefix(strings.ToLower(strings.TrimSpace(metadata[0])), "image/") {
		return nil, ollamaCompatibilityError(providerID, valueField, "data URL must contain an image media type")
	}
	base64Encoded := false
	for _, item := range metadata[1:] {
		if strings.EqualFold(strings.TrimSpace(item), "base64") {
			base64Encoded = true
			break
		}
	}
	if !base64Encoded {
		return nil, ollamaCompatibilityError(providerID, valueField, "Ollama requires base64-encoded image data")
	}

	payload := imageURL[comma+1:]
	decoded, err := base64.StdEncoding.DecodeString(payload)
	if err != nil {
		decoded, err = base64.RawStdEncoding.DecodeString(payload)
	}
	if err != nil || len(decoded) == 0 {
		return nil, ollamaCompatibilityError(providerID, valueField, "image data URL contains invalid or empty base64 data")
	}
	return decoded, nil
}

func translateOllamaToolCalls(toolCalls []models.ToolCall, field, providerID string) ([]ollamaToolCall, error) {
	if len(toolCalls) == 0 {
		return nil, nil
	}
	out := make([]ollamaToolCall, 0, len(toolCalls))
	for i := range toolCalls {
		callPath := fmt.Sprintf("%s[%d]", field, i)
		call := toolCalls[i]
		callType := strings.ToLower(strings.TrimSpace(call.Type))
		if callType != "" && callType != "function" {
			return nil, ollamaCompatibilityError(providerID, callPath+".type", "Ollama only supports function tool calls")
		}
		if strings.TrimSpace(call.Function.Name) == "" {
			return nil, ollamaCompatibilityError(providerID, callPath+".function.name", "Ollama requires a function name")
		}

		arguments := strings.TrimSpace(call.Function.Arguments)
		if arguments == "" {
			arguments = "{}"
		}
		var object map[string]interface{}
		if err := json.Unmarshal([]byte(arguments), &object); err != nil || object == nil {
			return nil, ollamaCompatibilityError(providerID, callPath+".function.arguments", "Ollama requires tool arguments to be a JSON object")
		}

		index := i
		if call.Index != nil {
			index = *call.Index
		}
		if index < 0 {
			return nil, ollamaCompatibilityError(providerID, callPath+".index", "Ollama requires a non-negative tool-call index")
		}
		out = append(out, ollamaToolCall{
			ID: call.ID,
			Function: ollamaToolFunction{
				Name:      call.Function.Name,
				Arguments: json.RawMessage(arguments),
				Index:     &index,
			},
		})
	}
	return out, nil
}

func firstUnsupportedOllamaField(object map[string]interface{}, allowed ...string) string {
	allowedSet := make(map[string]struct{}, len(allowed))
	for _, field := range allowed {
		allowedSet[field] = struct{}{}
	}
	unsupported := make([]string, 0)
	for field := range object {
		if _, ok := allowedSet[field]; !ok {
			unsupported = append(unsupported, field)
		}
	}
	if len(unsupported) == 0 {
		return ""
	}
	sort.Strings(unsupported)
	return unsupported[0]
}

func normalizeOllamaEmbeddingInput(input interface{}) (interface{}, error) {
	switch v := input.(type) {
	case string:
		return v, nil
	case []interface{}:
		out := make([]string, 0, len(v))
		for i := range v {
			s, ok := v[i].(string)
			if !ok {
				return nil, fmt.Errorf("ollama embeddings only supports string or array of strings input")
			}
			out = append(out, s)
		}
		return out, nil
	default:
		return nil, fmt.Errorf("ollama embeddings only supports string or array of strings input")
	}
}

func mapOllamaDoneReason(reason string) *string {
	r := strings.TrimSpace(reason)
	if r == "" {
		return nil
	}
	mapped := "stop"
	switch r {
	case "stop":
		mapped = "stop"
	case "length":
		mapped = "length"
	case "tool_calls", "tool_call":
		mapped = "tool_calls"
	default:
		mapped = r
	}
	return &mapped
}

func resolveOllamaToolChoice(tools []models.Tool, choice interface{}, providerID string) ([]models.Tool, string, error) {
	mode, functionName, err := parseOllamaToolChoice(choice)
	if err != nil {
		return nil, "", err
	}
	if mode == "" {
		mode = "auto"
	}

	switch mode {
	case "auto":
		return tools, mode, nil
	case "none":
		return nil, mode, nil
	case "required":
		if len(tools) == 0 {
			return nil, "", &ProviderError{
				StatusCode: 400,
				Message:    "tool_choice=required requires at least one tool",
				Type:       "invalid_request_error",
				Provider:   "ollama",
			}
		}
		return nil, "", ollamaCompatibilityError(providerID, "tool_choice", "Ollama /api/chat cannot enforce required tool use")
	case "function":
		if len(tools) == 0 {
			return nil, "", &ProviderError{
				StatusCode: 400,
				Message:    "tool_choice=function requires at least one tool",
				Type:       "invalid_request_error",
				Provider:   "ollama",
			}
		}
		found := false
		for i := range tools {
			if strings.TrimSpace(tools[i].Function.Name) == functionName {
				found = true
				break
			}
		}
		if !found {
			return nil, "", &ProviderError{
				StatusCode: 400,
				Message:    fmt.Sprintf("tool_choice references unknown tool %q", functionName),
				Type:       "invalid_request_error",
				Provider:   "ollama",
			}
		}
		return nil, "", ollamaCompatibilityError(providerID, "tool_choice", "Ollama /api/chat cannot enforce a named function tool choice")
	default:
		return nil, "", &ProviderError{
			StatusCode: 400,
			Message:    fmt.Sprintf("unsupported tool_choice %q for ollama", mode),
			Type:       "invalid_request_error",
			Provider:   "ollama",
		}
	}
}

func parseOllamaToolChoice(choice interface{}) (mode string, functionName string, err error) {
	if choice == nil {
		return "auto", "", nil
	}
	if s, ok := choice.(string); ok {
		switch strings.TrimSpace(strings.ToLower(s)) {
		case "", "auto":
			return "auto", "", nil
		case "none":
			return "none", "", nil
		case "required", "any":
			return "required", "", nil
		default:
			return "", "", &ProviderError{
				StatusCode: 400,
				Message:    fmt.Sprintf("unsupported tool_choice %q for ollama", s),
				Type:       "invalid_request_error",
				Provider:   "ollama",
			}
		}
	}

	var obj struct {
		Type     string `json:"type"`
		Function *struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	b, marshalErr := json.Marshal(choice)
	if marshalErr != nil {
		return "", "", &ProviderError{
			StatusCode: 400,
			Message:    "invalid tool_choice payload",
			Type:       "invalid_request_error",
			Provider:   "ollama",
		}
	}
	if unmarshalErr := json.Unmarshal(b, &obj); unmarshalErr != nil {
		return "", "", &ProviderError{
			StatusCode: 400,
			Message:    "invalid tool_choice payload",
			Type:       "invalid_request_error",
			Provider:   "ollama",
		}
	}

	switch strings.TrimSpace(strings.ToLower(obj.Type)) {
	case "", "auto":
		return "auto", "", nil
	case "none":
		return "none", "", nil
	case "required", "any":
		return "required", "", nil
	case "function":
		if obj.Function == nil || strings.TrimSpace(obj.Function.Name) == "" {
			return "", "", &ProviderError{
				StatusCode: 400,
				Message:    "tool_choice.function.name is required",
				Type:       "invalid_request_error",
				Provider:   "ollama",
			}
		}
		return "function", strings.TrimSpace(obj.Function.Name), nil
	default:
		return "", "", &ProviderError{
			StatusCode: 400,
			Message:    fmt.Sprintf("unsupported tool_choice type %q for ollama", obj.Type),
			Type:       "invalid_request_error",
			Provider:   "ollama",
		}
	}
}

func ollamaMessageRoles(messages []ollamaMessage) []string {
	if len(messages) == 0 {
		return nil
	}
	roles := make([]string, 0, len(messages))
	for i := range messages {
		roles = append(roles, strings.TrimSpace(messages[i].Role))
	}
	return roles
}

func ollamaToolNames(tools []models.Tool) []string {
	if len(tools) == 0 {
		return nil
	}
	names := make([]string, 0, len(tools))
	for i := range tools {
		names = append(names, strings.TrimSpace(tools[i].Function.Name))
	}
	return names
}
