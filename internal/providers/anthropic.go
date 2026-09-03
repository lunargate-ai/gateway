package providers

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/safeurl"
	"github.com/lunargate-ai/gateway/pkg/models"
)

// AnthropicTranslator handles translation between OpenAI format and Anthropic's Messages API.
type AnthropicTranslator struct {
	cfg config.ProviderConfig
}

func NewAnthropicTranslator(cfg config.ProviderConfig) *AnthropicTranslator {
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.anthropic.com"
	}
	if cfg.DefaultModel == "" {
		cfg.DefaultModel = "claude-sonnet-4-6"
	}
	if cfg.APIVersion == "" {
		cfg.APIVersion = anthropicDefaultAPIVersion
	}
	return &AnthropicTranslator{cfg: cfg}
}

func (t *AnthropicTranslator) Name() string {
	return "anthropic"
}

func (t *AnthropicTranslator) DefaultModel() string {
	return t.cfg.DefaultModel
}

func (t *AnthropicTranslator) BaseURL() string {
	return strings.TrimRight(strings.TrimSpace(t.cfg.BaseURL), "/")
}

// --- Anthropic-specific request/response types ---

type anthropicRequest struct {
	Model         string                  `json:"model"`
	MaxTokens     int                     `json:"max_tokens"`
	Messages      []anthropicMessage      `json:"messages"`
	System        []anthropicContentBlock `json:"system,omitempty"`
	Metadata      *anthropicMetadata      `json:"metadata,omitempty"`
	OutputConfig  *anthropicOutputConfig  `json:"output_config,omitempty"`
	Thinking      *anthropicThinking      `json:"thinking,omitempty"`
	Temperature   *float64                `json:"temperature,omitempty"`
	TopP          *float64                `json:"top_p,omitempty"`
	TopK          *int                    `json:"top_k,omitempty"`
	Stream        bool                    `json:"stream,omitempty"`
	StopSequences []string                `json:"stop_sequences,omitempty"`
	Tools         []anthropicTool         `json:"tools,omitempty"`
	ToolChoice    interface{}             `json:"tool_choice,omitempty"`
}

type anthropicMetadata struct {
	UserID string `json:"user_id,omitempty"`
}

type anthropicOutputConfig struct {
	Effort string                     `json:"effort,omitempty"`
	Format *anthropicJSONOutputFormat `json:"format,omitempty"`
}

type anthropicThinking struct {
	Type string `json:"type"`
}

type anthropicJSONOutputFormat struct {
	Type   string      `json:"type"`
	Schema interface{} `json:"schema"`
}

type anthropicMessage struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"`
}

type anthropicResponse struct {
	ID          string                  `json:"id"`
	Type        string                  `json:"type"`
	Role        string                  `json:"role"`
	Content     []anthropicContentBlock `json:"content"`
	Model       string                  `json:"model"`
	StopReason  *string                 `json:"stop_reason"`
	StopDetails *anthropicStopDetails   `json:"stop_details"`
	Usage       anthropicUsage          `json:"usage"`
}

type anthropicStopDetails struct {
	Type        string  `json:"type"`
	Category    *string `json:"category"`
	Explanation *string `json:"explanation"`
}

type anthropicContentBlock struct {
	Type      string      `json:"type"`
	Text      string      `json:"text,omitempty"`
	Source    interface{} `json:"source,omitempty"`
	ID        string      `json:"id,omitempty"`
	Name      string      `json:"name,omitempty"`
	Input     interface{} `json:"input,omitempty"`
	ToolUseID string      `json:"tool_use_id,omitempty"`
	Content   interface{} `json:"content,omitempty"`
	IsError   *bool       `json:"is_error,omitempty"`
}

type anthropicTool struct {
	Name        string      `json:"name"`
	Description string      `json:"description,omitempty"`
	InputSchema interface{} `json:"input_schema,omitempty"`
	Strict      *bool       `json:"strict,omitempty"`
}

type anthropicUsage struct {
	InputTokens              int                     `json:"input_tokens"`
	OutputTokens             int                     `json:"output_tokens"`
	CacheCreationInputTokens int                     `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int                     `json:"cache_read_input_tokens"`
	CacheCreation            *anthropicCacheCreation `json:"cache_creation,omitempty"`
}

type anthropicCacheCreation struct {
	Ephemeral5mInputTokens int `json:"ephemeral_5m_input_tokens"`
	Ephemeral1hInputTokens int `json:"ephemeral_1h_input_tokens"`
}

type anthropicErrorResponse struct {
	Type  string `json:"type"`
	Error struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

// --- Interface implementation ---

// ValidateRequestCompatibility verifies that an OpenAI-compatible request can
// be represented by this exact Anthropic target without silently dropping a
// client control.
func (t *AnthropicTranslator) ValidateRequestCompatibility(providerID string, req *models.UnifiedRequest) error {
	if req == nil {
		return nil
	}
	providerID = strings.TrimSpace(providerID)
	if providerID == "" {
		providerID = "anthropic"
	}
	if err := validateTranslatedChatRawControls(providerID, req); err != nil {
		return err
	}

	unsupported := func(field, reason string) error {
		return &models.CompatibilityError{Field: field, Provider: providerID, Reason: reason}
	}
	if req.N != nil && *req.N != 1 {
		return unsupported("n", "Anthropic Messages returns one completion per request")
	}
	if req.PresencePenalty != nil {
		return unsupported("presence_penalty", "Anthropic Messages has no equivalent presence penalty")
	}
	if req.FrequencyPenalty != nil {
		return unsupported("frequency_penalty", "Anthropic Messages has no equivalent frequency penalty")
	}
	if req.LogitBias != nil {
		return unsupported("logit_bias", "Anthropic Messages has no equivalent token bias control")
	}
	if req.Seed != nil {
		return unsupported("seed", "Anthropic Messages has no deterministic seed control")
	}
	if req.Store != nil && *req.Store && !strings.EqualFold(strings.TrimSpace(req.SourceRequestType), "responses") {
		return unsupported("store", "Anthropic Messages has no per-request storage control")
	}
	if req.PreviousResponseID != "" {
		return unsupported("previous_response_id", "Anthropic Messages has no native Responses continuation")
	}

	if req.Stop != nil {
		if _, ok := mapAnthropicStopSequences(req.Stop); !ok {
			return unsupported("stop", "expected a string or an array of strings")
		}
	}
	if len(req.Functions) > 0 {
		return unsupported("functions", "normalize legacy functions into tools before using Anthropic")
	}
	if req.FunctionCall != nil {
		return unsupported("function_call", "normalize legacy function_call into tool_choice before using Anthropic")
	}
	if err := validateTranslatedChatTypedToolChoice(providerID, req.ToolChoice); err != nil {
		return err
	}
	if err := validateAnthropicTools(providerID, req.Tools); err != nil {
		return err
	}
	if req.ToolChoice != nil {
		if _, ok := mapOpenAIToolChoiceToAnthropic(req.ToolChoice); !ok {
			return unsupported("tool_choice", "the requested tool selection mode has no Anthropic equivalent")
		}
	}

	if effort := strings.ToLower(strings.TrimSpace(req.ReasoningEffort)); effort != "" {
		if !t.cfg.Capabilities.ReasoningEffort {
			return unsupported("reasoning_effort", "enable provider capability reasoning_effort for a model with output_config.effort support")
		}
		if !isAnthropicEffort(effort) {
			return unsupported("reasoning_effort", fmt.Sprintf("unsupported Anthropic effort level %q", req.ReasoningEffort))
		}
		if !anthropicEffortEnabled(t.cfg.Capabilities, effort) {
			return unsupported("reasoning_effort", fmt.Sprintf("Anthropic effort level %q is not enabled for this provider", req.ReasoningEffort))
		}
	}

	if req.ResponseFormat != nil {
		switch formatType := strings.ToLower(strings.TrimSpace(req.ResponseFormat.Type)); formatType {
		case "text":
			// Anthropic's default text response is equivalent.
		case "json_schema":
			if !t.cfg.Capabilities.StructuredOutputs {
				return unsupported("response_format", "enable provider capability structured_outputs for a compatible Anthropic model")
			}
			if _, err := translatedChatAnnotatedJSONSchema(providerID, req.ResponseFormat.JSONSchema); err != nil {
				return err
			}
		default:
			return unsupported("response_format", fmt.Sprintf("Anthropic Messages has no equivalent for response format %q", req.ResponseFormat.Type))
		}
	}

	return validateAnthropicMessageInput(providerID, req.Messages)
}

func validateAnthropicTools(providerID string, tools []models.Tool) error {
	unsupported := func(field, reason string) error {
		return &models.CompatibilityError{Field: field, Provider: providerID, Reason: reason}
	}
	for index := range tools {
		path := fmt.Sprintf("tools[%d]", index)
		tool := tools[index]
		toolType := strings.ToLower(strings.TrimSpace(tool.Type))
		if toolType != "" && toolType != "function" {
			return unsupported(path+".type", "Anthropic only supports function tools")
		}
		if strings.TrimSpace(tool.Function.Name) == "" {
			return unsupported(path+".function.name", "Anthropic requires a function name")
		}
		if tool.Function.Parameters != nil {
			encoded, err := json.Marshal(tool.Function.Parameters)
			if err != nil {
				return unsupported(path+".function.parameters", "function parameters must be a JSON schema object")
			}
			var schema map[string]interface{}
			if err := json.Unmarshal(encoded, &schema); err != nil || schema == nil {
				return unsupported(path+".function.parameters", "function parameters must be a JSON schema object")
			}
		}
	}
	return nil
}

func validateAnthropicMessageInput(providerID string, messages []models.Message) error {
	unsupported := func(field, reason string) error {
		return &models.CompatibilityError{Field: field, Provider: providerID, Reason: reason}
	}

	for messageIndex := range messages {
		messagePath := fmt.Sprintf("messages[%d]", messageIndex)
		message := messages[messageIndex]
		switch message.Role {
		case "system", "developer", "user", "assistant", "tool":
		default:
			return unsupported(messagePath+".role", "Anthropic Messages supports system, developer, user, assistant, and tool messages")
		}
		if message.Name != "" {
			return unsupported(messagePath+".name", "Anthropic Messages cannot preserve Chat Completions participant names")
		}
		if message.Refusal != "" {
			return unsupported(messagePath+".refusal", "Anthropic Messages has no assistant refusal-history field")
		}
		if message.ReasoningContent != "" {
			return unsupported(messagePath+".reasoning_content", "Anthropic Messages requires signed thinking blocks for assistant reasoning history")
		}
		if message.FunctionCall != nil {
			return unsupported(messagePath+".function_call", "normalize legacy function_call into assistant.tool_calls before using Anthropic")
		}
		if message.ToolCallID != "" && message.Role != "tool" {
			return unsupported(messagePath+".tool_call_id", "Anthropic only represents tool_call_id on tool-result messages")
		}
		if len(message.ToolCalls) > 0 && message.Role != "assistant" {
			return unsupported(messagePath+".tool_calls", "Anthropic tool calls belong to assistant messages")
		}
		if message.Role == "tool" && strings.TrimSpace(message.ToolCallID) == "" {
			return unsupported(messagePath+".tool_call_id", "Anthropic tool results require the originating tool-call ID")
		}

		switch message.Content.(type) {
		case nil, string, []interface{}:
		default:
			return unsupported(messagePath+".content", "Anthropic message content must be a string or an array of supported content parts")
		}

		parts, ok := message.Content.([]interface{})
		if ok {
			for partIndex, part := range parts {
				partPath := fmt.Sprintf("%s.content[%d]", messagePath, partIndex)
				encoded, err := json.Marshal(part)
				if err != nil {
					return unsupported(partPath, "Anthropic content parts must be JSON objects")
				}
				var object map[string]interface{}
				if err := json.Unmarshal(encoded, &object); err != nil || object == nil {
					return unsupported(partPath, "Anthropic content parts must be JSON objects")
				}

				partType, ok := object["type"].(string)
				if !ok || strings.TrimSpace(partType) == "" {
					return unsupported(partPath+".type", "content part type is required")
				}
				partType = strings.ToLower(strings.TrimSpace(partType))
				switch partType {
				case "text", "input_text":
					if field := firstUnsupportedTranslatedChatField(object, "text", "type"); field != "" {
						return unsupported(partPath+"."+field, "Anthropic text parts cannot preserve this field")
					}
					if _, ok := object["text"].(string); !ok {
						return unsupported(partPath+".text", "Anthropic text parts require string text")
					}
				case "image_url", "image", "input_image":
					if message.Role != "user" {
						return unsupported(partPath+".type", "Anthropic only accepts image history in user messages")
					}
					if err := validateAnthropicImageContentPart(providerID, object, partPath, partType); err != nil {
						return err
					}
				default:
					return unsupported(partPath+".type", "Anthropic Messages cannot represent this content part type")
				}
			}
		}
		if (message.Role == "user" || message.Role == "assistant") && len(message.ToolCalls) == 0 && anthropicMessageContentIsEmpty(message.Content) {
			return unsupported(messagePath+".content", "Anthropic requires non-empty user and assistant message content when no tool calls are present")
		}

		for toolIndex := range message.ToolCalls {
			callPath := fmt.Sprintf("%s.tool_calls[%d]", messagePath, toolIndex)
			call := message.ToolCalls[toolIndex]
			callType := strings.ToLower(strings.TrimSpace(call.Type))
			if callType != "" && callType != "function" {
				return unsupported(callPath+".type", "Anthropic only supports function tool calls")
			}
			if strings.TrimSpace(call.Function.Name) == "" {
				return unsupported(callPath+".function.name", "Anthropic requires a function name")
			}
			arguments := strings.TrimSpace(call.Function.Arguments)
			if arguments == "" {
				arguments = "{}"
			}
			var input map[string]interface{}
			if err := json.Unmarshal([]byte(arguments), &input); err != nil || input == nil {
				return unsupported(callPath+".function.arguments", "Anthropic requires tool arguments to be a JSON object")
			}
		}
	}

	return nil
}

func anthropicMessageContentIsEmpty(content interface{}) bool {
	switch value := content.(type) {
	case nil:
		return true
	case string:
		return value == ""
	case []interface{}:
		for _, part := range value {
			encoded, err := json.Marshal(part)
			if err != nil {
				return false
			}
			var object map[string]interface{}
			if err := json.Unmarshal(encoded, &object); err != nil {
				return false
			}
			partType, _ := object["type"].(string)
			switch strings.ToLower(strings.TrimSpace(partType)) {
			case "text", "input_text":
				if text, _ := object["text"].(string); text != "" {
					return false
				}
			case "image_url", "image", "input_image":
				return false
			}
		}
		return true
	default:
		return false
	}
}

func validateAnthropicImageContentPart(
	providerID string,
	object map[string]interface{},
	partPath string,
	partType string,
) error {
	unsupported := func(field, reason string) error {
		return &models.CompatibilityError{Field: field, Provider: providerID, Reason: reason}
	}
	allowed := []string{"image_url", "type"}
	if partType != "image_url" {
		allowed = []string{"detail", "image_url", "type", "url"}
	}
	if field := firstUnsupportedTranslatedChatField(object, allowed...); field != "" {
		return unsupported(partPath+"."+field, "Anthropic image parts cannot preserve this field")
	}
	if detail, exists := object["detail"]; exists {
		if !isAutomaticImageDetail(detail) {
			return unsupported(partPath+".detail", "Anthropic cannot enforce OpenAI image detail settings")
		}
	}

	urlValue := ""
	if rawURL, exists := object["url"]; exists {
		value, ok := rawURL.(string)
		if !ok || strings.TrimSpace(value) == "" {
			return unsupported(partPath+".url", "Anthropic image URL must be a non-empty string")
		}
		urlValue = strings.TrimSpace(value)
	}
	imageURLValue := ""
	imageURLField := partPath + ".image_url"
	if reference, exists := object["image_url"]; exists {
		value, err := anthropicImageReferenceURL(providerID, reference, partPath+".image_url")
		if err != nil {
			return err
		}
		imageURLValue = value
		if _, ok := reference.(map[string]interface{}); ok {
			imageURLField += ".url"
		}
	}
	if urlValue == "" && imageURLValue == "" {
		return unsupported(partPath+".image_url", "Anthropic image parts require a non-empty image URL")
	}
	if urlValue != "" && imageURLValue != "" && urlValue != imageURLValue {
		return unsupported(partPath+".image_url", "image_url conflicts with url and would otherwise be ignored")
	}
	selectedURL := urlValue
	selectedField := partPath + ".url"
	if selectedURL == "" {
		selectedURL = imageURLValue
		selectedField = imageURLField
	}
	if !isAnthropicImageURL(selectedURL) {
		return unsupported(selectedField, "Anthropic images require an HTTP(S) URL or a base64 data URL")
	}
	return nil
}

func anthropicImageReferenceURL(providerID string, reference interface{}, path string) (string, error) {
	unsupported := func(field, reason string) error {
		return &models.CompatibilityError{Field: field, Provider: providerID, Reason: reason}
	}
	switch value := reference.(type) {
	case string:
		if strings.TrimSpace(value) == "" {
			return "", unsupported(path, "Anthropic image URL must be a non-empty string")
		}
		return strings.TrimSpace(value), nil
	case map[string]interface{}:
		if field := firstUnsupportedTranslatedChatField(value, "detail", "url"); field != "" {
			return "", unsupported(path+"."+field, "Anthropic image URLs cannot preserve this field")
		}
		if detail, exists := value["detail"]; exists && !isAutomaticImageDetail(detail) {
			return "", unsupported(path+".detail", "Anthropic cannot enforce OpenAI image detail settings")
		}
		rawURL, ok := value["url"].(string)
		if !ok || strings.TrimSpace(rawURL) == "" {
			return "", unsupported(path+".url", "Anthropic image URL must be a non-empty string")
		}
		return strings.TrimSpace(rawURL), nil
	default:
		return "", unsupported(path, "Anthropic image_url must be a string or an object containing url")
	}
}

func isAnthropicImageURL(value string) bool {
	lower := strings.ToLower(strings.TrimSpace(value))
	if strings.HasPrefix(lower, "http://") || strings.HasPrefix(lower, "https://") {
		return true
	}
	if strings.HasPrefix(lower, "data:") {
		_, _, ok := parseDataURL(value)
		return ok
	}
	return false
}

func isAutomaticImageDetail(value interface{}) bool {
	detail, ok := value.(string)
	return ok && (strings.TrimSpace(detail) == "" || strings.EqualFold(strings.TrimSpace(detail), "auto"))
}

func (t *AnthropicTranslator) TranslateRequest(ctx context.Context, req *models.UnifiedRequest) (*http.Request, error) {
	if err := t.ValidateRequestCompatibility("anthropic", req); err != nil {
		return nil, err
	}

	var systemPrompt []anthropicContentBlock
	var messages []anthropicMessage

	for _, msg := range req.Messages {
		// Anthropic represents both OpenAI instruction roles in its top-level
		// system field. Keep each source content segment distinct and ordered.
		if msg.Role == "system" || msg.Role == "developer" {
			blocks, err := openAIInstructionToAnthropicBlocks(&msg)
			if err != nil {
				return nil, fmt.Errorf("failed to translate %s instruction to anthropic blocks: %w", msg.Role, err)
			}
			systemPrompt = append(systemPrompt, blocks...)
			continue
		}

		switch msg.Role {
		case "user", "assistant":
			blocks, err := openAIMessageToAnthropicBlocks(&msg)
			if err != nil {
				return nil, fmt.Errorf("failed to translate message to anthropic blocks: %w", err)
			}
			messages = append(messages, anthropicMessage{Role: msg.Role, Content: blocks})

		case "tool":
			blocks, err := openAIToolResultToAnthropicBlocks(&msg)
			if err != nil {
				return nil, fmt.Errorf("failed to translate tool result to anthropic blocks: %w", err)
			}
			// Tool results are provided as a user message containing tool_result block(s).
			messages = append(messages, anthropicMessage{Role: "user", Content: blocks})
		}
	}

	maxTokens := 4096
	if req.MaxTokens != nil {
		maxTokens = *req.MaxTokens
	}
	temperature := req.Temperature
	if temperature == nil && t.cfg.Temperature != nil {
		v := *t.cfg.Temperature
		temperature = &v
	}
	topP := req.TopP
	if topP == nil && t.cfg.TopP != nil {
		v := *t.cfg.TopP
		topP = &v
	}
	topK := req.TopK
	if topK == nil && t.cfg.TopK != nil {
		v := *t.cfg.TopK
		topK = &v
	}
	var metadata *anthropicMetadata
	if userID := strings.TrimSpace(req.User); userID != "" {
		metadata = &anthropicMetadata{UserID: userID}
	}
	var outputConfig *anthropicOutputConfig
	var thinking *anthropicThinking
	if effort := strings.ToLower(strings.TrimSpace(req.ReasoningEffort)); effort != "" {
		outputConfig = &anthropicOutputConfig{Effort: effort}
		if t.cfg.Capabilities.AdaptiveThinking {
			thinking = &anthropicThinking{Type: "adaptive"}
		}
	}
	if req.ResponseFormat != nil && strings.EqualFold(strings.TrimSpace(req.ResponseFormat.Type), "json_schema") {
		schema, err := translatedChatAnnotatedJSONSchema("anthropic", req.ResponseFormat.JSONSchema)
		if err != nil {
			return nil, err
		}
		if outputConfig == nil {
			outputConfig = &anthropicOutputConfig{}
		}
		outputConfig.Format = &anthropicJSONOutputFormat{
			Type:   "json_schema",
			Schema: schema,
		}
	}
	toolChoice, _ := mapOpenAIToolChoiceToAnthropic(req.ToolChoice)
	stopSequences, _ := mapAnthropicStopSequences(req.Stop)

	anthropicReq := anthropicRequest{
		Model:         req.Model,
		MaxTokens:     maxTokens,
		Messages:      messages,
		System:        systemPrompt,
		Metadata:      metadata,
		OutputConfig:  outputConfig,
		Thinking:      thinking,
		Temperature:   temperature,
		TopP:          topP,
		TopK:          topK,
		Stream:        req.Stream,
		StopSequences: stopSequences,
		Tools:         mapOpenAIToolsToAnthropic(req.Tools),
		ToolChoice:    toolChoice,
	}

	body, err := json.Marshal(anthropicReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal anthropic request: %w", err)
	}

	endpoint, err := safeurl.JoinHTTPPath(t.cfg.BaseURL, "v1/messages")
	if err != nil {
		return nil, fmt.Errorf("failed to build anthropic endpoint: %w", err)
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create anthropic http request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-api-key", t.cfg.APIKey)
	httpReq.Header.Set("anthropic-version", t.cfg.APIVersion)
	applyUpstreamRequestHeaders(ctx, httpReq, "Anthropic-Beta")

	return httpReq, nil
}

func openAIInstructionToAnthropicBlocks(msg *models.Message) ([]anthropicContentBlock, error) {
	blocks, err := openAIMessageToAnthropicBlocks(msg)
	if err != nil {
		return nil, err
	}

	instructions := make([]anthropicContentBlock, 0, len(blocks))
	for _, block := range blocks {
		if block.Type != "text" {
			return nil, fmt.Errorf("unsupported %s instruction content block %q", msg.Role, block.Type)
		}
		if block.Text != "" {
			instructions = append(instructions, block)
		}
	}
	return instructions, nil
}

func (t *AnthropicTranslator) ParseResponse(resp *http.Response) (*models.UnifiedResponse, error) {
	body, err := readUpstreamResponseBody(resp, "anthropic")
	if err != nil {
		return nil, fmt.Errorf("failed to read anthropic response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		var errResp anthropicErrorResponse
		if jsonErr := json.Unmarshal(body, &errResp); jsonErr == nil {
			return nil, &ProviderError{
				StatusCode: resp.StatusCode,
				Message:    errResp.Error.Message,
				Type:       errResp.Error.Type,
				Provider:   "anthropic",
			}
		}
		return nil, &ProviderError{
			StatusCode: resp.StatusCode,
			Message:    string(body),
			Provider:   "anthropic",
		}
	}

	var anthropicResp anthropicResponse
	if err := json.Unmarshal(body, &anthropicResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal anthropic response: %w", err)
	}

	return t.toUnified(&anthropicResp), nil
}

func (t *AnthropicTranslator) toUnified(resp *anthropicResponse) *models.UnifiedResponse {
	var text strings.Builder
	toolCalls := make([]models.ToolCall, 0, 4)
	toolIdx := 0
	for _, c := range resp.Content {
		switch c.Type {
		case "text":
			text.WriteString(c.Text)
		case "tool_use":
			args := "{}"
			if c.Input != nil {
				if b, err := json.Marshal(c.Input); err == nil {
					args = string(b)
				}
			}
			idx := toolIdx
			toolIdx++
			toolCalls = append(toolCalls, models.ToolCall{
				Index: &idx,
				ID:    c.ID,
				Type:  "function",
				Function: models.ToolCallFunction{
					Name:      c.Name,
					Arguments: args,
				},
			})
		}
	}

	finishReason := mapAnthropicStopReason(resp.StopReason, resp.StopDetails)
	refusal := anthropicRefusalExplanation(resp.StopReason, resp.StopDetails)
	if isAnthropicRefusal(resp.StopReason, resp.StopDetails) {
		// Anthropic refusal output may contain a partial answer or tool call that
		// must not be used. Preserve only the refusal signal and explanation.
		text.Reset()
		toolCalls = nil
	}

	return &models.UnifiedResponse{
		ID:      resp.ID,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   resp.Model,
		Choices: []models.Choice{
			{
				Index: 0,
				Message: &models.Message{
					Role:    "assistant",
					Content: text.String(),
					Refusal: refusal,
					ToolCalls: func() []models.ToolCall {
						if len(toolCalls) == 0 {
							return nil
						}
						return toolCalls
					}(),
				},
				FinishReason: finishReason,
			},
		},
		Usage: anthropicUsageToUnified(resp.Usage),
	}
}

func anthropicUsageToUnified(usage anthropicUsage) *models.Usage {
	cacheWrite5m := 0
	cacheWrite1h := 0
	if usage.CacheCreation != nil {
		cacheWrite5m = models.NonNegativeTokenCount(usage.CacheCreation.Ephemeral5mInputTokens)
		cacheWrite1h = models.NonNegativeTokenCount(usage.CacheCreation.Ephemeral1hInputTokens)
	}
	cacheWrite := models.NonNegativeTokenCount(usage.CacheCreationInputTokens)
	if classified := models.SaturatingTokenSum(cacheWrite5m, cacheWrite1h); classified > cacheWrite {
		cacheWrite = classified
	}
	cacheRead := models.NonNegativeTokenCount(usage.CacheReadInputTokens)
	uncachedInput := models.NonNegativeTokenCount(usage.InputTokens)
	output := models.NonNegativeTokenCount(usage.OutputTokens)
	input := models.SaturatingTokenSum(uncachedInput, cacheRead, cacheWrite)

	result := &models.Usage{
		PromptTokens:     input,
		CompletionTokens: output,
		TotalTokens:      models.SaturatingTokenSum(input, output),
	}
	if cacheRead > 0 || cacheWrite > 0 {
		result.PromptTokensDetails = &models.InputTokensDetails{
			CachedTokens:       cacheRead,
			CacheWriteTokens:   cacheWrite,
			CacheWriteTokens5m: cacheWrite5m,
			CacheWriteTokens1h: cacheWrite1h,
		}
	}
	return result
}

func anthropicUsageHasTokens(usage anthropicUsage) bool {
	if usage.InputTokens != 0 || usage.OutputTokens != 0 || usage.CacheCreationInputTokens != 0 || usage.CacheReadInputTokens != 0 {
		return true
	}
	return usage.CacheCreation != nil &&
		(usage.CacheCreation.Ephemeral5mInputTokens != 0 || usage.CacheCreation.Ephemeral1hInputTokens != 0)
}

func (t *AnthropicTranslator) ParseStreamChunk(data []byte) (*models.StreamChunk, error) {
	return nil, fmt.Errorf("anthropic streaming requires a per-request stream translator")
}

func (t *AnthropicTranslator) SupportsStreaming() bool {
	return true
}

func (t *AnthropicTranslator) Models() []models.ModelInfo {
	now := time.Now().Unix()
	return []models.ModelInfo{
		{ID: "claude-fable-5-1", Object: "model", Created: now, OwnedBy: "anthropic"},
		{ID: "claude-opus-5", Object: "model", Created: now, OwnedBy: "anthropic"},
		{ID: "claude-sonnet-5", Object: "model", Created: now, OwnedBy: "anthropic"},
		{ID: "claude-sonnet-4-6", Object: "model", Created: now, OwnedBy: "anthropic"},
		{ID: "claude-haiku-4-5-20251001", Object: "model", Created: now, OwnedBy: "anthropic"},
	}
}

func mapAnthropicStopReason(reason *string, details *anthropicStopDetails) *string {
	if reason == nil {
		return nil
	}
	if details != nil && strings.EqualFold(strings.TrimSpace(details.Type), "refusal") {
		mapped := "content_filter"
		return &mapped
	}
	mapped := "stop"
	switch strings.ToLower(strings.TrimSpace(*reason)) {
	case "end_turn":
		mapped = "stop"
	case "max_tokens", "model_context_window_exceeded":
		mapped = "length"
	case "pause_turn":
		// Chat Completions has no server-tool continuation reason. "length" is
		// its standard signal that the otherwise valid response is incomplete.
		mapped = "length"
	case "stop_sequence":
		mapped = "stop"
	case "tool_use":
		mapped = "tool_calls"
	case "refusal":
		mapped = "content_filter"
	default:
		mapped = *reason
	}
	return &mapped
}

func anthropicRefusalExplanation(reason *string, details *anthropicStopDetails) string {
	if details == nil || details.Explanation == nil {
		return ""
	}
	if !isAnthropicRefusal(reason, details) {
		return ""
	}
	return *details.Explanation
}

func isAnthropicRefusal(reason *string, details *anthropicStopDetails) bool {
	if details != nil && strings.EqualFold(strings.TrimSpace(details.Type), "refusal") {
		return true
	}
	return reason != nil && strings.EqualFold(strings.TrimSpace(*reason), "refusal")
}

func contentToString(v interface{}) string {
	switch vv := v.(type) {
	case string:
		return vv
	case nil:
		return ""
	default:
		b, err := json.Marshal(v)
		if err != nil {
			return ""
		}
		return string(b)
	}
}

func parseDataURL(dataURL string) (mediaType string, data string, ok bool) {
	dataURL = strings.TrimSpace(dataURL)
	if len(dataURL) < len("data:") || !strings.EqualFold(dataURL[:len("data:")], "data:") {
		return "", "", false
	}
	parts := strings.SplitN(dataURL, ",", 2)
	if len(parts) != 2 {
		return "", "", false
	}
	meta := parts[0][len("data:"):]
	data = strings.TrimSpace(parts[1])

	if data == "" {
		return "", "", false
	}

	semi := strings.IndexByte(meta, ';')
	if semi < 0 {
		return "", "", false
	}
	mediaType = strings.ToLower(strings.TrimSpace(meta[:semi]))
	if !isAnthropicBase64ImageMediaType(mediaType) {
		return "", "", false
	}
	flags := strings.Split(meta[semi+1:], ";")
	hasBase64 := false
	for _, flag := range flags {
		if strings.EqualFold(strings.TrimSpace(flag), "base64") {
			hasBase64 = true
			break
		}
	}
	if !hasBase64 {
		return "", "", false
	}
	if _, err := base64.StdEncoding.DecodeString(data); err != nil {
		return "", "", false
	}

	return mediaType, data, true
}

func isAnthropicBase64ImageMediaType(mediaType string) bool {
	switch mediaType {
	case "image/jpeg", "image/png", "image/gif", "image/webp":
		return true
	default:
		return false
	}
}

func openAIMessageToAnthropicBlocks(msg *models.Message) ([]anthropicContentBlock, error) {
	blocks := make([]anthropicContentBlock, 0, 4)

	switch c := msg.Content.(type) {
	case string:
		if c != "" {
			blocks = append(blocks, anthropicContentBlock{Type: "text", Text: c})
		}
	case []interface{}:
		for _, part := range c {
			b, err := json.Marshal(part)
			if err != nil {
				continue
			}
			var obj map[string]interface{}
			if err := json.Unmarshal(b, &obj); err != nil {
				continue
			}
			pt, _ := obj["type"].(string)
			switch pt {
			case "text", "input_text":
				if text, ok := obj["text"].(string); ok {
					if text != "" {
						blocks = append(blocks, anthropicContentBlock{Type: "text", Text: text})
					}
				}
			case "image_url", "image", "input_image":
				urlVal := ""
				if u, ok := obj["url"].(string); ok {
					urlVal = u
				}
				if urlVal == "" {
					switch iv := obj["image_url"].(type) {
					case string:
						urlVal = iv
					case map[string]interface{}:
						if u, ok := iv["url"].(string); ok {
							urlVal = u
						}
					}
				}

				urlVal = strings.TrimSpace(urlVal)
				if urlVal == "" {
					return nil, fmt.Errorf("image content part missing url")
				}

				if mediaType, data, ok := parseDataURL(urlVal); ok {
					blocks = append(blocks, anthropicContentBlock{
						Type: "image",
						Source: map[string]interface{}{
							"type":       "base64",
							"media_type": mediaType,
							"data":       data,
						},
					})
				} else {
					blocks = append(blocks, anthropicContentBlock{
						Type: "image",
						Source: map[string]interface{}{
							"type": "url",
							"url":  urlVal,
						},
					})
				}
			default:
				if text := contentToString(part); text != "" {
					blocks = append(blocks, anthropicContentBlock{Type: "text", Text: text})
				}
			}
		}
	default:
		if s := contentToString(msg.Content); s != "" {
			blocks = append(blocks, anthropicContentBlock{Type: "text", Text: s})
		}
	}

	if msg.Role == "assistant" {
		for _, tc := range msg.ToolCalls {
			input := interface{}(map[string]interface{}{})
			if strings.TrimSpace(tc.Function.Arguments) != "" {
				if err := decodeJSONPreserveNumbers([]byte(tc.Function.Arguments), &input); err != nil {
					return nil, fmt.Errorf("invalid tool arguments JSON for %s: %w", tc.Function.Name, err)
				}
			}
			blocks = append(blocks, anthropicContentBlock{
				Type:  "tool_use",
				ID:    tc.ID,
				Name:  tc.Function.Name,
				Input: input,
			})
		}
	}

	if len(blocks) == 0 {
		blocks = append(blocks, anthropicContentBlock{Type: "text", Text: ""})
	}

	return blocks, nil
}

func openAIToolResultToAnthropicBlocks(msg *models.Message) ([]anthropicContentBlock, error) {
	if msg.ToolCallID == "" {
		return nil, fmt.Errorf("tool result message missing tool_call_id")
	}
	content, err := openAIToolResultContentToAnthropic(msg.Content)
	if err != nil {
		return nil, err
	}
	return []anthropicContentBlock{{
		Type:      "tool_result",
		ToolUseID: msg.ToolCallID,
		Content:   content,
	}}, nil
}

func openAIToolResultContentToAnthropic(content interface{}) (interface{}, error) {
	switch value := content.(type) {
	case nil:
		return nil, nil
	case string:
		if value == "" {
			return nil, nil
		}
		return value, nil
	case []interface{}:
		blocks := make([]anthropicContentBlock, 0, len(value))
		for index, part := range value {
			encoded, err := json.Marshal(part)
			if err != nil {
				return nil, fmt.Errorf("invalid tool result content part %d: %w", index, err)
			}
			var object map[string]interface{}
			if err := json.Unmarshal(encoded, &object); err != nil || object == nil {
				return nil, fmt.Errorf("invalid tool result content part %d", index)
			}
			partType, _ := object["type"].(string)
			if partType != "text" && partType != "input_text" {
				return nil, fmt.Errorf("unsupported tool result content part %q", partType)
			}
			text, ok := object["text"].(string)
			if !ok {
				return nil, fmt.Errorf("tool result content part %d requires string text", index)
			}
			if text == "" {
				continue
			}
			blocks = append(blocks, anthropicContentBlock{Type: "text", Text: text})
		}
		if len(blocks) == 0 {
			return nil, nil
		}
		return blocks, nil
	default:
		return nil, fmt.Errorf("unsupported tool result content type %T", content)
	}
}

func mapOpenAIToolsToAnthropic(tools []models.Tool) []anthropicTool {
	if len(tools) == 0 {
		return nil
	}
	out := make([]anthropicTool, 0, len(tools))
	for _, t := range tools {
		if t.Type != "" && t.Type != "function" {
			continue
		}
		out = append(out, anthropicTool{
			Name:        t.Function.Name,
			Description: t.Function.Description,
			InputSchema: t.Function.Parameters,
			Strict:      t.Function.Strict,
		})
	}
	return out
}

func mapOpenAIToolChoiceToAnthropic(choice interface{}) (interface{}, bool) {
	if choice == nil {
		return nil, true
	}
	if s, ok := choice.(string); ok {
		switch strings.ToLower(strings.TrimSpace(s)) {
		case "auto":
			return map[string]interface{}{"type": "auto"}, true
		case "none":
			return map[string]interface{}{"type": "none"}, true
		case "any":
			return map[string]interface{}{"type": "any"}, true
		case "required":
			return map[string]interface{}{"type": "any"}, true
		default:
			return nil, false
		}
	}

	b, err := json.Marshal(choice)
	if err != nil {
		return nil, false
	}
	var obj struct {
		Type     string `json:"type"`
		Function *struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	if err := json.Unmarshal(b, &obj); err != nil {
		return nil, false
	}
	switch strings.ToLower(strings.TrimSpace(obj.Type)) {
	case "function":
		if obj.Function == nil || strings.TrimSpace(obj.Function.Name) == "" {
			return nil, false
		}
		return map[string]interface{}{
			"type": "tool",
			"name": strings.TrimSpace(obj.Function.Name),
		}, true
	case "required", "any":
		return map[string]interface{}{"type": "any"}, true
	case "auto":
		return map[string]interface{}{"type": "auto"}, true
	case "none":
		return map[string]interface{}{"type": "none"}, true
	default:
		return nil, false
	}
}

func mapAnthropicStopSequences(stop interface{}) ([]string, bool) {
	switch value := stop.(type) {
	case nil:
		return nil, true
	case string:
		return []string{value}, true
	case []string:
		return append([]string(nil), value...), true
	case []interface{}:
		sequences := make([]string, 0, len(value))
		for _, item := range value {
			sequence, ok := item.(string)
			if !ok {
				return nil, false
			}
			sequences = append(sequences, sequence)
		}
		return sequences, true
	default:
		return nil, false
	}
}

func isAnthropicEffort(effort string) bool {
	switch strings.ToLower(strings.TrimSpace(effort)) {
	case "low", "medium", "high", "xhigh", "max":
		return true
	default:
		return false
	}
}

func anthropicEffortEnabled(capabilities config.ProviderCapabilities, effort string) bool {
	effort = strings.ToLower(strings.TrimSpace(effort))
	if !capabilities.ReasoningEffort || !isAnthropicEffort(effort) {
		return false
	}
	if len(capabilities.ReasoningEffortLevels) == 0 {
		switch effort {
		case "low", "medium", "high":
			return true
		default:
			return false
		}
	}
	for _, enabled := range capabilities.ReasoningEffortLevels {
		if strings.EqualFold(strings.TrimSpace(enabled), effort) {
			return true
		}
	}
	return false
}
