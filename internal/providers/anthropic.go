package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/lunargate-ai/gateway/internal/config"
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
		cfg.DefaultModel = "claude-3-sonnet-20240229"
	}
	if cfg.APIVersion == "" {
		cfg.APIVersion = "2023-06-01"
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
	Model         string             `json:"model"`
	MaxTokens     int                `json:"max_tokens"`
	Messages      []anthropicMessage `json:"messages"`
	System        string             `json:"system,omitempty"`
	Temperature   *float64           `json:"temperature,omitempty"`
	TopP          *float64           `json:"top_p,omitempty"`
	Stream        bool               `json:"stream,omitempty"`
	StopSequences []string           `json:"stop_sequences,omitempty"`
	Tools         []anthropicTool     `json:"tools,omitempty"`
	ToolChoice    interface{}         `json:"tool_choice,omitempty"`
}

type anthropicMessage struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"`
}

type anthropicResponse struct {
	ID           string               `json:"id"`
	Type         string               `json:"type"`
	Role         string               `json:"role"`
	Content      []anthropicContentBlock `json:"content"`
	Model        string               `json:"model"`
	StopReason   *string              `json:"stop_reason"`
	Usage        anthropicUsage       `json:"usage"`
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
}

type anthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

type anthropicErrorResponse struct {
	Type  string `json:"type"`
	Error struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

// --- Interface implementation ---

func (t *AnthropicTranslator) TranslateRequest(ctx context.Context, req *models.UnifiedRequest) (*http.Request, error) {
	var systemPrompt string
	var messages []anthropicMessage

	for _, msg := range req.Messages {
		// Anthropic uses top-level system prompt.
		if msg.Role == "system" {
			if s := contentToString(msg.Content); s != "" {
				if systemPrompt == "" {
					systemPrompt = s
				} else {
					systemPrompt += "\n" + s
				}
			}
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

	anthropicReq := anthropicRequest{
		Model:       req.Model,
		MaxTokens:   maxTokens,
		Messages:    messages,
		System:      systemPrompt,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Stream:      req.Stream,
		Tools:       mapOpenAIToolsToAnthropic(req.Tools),
		ToolChoice:  mapOpenAIToolChoiceToAnthropic(req.ToolChoice),
	}

	// Handle stop sequences
	if req.Stop != nil {
		switch v := req.Stop.(type) {
		case string:
			anthropicReq.StopSequences = []string{v}
		case []interface{}:
			for _, s := range v {
				if str, ok := s.(string); ok {
					anthropicReq.StopSequences = append(anthropicReq.StopSequences, str)
				}
			}
		}
	}

	body, err := json.Marshal(anthropicReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal anthropic request: %w", err)
	}

	endpoint := fmt.Sprintf("%s/v1/messages", t.cfg.BaseURL)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create anthropic http request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-api-key", t.cfg.APIKey)
	httpReq.Header.Set("anthropic-version", t.cfg.APIVersion)

	return httpReq, nil
}

func (t *AnthropicTranslator) ParseResponse(resp *http.Response) (*models.UnifiedResponse, error) {
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
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

	finishReason := mapAnthropicStopReason(resp.StopReason)

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
		Usage: &models.Usage{
			PromptTokens:     resp.Usage.InputTokens,
			CompletionTokens: resp.Usage.OutputTokens,
			TotalTokens:      resp.Usage.InputTokens + resp.Usage.OutputTokens,
		},
	}
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
		{ID: "claude-3-opus-20240229", Object: "model", Created: now, OwnedBy: "anthropic"},
		{ID: "claude-3-sonnet-20240229", Object: "model", Created: now, OwnedBy: "anthropic"},
		{ID: "claude-3-haiku-20240307", Object: "model", Created: now, OwnedBy: "anthropic"},
		{ID: "claude-3-5-sonnet-20241022", Object: "model", Created: now, OwnedBy: "anthropic"},
	}
}

func mapAnthropicStopReason(reason *string) *string {
	if reason == nil {
		return nil
	}
	mapped := "stop"
	switch *reason {
	case "end_turn":
		mapped = "stop"
	case "max_tokens":
		mapped = "length"
	case "stop_sequence":
		mapped = "stop"
	case "tool_use":
		mapped = "tool_calls"
	default:
		mapped = *reason
	}
	return &mapped
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
	if !strings.HasPrefix(dataURL, "data:") {
		return "", "", false
	}
	parts := strings.SplitN(dataURL, ",", 2)
	if len(parts) != 2 {
		return "", "", false
	}
	meta := strings.TrimPrefix(parts[0], "data:")
	data = strings.TrimSpace(parts[1])

	if data == "" {
		return "", "", false
	}

	semi := strings.IndexByte(meta, ';')
	if semi < 0 {
		return "", "", false
	}
	mediaType = meta[:semi]
	flags := meta[semi+1:]
	if mediaType == "" {
		return "", "", false
	}
	if !strings.Contains(flags, "base64") {
		return "", "", false
	}

	return mediaType, data, true
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
			if tc.Function.Name == "" {
				continue
			}
			var input interface{}
			if tc.Function.Arguments != "" {
				if err := json.Unmarshal([]byte(tc.Function.Arguments), &input); err != nil {
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
	content := contentToString(msg.Content)
	return []anthropicContentBlock{{
		Type:      "tool_result",
		ToolUseID: msg.ToolCallID,
		Content:   content,
	}}, nil
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
		})
	}
	return out
}

func mapOpenAIToolChoiceToAnthropic(choice interface{}) interface{} {
	if choice == nil {
		return nil
	}
	if s, ok := choice.(string); ok {
		switch s {
		case "auto":
			return map[string]interface{}{"type": "auto"}
		case "none":
			return map[string]interface{}{"type": "none"}
		case "any":
			return map[string]interface{}{"type": "any"}
		case "required":
			return map[string]interface{}{"type": "any"}
		default:
			return nil
		}
	}

	b, err := json.Marshal(choice)
	if err != nil {
		return nil
	}
	var obj struct {
		Type     string `json:"type"`
		Function *struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	if err := json.Unmarshal(b, &obj); err != nil {
		return nil
	}
	if obj.Type == "function" && obj.Function != nil && obj.Function.Name != "" {
		return map[string]interface{}{
			"type": "tool",
			"name": obj.Function.Name,
		}
	}
	if obj.Type == "required" {
		return map[string]interface{}{"type": "any"}
	}
	return nil
}
