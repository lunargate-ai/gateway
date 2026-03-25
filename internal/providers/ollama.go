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
	Messages []ollamaMessage         `json:"messages"`
	Stream   bool                   `json:"stream"`
	Tools    []models.Tool          `json:"tools,omitempty"`
	Format   interface{}            `json:"format,omitempty"`
	Options  map[string]interface{} `json:"options,omitempty"`
}

type ollamaMessage struct {
	Role      string           `json:"role"`
	Content   string           `json:"content"`
	Thinking  string           `json:"thinking,omitempty"`
	Reasoning string           `json:"reasoning,omitempty"`
	ToolCalls []ollamaToolCall `json:"tool_calls,omitempty"`
}

type ollamaToolCall struct {
	Function ollamaToolFunction `json:"function"`
}

type ollamaToolFunction struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments,omitempty"`
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
	msgs := make([]ollamaMessage, 0, len(req.Messages))
	for i := range req.Messages {
		m := req.Messages[i]
		msgs = append(msgs, ollamaMessage{Role: m.Role, Content: messageContentToString(m.Content)})
	}

	options := make(map[string]interface{}, 4)
	if req.Temperature != nil {
		options["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		options["top_p"] = *req.TopP
	}
	if req.MaxTokens != nil {
		options["num_predict"] = *req.MaxTokens
	}
	if len(options) == 0 {
		options = nil
	}

	var format interface{}
	if req.ResponseFormat != nil {
		if strings.TrimSpace(req.ResponseFormat.Type) == "json" || strings.TrimSpace(req.ResponseFormat.Type) == "json_object" {
			format = "json"
		}
	}

	ollamaReq := ollamaChatRequest{
		Model:    req.Model,
		Messages: msgs,
		Stream:   req.Stream,
		Tools:    req.Tools,
		Format:   format,
		Options:  options,
	}

	body, err := json.Marshal(ollamaReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal ollama request: %w", err)
	}

	endpoint := fmt.Sprintf("%s/api/chat", strings.TrimRight(strings.TrimSpace(t.cfg.BaseURL), "/"))
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create ollama http request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	return httpReq, nil
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

	endpoint := fmt.Sprintf("%s/api/embed", strings.TrimRight(strings.TrimSpace(t.cfg.BaseURL), "/"))
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create ollama embeddings http request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	return httpReq, nil
}

func (t *OllamaTranslator) ParseResponse(resp *http.Response) (*models.UnifiedResponse, error) {
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
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
			TotalTokens:      result.PromptEvalCount + result.EvalCount,
		}
	}

	id := fmt.Sprintf("chatcmpl_ollama_%d", time.Now().UnixNano())
	created := time.Now().Unix()

	toolCalls := make([]models.ToolCall, 0, len(result.Message.ToolCalls))
	for i := range result.Message.ToolCalls {
		idx := i
		callID := fmt.Sprintf("call_%s_%d", id, i)
		args := "{}"
		if len(result.Message.ToolCalls[i].Function.Arguments) > 0 {
			args = string(result.Message.ToolCalls[i].Function.Arguments)
		}
		toolCalls = append(toolCalls, models.ToolCall{
			Index: &idx,
			ID:    callID,
			Type:  "function",
			Function: models.ToolCallFunction{
				Name:      result.Message.ToolCalls[i].Function.Name,
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

func (t *OllamaTranslator) ParseEmbeddingsResponse(resp *http.Response) (*models.EmbeddingsResponse, error) {
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
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
		data = append(data, models.EmbeddingData{Object: "embedding", Embedding: vector, Index: i})
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

func messageContentToString(content interface{}) string {
	switch v := content.(type) {
	case string:
		return v
	case nil:
		return ""
	case []interface{}:
		var b strings.Builder
		for i := range v {
			m, ok := v[i].(map[string]interface{})
			if !ok {
				continue
			}
			pt, _ := m["type"].(string)
			if pt != "text" {
				continue
			}
			txt, _ := m["text"].(string)
			if txt == "" {
				continue
			}
			b.WriteString(txt)
		}
		return b.String()
	default:
		return ""
	}
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
