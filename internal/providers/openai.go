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

	body, err := json.Marshal(&reqCopy)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal openai request: %w", err)
	}

	endpoint := fmt.Sprintf("%s/chat/completions", t.cfg.BaseURL)
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
