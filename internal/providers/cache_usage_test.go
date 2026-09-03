package providers

import (
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestOpenAIParsesPromptCacheUsage(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{
			name: "chat completions",
			body: `{"id":"chatcmpl-cache","object":"chat.completion","created":1,"model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"completion_tokens":20,"total_tokens":120,"prompt_tokens_details":{"cached_tokens":40,"cache_write_tokens":30}}}`,
		},
		{
			name: "responses",
			body: `{"id":"resp-cache","object":"response","created_at":1,"status":"completed","model":"gpt-4o","output":[{"type":"message","id":"msg-cache","status":"completed","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],"output_text":"ok","usage":{"input_tokens":100,"output_tokens":20,"total_tokens":120,"input_tokens_details":{"cached_tokens":40,"cache_write_tokens":30}}}`,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			response, err := NewOpenAITranslator(config.ProviderConfig{}).ParseResponse(&http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(strings.NewReader(test.body)),
			})
			if err != nil {
				t.Fatalf("ParseResponse returned error: %v", err)
			}
			assertCacheUsage(t, response.Usage, 100, 20, 40, 30, 0, 0)
		})
	}
}

func TestAnthropicParsesCacheUsageWithoutDoubleCounting(t *testing.T) {
	body := `{
		"id":"msg-cache","type":"message","role":"assistant","model":"claude-sonnet-4-6",
		"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn",
		"usage":{"input_tokens":100,"output_tokens":20,"cache_read_input_tokens":40,"cache_creation_input_tokens":30,"cache_creation":{"ephemeral_5m_input_tokens":10,"ephemeral_1h_input_tokens":5}}
	}`
	response, err := NewAnthropicTranslator(config.ProviderConfig{}).ParseResponse(&http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(body)),
	})
	if err != nil {
		t.Fatalf("ParseResponse returned error: %v", err)
	}
	assertCacheUsage(t, response.Usage, 170, 20, 40, 30, 10, 5)
}

func TestAnthropicStreamParsesCacheUsage(t *testing.T) {
	translator := NewAnthropicStreamTranslator(NewAnthropicTranslator(config.ProviderConfig{}))
	start, err := translator.ParseStreamChunk([]byte(`{
		"type":"message_start",
		"message":{"id":"msg-cache","type":"message","role":"assistant","model":"claude-sonnet-4-6","content":[],"usage":{"input_tokens":100,"output_tokens":0,"cache_read_input_tokens":40,"cache_creation_input_tokens":30,"cache_creation":{"ephemeral_5m_input_tokens":10,"ephemeral_1h_input_tokens":5}}}
	}`))
	if err != nil {
		t.Fatalf("parse message_start: %v", err)
	}
	assertCacheUsage(t, start.Usage, 170, 0, 40, 30, 10, 5)

	delta, err := translator.ParseStreamChunk([]byte(`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":20}}`))
	if err != nil {
		t.Fatalf("parse message_delta: %v", err)
	}
	assertCacheUsage(t, delta.Usage, 0, 20, 0, 0, 0, 0)
}

func assertCacheUsage(t *testing.T, usage *models.Usage, input, output, cached, cacheWrite, cacheWrite5m, cacheWrite1h int) {
	t.Helper()
	if usage == nil {
		t.Fatal("usage is nil")
	}
	models.NormalizeUsage(usage)
	if usage.PromptTokens != input || usage.CompletionTokens != output || usage.TotalTokens != models.SaturatingTokenSum(input, output) {
		t.Fatalf("usage totals = %#v, want input=%d output=%d", usage, input, output)
	}
	accounting := models.TokenUsageFromUsage(usage)
	if accounting.CachedInputTokens != cached || accounting.CacheWriteInputTokens != cacheWrite ||
		accounting.CacheWriteInputTokens5m != cacheWrite5m || accounting.CacheWriteInputTokens1h != cacheWrite1h {
		t.Fatalf("cache usage = %#v, want cached=%d write=%d 5m=%d 1h=%d", accounting, cached, cacheWrite, cacheWrite5m, cacheWrite1h)
	}
}
