package providers

import (
	"context"
	"net/http"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestOpenAITranslatorForwardsOnlySupportedControlHeaders(t *testing.T) {
	headers := http.Header{
		"Authorization":       {"Bearer client-secret"},
		"Cookie":              {"session=client-secret"},
		"Idempotency-Key":     {"request-one", "request-two"},
		"OpenAI-Beta":         {"responses=v1", "assistants=v2"},
		"Anthropic-Beta":      {"prompt-caching-2024-07-31"},
		"OpenAI-Organization": {"client-organization"},
		"X-Api-Key":           {"client-secret"},
	}
	ctx := WithUpstreamRequestHeaders(context.Background(), headers)
	translator := NewOpenAITranslator(config.ProviderConfig{
		APIKey:       "configured-secret",
		BaseURL:      "https://api.openai.com/v1",
		Organization: "configured-organization",
	})

	assertRequest := func(t *testing.T, request *http.Request) {
		t.Helper()
		if got := request.Header.Values("Idempotency-Key"); len(got) != 2 || got[0] != "request-one" || got[1] != "request-two" {
			t.Fatalf("Idempotency-Key = %#v", got)
		}
		if got := request.Header.Values("OpenAI-Beta"); len(got) != 2 || got[0] != "responses=v1" || got[1] != "assistants=v2" {
			t.Fatalf("OpenAI-Beta = %#v", got)
		}
		if got := request.Header.Get("Authorization"); got != "Bearer configured-secret" {
			t.Fatalf("Authorization = %q", got)
		}
		if got := request.Header.Get("OpenAI-Organization"); got != "configured-organization" {
			t.Fatalf("OpenAI-Organization = %q", got)
		}
		for _, name := range []string{"Anthropic-Beta", "Cookie", "X-Api-Key"} {
			if got := request.Header.Values(name); len(got) != 0 {
				t.Fatalf("unsafe header %s = %#v", name, got)
			}
		}
	}

	chatRequest, err := translator.TranslateRequest(ctx, &models.UnifiedRequest{
		Model:    "gpt-5.4",
		Messages: []models.Message{{Role: "user", Content: "hello"}},
	})
	if err != nil {
		t.Fatalf("TranslateRequest: %v", err)
	}
	assertRequest(t, chatRequest)

	responsesRequest, err := translator.TranslateRequest(WithUpstreamRequestType(ctx, "responses"), &models.UnifiedRequest{
		Model:    "gpt-5.4",
		Messages: []models.Message{{Role: "user", Content: "hello"}},
	})
	if err != nil {
		t.Fatalf("TranslateRequest responses: %v", err)
	}
	assertRequest(t, responsesRequest)

	embeddingsRequest, err := translator.TranslateEmbeddingsRequest(ctx, &models.EmbeddingsRequest{
		Model: "text-embedding-3-small",
		Input: "hello",
	})
	if err != nil {
		t.Fatalf("TranslateEmbeddingsRequest: %v", err)
	}
	assertRequest(t, embeddingsRequest)
}

func TestAnthropicTranslatorForwardsOnlyAnthropicBeta(t *testing.T) {
	headers := http.Header{
		"Authorization":   {"Bearer client-secret"},
		"Cookie":          {"session=client-secret"},
		"Idempotency-Key": {"must-not-forward"},
		"OpenAI-Beta":     {"must-not-forward"},
		"Anthropic-Beta":  {"prompt-caching-2024-07-31", "interleaved-thinking-2025-05-14"},
		"X-Api-Key":       {"client-secret"},
	}
	ctx := WithUpstreamRequestHeaders(context.Background(), headers)
	translator := NewAnthropicTranslator(config.ProviderConfig{
		APIKey:     "configured-secret",
		BaseURL:    "https://api.anthropic.com",
		APIVersion: "2023-06-01",
	})

	request, err := translator.TranslateRequest(ctx, &models.UnifiedRequest{
		Model:    "claude-sonnet-4-5",
		Messages: []models.Message{{Role: "user", Content: "hello"}},
	})
	if err != nil {
		t.Fatalf("TranslateRequest: %v", err)
	}
	if got := request.Header.Values("Anthropic-Beta"); len(got) != 2 || got[0] != "prompt-caching-2024-07-31" || got[1] != "interleaved-thinking-2025-05-14" {
		t.Fatalf("Anthropic-Beta = %#v", got)
	}
	if got := request.Header.Get("X-Api-Key"); got != "configured-secret" {
		t.Fatalf("X-Api-Key = %q", got)
	}
	if got := request.Header.Get("Anthropic-Version"); got != "2023-06-01" {
		t.Fatalf("Anthropic-Version = %q", got)
	}
	for _, name := range []string{"Authorization", "Cookie", "Idempotency-Key", "OpenAI-Beta"} {
		if got := request.Header.Values(name); len(got) != 0 {
			t.Fatalf("unsafe header %s = %#v", name, got)
		}
	}
}

func TestOllamaTranslatorDoesNotForwardProviderControlHeaders(t *testing.T) {
	ctx := WithUpstreamRequestHeaders(context.Background(), http.Header{
		"Idempotency-Key": {"must-not-forward"},
		"OpenAI-Beta":     {"must-not-forward"},
		"Anthropic-Beta":  {"must-not-forward"},
	})
	translator := NewOllamaTranslator(config.ProviderConfig{
		BaseURL: "http://localhost:11434",
	})

	request, err := translator.TranslateRequest(ctx, &models.UnifiedRequest{
		Model:    "llama3.2",
		Messages: []models.Message{{Role: "user", Content: "hello"}},
	})
	if err != nil {
		t.Fatalf("TranslateRequest: %v", err)
	}
	for _, name := range []string{"Idempotency-Key", "OpenAI-Beta", "Anthropic-Beta"} {
		if got := request.Header.Values(name); len(got) != 0 {
			t.Fatalf("header %s = %#v", name, got)
		}
	}
}
