package providers

import (
	"context"
	"net/http"
	"strings"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestProviderEndpointsPreserveBaseQueryAfterJoiningPath(t *testing.T) {
	const baseURL = "https://url-user:url-password@example.test/root/?api_key=query-secret#unused-fragment"
	chatRequest := &models.UnifiedRequest{
		Model:    "test-model",
		Messages: []models.Message{{Role: "user", Content: "hello"}},
	}
	embeddingsRequest := &models.EmbeddingsRequest{Model: "test-model", Input: "hello"}

	tests := []struct {
		name     string
		build    func() (*http.Request, error)
		wantPath string
	}{
		{
			name: "OpenAI Chat Completions",
			build: func() (*http.Request, error) {
				return NewOpenAITranslator(config.ProviderConfig{BaseURL: baseURL}).TranslateRequest(context.Background(), chatRequest)
			},
			wantPath: "/root/chat/completions",
		},
		{
			name: "OpenAI Responses",
			build: func() (*http.Request, error) {
				ctx := WithUpstreamRequestType(context.Background(), "responses")
				return NewOpenAITranslator(config.ProviderConfig{BaseURL: baseURL}).TranslateRequest(ctx, chatRequest)
			},
			wantPath: "/root/responses",
		},
		{
			name: "OpenAI embeddings",
			build: func() (*http.Request, error) {
				return NewOpenAITranslator(config.ProviderConfig{BaseURL: baseURL}).TranslateEmbeddingsRequest(context.Background(), embeddingsRequest)
			},
			wantPath: "/root/embeddings",
		},
		{
			name: "Anthropic messages",
			build: func() (*http.Request, error) {
				return NewAnthropicTranslator(config.ProviderConfig{BaseURL: baseURL}).TranslateRequest(context.Background(), chatRequest)
			},
			wantPath: "/root/v1/messages",
		},
		{
			name: "Ollama chat",
			build: func() (*http.Request, error) {
				return NewOllamaTranslator(config.ProviderConfig{BaseURL: baseURL}).TranslateRequest(context.Background(), chatRequest)
			},
			wantPath: "/root/api/chat",
		},
		{
			name: "Ollama embeddings",
			build: func() (*http.Request, error) {
				return NewOllamaTranslator(config.ProviderConfig{BaseURL: baseURL}).TranslateEmbeddingsRequest(context.Background(), embeddingsRequest)
			},
			wantPath: "/root/api/embed",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request, err := test.build()
			if err != nil {
				t.Fatalf("build provider request: %v", err)
			}
			if got := request.URL.Path; got != test.wantPath {
				t.Fatalf("path = %q, want %q", got, test.wantPath)
			}
			if got := request.URL.RawQuery; got != "api_key=query-secret" {
				t.Fatalf("query = %q, want preserved base query", got)
			}
			if request.URL.User == nil || request.URL.User.Username() != "url-user" {
				t.Fatalf("userinfo was not preserved for transport: %#v", request.URL.User)
			}
			if request.URL.Fragment != "" {
				t.Fatalf("fragment = %q, want discarded", request.URL.Fragment)
			}
		})
	}
}

func TestProviderEndpointBuildErrorDoesNotEchoBaseURL(t *testing.T) {
	const secret = "provider-base-url-secret"
	translator := NewOpenAITranslator(config.ProviderConfig{
		BaseURL: "https://example.test/%zz?api_key=" + secret,
	})

	_, err := translator.TranslateRequest(context.Background(), &models.UnifiedRequest{
		Model:    "test-model",
		Messages: []models.Message{{Role: "user", Content: "hello"}},
	})
	if err == nil {
		t.Fatal("invalid provider base URL returned no error")
	}
	if strings.Contains(err.Error(), secret) {
		t.Fatalf("endpoint error leaked base URL: %v", err)
	}
}
