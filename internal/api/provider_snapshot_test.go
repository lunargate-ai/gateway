package api

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/middleware"
	"github.com/lunargate-ai/gateway/internal/observability"
	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/internal/resilience"
	"github.com/lunargate-ai/gateway/internal/routing"
	"github.com/lunargate-ai/gateway/internal/streaming"
	"github.com/lunargate-ai/gateway/pkg/models"
	"github.com/prometheus/client_golang/prometheus"
)

func TestChatCompletionsPinsProviderSnapshotAcrossReload(t *testing.T) {
	var registry *providers.Registry
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if ok := registry.UpdateProvidersConfig(map[string]config.ProviderConfig{
			"mutable": {Type: "anthropic", APIKey: "new-key"},
		}); !ok {
			http.Error(w, "reload failed", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"chatcmpl-stable","object":"chat.completion","created":1,"model":"gpt-stable","choices":[{"index":0,"message":{"role":"assistant","content":"stable-response"},"finish_reason":"stop"}]}`)
	}))
	defer upstream.Close()

	registry = providers.NewRegistry(map[string]config.ProviderConfig{
		"mutable": {Type: "openai", APIKey: "old-key", BaseURL: upstream.URL},
	})
	handler := newProviderSnapshotTestHandler(t, registry, "/v1/chat/completions", "gpt-stable")
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(
		`{"model":"gpt-stable","messages":[{"role":"user","content":"hello"}]}`,
	))
	rec := httptest.NewRecorder()

	handler.ChatCompletions(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	var response models.UnifiedResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(response.Choices) != 1 || response.Choices[0].Message == nil || response.Choices[0].Message.Content != "stable-response" {
		t.Fatalf("response parsed with wrong registry generation: %#v", response)
	}
}

func TestStreamingPinsProviderSnapshotAcrossReload(t *testing.T) {
	var registry *providers.Registry
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if ok := registry.UpdateProvidersConfig(map[string]config.ProviderConfig{
			"mutable": {Type: "ollama"},
		}); !ok {
			http.Error(w, "reload failed", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"id\":\"chatcmpl-stable\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-stable\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"stable-stream\"},\"finish_reason\":null}]}\n\n")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()

	registry = providers.NewRegistry(map[string]config.ProviderConfig{
		"mutable": {Type: "openai", APIKey: "old-key", BaseURL: upstream.URL},
	})
	handler := newProviderSnapshotTestHandler(t, registry, "/v1/chat/completions", "gpt-stable")
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(
		`{"model":"gpt-stable","stream":true,"messages":[{"role":"user","content":"hello"}]}`,
	))
	rec := httptest.NewRecorder()

	handler.ChatCompletions(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "stable-stream") {
		t.Fatalf("stream parsed with wrong registry generation: %s", rec.Body.String())
	}
}

func TestEmbeddingsPinsProviderSnapshotAcrossReload(t *testing.T) {
	var registry *providers.Registry
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if ok := registry.UpdateProvidersConfig(map[string]config.ProviderConfig{
			"mutable": {Type: "anthropic", APIKey: "new-key"},
		}); !ok {
			http.Error(w, "reload failed", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"object":"list","data":[{"object":"embedding","embedding":[0.25,0.75],"index":0}],"model":"embedding-stable","usage":{"prompt_tokens":2,"total_tokens":2}}`)
	}))
	defer upstream.Close()

	registry = providers.NewRegistry(map[string]config.ProviderConfig{
		"mutable": {Type: "openai", APIKey: "old-key", BaseURL: upstream.URL},
	})
	handler := newProviderSnapshotTestHandler(t, registry, "/v1/embeddings", "embedding-stable")
	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewBufferString(
		`{"model":"embedding-stable","input":"hello"}`,
	))
	rec := httptest.NewRecorder()

	handler.Embeddings(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	var response models.EmbeddingsResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(response.Data) != 1 {
		t.Fatalf("response parsed with wrong registry generation: %#v", response)
	}
}

func newProviderSnapshotTestHandler(t *testing.T, registry *providers.Registry, path string, model string) *Handler {
	t.Helper()
	router := routing.NewEngine(config.RoutingConfig{
		DefaultStrategy: "weighted",
		Routes: []config.RouteConfig{{
			Name:    "snapshot-test",
			Match:   config.MatchConfig{Path: path},
			Targets: []config.TargetConfig{{Provider: "mutable", Model: model, Weight: 1}},
		}},
	})
	cache := middleware.NewCache(config.CacheConfig{Enabled: false})
	t.Cleanup(cache.Stop)
	return NewHandler(
		registry,
		router,
		resilience.NewFallbackExecutor(
			resilience.NewRetrier(config.RetryConfig{Enabled: false}),
			resilience.NewCircuitBreakerManager(),
		),
		cache,
		streaming.NewHandler(),
		observability.NewMetricsWithRegisterer(prometheus.NewRegistry()),
		nil,
		nil,
		nil,
	)
}
