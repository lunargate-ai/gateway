package api

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
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

func TestChatCircuitBreakerMovesToChangedProviderIdentity(t *testing.T) {
	var failedCalls atomic.Int32
	failingUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		failedCalls.Add(1)
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = io.WriteString(w, `{"error":{"message":"unavailable","type":"server_error"}}`)
	}))
	defer failingUpstream.Close()

	var successCalls atomic.Int32
	successUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		successCalls.Add(1)
		if got := r.Header.Get("Authorization"); got != "Bearer secret-new" {
			http.Error(w, "wrong provider credential", http.StatusUnauthorized)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"chatcmpl-new","object":"chat.completion","created":1,"model":"gpt-test","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer successUpstream.Close()

	initial := map[string]config.ProviderConfig{
		"shared": {
			Type:         "openai",
			APIKey:       "secret-old",
			BaseURL:      failingUpstream.URL,
			DefaultModel: "gpt-test",
		},
	}
	registry := providers.NewRegistry(initial)
	handler := newCircuitBreakerIdentityTestHandler(t, registry, initial, "/v1/chat/completions", "gpt-test")

	for i := 0; i < 5; i++ {
		if status := performChatIdentityRequest(handler); status == http.StatusOK {
			t.Fatalf("failure request %d unexpectedly succeeded", i+1)
		}
	}
	if got := failedCalls.Load(); got != 5 {
		t.Fatalf("failing upstream calls = %d, want 5", got)
	}

	identical := map[string]config.ProviderConfig{"shared": initial["shared"]}
	if changed := registry.UpdateProvidersConfig(identical); changed {
		t.Fatal("identical provider reload reported a change")
	}
	if status := performChatIdentityRequest(handler); status == http.StatusOK {
		t.Fatal("open identity unexpectedly recovered after no-op reload")
	}
	if got := failedCalls.Load(); got != 5 {
		t.Fatalf("no-op reload bypassed open breaker; calls = %d", got)
	}

	modelOnly := initial["shared"]
	modelOnly.DefaultModel = "gpt-other"
	modelOnlyConfig := map[string]config.ProviderConfig{"shared": modelOnly}
	if changed := registry.UpdateProvidersConfig(modelOnlyConfig); !changed {
		t.Fatal("model-only provider reload was not applied")
	}
	handler.UpdateProviderConfigs(modelOnlyConfig)
	if status := performChatIdentityRequest(handler); status == http.StatusOK {
		t.Fatal("open identity unexpectedly recovered after model-only reload")
	}
	if got := failedCalls.Load(); got != 5 {
		t.Fatalf("model-only reload bypassed open breaker; calls = %d", got)
	}

	changed := modelOnly
	changed.APIKey = "secret-new"
	changed.BaseURL = successUpstream.URL
	changedConfig := map[string]config.ProviderConfig{"shared": changed}
	if applied := registry.UpdateProvidersConfig(changedConfig); !applied {
		t.Fatal("provider identity reload was not applied")
	}
	handler.UpdateProviderConfigs(changedConfig)
	if status := performChatIdentityRequest(handler); status != http.StatusOK {
		t.Fatalf("changed provider identity status = %d, want 200", status)
	}
	if got := successCalls.Load(); got != 1 {
		t.Fatalf("changed provider identity calls = %d, want 1", got)
	}
}

func TestEmbeddingsCircuitBreakerMovesToChangedProviderIdentity(t *testing.T) {
	var failedCalls atomic.Int32
	failingUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		failedCalls.Add(1)
		w.WriteHeader(http.StatusBadGateway)
	}))
	defer failingUpstream.Close()

	var successCalls atomic.Int32
	successUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		successCalls.Add(1)
		if got := r.Header.Get("Authorization"); got != "Bearer embedding-secret-new" {
			http.Error(w, "wrong provider credential", http.StatusUnauthorized)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0.25,0.75]}],"model":"embedding-test","usage":{"prompt_tokens":1,"total_tokens":1}}`)
	}))
	defer successUpstream.Close()

	initial := map[string]config.ProviderConfig{
		"shared": {
			Type:         "openai",
			APIKey:       "embedding-secret-old",
			BaseURL:      failingUpstream.URL,
			DefaultModel: "embedding-test",
		},
	}
	registry := providers.NewRegistry(initial)
	handler := newCircuitBreakerIdentityTestHandler(t, registry, initial, "/v1/embeddings", "embedding-test")

	for i := 0; i < 5; i++ {
		if status := performEmbeddingsIdentityRequest(handler); status == http.StatusOK {
			t.Fatalf("failure request %d unexpectedly succeeded", i+1)
		}
	}
	if got := failedCalls.Load(); got != 5 {
		t.Fatalf("failing upstream calls = %d, want 5", got)
	}

	changed := initial["shared"]
	changed.APIKey = "embedding-secret-new"
	changed.BaseURL = successUpstream.URL
	changedConfig := map[string]config.ProviderConfig{"shared": changed}
	if applied := registry.UpdateProvidersConfig(changedConfig); !applied {
		t.Fatal("provider identity reload was not applied")
	}
	handler.UpdateProviderConfigs(changedConfig)
	if status := performEmbeddingsIdentityRequest(handler); status != http.StatusOK {
		t.Fatalf("changed provider identity status = %d, want 200", status)
	}
	if got := successCalls.Load(); got != 1 {
		t.Fatalf("changed provider identity calls = %d, want 1", got)
	}
}

func TestCircuitBreakerTargetSnapshotsPinProviderGeneration(t *testing.T) {
	registry := providers.NewRegistry(map[string]config.ProviderConfig{
		"primary":  {Type: "openai", APIKey: "old", BaseURL: "https://old.example/v1"},
		"fallback": {Type: "anthropic", APIKey: "fallback"},
	})
	handler := &Handler{registry: registry}
	resolved := &routing.ResolvedRoute{
		Target:    routing.Target{Provider: "primary", Model: "model-a"},
		Fallbacks: []routing.Target{{Provider: "fallback", Model: "model-b"}},
	}
	ctx := handler.withCircuitBreakerTargetSnapshots(context.Background(), resolved)
	oldKey := resolved.Target.CircuitBreakerKey()
	if oldKey == "" || oldKey == "primary" {
		t.Fatal("primary breaker key was not enriched")
	}
	if resolved.Fallbacks[0].CircuitBreakerKey() == oldKey {
		t.Fatal("primary and fallback share breaker identity")
	}

	changed := map[string]config.ProviderConfig{
		"primary":  {Type: "openai", APIKey: "new", BaseURL: "https://new.example/v1"},
		"fallback": {Type: "anthropic", APIKey: "fallback"},
	}
	if applied := registry.UpdateProvidersConfig(changed); !applied {
		t.Fatal("provider identity reload was not applied")
	}
	pinned, ok := circuitBreakerTargetSnapshotFromContext(ctx, resolved.Target)
	if !ok || pinned.Translator.BaseURL() != "https://old.example/v1" {
		t.Fatal("request did not retain old provider snapshot")
	}

	newResolved := &routing.ResolvedRoute{Target: routing.Target{Provider: "primary", Model: "model-a"}}
	newCtx := handler.withCircuitBreakerTargetSnapshots(context.Background(), newResolved)
	if newResolved.Target.CircuitBreakerKey() == oldKey {
		t.Fatal("new provider account retained old breaker identity")
	}
	current, ok := circuitBreakerTargetSnapshotFromContext(newCtx, newResolved.Target)
	if !ok || current.Translator.BaseURL() != "https://new.example/v1" {
		t.Fatal("new request did not retain current provider snapshot")
	}
}

func newCircuitBreakerIdentityTestHandler(
	t *testing.T,
	registry *providers.Registry,
	providerConfigs map[string]config.ProviderConfig,
	path string,
	model string,
) *Handler {
	t.Helper()
	router := routing.NewEngine(config.RoutingConfig{
		DefaultStrategy: "weighted",
		Routes: []config.RouteConfig{{
			Name:    "identity-test",
			Match:   config.MatchConfig{Path: path},
			Targets: []config.TargetConfig{{Provider: "shared", Model: model, Weight: 1}},
		}},
	})
	cache := middleware.NewCache(config.CacheConfig{Enabled: false})
	t.Cleanup(cache.Stop)
	handler := NewHandler(
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
	handler.UpdateProviderConfigs(providerConfigs)
	return handler
}

func performChatIdentityRequest(handler *Handler) int {
	payload := models.UnifiedRequest{
		Model:    "gpt-test",
		Messages: []models.Message{{Role: "user", Content: "hello"}},
	}
	body, _ := json.Marshal(payload)
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	recorder := httptest.NewRecorder()
	handler.ChatCompletions(recorder, request)
	return recorder.Code
}

func performEmbeddingsIdentityRequest(handler *Handler) int {
	request := httptest.NewRequest(http.MethodPost, "/v1/embeddings", strings.NewReader(
		`{"model":"embedding-test","input":"hello"}`,
	))
	recorder := httptest.NewRecorder()
	handler.Embeddings(recorder, request)
	return recorder.Code
}
