package api

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/middleware"
	"github.com/lunargate-ai/gateway/internal/observability"
	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/internal/resilience"
	"github.com/lunargate-ai/gateway/internal/routing"
	"github.com/lunargate-ai/gateway/internal/streaming"
	"github.com/prometheus/client_golang/prometheus"
)

func TestChatCompletionsStoreTrueDisablesRetryAndFallback(t *testing.T) {
	primaryCalls := 0
	primary := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		primaryCalls++
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error":{"message":"stored before failure","type":"server_error"}}`))
	}))
	defer primary.Close()

	fallbackCalls := 0
	fallback := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fallbackCalls++
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-fallback","object":"chat.completion","created":1,"model":"fallback-model","choices":[{"index":0,"message":{"role":"assistant","content":"unexpected"},"finish_reason":"stop"}]}`))
	}))
	defer fallback.Close()

	providerConfigs := map[string]config.ProviderConfig{
		"primary":  {Type: "openai", APIKey: "dummy", BaseURL: primary.URL},
		"fallback": {Type: "openai", APIKey: "dummy", BaseURL: fallback.URL},
	}
	registry := providers.NewRegistry(providerConfigs)
	router := routing.NewEngine(config.RoutingConfig{
		DefaultStrategy: "weighted",
		Routes: []config.RouteConfig{{
			Name:     "default",
			Match:    config.MatchConfig{Path: "*"},
			Targets:  []config.TargetConfig{{Provider: "primary", Model: "primary-model", Weight: 1}},
			Fallback: []config.TargetConfig{{Provider: "fallback", Model: "fallback-model", Weight: 1}},
		}},
	})
	handler := NewHandler(
		registry,
		router,
		resilience.NewFallbackExecutor(
			resilience.NewRetrier(config.RetryConfig{
				Enabled:         true,
				MaxAttempts:     3,
				RetryableErrors: []int{http.StatusInternalServerError},
			}),
			resilience.NewCircuitBreakerManager(),
		),
		middleware.NewCache(config.CacheConfig{Enabled: false}),
		streaming.NewHandler(),
		observability.NewMetricsWithRegisterer(prometheus.NewRegistry()),
		nil,
		nil,
		nil,
	)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(
		`{"model":"primary-model","store":true,"messages":[{"role":"user","content":"hello"}]}`,
	))
	recorder := httptest.NewRecorder()
	handler.ChatCompletions(recorder, req)

	if recorder.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d, want %d; body=%s", recorder.Code, http.StatusInternalServerError, recorder.Body.String())
	}
	if primaryCalls != 1 {
		t.Fatalf("primary calls = %d, want exactly 1 for store:true", primaryCalls)
	}
	if fallbackCalls != 0 {
		t.Fatalf("fallback calls = %d, want 0 for store:true", fallbackCalls)
	}
}
