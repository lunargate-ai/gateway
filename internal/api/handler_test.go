package api

import (
	"bytes"
	"encoding/json"
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
	"github.com/lunargate-ai/gateway/pkg/models"

	"github.com/prometheus/client_golang/prometheus"
)

func TestChatCompletions_RequestBodyTooLarge_Returns413(t *testing.T) {
	reg := providers.NewRegistry(map[string]config.ProviderConfig{})
	router := routing.NewEngine(config.RoutingConfig{DefaultStrategy: "weighted", Routes: []config.RouteConfig{}})
	retrier := resilience.NewRetrier(config.RetryConfig{Enabled: false})
	cbm := resilience.NewCircuitBreakerManager()
	fb := resilience.NewFallbackExecutor(retrier, cbm)
	cache := middleware.NewCache(config.CacheConfig{Enabled: false})
	streamer := streaming.NewHandler()
	metrics := observability.NewMetricsWithRegisterer(prometheus.NewRegistry())
	h := NewHandler(reg, router, fb, cache, streamer, metrics, nil, nil, nil)

	bigContent := string(bytes.Repeat([]byte("a"), (10<<20)+1024))
	payload := []byte(`{"model":"mock-gpt","messages":[{"role":"user","content":"` + bigContent + `"}]}`)
	req := httptest.NewRequest(http.MethodPost, "http://example.com/v1/chat/completions", bytes.NewReader(payload))
	rec := httptest.NewRecorder()

	h.ChatCompletions(rec, req)

	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("expected status %d, got %d", http.StatusRequestEntityTooLarge, rec.Code)
	}
}

func TestChatCompletions_ProviderErrorPassthrough(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":{"message":"bad key","type":"invalid_api_key"}}`))
	}))
	defer upstream.Close()

	providerID := "openai"
	cfgProviders := map[string]config.ProviderConfig{
		providerID: {Type: "openai", APIKey: "dummy", BaseURL: upstream.URL},
	}
	reg := providers.NewRegistry(cfgProviders)

	router := routing.NewEngine(config.RoutingConfig{
		DefaultStrategy: "weighted",
		Routes: []config.RouteConfig{
			{
				Name:    "default",
				Match:   config.MatchConfig{Path: "*"},
				Targets: []config.TargetConfig{{Provider: providerID, Model: "gpt-4-turbo", Weight: 1}},
			},
		},
	})

	retrier := resilience.NewRetrier(config.RetryConfig{Enabled: true, MaxAttempts: 1, RetryableErrors: []int{429, 500, 502, 503, 504}})
	cbm := resilience.NewCircuitBreakerManager()
	fb := resilience.NewFallbackExecutor(retrier, cbm)
	cache := middleware.NewCache(config.CacheConfig{Enabled: false})
	streamer := streaming.NewHandler()
	metrics := observability.NewMetricsWithRegisterer(prometheus.NewRegistry())
	h := NewHandler(reg, router, fb, cache, streamer, metrics, nil, nil, nil)

	payload := models.UnifiedRequest{Model: "gpt-4-turbo", Messages: []models.Message{{Role: "user", Content: "hi"}}}
	b, _ := json.Marshal(payload)
	req := httptest.NewRequest(http.MethodPost, "http://example.com/v1/chat/completions", bytes.NewReader(b))
	rec := httptest.NewRecorder()

	h.ChatCompletions(rec, req)

	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("expected status %d, got %d", http.StatusUnauthorized, rec.Code)
	}

	var resp models.ErrorResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to unmarshal response: %v", err)
	}
	if resp.Error.Type != "invalid_api_key" {
		t.Fatalf("expected error type %q, got %q", "invalid_api_key", resp.Error.Type)
	}
	if resp.Error.Message != "bad key" {
		t.Fatalf("expected error message %q, got %q", "bad key", resp.Error.Message)
	}
}

func TestChatCompletions_SetsTimingHeaders(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"mock-gpt","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	defer upstream.Close()

	providerID := "openai"
	cfgProviders := map[string]config.ProviderConfig{
		providerID: {Type: "openai", APIKey: "dummy", BaseURL: upstream.URL},
	}
	reg := providers.NewRegistry(cfgProviders)

	router := routing.NewEngine(config.RoutingConfig{
		DefaultStrategy: "weighted",
		Routes: []config.RouteConfig{
			{
				Name:    "default",
				Match:   config.MatchConfig{Path: "*"},
				Targets: []config.TargetConfig{{Provider: providerID, Model: "mock-gpt", Weight: 1}},
			},
		},
	})

	retrier := resilience.NewRetrier(config.RetryConfig{Enabled: false})
	cbm := resilience.NewCircuitBreakerManager()
	fb := resilience.NewFallbackExecutor(retrier, cbm)
	cache := middleware.NewCache(config.CacheConfig{Enabled: false})
	streamer := streaming.NewHandler()
	metrics := observability.NewMetricsWithRegisterer(prometheus.NewRegistry())
	h := NewHandler(reg, router, fb, cache, streamer, metrics, nil, nil, nil)

	payload := models.UnifiedRequest{Model: "mock-gpt", Messages: []models.Message{{Role: "user", Content: "hi"}}}
	b, _ := json.Marshal(payload)
	req := httptest.NewRequest(http.MethodPost, "http://example.com/v1/chat/completions", bytes.NewReader(b))
	rec := httptest.NewRecorder()

	h.ChatCompletions(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}
	if rec.Header().Get("X-LunarGate-Overhead-Duration-Ms") == "" {
		t.Fatalf("expected X-LunarGate-Overhead-Duration-Ms header to be set")
	}
	if rec.Header().Get("X-LunarGate-Latency-Ms") == "" {
		t.Fatalf("expected X-LunarGate-Latency-Ms header to be set")
	}
}
