package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/middleware"
	"github.com/lunargate-ai/gateway/internal/observability"
	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/internal/resilience"
	"github.com/lunargate-ai/gateway/internal/routing"
	"github.com/lunargate-ai/gateway/internal/streaming"
	"github.com/lunargate-ai/gateway/pkg/models"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
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

func TestChatCompletions_RetryExhausted429PreservesUpstreamStatus(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"message":"too many requests","type":"rate_limit_error"}}`))
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

	retrier := resilience.NewRetrier(config.RetryConfig{
		Enabled:         true,
		MaxAttempts:     1,
		RetryableErrors: []int{http.StatusTooManyRequests},
	})
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

	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("expected status %d, got %d", http.StatusTooManyRequests, rec.Code)
	}

	var resp models.ErrorResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to unmarshal response: %v", err)
	}
	if resp.Error.Type != "rate_limit_error" {
		t.Fatalf("expected error type %q, got %q", "rate_limit_error", resp.Error.Type)
	}
	if resp.Error.Message != "too many requests" {
		t.Fatalf("expected error message %q, got %q", "too many requests", resp.Error.Message)
	}
}

func TestChatCompletions_NoRetryDisablesPerTargetRetriesAndStillFallsBack(t *testing.T) {
	primaryCalls := 0
	primaryModel := ""
	primaryUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		primaryCalls++
		var payload struct {
			Model string `json:"model"`
		}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatalf("decode primary request: %v", err)
		}
		primaryModel = payload.Model
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error":{"message":"try fallback","type":"server_error"}}`))
	}))
	defer primaryUpstream.Close()

	fallbackCalls := 0
	fallbackModel := ""
	fallbackUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fallbackCalls++
		var payload struct {
			Model string `json:"model"`
		}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatalf("decode fallback request: %v", err)
		}
		fallbackModel = payload.Model
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-fallback","object":"chat.completion","created":1,"model":"fallback-gpt","choices":[{"index":0,"message":{"role":"assistant","content":"ok-from-fallback"},"finish_reason":"stop"}]}`))
	}))
	defer fallbackUpstream.Close()

	cfgProviders := map[string]config.ProviderConfig{
		"primary":  {Type: "openai", APIKey: "dummy", BaseURL: primaryUpstream.URL},
		"fallback": {Type: "openai", APIKey: "dummy", BaseURL: fallbackUpstream.URL},
	}
	reg := providers.NewRegistry(cfgProviders)

	router := routing.NewEngine(config.RoutingConfig{
		DefaultStrategy: "weighted",
		Routes: []config.RouteConfig{
			{
				Name:    "default",
				Match:   config.MatchConfig{Path: "*"},
				Targets: []config.TargetConfig{{Provider: "primary", Model: "gpt-primary", Weight: 1}},
				Fallback: []config.TargetConfig{
					{Provider: "fallback", Model: "gpt-fallback", Weight: 1},
				},
			},
		},
	})

	retrier := resilience.NewRetrier(config.RetryConfig{Enabled: true, MaxAttempts: 3, RetryableErrors: []int{500}})
	cbm := resilience.NewCircuitBreakerManager()
	fb := resilience.NewFallbackExecutor(retrier, cbm)
	cache := middleware.NewCache(config.CacheConfig{Enabled: false})
	streamer := streaming.NewHandler()
	metrics := observability.NewMetricsWithRegisterer(prometheus.NewRegistry())
	h := NewHandler(reg, router, fb, cache, streamer, metrics, nil, nil, nil)

	payload := models.UnifiedRequest{Messages: []models.Message{{Role: "user", Content: "hi"}}}
	b, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("failed to marshal payload: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "http://example.com/v1/chat/completions", bytes.NewReader(b))
	req.Header.Set("X-LunarGate-No-Retry", "true")
	rec := httptest.NewRecorder()

	h.ChatCompletions(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}
	if primaryCalls != 1 {
		t.Fatalf("expected exactly one primary attempt with X-LunarGate-No-Retry, got %d", primaryCalls)
	}
	if fallbackCalls != 1 {
		t.Fatalf("expected fallback to run after the single primary failure, got %d calls", fallbackCalls)
	}
	if primaryModel != "gpt-primary" {
		t.Fatalf("primary upstream model = %q, want %q", primaryModel, "gpt-primary")
	}
	if fallbackModel != "gpt-fallback" {
		t.Fatalf("fallback upstream model = %q, want %q", fallbackModel, "gpt-fallback")
	}
	if got := rec.Header().Get("X-LunarGate-Provider"); got != "fallback" {
		t.Fatalf("expected fallback provider in response header, got %q", got)
	}
	if gauge := testutil.ToFloat64(metrics.CircuitBreakerState.WithLabelValues("fallback")); gauge != 0 {
		t.Fatalf("expected fallback circuit breaker gauge to remain closed, got %v", gauge)
	}
}

func TestCallEmbeddingsProviderBindsTargetModel(t *testing.T) {
	upstreamModel := ""
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload struct {
			Model string `json:"model"`
		}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatalf("decode embeddings request: %v", err)
		}
		upstreamModel = payload.Model
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"object":"list","data":[],"model":"embedding-fallback","usage":{"prompt_tokens":1,"total_tokens":1}}`))
	}))
	defer upstream.Close()

	reg := providers.NewRegistry(map[string]config.ProviderConfig{
		"fallback": {Type: "openai", APIKey: "dummy", BaseURL: upstream.URL},
	})
	h := &Handler{registry: reg, providerClients: newProviderClientRegistry(nil)}
	req := &models.EmbeddingsRequest{Model: "primary/embedding-primary", Input: "hello"}
	resp, err := h.callEmbeddingsProvider(context.Background(), routing.Target{
		Provider: "fallback",
		Model:    "embedding-fallback",
	}, req, nil)
	if err != nil {
		t.Fatalf("callEmbeddingsProvider returned error: %v", err)
	}
	defer resp.Body.Close()

	if upstreamModel != "embedding-fallback" {
		t.Fatalf("fallback upstream model = %q, want %q", upstreamModel, "embedding-fallback")
	}
}

func TestChatCompletionsRejectsUnavailableBareModel(t *testing.T) {
	h := newUnavailableModelTestHandler(t)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(
		`{"model":"gpt-4o","messages":[{"role":"user","content":"hello"}]}`,
	))
	rec := httptest.NewRecorder()

	h.ChatCompletions(rec, req)
	assertModelNotFoundError(t, rec)
}

func TestChatCompletionsNormalizesAutoModelHeader(t *testing.T) {
	upstreamModel := ""
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload struct {
			Model string `json:"model"`
		}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatalf("decode upstream request: %v", err)
		}
		upstreamModel = payload.Model
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-auto","object":"chat.completion","created":1,"model":"mock-gpt","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	defer upstream.Close()

	reg := providers.NewRegistry(map[string]config.ProviderConfig{
		"openai": {Type: "openai", APIKey: "dummy", BaseURL: upstream.URL},
	})
	router := routing.NewEngine(config.RoutingConfig{
		DefaultStrategy: "weighted",
		Routes: []config.RouteConfig{{
			Name:    "default",
			Match:   config.MatchConfig{Path: "*"},
			Targets: []config.TargetConfig{{Provider: "openai", Model: "mock-gpt", Weight: 1}},
		}},
	})
	h := NewHandler(
		reg,
		router,
		resilience.NewFallbackExecutor(
			resilience.NewRetrier(config.RetryConfig{Enabled: false}),
			resilience.NewCircuitBreakerManager(),
		),
		middleware.NewCache(config.CacheConfig{Enabled: false}),
		streaming.NewHandler(),
		observability.NewMetricsWithRegisterer(prometheus.NewRegistry()),
		nil,
		nil,
		nil,
	)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(
		`{"model":"client-placeholder","messages":[{"role":"user","content":"hello"}]}`,
	))
	req.Header.Set("X-LunarGate-Model", "lunargate/auto")
	rec := httptest.NewRecorder()

	h.ChatCompletions(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d; body=%s", rec.Code, http.StatusOK, rec.Body.String())
	}
	if upstreamModel != "mock-gpt" {
		t.Fatalf("upstream model = %q, want route-selected model", upstreamModel)
	}
}

func TestEmbeddingsRejectsUnavailableBareModel(t *testing.T) {
	h := newUnavailableModelTestHandler(t)
	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", strings.NewReader(
		`{"model":"text-embedding-ada-002","input":"hello"}`,
	))
	rec := httptest.NewRecorder()

	h.Embeddings(rec, req)
	assertModelNotFoundError(t, rec)
}

func TestResolveEmbeddingsRouteDoesNotBypassTargetPolicy(t *testing.T) {
	reg := providers.NewRegistry(map[string]config.ProviderConfig{
		"allowed": {Type: "openai", APIKey: "dummy", BaseURL: "http://127.0.0.1:1"},
		"blocked": {Type: "openai", APIKey: "dummy", BaseURL: "http://127.0.0.1:2"},
	})
	router := routing.NewEngine(config.RoutingConfig{
		DefaultStrategy: "weighted",
		Routes: []config.RouteConfig{{
			Name:    "embeddings-policy",
			Match:   config.MatchConfig{Path: "/v1/embeddings"},
			Targets: []config.TargetConfig{{Provider: "allowed", Model: "text-embedding-3-small", Weight: 1}},
		}},
	})
	h := &Handler{registry: reg, router: router}

	_, err := h.resolveEmbeddingsRoute(context.Background(), "/v1/embeddings", map[string]string{
		"x-lunargate-provider": "blocked",
		"x-lunargate-model":    "blocked/text-embedding-3-small",
	}, "blocked")
	var unavailable *routing.RequestedTargetUnavailableError
	if !errors.As(err, &unavailable) {
		t.Fatalf("error = %T %v, want RequestedTargetUnavailableError", err, err)
	}
}

func newUnavailableModelTestHandler(t *testing.T) *Handler {
	t.Helper()
	reg := providers.NewRegistry(map[string]config.ProviderConfig{
		"openai": {Type: "openai", APIKey: "dummy", BaseURL: "http://127.0.0.1:1"},
	})
	router := routing.NewEngine(config.RoutingConfig{
		DefaultStrategy: "weighted",
		Routes: []config.RouteConfig{{
			Name:    "default",
			Match:   config.MatchConfig{Path: "*"},
			Targets: []config.TargetConfig{{Provider: "openai", Model: "gpt-5.4", Weight: 1}},
		}},
	})
	retrier := resilience.NewRetrier(config.RetryConfig{Enabled: false})
	return NewHandler(
		reg,
		router,
		resilience.NewFallbackExecutor(retrier, resilience.NewCircuitBreakerManager()),
		middleware.NewCache(config.CacheConfig{Enabled: false}),
		streaming.NewHandler(),
		observability.NewMetricsWithRegisterer(prometheus.NewRegistry()),
		nil,
		nil,
		nil,
	)
}

func assertModelNotFoundError(t *testing.T, rec *httptest.ResponseRecorder) {
	t.Helper()
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d; body=%s", rec.Code, http.StatusBadRequest, rec.Body.String())
	}
	var response models.ErrorResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if response.Error.Param == nil || *response.Error.Param != "model" {
		t.Fatalf("param = %#v, want model", response.Error.Param)
	}
	if response.Error.Code == nil || *response.Error.Code != "model_not_found" {
		t.Fatalf("code = %#v, want model_not_found", response.Error.Code)
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

func TestChatCompletions_TTFTTimeoutAfterHeaders(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		time.Sleep(150 * time.Millisecond)
		_, _ = io.WriteString(w, `{"id":"cmpl-timeout","object":"chat.completion","created":1,"model":"gpt-4-turbo","choices":[{"index":0,"message":{"role":"assistant","content":"late"},"finish_reason":"stop"}]}`)
	}))
	defer upstream.Close()

	providerID := "openai"
	cfgProviders := map[string]config.ProviderConfig{
		providerID: {Type: "openai", APIKey: "dummy", BaseURL: upstream.URL, Timeout: 50 * time.Millisecond},
	}
	reg := providers.NewRegistry(cfgProviders)
	router := routing.NewEngine(config.RoutingConfig{
		DefaultStrategy: "weighted",
		Routes: []config.RouteConfig{{
			Name:    "default",
			Match:   config.MatchConfig{Path: "*"},
			Targets: []config.TargetConfig{{Provider: providerID, Model: "gpt-4-turbo", Weight: 1}},
		}},
	})
	retrier := resilience.NewRetrier(config.RetryConfig{Enabled: false})
	cbm := resilience.NewCircuitBreakerManager()
	fb := resilience.NewFallbackExecutor(retrier, cbm)
	cache := middleware.NewCache(config.CacheConfig{Enabled: false})
	streamer := streaming.NewHandler()
	metrics := observability.NewMetricsWithRegisterer(prometheus.NewRegistry())
	h := NewHandler(reg, router, fb, cache, streamer, metrics, nil, nil, nil)
	h.UpdateProviderConfigs(cfgProviders)

	payload := models.UnifiedRequest{Model: "gpt-4-turbo", Messages: []models.Message{{Role: "user", Content: "hi"}}}
	b, _ := json.Marshal(payload)
	req := httptest.NewRequest(http.MethodPost, "http://example.com/v1/chat/completions", bytes.NewReader(b))
	rec := httptest.NewRecorder()

	h.ChatCompletions(rec, req)

	if rec.Code != http.StatusBadGateway {
		t.Fatalf("expected status %d, got %d", http.StatusBadGateway, rec.Code)
	}

	var resp models.ErrorResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to unmarshal response: %v", err)
	}
	if resp.Error.Type != "upstream_timeout" {
		t.Fatalf("expected error type %q, got %q", "upstream_timeout", resp.Error.Type)
	}
}

func TestChatCompletions_TTFTTimeoutClearsAfterFirstByte(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		_, _ = io.WriteString(w, `{`)
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		time.Sleep(150 * time.Millisecond)
		_, _ = io.WriteString(w, `"id":"cmpl-ok","object":"chat.completion","created":1,"model":"gpt-4-turbo","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}]}`)
	}))
	defer upstream.Close()

	providerID := "openai"
	cfgProviders := map[string]config.ProviderConfig{
		providerID: {Type: "openai", APIKey: "dummy", BaseURL: upstream.URL, Timeout: 50 * time.Millisecond},
	}
	reg := providers.NewRegistry(cfgProviders)
	router := routing.NewEngine(config.RoutingConfig{
		DefaultStrategy: "weighted",
		Routes: []config.RouteConfig{{
			Name:    "default",
			Match:   config.MatchConfig{Path: "*"},
			Targets: []config.TargetConfig{{Provider: providerID, Model: "gpt-4-turbo", Weight: 1}},
		}},
	})
	retrier := resilience.NewRetrier(config.RetryConfig{Enabled: false})
	cbm := resilience.NewCircuitBreakerManager()
	fb := resilience.NewFallbackExecutor(retrier, cbm)
	cache := middleware.NewCache(config.CacheConfig{Enabled: false})
	streamer := streaming.NewHandler()
	metrics := observability.NewMetricsWithRegisterer(prometheus.NewRegistry())
	h := NewHandler(reg, router, fb, cache, streamer, metrics, nil, nil, nil)
	h.UpdateProviderConfigs(cfgProviders)

	payload := models.UnifiedRequest{Model: "gpt-4-turbo", Messages: []models.Message{{Role: "user", Content: "hi"}}}
	b, _ := json.Marshal(payload)
	req := httptest.NewRequest(http.MethodPost, "http://example.com/v1/chat/completions", bytes.NewReader(b))
	rec := httptest.NewRecorder()

	h.ChatCompletions(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d body=%s", http.StatusOK, rec.Code, rec.Body.String())
	}

	var resp models.UnifiedResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to unmarshal response: %v", err)
	}
	if len(resp.Choices) != 1 || resp.Choices[0].Message == nil {
		t.Fatalf("expected one response choice with a message, got %#v", resp.Choices)
	}
	if got := resp.Choices[0].Message.Content; got != "hello" {
		t.Fatalf("expected content %q, got %#v", "hello", got)
	}
}

func TestChatCompletions_TotalTimeoutCutsAfterFirstByte(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		_, _ = io.WriteString(w, `{`)
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		time.Sleep(150 * time.Millisecond)
		_, _ = io.WriteString(w, `"id":"cmpl-total-timeout","object":"chat.completion","created":1,"model":"gpt-4-turbo","choices":[{"index":0,"message":{"role":"assistant","content":"too late"},"finish_reason":"stop"}]}`)
	}))
	defer upstream.Close()

	providerID := "openai"
	cfgProviders := map[string]config.ProviderConfig{
		providerID: {
			Type:        "openai",
			APIKey:      "dummy",
			BaseURL:     upstream.URL,
			Timeout:     50 * time.Millisecond,
			TimeoutMode: "total",
		},
	}
	reg := providers.NewRegistry(cfgProviders)
	router := routing.NewEngine(config.RoutingConfig{
		DefaultStrategy: "weighted",
		Routes: []config.RouteConfig{{
			Name:    "default",
			Match:   config.MatchConfig{Path: "*"},
			Targets: []config.TargetConfig{{Provider: providerID, Model: "gpt-4-turbo", Weight: 1}},
		}},
	})
	retrier := resilience.NewRetrier(config.RetryConfig{Enabled: false})
	cbm := resilience.NewCircuitBreakerManager()
	fb := resilience.NewFallbackExecutor(retrier, cbm)
	cache := middleware.NewCache(config.CacheConfig{Enabled: false})
	streamer := streaming.NewHandler()
	metrics := observability.NewMetricsWithRegisterer(prometheus.NewRegistry())
	h := NewHandler(reg, router, fb, cache, streamer, metrics, nil, nil, nil)
	h.UpdateProviderConfigs(cfgProviders)

	payload := models.UnifiedRequest{Model: "gpt-4-turbo", Messages: []models.Message{{Role: "user", Content: "hi"}}}
	b, _ := json.Marshal(payload)
	req := httptest.NewRequest(http.MethodPost, "http://example.com/v1/chat/completions", bytes.NewReader(b))
	rec := httptest.NewRecorder()

	h.ChatCompletions(rec, req)

	if rec.Code != http.StatusBadGateway {
		t.Fatalf("expected status %d, got %d", http.StatusBadGateway, rec.Code)
	}

	var resp models.ErrorResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to unmarshal response: %v", err)
	}
	if resp.Error.Type != "upstream_timeout" {
		t.Fatalf("expected error type %q, got %q", "upstream_timeout", resp.Error.Type)
	}
	if resp.Error.Message != "provider timed out before full response completed" {
		t.Fatalf("expected total-timeout message, got %q", resp.Error.Message)
	}
}

func TestResponses_MapsToChatCompletions(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"mock-gpt","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`))
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

	payload := []byte(`{"model":"lunargate/auto","input":[{"role":"user","content":[{"type":"input_text","text":"Say hi"}]}]}`)
	req := httptest.NewRequest(http.MethodPost, "http://example.com/v1/responses", bytes.NewReader(payload))
	rec := httptest.NewRecorder()

	h.Responses(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}

	var out models.ResponsesResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("failed to unmarshal responses payload: %v", err)
	}
	if out.Object != "response" {
		t.Fatalf("expected response object, got %q", out.Object)
	}
	if got := responsesTextFromTypedResponseForTest(&out); got != "ok" {
		t.Fatalf("expected output text %q, got %q", "ok", got)
	}
	if rec.Header().Get("X-LunarGate-Provider") == "" {
		t.Fatalf("expected X-LunarGate-Provider header to be set")
	}
}

func TestResponses_RoutesViaChatPathWhenOnlyChatRouteConfigured(t *testing.T) {
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
				Name:    "chat-only-route",
				Match:   config.MatchConfig{Path: "/v1/chat/completions"},
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

	payload := []byte(`{"model":"lunargate/auto","input":[{"role":"user","content":[{"type":"input_text","text":"Say hi"}]}]}`)
	req := httptest.NewRequest(http.MethodPost, "http://example.com/v1/responses", bytes.NewReader(payload))
	rec := httptest.NewRecorder()

	h.Responses(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}
	if got := rec.Header().Get("X-LunarGate-Route"); got != "chat-only-route" {
		t.Fatalf("expected routed via chat-only-route, got %q", got)
	}

	var out models.ResponsesResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("failed to unmarshal responses payload: %v", err)
	}
	if got := responsesTextFromTypedResponseForTest(&out); got != "ok" {
		t.Fatalf("expected output text %q, got %q", "ok", got)
	}
}

func TestResponses_RoutesViaResponsesPathWhenResponsesRouteConfigured(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-2","object":"chat.completion","created":1,"model":"mock-gpt","choices":[{"index":0,"message":{"role":"assistant","content":"ok-responses-route"},"finish_reason":"stop"}]}`))
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
				Name:    "responses-only-route",
				Match:   config.MatchConfig{Path: "/v1/responses"},
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

	payload := []byte(`{"model":"lunargate/auto","input":[{"role":"user","content":[{"type":"input_text","text":"Say hi"}]}]}`)
	req := httptest.NewRequest(http.MethodPost, "http://example.com/v1/responses", bytes.NewReader(payload))
	rec := httptest.NewRecorder()

	h.Responses(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}
	if got := rec.Header().Get("X-LunarGate-Route"); got != "responses-only-route" {
		t.Fatalf("expected routed via responses-only-route, got %q", got)
	}

	var out models.ResponsesResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("failed to unmarshal responses payload: %v", err)
	}
	if got := responsesTextFromTypedResponseForTest(&out); got != "ok-responses-route" {
		t.Fatalf("expected output text %q, got %q", "ok-responses-route", got)
	}
}

func TestResponses_StreamPassthrough(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"mock-gpt\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
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
				Name:    "responses-default",
				Match:   config.MatchConfig{Path: "/v1/responses"},
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

	payload := []byte(`{"model":"lunargate/auto","stream":true,"input":[{"role":"user","content":[{"type":"input_text","text":"Say hi"}]}]}`)
	req := httptest.NewRequest(http.MethodPost, "http://example.com/v1/responses", bytes.NewReader(payload))
	rec := httptest.NewRecorder()

	h.Responses(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}
	body := rec.Body.String()
	if !strings.Contains(body, "data: ") {
		t.Fatalf("expected streaming body to contain SSE data, got %q", body)
	}
	if !strings.Contains(body, `"type":"response.completed"`) {
		t.Fatalf("expected responses stream to emit response.completed event, got %q", body)
	}
	if strings.Contains(body, "streaming responses are not supported yet") {
		t.Fatalf("unexpected legacy streaming error in body")
	}
}

func TestResponses_StreamToolCallLifecycle(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		chunk1 := map[string]interface{}{
			"id":      "chatcmpl-tool",
			"object":  "chat.completion.chunk",
			"created": 1,
			"model":   "mock-gpt",
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"delta": map[string]interface{}{
						"tool_calls": []map[string]interface{}{
							{
								"index": 0,
								"id":    "call_1",
								"type":  "function",
								"function": map[string]interface{}{
									"name":      "exec_command",
									"arguments": "{\"cmd\":\"pwd\"",
								},
							},
						},
					},
					"finish_reason": nil,
				},
			},
		}
		chunk1Bytes, _ := json.Marshal(chunk1)
		_, _ = w.Write([]byte("data: " + string(chunk1Bytes) + "\n\n"))

		chunk2 := map[string]interface{}{
			"id":      "chatcmpl-tool",
			"object":  "chat.completion.chunk",
			"created": 1,
			"model":   "mock-gpt",
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"delta": map[string]interface{}{
						"tool_calls": []map[string]interface{}{
							{
								"index": 0,
								"id":    "call_1",
								"type":  "function",
								"function": map[string]interface{}{
									"arguments": "}",
								},
							},
						},
					},
					"finish_reason": "tool_calls",
				},
			},
		}
		chunk2Bytes, _ := json.Marshal(chunk2)
		_, _ = w.Write([]byte("data: " + string(chunk2Bytes) + "\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
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
				Name:    "responses-default",
				Match:   config.MatchConfig{Path: "/v1/responses"},
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

	payload := []byte(`{"model":"lunargate/auto","stream":true,"input":[{"role":"user","content":[{"type":"input_text","text":"run pwd"}]}]}`)
	req := httptest.NewRequest(http.MethodPost, "http://example.com/v1/responses", bytes.NewReader(payload))
	rec := httptest.NewRecorder()

	h.Responses(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}
	body := rec.Body.String()
	if !strings.Contains(body, `"type":"response.function_call_arguments.delta"`) {
		t.Fatalf("expected function_call_arguments.delta event, got %q", body)
	}
	if !strings.Contains(body, `"type":"response.function_call_arguments.done"`) {
		t.Fatalf("expected function_call_arguments.done event, got %q", body)
	}
	if !strings.Contains(body, `"type":"response.output_item.done"`) {
		t.Fatalf("expected output_item.done event, got %q", body)
	}
	if !strings.Contains(body, `"type":"response.completed"`) {
		t.Fatalf("expected response.completed event, got %q", body)
	}
}

func TestResponses_StreamMultipleToolCallsLifecycle(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		chunk1 := map[string]interface{}{
			"id":      "chatcmpl-tool-multi",
			"object":  "chat.completion.chunk",
			"created": 1,
			"model":   "mock-gpt",
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"delta": map[string]interface{}{
						"tool_calls": []map[string]interface{}{
							{
								"index": 0,
								"id":    "call_1",
								"type":  "function",
								"function": map[string]interface{}{
									"name":      "exec_a",
									"arguments": "{\"x\":1",
								},
							},
							{
								"index": 1,
								"id":    "call_2",
								"type":  "function",
								"function": map[string]interface{}{
									"name":      "exec_b",
									"arguments": "{\"y\":2",
								},
							},
						},
					},
					"finish_reason": nil,
				},
			},
		}
		chunk1Bytes, _ := json.Marshal(chunk1)
		_, _ = w.Write([]byte("data: " + string(chunk1Bytes) + "\n\n"))

		chunk2 := map[string]interface{}{
			"id":      "chatcmpl-tool-multi",
			"object":  "chat.completion.chunk",
			"created": 1,
			"model":   "mock-gpt",
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"delta": map[string]interface{}{
						"tool_calls": []map[string]interface{}{
							{
								"index": 0,
								"id":    "call_1",
								"type":  "function",
								"function": map[string]interface{}{
									"arguments": "}",
								},
							},
							{
								"index": 1,
								"id":    "call_2",
								"type":  "function",
								"function": map[string]interface{}{
									"arguments": "}",
								},
							},
						},
					},
					"finish_reason": "tool_calls",
				},
			},
		}
		chunk2Bytes, _ := json.Marshal(chunk2)
		_, _ = w.Write([]byte("data: " + string(chunk2Bytes) + "\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
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
				Name:    "responses-default",
				Match:   config.MatchConfig{Path: "/v1/responses"},
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

	payload := []byte(`{"model":"lunargate/auto","stream":true,"input":[{"role":"user","content":[{"type":"input_text","text":"run tools"}]}]}`)
	req := httptest.NewRequest(http.MethodPost, "http://example.com/v1/responses", bytes.NewReader(payload))
	rec := httptest.NewRecorder()

	h.Responses(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}
	body := rec.Body.String()
	if strings.Count(body, `"type":"response.function_call_arguments.done"`) != 2 {
		t.Fatalf("expected two function_call_arguments.done events, got %q", body)
	}
	if !strings.Contains(body, `"call_id":"call_1"`) || !strings.Contains(body, `"call_id":"call_2"`) {
		t.Fatalf("expected both call_ids in stream output, got %q", body)
	}
	if !strings.Contains(body, `"type":"response.completed"`) {
		t.Fatalf("expected response.completed event, got %q", body)
	}
}

func TestResponses_PreviousResponseIDResolvedLocallyForNonStreamFollowUp(t *testing.T) {
	var capturedBodies []map[string]interface{}
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("failed to read upstream body: %v", err)
		}
		var payload map[string]interface{}
		if err := json.Unmarshal(body, &payload); err != nil {
			t.Fatalf("failed to decode upstream body: %v", err)
		}
		capturedBodies = append(capturedBodies, payload)

		w.Header().Set("Content-Type", "application/json")
		switch len(capturedBodies) {
		case 1:
			_, _ = w.Write([]byte(`{"id":"chatcmpl-tool-1","object":"chat.completion","created":1,"model":"mock-gpt","choices":[{"index":0,"message":{"role":"assistant","content":"","tool_calls":[{"id":"call_time_1","type":"function","function":{"name":"get_current_time","arguments":"{\"format\":\"iso\"}"}}]},"finish_reason":"tool_calls"}]}`))
		case 2:
			_, _ = w.Write([]byte(`{"id":"chatcmpl-tool-2","object":"chat.completion","created":2,"model":"mock-gpt","choices":[{"index":0,"message":{"role":"assistant","content":"The current time is 2026-04-09T16:51:50Z."},"finish_reason":"stop"}]}`))
		default:
			t.Fatalf("unexpected upstream call %d", len(capturedBodies))
		}
	}))
	defer upstream.Close()

	providerID := "openai"
	cfgProviders := map[string]config.ProviderConfig{
		providerID: {Type: "openai", APIKey: "dummy", BaseURL: upstream.URL},
	}
	reg := providers.NewRegistry(cfgProviders)
	router := routing.NewEngine(config.RoutingConfig{
		DefaultStrategy: "weighted",
		Routes: []config.RouteConfig{{
			Name:    "responses-default",
			Match:   config.MatchConfig{Path: "/v1/responses"},
			Targets: []config.TargetConfig{{Provider: providerID, Model: "mock-gpt", Weight: 1}},
		}},
	})
	retrier := resilience.NewRetrier(config.RetryConfig{Enabled: false})
	cbm := resilience.NewCircuitBreakerManager()
	fb := resilience.NewFallbackExecutor(retrier, cbm)
	cache := middleware.NewCache(config.CacheConfig{Enabled: false})
	streamer := streaming.NewHandler()
	metrics := observability.NewMetricsWithRegisterer(prometheus.NewRegistry())
	h := NewHandler(reg, router, fb, cache, streamer, metrics, nil, nil, nil)
	h.UpdateProviderConfigs(cfgProviders)

	firstPayload := []byte(`{"model":"lunargate/auto","input":"Call get_current_time once and then answer with a short sentence that includes the returned timestamp.","tools":[{"type":"function","name":"get_current_time","description":"Return the current UTC time in ISO 8601 format.","parameters":{"type":"object","properties":{"format":{"type":"string"}}}}],"tool_choice":"auto"}`)
	firstReq := httptest.NewRequest(http.MethodPost, "http://example.com/v1/responses", bytes.NewReader(firstPayload))
	firstRec := httptest.NewRecorder()
	h.Responses(firstRec, firstReq)

	if firstRec.Code != http.StatusOK {
		t.Fatalf("expected first status %d, got %d", http.StatusOK, firstRec.Code)
	}
	var firstResp models.ResponsesResponse
	if err := json.Unmarshal(firstRec.Body.Bytes(), &firstResp); err != nil {
		t.Fatalf("failed to unmarshal first responses payload: %v", err)
	}
	if !strings.HasPrefix(firstResp.ID, "resp_") {
		t.Fatalf("translated response ID = %q, want resp_ prefix", firstResp.ID)
	}

	secondPayload := []byte(fmt.Sprintf(`{"model":"lunargate/auto","previous_response_id":%q,"input":[{"type":"function_call_output","call_id":"call_time_1","output":"{\"iso\":\"2026-04-09T16:51:50Z\"}"}],"tools":[{"type":"function","name":"get_current_time","description":"Return the current UTC time in ISO 8601 format.","parameters":{"type":"object","properties":{"format":{"type":"string"}}}}],"tool_choice":"auto"}`, firstResp.ID))
	secondReq := httptest.NewRequest(http.MethodPost, "http://example.com/v1/responses", bytes.NewReader(secondPayload))
	secondRec := httptest.NewRecorder()
	h.Responses(secondRec, secondReq)

	if secondRec.Code != http.StatusOK {
		t.Fatalf("expected second status %d, got %d", http.StatusOK, secondRec.Code)
	}
	var secondResp models.ResponsesResponse
	if err := json.Unmarshal(secondRec.Body.Bytes(), &secondResp); err != nil {
		t.Fatalf("failed to unmarshal second responses payload: %v", err)
	}
	if got := responsesTextFromTypedResponseForTest(&secondResp); got != "The current time is 2026-04-09T16:51:50Z." {
		t.Fatalf("expected final assistant text, got %q", got)
	}

	if len(capturedBodies) != 2 {
		t.Fatalf("expected two upstream calls, got %d", len(capturedBodies))
	}
	messages, _ := capturedBodies[1]["messages"].([]interface{})
	if len(messages) != 3 {
		t.Fatalf("expected follow-up request to include prior history plus tool output, got %d messages", len(messages))
	}
	assistant, _ := messages[1].(map[string]interface{})
	assistantToolCalls, _ := assistant["tool_calls"].([]interface{})
	if len(assistantToolCalls) != 1 {
		t.Fatalf("expected assistant tool call history in follow-up request, got %v", assistant["tool_calls"])
	}
	toolMsg, _ := messages[2].(map[string]interface{})
	if got, _ := toolMsg["role"].(string); got != "tool" {
		t.Fatalf("expected final follow-up message role=tool, got %q", got)
	}
	if got, _ := toolMsg["tool_call_id"].(string); got != "call_time_1" {
		t.Fatalf("expected tool_call_id call_time_1, got %q", got)
	}
	if got, _ := toolMsg["content"].(string); got != `{"iso":"2026-04-09T16:51:50Z"}` {
		t.Fatalf("expected tool output content to be preserved, got %q", got)
	}
	_ = firstResp
}

func TestResponses_PreviousResponseIDResolvedLocallyForStreamFollowUp(t *testing.T) {
	var capturedBodies []map[string]interface{}
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("failed to read upstream body: %v", err)
		}
		var payload map[string]interface{}
		if err := json.Unmarshal(body, &payload); err != nil {
			t.Fatalf("failed to decode upstream body: %v", err)
		}
		capturedBodies = append(capturedBodies, payload)

		w.Header().Set("Content-Type", "text/event-stream")
		switch len(capturedBodies) {
		case 1:
			_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-stream-tool-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"mock-gpt\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_time_stream_1\",\"type\":\"function\",\"function\":{\"name\":\"get_current_time\",\"arguments\":\"{\\\"format\\\":\\\"iso\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n"))
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
		case 2:
			_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-stream-tool-2\",\"object\":\"chat.completion.chunk\",\"created\":2,\"model\":\"mock-gpt\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"The current time is 2026-04-09T16:51:50Z.\"},\"finish_reason\":\"stop\"}]}\n\n"))
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
		default:
			t.Fatalf("unexpected upstream call %d", len(capturedBodies))
		}
	}))
	defer upstream.Close()

	providerID := "openai"
	cfgProviders := map[string]config.ProviderConfig{
		providerID: {Type: "openai", APIKey: "dummy", BaseURL: upstream.URL},
	}
	reg := providers.NewRegistry(cfgProviders)
	router := routing.NewEngine(config.RoutingConfig{
		DefaultStrategy: "weighted",
		Routes: []config.RouteConfig{{
			Name:    "responses-default",
			Match:   config.MatchConfig{Path: "/v1/responses"},
			Targets: []config.TargetConfig{{Provider: providerID, Model: "mock-gpt", Weight: 1}},
		}},
	})
	retrier := resilience.NewRetrier(config.RetryConfig{Enabled: false})
	cbm := resilience.NewCircuitBreakerManager()
	fb := resilience.NewFallbackExecutor(retrier, cbm)
	cache := middleware.NewCache(config.CacheConfig{Enabled: false})
	streamer := streaming.NewHandler()
	metrics := observability.NewMetricsWithRegisterer(prometheus.NewRegistry())
	h := NewHandler(reg, router, fb, cache, streamer, metrics, nil, nil, nil)
	h.UpdateProviderConfigs(cfgProviders)

	firstPayload := []byte(`{"model":"lunargate/auto","stream":true,"input":"Call get_current_time once and then answer with a short sentence that includes the returned timestamp.","tools":[{"type":"function","name":"get_current_time","description":"Return the current UTC time in ISO 8601 format.","parameters":{"type":"object","properties":{"format":{"type":"string"}}}}],"tool_choice":"auto"}`)
	firstReq := httptest.NewRequest(http.MethodPost, "http://example.com/v1/responses", bytes.NewReader(firstPayload))
	firstRec := httptest.NewRecorder()
	h.Responses(firstRec, firstReq)

	if firstRec.Code != http.StatusOK {
		t.Fatalf("expected first status %d, got %d", http.StatusOK, firstRec.Code)
	}
	firstEvents := decodeSSEEvents(t, firstRec.Body.String())
	var firstResponseID string
	for _, evt := range firstEvents {
		if typ, _ := evt["type"].(string); typ != "response.completed" {
			continue
		}
		responseObj, _ := evt["response"].(map[string]interface{})
		firstResponseID, _ = responseObj["id"].(string)
	}
	if !strings.HasPrefix(firstResponseID, "resp_") {
		t.Fatalf("translated streaming response ID = %q, want resp_ prefix", firstResponseID)
	}

	secondPayload := []byte(fmt.Sprintf(`{"model":"lunargate/auto","stream":true,"previous_response_id":%q,"input":[{"type":"function_call_output","call_id":"call_time_stream_1","output":"{\"iso\":\"2026-04-09T16:51:50Z\"}"}],"tools":[{"type":"function","name":"get_current_time","description":"Return the current UTC time in ISO 8601 format.","parameters":{"type":"object","properties":{"format":{"type":"string"}}}}],"tool_choice":"auto"}`, firstResponseID))
	secondReq := httptest.NewRequest(http.MethodPost, "http://example.com/v1/responses", bytes.NewReader(secondPayload))
	secondRec := httptest.NewRecorder()
	h.Responses(secondRec, secondReq)

	if secondRec.Code != http.StatusOK {
		t.Fatalf("expected second status %d, got %d", http.StatusOK, secondRec.Code)
	}
	secondBody := secondRec.Body.String()
	if !strings.Contains(secondBody, `"The current time is 2026-04-09T16:51:50Z."`) {
		t.Fatalf("expected streamed final assistant text, got %q", secondBody)
	}

	if len(capturedBodies) != 2 {
		t.Fatalf("expected two upstream calls, got %d", len(capturedBodies))
	}
	messages, _ := capturedBodies[1]["messages"].([]interface{})
	if len(messages) != 3 {
		t.Fatalf("expected streamed follow-up request to include prior history plus tool output, got %d messages", len(messages))
	}
	assistant, _ := messages[1].(map[string]interface{})
	assistantToolCalls, _ := assistant["tool_calls"].([]interface{})
	if len(assistantToolCalls) != 1 {
		t.Fatalf("expected assistant tool call history in streamed follow-up request, got %v", assistant["tool_calls"])
	}
}

func TestCopyHeaders_PreservesExistingDestinationHeaders(t *testing.T) {
	dst := http.Header{}
	dst.Set("X-Keep", "keep")

	src := http.Header{}
	src.Set("X-Keep", "replace")
	src.Set("X-New", "new-value")
	src.Set("Content-Length", "123")
	src.Set("Content-Encoding", "gzip")
	src.Set("Set-Cookie", "session=secret")
	src.Set("Connection", "X-Hop")
	src.Set("X-Hop", "private")

	copyHeaders(dst, src)

	if got := dst.Get("X-Keep"); got != "keep" {
		t.Fatalf("expected existing destination header to be preserved, got %q", got)
	}
	if got := dst.Get("X-New"); got != "new-value" {
		t.Fatalf("expected new source header to be copied, got %q", got)
	}
	if got := dst.Get("Content-Length"); got != "" {
		t.Fatalf("expected Content-Length to be skipped, got %q", got)
	}
	for _, key := range []string{"Content-Encoding", "Set-Cookie", "Connection", "X-Hop"} {
		if got := dst.Get(key); got != "" {
			t.Fatalf("expected unsafe header %s to be skipped, got %q", key, got)
		}
	}
}

func TestChatCompletions_UpstreamRequestTypeResponses_UsesResponsesEndpoint(t *testing.T) {
	var capturedPath string
	var capturedBody map[string]interface{}
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedPath = r.URL.Path
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("failed to read upstream body: %v", err)
		}
		if err := json.Unmarshal(body, &capturedBody); err != nil {
			t.Fatalf("failed to decode upstream body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"resp_1","object":"response","created_at":1,"status":"completed","model":"gpt-5.3-codex","output":[],"output_text":"ok","usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`))
	}))
	defer upstream.Close()

	providerID := "openai"
	reg := providers.NewRegistry(map[string]config.ProviderConfig{
		providerID: {Type: "openai", APIKey: "dummy", BaseURL: upstream.URL + "/v1"},
	})
	router := routing.NewEngine(config.RoutingConfig{
		DefaultStrategy: "weighted",
		Routes: []config.RouteConfig{{
			Name:  "responses-upstream-route",
			Match: config.MatchConfig{Path: "*"},
			Targets: []config.TargetConfig{{
				Provider:            providerID,
				Model:               "gpt-5.3-codex",
				Weight:              1,
				UpstreamRequestType: "responses",
			}},
		}},
	})
	retrier := resilience.NewRetrier(config.RetryConfig{Enabled: false})
	cbm := resilience.NewCircuitBreakerManager()
	fb := resilience.NewFallbackExecutor(retrier, cbm)
	cache := middleware.NewCache(config.CacheConfig{Enabled: false})
	streamer := streaming.NewHandler()
	metrics := observability.NewMetricsWithRegisterer(prometheus.NewRegistry())
	h := NewHandler(reg, router, fb, cache, streamer, metrics, nil, nil, nil)

	payload := []byte(`{"model":"lunargate/auto","messages":[{"role":"user","content":"hello"}],"tools":[{"type":"function","function":{"name":"exec_command","description":"run command","parameters":{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}}}],"tool_choice":"auto"}`)
	req := httptest.NewRequest(http.MethodPost, "http://example.com/v1/chat/completions", bytes.NewReader(payload))
	rec := httptest.NewRecorder()

	h.ChatCompletions(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}
	if capturedPath != "/v1/responses" {
		t.Fatalf("expected upstream path /v1/responses, got %q", capturedPath)
	}
	if capturedBody == nil {
		t.Fatalf("expected captured upstream payload")
	}
	if _, ok := capturedBody["input"]; !ok {
		t.Fatalf("expected responses upstream payload with input")
	}
	if _, ok := capturedBody["messages"]; ok {
		t.Fatalf("did not expect chat-completions messages field in responses payload")
	}
	if choice, _ := capturedBody["tool_choice"].(string); choice != "auto" {
		t.Fatalf("expected responses upstream tool_choice=auto to be preserved, got %q", choice)
	}
}

func TestResponses_NativeResponsesStreamPreservesRepeatedDeltasWithoutSnapshots(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/responses" {
			t.Fatalf("expected upstream path /v1/responses, got %q", r.URL.Path)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		events := []string{
			`{"type":"response.created","sequence_number":0,"response":{"id":"resp_native","object":"response","created_at":123,"status":"in_progress","model":"gpt-native","output":[]}}`,
			`{"type":"response.output_text.delta","sequence_number":1,"response_id":"resp_native","item_id":"msg_native","output_index":0,"content_index":0,"delta":"ha"}`,
			`{"type":"response.output_text.delta","sequence_number":2,"response_id":"resp_native","item_id":"msg_native","output_index":0,"content_index":0,"delta":"ha"}`,
			`{"type":"response.output_text.done","sequence_number":3,"response_id":"resp_native","item_id":"msg_native","output_index":0,"content_index":0,"text":"haha"}`,
			`{"type":"response.content_part.done","sequence_number":4,"response_id":"resp_native","item_id":"msg_native","output_index":0,"content_index":0,"part":{"type":"output_text","text":"haha"}}`,
			`{"type":"response.output_item.done","sequence_number":5,"response_id":"resp_native","output_index":0,"item":{"id":"msg_native","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"haha"}]}}`,
			`{"type":"response.completed","sequence_number":6,"response":{"id":"resp_native","object":"response","created_at":123,"status":"completed","model":"gpt-native","output":[{"id":"msg_native","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"haha"}]}],"output_text":"haha","usage":{"input_tokens":3,"output_tokens":2,"total_tokens":5}}}`,
		}
		for _, event := range events {
			_, _ = w.Write([]byte("data: " + event + "\n\n"))
		}
	}))
	defer upstream.Close()

	providerID := "openai"
	reg := providers.NewRegistry(map[string]config.ProviderConfig{
		providerID: {Type: "openai", APIKey: "dummy", BaseURL: upstream.URL + "/v1"},
	})
	router := routing.NewEngine(config.RoutingConfig{
		DefaultStrategy: "weighted",
		Routes: []config.RouteConfig{{
			Name:  "native-responses-stream",
			Match: config.MatchConfig{Path: "/v1/responses"},
			Targets: []config.TargetConfig{{
				Provider:            providerID,
				Model:               "gpt-native",
				Weight:              1,
				UpstreamRequestType: "responses",
			}},
		}},
	})
	retrier := resilience.NewRetrier(config.RetryConfig{Enabled: false})
	cbm := resilience.NewCircuitBreakerManager()
	fb := resilience.NewFallbackExecutor(retrier, cbm)
	cache := middleware.NewCache(config.CacheConfig{Enabled: false})
	streamer := streaming.NewHandler()
	metrics := observability.NewMetricsWithRegisterer(prometheus.NewRegistry())
	h := NewHandler(reg, router, fb, cache, streamer, metrics, nil, nil, nil)

	payload := []byte(`{"model":"lunargate/auto","stream":true,"input":"laugh"}`)
	req := httptest.NewRequest(http.MethodPost, "http://example.com/v1/responses", bytes.NewReader(payload))
	rec := httptest.NewRecorder()
	h.Responses(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d: %s", http.StatusOK, rec.Code, rec.Body.String())
	}
	events := decodeSSEEvents(t, rec.Body.String())
	var deltas []string
	var completed map[string]interface{}
	for _, event := range events {
		switch event["type"] {
		case "response.output_text.delta":
			if delta, _ := event["delta"].(string); delta != "" {
				deltas = append(deltas, delta)
			}
		case "response.completed":
			completed, _ = event["response"].(map[string]interface{})
		}
	}
	if len(deltas) != 2 || deltas[0] != "ha" || deltas[1] != "ha" {
		t.Fatalf("text deltas = %#v, want two repeated true deltas", deltas)
	}
	if completed == nil || completed["output_text"] != "haha" {
		t.Fatalf("completed response = %#v, want output_text haha", completed)
	}
	usage, _ := completed["usage"].(map[string]interface{})
	if usage == nil || usage["input_tokens"] != float64(3) || usage["output_tokens"] != float64(2) {
		t.Fatalf("completed usage = %#v", usage)
	}
}
