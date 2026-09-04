package api

import (
	"bytes"
	"encoding/json"
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
	"github.com/prometheus/client_golang/prometheus/testutil"
)

func newResilienceClassificationHandler(
	t *testing.T,
	providerConfigs map[string]config.ProviderConfig,
	route config.RouteConfig,
	retryConfig config.RetryConfig,
) (*Handler, *resilience.CircuitBreakerManager, *observability.Metrics) {
	t.Helper()

	cache := middleware.NewCache(config.CacheConfig{Enabled: false})
	t.Cleanup(cache.Stop)
	metrics := observability.NewMetricsWithRegisterer(prometheus.NewRegistry())
	cbm := resilience.NewCircuitBreakerManager()
	handler := NewHandler(
		providers.NewRegistry(providerConfigs),
		routing.NewEngine(config.RoutingConfig{
			DefaultStrategy: "weighted",
			Routes:          []config.RouteConfig{route},
		}),
		resilience.NewFallbackExecutor(resilience.NewRetrier(retryConfig), cbm),
		cache,
		streaming.NewHandler(),
		metrics,
		nil,
		nil,
		nil,
	)
	return handler, cbm, metrics
}

func TestChatCompletions_TranslationErrorStopsRetryAndFallback(t *testing.T) {
	var fallbackCalls atomic.Int32
	fallbackUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fallbackCalls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-fallback","object":"chat.completion","choices":[{"message":{"role":"assistant","content":"unexpected"}}]}`))
	}))
	defer fallbackUpstream.Close()

	handler, cbm, metrics := newResilienceClassificationHandler(t, map[string]config.ProviderConfig{
		"primary":  {Type: "ollama", BaseURL: "http://127.0.0.1:1"},
		"fallback": {Type: "openai", APIKey: "dummy", BaseURL: fallbackUpstream.URL},
	}, config.RouteConfig{
		Name:     "default",
		Match:    config.MatchConfig{Path: "*"},
		Targets:  []config.TargetConfig{{Provider: "primary", Model: "llama3", Weight: 1}},
		Fallback: []config.TargetConfig{{Provider: "fallback", Model: "gpt-fallback", Weight: 1}},
	}, config.RetryConfig{
		Enabled:         true,
		MaxAttempts:     3,
		RetryableErrors: []int{http.StatusTooManyRequests, http.StatusInternalServerError},
	})

	payload := []byte(`{
		"messages":[{"role":"user","content":"hi"}],
		"tools":[{"type":"function","function":{"name":"known","parameters":{"type":"object"}}}],
		"tool_choice":{"type":"function","function":{"name":"missing"}}
	}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(payload))
	rec := httptest.NewRecorder()
	handler.ChatCompletions(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400; body=%s", rec.Code, rec.Body.String())
	}
	var response models.ErrorResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if response.Error.Type != "invalid_request_error" || !strings.Contains(response.Error.Message, `unknown tool "missing"`) {
		t.Fatalf("error = %#v, want invalid unknown-tool request", response.Error)
	}
	if calls := fallbackCalls.Load(); calls != 0 {
		t.Fatalf("fallback calls = %d, want 0", calls)
	}
	if counts := cbm.Get("primary").Counts(); counts.TotalFailures != 0 {
		t.Fatalf("translation error provider failures = %d, want 0", counts.TotalFailures)
	}
	if got := testutil.ToFloat64(metrics.ProviderErrors.WithLabelValues("primary", "all_failed")); got != 0 {
		t.Fatalf("all_failed metric = %v, want 0", got)
	}
}

func TestEmbeddings_TranslationErrorStopsRetryAndFallback(t *testing.T) {
	var fallbackCalls atomic.Int32
	fallbackUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fallbackCalls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"object":"list","data":[],"model":"embed","usage":{"prompt_tokens":0,"total_tokens":0}}`))
	}))
	defer fallbackUpstream.Close()

	handler, cbm, metrics := newResilienceClassificationHandler(t, map[string]config.ProviderConfig{
		"primary":  {Type: "ollama", BaseURL: "http://127.0.0.1:1"},
		"fallback": {Type: "openai", APIKey: "dummy", BaseURL: fallbackUpstream.URL},
	}, config.RouteConfig{
		Name:     "embeddings",
		Match:    config.MatchConfig{Path: "/v1/embeddings"},
		Targets:  []config.TargetConfig{{Provider: "primary", Model: "embed", Weight: 1}},
		Fallback: []config.TargetConfig{{Provider: "fallback", Model: "embed", Weight: 1}},
	}, config.RetryConfig{
		Enabled:         true,
		MaxAttempts:     3,
		RetryableErrors: []int{http.StatusTooManyRequests, http.StatusInternalServerError},
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewReader([]byte(`{"model":"embed","input":42}`)))
	rec := httptest.NewRecorder()
	handler.Embeddings(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400; body=%s", rec.Code, rec.Body.String())
	}
	var response models.ErrorResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if response.Error.Type != "invalid_request_error" || !strings.Contains(response.Error.Message, "accepts only a string or an array of strings") {
		t.Fatalf("error = %#v, want invalid embeddings input", response.Error)
	}
	if response.Error.Param == nil || *response.Error.Param != "input" {
		t.Fatalf("error param = %#v, want input", response.Error.Param)
	}
	if response.Error.Code == nil || *response.Error.Code != "unsupported_feature" {
		t.Fatalf("error code = %#v, want unsupported_feature", response.Error.Code)
	}
	if calls := fallbackCalls.Load(); calls != 0 {
		t.Fatalf("fallback calls = %d, want 0", calls)
	}
	if counts := cbm.Get("primary").Counts(); counts.TotalFailures != 0 {
		t.Fatalf("translation error provider failures = %d, want 0", counts.TotalFailures)
	}
	if got := testutil.ToFloat64(metrics.ProviderErrors.WithLabelValues("primary", "all_failed")); got != 0 {
		t.Fatalf("all_failed metric = %v, want 0", got)
	}
}

func TestChatCompletions_NonRetryable4xxSkipsRetryAndFallback(t *testing.T) {
	var primaryCalls atomic.Int32
	primaryUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		primaryCalls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"message":"bad input","type":"invalid_request_error"}}`))
	}))
	defer primaryUpstream.Close()

	var fallbackCalls atomic.Int32
	fallbackUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fallbackCalls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-fallback","object":"chat.completion","choices":[{"message":{"role":"assistant","content":"unexpected"}}]}`))
	}))
	defer fallbackUpstream.Close()

	handler, cbm, _ := newResilienceClassificationHandler(t, map[string]config.ProviderConfig{
		"primary":  {Type: "openai", APIKey: "dummy", BaseURL: primaryUpstream.URL},
		"fallback": {Type: "openai", APIKey: "dummy", BaseURL: fallbackUpstream.URL},
	}, config.RouteConfig{
		Name:     "default",
		Match:    config.MatchConfig{Path: "*"},
		Targets:  []config.TargetConfig{{Provider: "primary", Model: "gpt-primary", Weight: 1}},
		Fallback: []config.TargetConfig{{Provider: "fallback", Model: "gpt-fallback", Weight: 1}},
	}, config.RetryConfig{
		Enabled:         true,
		MaxAttempts:     3,
		RetryableErrors: []int{http.StatusTooManyRequests, http.StatusInternalServerError},
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader([]byte(`{"messages":[{"role":"user","content":"hi"}]}`)))
	rec := httptest.NewRecorder()
	handler.ChatCompletions(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400; body=%s", rec.Code, rec.Body.String())
	}
	if calls := primaryCalls.Load(); calls != 1 {
		t.Fatalf("primary calls = %d, want 1", calls)
	}
	if calls := fallbackCalls.Load(); calls != 0 {
		t.Fatalf("fallback calls = %d, want 0", calls)
	}
	if counts := cbm.Get("primary").Counts(); counts.TotalFailures != 0 {
		t.Fatalf("client error provider failures = %d, want 0", counts.TotalFailures)
	}
}

func TestChatCompletions_Configured429RetriesAndFallsBackWithoutTrippingCircuit(t *testing.T) {
	var primaryCalls atomic.Int32
	primaryUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		primaryCalls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"message":"rate limited","type":"rate_limit_error"}}`))
	}))
	defer primaryUpstream.Close()

	var fallbackCalls atomic.Int32
	fallbackUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fallbackCalls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-fallback","object":"chat.completion","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	defer fallbackUpstream.Close()

	handler, cbm, _ := newResilienceClassificationHandler(t, map[string]config.ProviderConfig{
		"primary":  {Type: "openai", APIKey: "dummy", BaseURL: primaryUpstream.URL},
		"fallback": {Type: "openai", APIKey: "dummy", BaseURL: fallbackUpstream.URL},
	}, config.RouteConfig{
		Name:     "default",
		Match:    config.MatchConfig{Path: "*"},
		Targets:  []config.TargetConfig{{Provider: "primary", Model: "gpt-primary", Weight: 1}},
		Fallback: []config.TargetConfig{{Provider: "fallback", Model: "gpt-fallback", Weight: 1}},
	}, config.RetryConfig{
		Enabled:         true,
		MaxAttempts:     2,
		InitialDelay:    0,
		MaxDelay:        0,
		Multiplier:      1,
		RetryableErrors: []int{http.StatusTooManyRequests},
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader([]byte(`{"messages":[{"role":"user","content":"hi"}]}`)))
	rec := httptest.NewRecorder()
	handler.ChatCompletions(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	if calls := primaryCalls.Load(); calls != 2 {
		t.Fatalf("primary calls = %d, want 2 configured attempts", calls)
	}
	if calls := fallbackCalls.Load(); calls != 1 {
		t.Fatalf("fallback calls = %d, want 1", calls)
	}
	if counts := cbm.Get("primary").Counts(); counts.TotalFailures != 0 {
		t.Fatalf("429 provider failures = %d, want 0", counts.TotalFailures)
	}
}
