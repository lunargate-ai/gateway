package api

import (
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
	"github.com/prometheus/client_golang/prometheus"
)

func TestChatCompletionsCollectorDoesNotPriceCustomOpenAICompatibleFallback(t *testing.T) {
	var primaryCalls atomic.Int32
	primary := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		primaryCalls.Add(1)
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer primary.Close()

	var fallbackCalls atomic.Int32
	fallback := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fallbackCalls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-cost","object":"chat.completion","model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1000000,"completion_tokens":1000000,"total_tokens":2000000}}`))
	}))
	defer fallback.Close()

	capture := newCollectorCapture(t, false, false)
	registry := providers.NewRegistry(map[string]config.ProviderConfig{
		"primary":  {Type: "anthropic", APIKey: "test", BaseURL: primary.URL},
		"fallback": {Type: "openai", APIKey: "test", BaseURL: fallback.URL},
	})
	router := routing.NewEngine(config.RoutingConfig{
		DefaultStrategy: "weighted",
		Routes: []config.RouteConfig{{
			Name:     "chat-cost",
			Match:    config.MatchConfig{Path: "/v1/chat/completions"},
			Targets:  []config.TargetConfig{{Provider: "primary", Model: "claude-3-opus-20240229", Weight: 1}},
			Fallback: []config.TargetConfig{{Provider: "fallback", Model: "gpt-4o", Weight: 1}},
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
		capture.client,
		nil,
		nil,
	)

	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/chat/completions",
		strings.NewReader(`{"messages":[{"role":"user","content":"hello"}]}`),
	)
	handler.ChatCompletions(recorder, request)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", recorder.Code, recorder.Body.String())
	}
	if got := recorder.Header().Get("X-LunarGate-Provider"); got != "fallback" {
		t.Fatalf("provider header = %q, want fallback", got)
	}
	if got := recorder.Header().Get("X-LunarGate-Model"); got != "fallback/gpt-4o" {
		t.Fatalf("model header = %q, want fallback/gpt-4o", got)
	}
	if got := primaryCalls.Load(); got != 1 {
		t.Fatalf("primary calls = %d, want 1", got)
	}
	if got := fallbackCalls.Load(); got != 1 {
		t.Fatalf("fallback calls = %d, want 1", got)
	}

	_, metric, _ := capture.waitForTraceAndMetric(t)
	if got := metric["provider"]; got != "fallback" {
		t.Fatalf("metric provider = %#v, want fallback", got)
	}
	if got := metric["model"]; got != "fallback/gpt-4o" {
		t.Fatalf("metric model = %#v, want fallback/gpt-4o", got)
	}
	if got := metric["tokens_input"]; got != float64(1_000_000) {
		t.Fatalf("metric tokens_input = %#v, want 1000000", got)
	}
	if got := metric["tokens_output"]; got != float64(1_000_000) {
		t.Fatalf("metric tokens_output = %#v, want 1000000", got)
	}
	if got := metric["cost_usd"]; got != float64(0) {
		t.Fatalf("metric cost_usd = %#v, want 0 for custom OpenAI-compatible provider", got)
	}
	if got := metric["fallback_used"]; got != true {
		t.Fatalf("metric fallback_used = %#v, want true", got)
	}
}
