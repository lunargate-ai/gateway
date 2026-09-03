package api

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/middleware"
	"github.com/lunargate-ai/gateway/internal/observability"
	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/internal/resilience"
	"github.com/lunargate-ai/gateway/internal/routing"
	"github.com/lunargate-ai/gateway/internal/streaming"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
)

func TestChatCompletionsFailureCollectorUsesLastFallbackModel(t *testing.T) {
	var primaryCalls atomic.Int32
	primary := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		primaryCalls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"type":"error","error":{"type":"api_error","message":"primary unavailable"}}`))
	}))
	defer primary.Close()

	var fallbackCalls atomic.Int32
	fallback := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fallbackCalls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error":{"message":"fallback unavailable","type":"server_error","code":"fallback_failed"}}`))
	}))
	defer fallback.Close()

	capture := newCollectorCapture(t, true, false)
	registry := providers.NewRegistry(map[string]config.ProviderConfig{
		"primary":  {Type: "anthropic", APIKey: "test", BaseURL: primary.URL},
		"fallback": {Type: "openai", APIKey: "test", BaseURL: fallback.URL},
	})
	router := routing.NewEngine(config.RoutingConfig{
		DefaultStrategy: "weighted",
		Routes: []config.RouteConfig{{
			Name:     "chat-failure",
			Match:    config.MatchConfig{Path: "/v1/chat/completions"},
			Targets:  []config.TargetConfig{{Provider: "primary", Model: "claude-3-opus-20240229", Weight: 1}},
			Fallback: []config.TargetConfig{{Provider: "fallback", Model: "gpt-4o", Weight: 1}},
		}},
	})
	cache := middleware.NewCache(config.CacheConfig{Enabled: false})
	t.Cleanup(cache.Stop)
	metrics := observability.NewMetricsWithRegisterer(prometheus.NewRegistry())
	handler := NewHandler(
		registry,
		router,
		resilience.NewFallbackExecutor(
			resilience.NewRetrier(config.RetryConfig{Enabled: false}),
			resilience.NewCircuitBreakerManager(),
		),
		cache,
		streaming.NewHandler(),
		metrics,
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

	if recorder.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d, want 500; body=%s", recorder.Code, recorder.Body.String())
	}
	if got := primaryCalls.Load(); got != 1 {
		t.Fatalf("primary calls = %d, want 1", got)
	}
	if got := fallbackCalls.Load(); got != 1 {
		t.Fatalf("fallback calls = %d, want 1", got)
	}
	if got := testutil.ToFloat64(metrics.ProviderErrors.WithLabelValues("fallback", "all_failed")); got != 1 {
		t.Fatalf("fallback all_failed metric = %v, want 1", got)
	}
	if got := testutil.ToFloat64(metrics.ProviderErrors.WithLabelValues("primary", "all_failed")); got != 0 {
		t.Fatalf("primary all_failed metric = %v, want 0", got)
	}

	metric, requestLog := waitForFailureCollectorEvents(t, capture)
	assertFailureCollectorTarget(t, "metric", metric)
	assertFailureCollectorTarget(t, "request log", requestLog)
}

func waitForFailureCollectorEvents(t *testing.T, capture *collectorCapture) (map[string]interface{}, map[string]interface{}) {
	t.Helper()
	var metric, requestLog map[string]interface{}
	timer := time.NewTimer(3 * time.Second)
	defer timer.Stop()
	for metric == nil || requestLog == nil {
		select {
		case result := <-capture.results:
			if result.err != nil {
				t.Fatalf("decode collector batch: %v", result.err)
			}
			for _, event := range result.batch.Events {
				switch event.Type {
				case "metric":
					metric = event.Data
				case "request_log":
					requestLog = event.Data
				}
			}
		case <-timer.C:
			t.Fatalf("timed out waiting for collector events: metric=%v request_log=%v", metric != nil, requestLog != nil)
		}
	}
	return metric, requestLog
}

func assertFailureCollectorTarget(t *testing.T, eventType string, event map[string]interface{}) {
	t.Helper()
	if got := event["provider"]; got != "fallback" {
		t.Fatalf("%s provider = %#v, want fallback", eventType, got)
	}
	if got := event["model"]; got != "fallback/gpt-4o" {
		t.Fatalf("%s model = %#v, want fallback/gpt-4o", eventType, got)
	}
	if got := event["fallback_used"]; got != true {
		t.Fatalf("%s fallback_used = %#v, want true", eventType, got)
	}
	tags, _ := event["tags"].(map[string]interface{})
	if got := tags["x-lunargate-resolved-provider"]; got != "fallback" {
		t.Fatalf("%s resolved provider = %#v, want fallback", eventType, got)
	}
	if got := tags["x-lunargate-resolved-model"]; got != "fallback/gpt-4o" {
		t.Fatalf("%s resolved model = %#v, want fallback/gpt-4o", eventType, got)
	}
}
