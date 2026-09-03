package api

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/observability"
	"github.com/lunargate-ai/gateway/internal/routing"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	dto "github.com/prometheus/client_model/go"
)

func TestChatCacheHitRecordsRequestMetricsAndCollectorEvents(t *testing.T) {
	var upstreamCalls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamCalls.Add(1)
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("upstream path = %q, want /v1/chat/completions", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"chatcmpl-cache","object":"chat.completion","created":1,"model":"gpt-cache","choices":[{"index":0,"message":{"role":"assistant","content":"cached"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`)
	}))
	defer upstream.Close()

	capture := newCollectorCapture(t, true, true)
	handler, metrics := newObservedOpenAIHandler(t, upstream.URL, config.TargetConfig{
		Provider: "openai",
		Model:    "gpt-cache",
		Weight:   1,
	}, capture.client, enabledTestCacheConfig())
	payload := `{"model":"gpt-cache","messages":[{"role":"user","content":"hello"}]}`

	first := httptest.NewRecorder()
	handler.ChatCompletions(first, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(payload)))
	if first.Code != http.StatusOK {
		t.Fatalf("first status = %d, want 200; body=%s", first.Code, first.Body.String())
	}
	if got := first.Header().Get("X-LunarGate-Cache-Status"); got != "MISS" {
		t.Fatalf("first cache status = %q, want MISS", got)
	}
	_, _, _ = capture.waitForRequestEvents(t)

	second := httptest.NewRecorder()
	handler.ChatCompletions(second, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(payload)))
	if second.Code != http.StatusOK {
		t.Fatalf("cached status = %d, want 200; body=%s", second.Code, second.Body.String())
	}
	if got := second.Header().Get("X-LunarGate-Cache-Status"); got != "HIT" {
		t.Fatalf("cached cache status = %q, want HIT", got)
	}
	if got := upstreamCalls.Load(); got != 1 {
		t.Fatalf("upstream calls = %d, want 1", got)
	}

	trace, metric, requestLog := capture.waitForRequestEvents(t)
	assertCapturedTraceRequestTypes(t, trace, "chat_completions", "chat_completions")
	assertCapturedRequestTypes(t, metric, "chat_completions", "chat_completions")
	assertCapturedRequestTypes(t, requestLog, "chat_completions", "chat_completions")
	assertCapturedCacheHit(t, metric)
	assertCapturedCacheHit(t, requestLog)
	assertCapturedSharedPayloads(t, requestLog, true, true)
	if got := metric["tokens_input"]; got != float64(3) {
		t.Fatalf("cached tokens_input = %#v, want 3", got)
	}
	if got := metric["tokens_output"]; got != float64(2) {
		t.Fatalf("cached tokens_output = %#v, want 2", got)
	}
	if got := metric["cost_usd"]; got != float64(0) {
		t.Fatalf("cached cost_usd = %#v, want 0", got)
	}
	assertRequestMetricCounts(t, metrics, "openai", "gpt-cache", "observed-route", 2)
}

func TestEmbeddingsCacheHitRecordsRequestMetricsAndCollectorEvents(t *testing.T) {
	var upstreamCalls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamCalls.Add(1)
		if r.URL.Path != "/v1/embeddings" {
			t.Errorf("upstream path = %q, want /v1/embeddings", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"object":"list","data":[{"object":"embedding","embedding":[0.1,0.2],"index":0}],"model":"text-embedding-cache","usage":{"prompt_tokens":2,"total_tokens":2}}`)
	}))
	defer upstream.Close()

	capture := newCollectorCapture(t, true, true)
	handler, metrics := newObservedOpenAIHandler(t, upstream.URL, config.TargetConfig{
		Provider: "openai",
		Model:    "text-embedding-cache",
		Weight:   1,
	}, capture.client, enabledTestCacheConfig())
	payload := `{"model":"text-embedding-cache","input":"hello"}`

	first := httptest.NewRecorder()
	handler.Embeddings(first, httptest.NewRequest(http.MethodPost, "/v1/embeddings", strings.NewReader(payload)))
	if first.Code != http.StatusOK {
		t.Fatalf("first status = %d, want 200; body=%s", first.Code, first.Body.String())
	}
	if got := first.Header().Get("X-LunarGate-Cache-Status"); got != "MISS" {
		t.Fatalf("first cache status = %q, want MISS", got)
	}
	_, _, _ = capture.waitForRequestEvents(t)

	second := httptest.NewRecorder()
	handler.Embeddings(second, httptest.NewRequest(http.MethodPost, "/v1/embeddings", strings.NewReader(payload)))
	if second.Code != http.StatusOK {
		t.Fatalf("cached status = %d, want 200; body=%s", second.Code, second.Body.String())
	}
	if got := second.Header().Get("X-LunarGate-Cache-Status"); got != "HIT" {
		t.Fatalf("cached cache status = %q, want HIT", got)
	}
	if got := upstreamCalls.Load(); got != 1 {
		t.Fatalf("upstream calls = %d, want 1", got)
	}

	trace, metric, requestLog := capture.waitForRequestEvents(t)
	assertCapturedTraceRequestTypes(t, trace, "embeddings", "embeddings")
	assertCapturedRequestTypes(t, metric, "embeddings", "embeddings")
	assertCapturedRequestTypes(t, requestLog, "embeddings", "embeddings")
	assertCapturedCacheHit(t, metric)
	assertCapturedCacheHit(t, requestLog)
	assertCapturedSharedPayloads(t, requestLog, true, true)
	if got := metric["tokens_input"]; got != float64(2) {
		t.Fatalf("cached tokens_input = %#v, want 2", got)
	}
	if got := metric["tokens_output"]; got != float64(0) {
		t.Fatalf("cached tokens_output = %#v, want 0", got)
	}
	if got := metric["cost_usd"]; got != float64(0) {
		t.Fatalf("cached cost_usd = %#v, want 0", got)
	}
	assertRequestMetricCounts(t, metrics, "openai", "text-embedding-cache", "observed-route", 2)
}

func TestCacheHitRequestLogHonorsSharingSettings(t *testing.T) {
	tests := []struct {
		name           string
		sharePrompts   bool
		shareResponses bool
		wantRequestLog bool
	}{
		{name: "metrics only"},
		{name: "prompts only", sharePrompts: true, wantRequestLog: true},
		{name: "responses only", shareResponses: true, wantRequestLog: true},
		{name: "prompts and responses", sharePrompts: true, shareResponses: true, wantRequestLog: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			capture := newCollectorCapture(t, tt.sharePrompts, tt.shareResponses)
			handler := &Handler{
				metrics:   observability.NewMetricsWithRegisterer(prometheus.NewRegistry()),
				collector: capture.client,
			}
			handler.recordCacheHit(context.Background(), cacheHitObservation{
				requestID:    "cache-sharing-test",
				startTime:    time.Now(),
				requestTypes: chatAPIRequestTypes("chat_completions", routing.Target{UpstreamRequestType: "chat_completions"}),
				provider:     "openai",
				model:        "openai/gpt-cache",
				metricsModel: "gpt-cache",
				route:        "observed-route",
				tags: map[string]string{
					"x-lunargate-request-type":          "chat_completions",
					"x-lunargate-upstream-request-type": "chat_completions",
				},
				request:  map[string]interface{}{"prompt": "shared only when enabled"},
				response: map[string]interface{}{"answer": "shared only when enabled"},
			})

			_, _, requestLog := capture.waitForTraceAndMetric(t)
			if tt.wantRequestLog && requestLog == nil {
				t.Fatal("request_log missing")
			}
			if !tt.wantRequestLog && requestLog != nil {
				t.Fatalf("unexpected request_log: %#v", requestLog)
			}
			if requestLog != nil {
				assertCapturedSharedPayloads(t, requestLog, tt.sharePrompts, tt.shareResponses)
			}
		})
	}
}

func enabledTestCacheConfig() config.CacheConfig {
	return config.CacheConfig{Enabled: true, TTL: time.Minute, MaxSize: 8}
}

func assertCapturedCacheHit(t *testing.T, data map[string]interface{}) {
	t.Helper()
	if got := data["cache_hit"]; got != true {
		t.Fatalf("cache_hit = %#v, want true", got)
	}
}

func assertCapturedSharedPayloads(t *testing.T, requestLog map[string]interface{}, wantRequest, wantResponse bool) {
	t.Helper()
	if got := requestLog["request"] != nil; got != wantRequest {
		t.Fatalf("request payload present = %v, want %v", got, wantRequest)
	}
	if got := requestLog["response"] != nil; got != wantResponse {
		t.Fatalf("response payload present = %v, want %v", got, wantResponse)
	}
}

func assertRequestMetricCounts(t *testing.T, metrics *observability.Metrics, provider, model, route string, want uint64) {
	t.Helper()
	if got := testutil.ToFloat64(metrics.RequestsTotal.WithLabelValues(provider, model, "200", route)); got != float64(want) {
		t.Fatalf("requests_total = %v, want %d", got, want)
	}

	observer := metrics.RequestDuration.WithLabelValues(provider, model)
	metric, ok := observer.(prometheus.Metric)
	if !ok {
		t.Fatalf("request duration observer %T does not expose a metric", observer)
	}
	var value dto.Metric
	if err := metric.Write(&value); err != nil {
		t.Fatalf("write request duration metric: %v", err)
	}
	if got := value.GetHistogram().GetSampleCount(); got != want {
		t.Fatalf("request_duration count = %d, want %d", got, want)
	}
}

func TestBoundedProviderErrorMetricTypeRejectsDynamicUpstreamLabels(t *testing.T) {
	if got := boundedProviderErrorMetricType(http.StatusBadRequest, "attacker-controlled-123"); got != "invalid_request" {
		t.Fatalf("400 label = %q, want invalid_request", got)
	}
	if got := boundedProviderErrorMetricType(http.StatusBadGateway, "attacker-controlled-456"); got != "upstream_error" {
		t.Fatalf("502 label = %q, want upstream_error", got)
	}
	if got := boundedProviderErrorMetricType(http.StatusBadGateway, "parse_error"); got != "parse_error" {
		t.Fatalf("trusted parser label = %q, want parse_error", got)
	}
}

func TestBoundedModelMetricLabelCollapsesUnconfiguredModels(t *testing.T) {
	if got := boundedModelMetricLabel(routing.Target{}, "attacker-model-123"); got != dynamicModelMetricLabel {
		t.Fatalf("dynamic model label = %q, want %q", got, dynamicModelMetricLabel)
	}
	configured := routing.Target{Model: "gpt-configured"}
	if got := boundedModelMetricLabel(configured, "gpt-configured"); got != "gpt-configured" {
		t.Fatalf("configured model label = %q, want gpt-configured", got)
	}
}
