package api

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/prometheus/client_golang/prometheus/testutil"
)

func TestChatCompletionsAccountsPromptCacheUsageOnce(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-cache","object":"chat.completion","model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1000000,"completion_tokens":1000000,"total_tokens":2000000,"prompt_tokens_details":{"cached_tokens":400000,"cache_write_tokens":100000}}}`))
	}))
	defer upstream.Close()

	capture := newCollectorCapture(t, false, false)
	handler, metrics := newObservedOpenAIHandler(
		t,
		upstream.URL,
		config.TargetConfig{Provider: "openai", Model: "gpt-4o", Weight: 1},
		capture.client,
		config.CacheConfig{Enabled: false},
	)

	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"messages":[{"role":"user","content":"hello"}]}`))
	handler.ChatCompletions(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", recorder.Code, recorder.Body.String())
	}

	_, metric, _ := capture.waitForTraceAndMetric(t)
	assertMetricNumber(t, metric, "tokens_input", 1_000_000)
	assertMetricNumber(t, metric, "tokens_output", 1_000_000)
	assertMetricNumber(t, metric, "tokens_input_cached", 400_000)
	assertMetricNumber(t, metric, "tokens_input_cache_write", 100_000)
	assertMetricNumber(t, metric, "cost_usd", 12)

	if got := testutil.ToFloat64(metrics.TokensTotal.WithLabelValues("openai", "gpt-4o", "input")); got != 1_000_000 {
		t.Fatalf("inclusive input metric = %v, want 1000000", got)
	}
	if got := testutil.ToFloat64(metrics.CacheTokensTotal.WithLabelValues("openai", "gpt-4o", "read")); got != 400_000 {
		t.Fatalf("cache read metric = %v, want 400000", got)
	}
	if got := testutil.ToFloat64(metrics.CacheTokensTotal.WithLabelValues("openai", "gpt-4o", "write")); got != 100_000 {
		t.Fatalf("cache write metric = %v, want 100000", got)
	}
}

func assertMetricNumber(t *testing.T, metric map[string]interface{}, field string, want float64) {
	t.Helper()
	if got := metric[field]; got != want {
		t.Fatalf("%s = %#v, want %v", field, got, want)
	}
}
