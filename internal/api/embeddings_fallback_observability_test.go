package api

import (
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
	"github.com/prometheus/client_golang/prometheus"
)

func TestEmbeddingsFallbackCollectorUsesServingCustomTarget(t *testing.T) {
	primary := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer primary.Close()
	fallback := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0.1]}],"model":"text-embedding-3-small","usage":{"prompt_tokens":1000000,"total_tokens":1000000}}`))
	}))
	defer fallback.Close()

	capture := newCollectorCapture(t, false, false)
	registry := providers.NewRegistry(map[string]config.ProviderConfig{
		"primary":  {Type: "openai", APIKey: "test", BaseURL: primary.URL},
		"fallback": {Type: "openai", APIKey: "test", BaseURL: fallback.URL},
	})
	router := routing.NewEngine(config.RoutingConfig{
		DefaultStrategy: "weighted",
		Routes: []config.RouteConfig{{
			Name:     "embeddings",
			Match:    config.MatchConfig{Path: "/v1/embeddings"},
			Targets:  []config.TargetConfig{{Provider: "primary", Model: "text-embedding-3-small", Weight: 1}},
			Fallback: []config.TargetConfig{{Provider: "fallback", Model: "text-embedding-3-small", Weight: 1}},
		}},
	})
	cache := middleware.NewCache(config.CacheConfig{Enabled: false})
	defer cache.Stop()
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
	request := httptest.NewRequest(http.MethodPost, "/v1/embeddings", strings.NewReader(`{"model":"text-embedding-3-small","input":"hello"}`))
	handler.Embeddings(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", recorder.Code, recorder.Body.String())
	}

	_, metric, _ := capture.waitForTraceAndMetric(t)
	if metric["provider"] != "fallback" || metric["model"] != "fallback/text-embedding-3-small" {
		t.Fatalf("metric target = %v/%v", metric["provider"], metric["model"])
	}
	if got := metric["cost_usd"]; got != float64(0) {
		t.Fatalf("metric cost_usd = %#v, want 0 for custom OpenAI-compatible provider", got)
	}
	tags, _ := metric["tags"].(map[string]interface{})
	if tags["x-lunargate-resolved-provider"] != "fallback" {
		t.Fatalf("resolved provider tag = %v, want fallback", tags["x-lunargate-resolved-provider"])
	}
	if tags["x-lunargate-resolved-model"] != "fallback/text-embedding-3-small" {
		t.Fatalf("resolved model tag = %v, want fallback/text-embedding-3-small", tags["x-lunargate-resolved-model"])
	}
}
