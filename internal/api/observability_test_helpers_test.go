package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
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
)

type capturedCollectorEvent struct {
	Type string                 `json:"type"`
	Data map[string]interface{} `json:"data"`
}

type capturedCollectorBatch struct {
	Events []capturedCollectorEvent `json:"events"`
}

type capturedCollectorResult struct {
	batch capturedCollectorBatch
	err   error
}

type collectorCapture struct {
	client  *observability.CollectorClient
	results chan capturedCollectorResult
}

func newCollectorCapture(t *testing.T, sharePrompts, shareResponses bool) *collectorCapture {
	t.Helper()
	results := make(chan capturedCollectorResult, 16)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var batch capturedCollectorBatch
		err := json.NewDecoder(r.Body).Decode(&batch)
		results <- capturedCollectorResult{batch: batch, err: err}
		w.WriteHeader(http.StatusAccepted)
	}))
	client := observability.NewCollectorClient(config.GeneralConfig{
		APIKey:     "collector-test-key",
		BackendURL: server.URL + "/v1",
	}, config.DataSharingConfig{
		Enabled:        true,
		SharePrompts:   sharePrompts,
		ShareResponses: shareResponses,
	}, "test")
	t.Cleanup(func() {
		client.Stop()
		server.Close()
	})
	return &collectorCapture{client: client, results: results}
}

func (c *collectorCapture) waitForRequestEvents(t *testing.T) (map[string]interface{}, map[string]interface{}, map[string]interface{}) {
	t.Helper()
	return c.waitForEvents(t, true)
}

func (c *collectorCapture) waitForTraceAndMetric(t *testing.T) (map[string]interface{}, map[string]interface{}, map[string]interface{}) {
	t.Helper()
	return c.waitForEvents(t, false)
}

func (c *collectorCapture) waitForEvents(t *testing.T, requireRequestLog bool) (map[string]interface{}, map[string]interface{}, map[string]interface{}) {
	t.Helper()
	var trace, metric, requestLog map[string]interface{}
	timer := time.NewTimer(3 * time.Second)
	defer timer.Stop()
	for trace == nil || metric == nil || (requireRequestLog && requestLog == nil) {
		select {
		case result := <-c.results:
			if result.err != nil {
				t.Fatalf("decode collector batch: %v", result.err)
			}
			for _, event := range result.batch.Events {
				switch event.Type {
				case "trace":
					trace = event.Data
				case "metric":
					metric = event.Data
				case "request_log":
					requestLog = event.Data
				}
			}
		case <-timer.C:
			t.Fatalf("timed out waiting for collector events: trace=%v metric=%v request_log=%v", trace != nil, metric != nil, requestLog != nil)
		}
	}
	return trace, metric, requestLog
}

func newObservedOpenAIHandler(
	t *testing.T,
	upstreamURL string,
	target config.TargetConfig,
	collector *observability.CollectorClient,
	cacheConfig config.CacheConfig,
) (*Handler, *observability.Metrics) {
	t.Helper()
	registry := providers.NewRegistry(map[string]config.ProviderConfig{
		target.Provider: {
			Type:    "openai",
			APIKey:  "provider-test-key",
			BaseURL: upstreamURL + "/v1",
		},
	})
	router := routing.NewEngine(config.RoutingConfig{
		DefaultStrategy: "weighted",
		Routes: []config.RouteConfig{{
			Name:    "observed-route",
			Match:   config.MatchConfig{Path: "*"},
			Targets: []config.TargetConfig{target},
		}},
	})
	cache := middleware.NewCache(cacheConfig)
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
		collector,
		nil,
		nil,
	)
	return handler, metrics
}

func assertCapturedRequestTypes(t *testing.T, data map[string]interface{}, client, upstream string) {
	t.Helper()
	if got := data["request_type"]; got != client {
		t.Fatalf("request_type = %#v, want %q", got, client)
	}
	if got := data["upstream_request_type"]; got != upstream {
		t.Fatalf("upstream_request_type = %#v, want %q", got, upstream)
	}
	tags, _ := data["tags"].(map[string]interface{})
	if tags == nil {
		t.Fatal("collector event has no tags")
	}
	if got := tags["x-lunargate-request-type"]; got != client {
		t.Fatalf("request type tag = %#v, want %q", got, client)
	}
	if got := tags["x-lunargate-upstream-request-type"]; got != upstream {
		t.Fatalf("upstream request type tag = %#v, want %q", got, upstream)
	}
}

func assertCapturedTraceRequestTypes(t *testing.T, data map[string]interface{}, client, upstream string) {
	t.Helper()
	tags, _ := data["tags"].(map[string]interface{})
	if tags == nil {
		t.Fatal("collector trace has no tags")
	}
	if got := tags["x-lunargate-request-type"]; got != client {
		t.Fatalf("trace request type tag = %#v, want %q", got, client)
	}
	if got := tags["x-lunargate-upstream-request-type"]; got != upstream {
		t.Fatalf("trace upstream request type tag = %#v, want %q", got, upstream)
	}
}
