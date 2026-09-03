package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
)

const (
	customMatchHeader = "x-customer-tier"
	customMatchSecret = "enterprise-secret"
)

func TestRoutingHeadersForRequestCanonicalValuesWin(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	req.Header.Set("X-Customer-Tier", customMatchSecret)
	req.Header.Set("X-LunarGate-Request-Type", requestTypeResponses)

	canonical := map[string]string{
		"x-lunargate-request-type": requestTypeChatCompletions,
	}
	headers := routingHeadersForRequest(req, []string{
		"X-Customer-Tier",
		"X-LunarGate-Request-Type",
	}, canonical)

	if got := headers[customMatchHeader]; got != customMatchSecret {
		t.Fatalf("custom match header = %q, want %q", got, customMatchSecret)
	}
	if got := headers["x-lunargate-request-type"]; got != requestTypeChatCompletions {
		t.Fatalf("request type = %q, want canonical %q", got, requestTypeChatCompletions)
	}
	if _, ok := extractHeaders(req)[customMatchHeader]; ok {
		t.Fatalf("custom match header entered collector allowlist")
	}
}

func TestChatCustomMatchHeaderRoutesWithoutCollectorLeak(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-route","object":"chat.completion","created":1,"model":"gpt-route","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	defer upstream.Close()

	capture := newCollectorCapture(t, true, true)
	handler, _ := newObservedOpenAIHandler(t, upstream.URL, config.TargetConfig{
		Provider: "openai",
		Model:    "gpt-route",
		Weight:   1,
	}, capture.client, config.CacheConfig{Enabled: false})
	handler.router.UpdateConfig(customHeaderRoutingConfig(
		"/v1/chat/completions",
		requestTypeChatCompletions,
		"openai",
		"gpt-route",
	))

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(
		`{"model":"gpt-route","messages":[{"role":"user","content":"hello"}]}`,
	))
	req.Header.Set("X-Customer-Tier", customMatchSecret)
	req.Header.Set("X-LunarGate-Request-Type", requestTypeResponses)
	recorder := httptest.NewRecorder()
	handler.ChatCompletions(recorder, req)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", recorder.Code, recorder.Body.String())
	}
	if got := recorder.Header().Get("X-LunarGate-Route"); got != "custom-header-route" {
		t.Fatalf("route = %q, want custom-header-route", got)
	}
	trace, metric, requestLog := capture.waitForRequestEvents(t)
	assertCollectorEventsExcludeCustomHeader(t, trace, metric, requestLog)
}

func TestEmbeddingsCustomMatchHeaderRoutes(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"object":"list","data":[{"object":"embedding","embedding":[0.1],"index":0}],"model":"embedding-route","usage":{"prompt_tokens":1,"total_tokens":1}}`))
	}))
	defer upstream.Close()

	handler, _ := newObservedOpenAIHandler(t, upstream.URL, config.TargetConfig{
		Provider: "openai",
		Model:    "embedding-route",
		Weight:   1,
	}, nil, config.CacheConfig{Enabled: false})
	handler.router.UpdateConfig(customHeaderRoutingConfig(
		"/v1/embeddings",
		requestTypeEmbeddings,
		"openai",
		"embedding-route",
	))

	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", strings.NewReader(
		`{"model":"embedding-route","input":"hello"}`,
	))
	req.Header.Set("X-Customer-Tier", customMatchSecret)
	req.Header.Set("X-LunarGate-Request-Type", requestTypeResponses)
	recorder := httptest.NewRecorder()
	handler.Embeddings(recorder, req)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", recorder.Code, recorder.Body.String())
	}
	if got := recorder.Header().Get("X-LunarGate-Route"); got != "custom-header-route" {
		t.Fatalf("route = %q, want custom-header-route", got)
	}
}

func customHeaderRoutingConfig(path, requestType, provider, model string) config.RoutingConfig {
	target := config.TargetConfig{Provider: provider, Model: model, Weight: 1}
	return config.RoutingConfig{
		DefaultStrategy: "weighted",
		Routes: []config.RouteConfig{
			{
				Name: "custom-header-route",
				Match: config.MatchConfig{
					Path: path,
					Headers: map[string]string{
						customMatchHeader:          customMatchSecret,
						"x-lunargate-request-type": requestType,
					},
				},
				Targets: []config.TargetConfig{target},
			},
			{
				Name:    "default-route",
				Match:   config.MatchConfig{Path: path},
				Targets: []config.TargetConfig{target},
			},
		},
	}
}

func assertCollectorEventsExcludeCustomHeader(t *testing.T, events ...map[string]interface{}) {
	t.Helper()
	payload, err := json.Marshal(events)
	if err != nil {
		t.Fatalf("marshal collector events: %v", err)
	}
	lower := strings.ToLower(string(payload))
	for _, forbidden := range []string{customMatchHeader, customMatchSecret} {
		if strings.Contains(lower, forbidden) {
			t.Fatalf("collector events exposed custom routing header %q: %s", forbidden, payload)
		}
	}
}
