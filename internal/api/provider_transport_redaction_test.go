package api

import (
	"bytes"
	"context"
	"errors"
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
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

func TestNativeResponseRequestPreservesBaseAndClientQueries(t *testing.T) {
	const baseURL = "https://url-user:url-password@private-provider.example/root/v1?api_key=query-secret#fragment-secret"
	providerConfigs := map[string]config.ProviderConfig{
		"custom": {Type: "openai", BaseURL: baseURL, APIKey: "header-secret"},
	}
	handler := &Handler{
		registry:        providers.NewRegistry(providerConfigs),
		providerClients: newProviderClientRegistry(providerConfigs),
	}
	setProviderTransportForTest(t, handler, "custom", providerURLRoundTripFunc(func(request *http.Request) (*http.Response, error) {
		if got, want := request.URL.Path, "/root/v1/responses/resp_1/input_items"; got != want {
			t.Fatalf("upstream path = %q, want %q", got, want)
		}
		if got, want := request.URL.RawQuery, "api_key=query-secret&limit=20&after=item_1"; got != want {
			t.Fatalf("upstream query = %q, want %q", got, want)
		}
		if request.URL.Fragment != "" {
			t.Fatalf("upstream fragment = %q, want discarded", request.URL.Fragment)
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     make(http.Header),
			Body:       http.NoBody,
			Request:    request,
		}, nil
	}))

	response, err := handler.nativeResponseRequest(
		context.Background(),
		http.MethodGet,
		responseBinding{Provider: "custom"},
		"responses/resp_1/input_items",
		"limit=20&after=item_1",
		nil,
		nil,
	)
	if err != nil {
		t.Fatalf("nativeResponseRequest returned error: %v", err)
	}
	defer response.Body.Close()
}

func TestChatTransportErrorRedactsProviderURLFromLogsAndCollector(t *testing.T) {
	const baseURL = "https://url-user:url-password@private-provider.example/root/v1?api_key=query-secret#fragment-secret"
	providerConfigs := map[string]config.ProviderConfig{
		"custom": {Type: "openai", BaseURL: baseURL, APIKey: "header-secret"},
	}
	registry := providers.NewRegistry(providerConfigs)
	router := routing.NewEngine(config.RoutingConfig{
		DefaultStrategy: "weighted",
		Routes: []config.RouteConfig{{
			Name:    "transport-error",
			Match:   config.MatchConfig{Path: "/v1/chat/completions"},
			Targets: []config.TargetConfig{{Provider: "custom", Model: "gpt-test", Weight: 1}},
		}},
	})
	cache := middleware.NewCache(config.CacheConfig{Enabled: false})
	t.Cleanup(cache.Stop)
	capture := newCollectorCapture(t, true, false)
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
	handler.UpdateProviderConfigs(providerConfigs)
	setProviderTransportForTest(t, handler, "custom", providerURLRoundTripFunc(func(request *http.Request) (*http.Response, error) {
		if got, want := request.URL.Path, "/root/v1/chat/completions"; got != want {
			t.Fatalf("upstream path = %q, want %q", got, want)
		}
		if got, want := request.URL.RawQuery, "api_key=query-secret"; got != want {
			t.Fatalf("upstream query = %q, want %q", got, want)
		}
		return nil, errors.New("connection-refused-category")
	}))

	var logs bytes.Buffer
	previousLogger := log.Logger
	log.Logger = zerolog.New(&logs)
	t.Cleanup(func() { log.Logger = previousLogger })

	recorder := httptest.NewRecorder()
	handler.ChatCompletions(recorder, httptest.NewRequest(
		http.MethodPost,
		"/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-test","messages":[{"role":"user","content":"hello"}]}`),
	))
	if recorder.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502; body=%s", recorder.Code, recorder.Body.String())
	}

	_, _, requestLog := capture.waitForRequestEvents(t)
	errorMessage, _ := requestLog["error_message"].(string)
	for source, value := range map[string]string{
		"gateway log":     logs.String(),
		"collector error": errorMessage,
		"client response": recorder.Body.String(),
	} {
		for _, secret := range []string{"url-user", "url-password", "query-secret", "fragment-secret", "header-secret"} {
			if strings.Contains(value, secret) {
				t.Fatalf("%s leaked %q: %s", source, secret, value)
			}
		}
	}
	if !strings.Contains(errorMessage, "connection-refused-category") {
		t.Fatalf("collector error lost transport category: %q", errorMessage)
	}
	if !strings.Contains(errorMessage, "https://private-provider.example/root/v1/chat/completions") {
		t.Fatalf("collector error lost sanitized endpoint context: %q", errorMessage)
	}
}

type providerURLRoundTripFunc func(*http.Request) (*http.Response, error)

func (f providerURLRoundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return f(request)
}

func setProviderTransportForTest(t *testing.T, handler *Handler, provider string, transport http.RoundTripper) {
	t.Helper()
	handler.providerClients.mu.Lock()
	defer handler.providerClients.mu.Unlock()
	clientConfig, ok := handler.providerClients.clients[provider]
	if !ok || clientConfig.client == nil {
		t.Fatalf("provider client %q is unavailable", provider)
	}
	client := *clientConfig.client
	client.Transport = transport
	clientConfig.client = &client
	handler.providerClients.clients[provider] = clientConfig
}
