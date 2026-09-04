package api

import (
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
)

func TestNativeResponseOperationsDoNotFollowRedirects(t *testing.T) {
	testCases := []struct {
		name         string
		method       string
		path         string
		body         string
		capabilities config.ProviderCapabilities
		responseID   string
	}{
		{
			name:         "retrieve",
			method:       http.MethodGet,
			path:         "/v1/responses/resp_redirect",
			capabilities: config.ProviderCapabilities{ResponsesLifecycle: true},
			responseID:   "resp_redirect",
		},
		{
			name:         "delete",
			method:       http.MethodDelete,
			path:         "/v1/responses/resp_redirect",
			capabilities: config.ProviderCapabilities{ResponsesLifecycle: true},
			responseID:   "resp_redirect",
		},
		{
			name:         "cancel",
			method:       http.MethodPost,
			path:         "/v1/responses/resp_redirect/cancel",
			body:         `{}`,
			capabilities: config.ProviderCapabilities{ResponsesLifecycle: true, ResponseCancellation: true},
			responseID:   "resp_redirect",
		},
		{
			name:         "input items",
			method:       http.MethodGet,
			path:         "/v1/responses/resp_redirect/input_items",
			capabilities: config.ProviderCapabilities{ResponsesLifecycle: true},
			responseID:   "resp_redirect",
		},
		{
			name:         "compact",
			method:       http.MethodPost,
			path:         "/v1/responses/compact",
			body:         `{"model":"native/gpt-native","input":"hello"}`,
			capabilities: config.ProviderCapabilities{ResponseCompaction: true},
		},
		{
			name:         "input tokens",
			method:       http.MethodPost,
			path:         "/v1/responses/input_tokens",
			body:         `{"model":"native/gpt-native","input":"hello"}`,
			capabilities: config.ProviderCapabilities{ResponseInputTokens: true},
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			var redirectCalls atomic.Int32
			redirectTarget := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				redirectCalls.Add(1)
				w.WriteHeader(http.StatusTeapot)
			}))
			defer redirectTarget.Close()

			const redirectBody = `{"redirect":"not followed","future_field":true}`
			var upstreamCalls atomic.Int32
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				upstreamCalls.Add(1)
				w.Header().Set("Content-Type", "application/json")
				w.Header().Set("Location", redirectTarget.URL+"/next")
				w.WriteHeader(http.StatusFound)
				_, _ = io.WriteString(w, redirectBody)
			}))
			defer upstream.Close()

			router, handler, cache := newNativeLifecycleRouter(t, upstream.URL+"/v1", map[string]config.ProviderCapabilities{
				"native": testCase.capabilities,
			})
			defer cache.Stop()
			if testCase.responseID != "" {
				handler.responseBindings.put(testCase.responseID, mustResponseBinding(t, handler, "native"))
			}

			request := httptest.NewRequest(testCase.method, testCase.path, strings.NewReader(testCase.body))
			response := httptest.NewRecorder()
			router.ServeHTTP(response, request)

			if response.Code != http.StatusFound {
				t.Fatalf("status = %d, want 302; body=%s", response.Code, response.Body.String())
			}
			if got := response.Body.String(); got != redirectBody {
				t.Fatalf("redirect body changed: %q", got)
			}
			if got := response.Header().Get("Location"); got != redirectTarget.URL+"/next" {
				t.Fatalf("Location = %q", got)
			}
			if got := upstreamCalls.Load(); got != 1 {
				t.Fatalf("upstream calls = %d, want exactly one", got)
			}
			if got := redirectCalls.Load(); got != 0 {
				t.Fatalf("redirect target calls = %d, want zero", got)
			}
		})
	}
}

func TestNativeResponseTransportErrorsAreSanitized(t *testing.T) {
	testCases := []struct {
		name         string
		method       string
		path         string
		body         string
		capabilities config.ProviderCapabilities
		responseID   string
	}{
		{
			name:         "retrieve",
			method:       http.MethodGet,
			path:         "/v1/responses/resp_transport",
			capabilities: config.ProviderCapabilities{ResponsesLifecycle: true},
			responseID:   "resp_transport",
		},
		{
			name:         "delete",
			method:       http.MethodDelete,
			path:         "/v1/responses/resp_transport",
			capabilities: config.ProviderCapabilities{ResponsesLifecycle: true},
			responseID:   "resp_transport",
		},
		{
			name:         "cancel",
			method:       http.MethodPost,
			path:         "/v1/responses/resp_transport/cancel",
			body:         `{}`,
			capabilities: config.ProviderCapabilities{ResponsesLifecycle: true, ResponseCancellation: true},
			responseID:   "resp_transport",
		},
		{
			name:         "input items",
			method:       http.MethodGet,
			path:         "/v1/responses/resp_transport/input_items",
			capabilities: config.ProviderCapabilities{ResponsesLifecycle: true},
			responseID:   "resp_transport",
		},
		{
			name:         "compact",
			method:       http.MethodPost,
			path:         "/v1/responses/compact",
			body:         `{"model":"native/gpt-native","input":"hello"}`,
			capabilities: config.ProviderCapabilities{ResponseCompaction: true},
		},
		{
			name:         "input tokens",
			method:       http.MethodPost,
			path:         "/v1/responses/input_tokens",
			body:         `{"model":"native/gpt-native","input":"hello"}`,
			capabilities: config.ProviderCapabilities{ResponseInputTokens: true},
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			providerConfig := config.ProviderConfig{
				Type:         "openai",
				APIKey:       "provider-secret",
				BaseURL:      "https://private-provider.example/top-secret-upstream-path/v1",
				DefaultModel: "gpt-native",
				Capabilities: testCase.capabilities,
			}
			router, handler, cache := newNativeLifecycleRouterFromConfigs(t, map[string]config.ProviderConfig{
				"native": providerConfig,
			})
			defer cache.Stop()
			if testCase.responseID != "" {
				handler.responseBindings.put(testCase.responseID, mustResponseBinding(t, handler, "native"))
			}

			var calls atomic.Int32
			setNativeResponseTestTransport(t, handler, "native", roundTripFunc(func(request *http.Request) (*http.Response, error) {
				calls.Add(1)
				return nil, fmt.Errorf("dial failed for %s: secret transport detail", request.URL.String())
			}))

			request := httptest.NewRequest(testCase.method, testCase.path, strings.NewReader(testCase.body))
			response := httptest.NewRecorder()
			router.ServeHTTP(response, request)

			if response.Code != http.StatusBadGateway {
				t.Fatalf("status = %d, want 502; body=%s", response.Code, response.Body.String())
			}
			if got := calls.Load(); got != 1 {
				t.Fatalf("transport calls = %d, want exactly one", got)
			}
			body := response.Body.String()
			for _, secret := range []string{"private-provider.example", "top-secret-upstream-path", "secret transport detail"} {
				if strings.Contains(body, secret) {
					t.Fatalf("transport detail %q leaked to client: %s", secret, body)
				}
			}
			decoded := decodeNativeLifecycleBody(t, body)
			errorEnvelope, ok := decoded["error"].(map[string]interface{})
			if !ok {
				t.Fatalf("error envelope = %#v", decoded["error"])
			}
			if errorEnvelope["type"] != "provider_error" {
				t.Fatalf("error type = %#v", errorEnvelope["type"])
			}
			if errorEnvelope["message"] != "upstream response provider request failed" {
				t.Fatalf("error message = %#v", errorEnvelope["message"])
			}
		})
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return f(request)
}

func setNativeResponseTestTransport(t *testing.T, handler *Handler, provider string, transport http.RoundTripper) {
	t.Helper()
	if handler == nil || handler.providerClients == nil {
		t.Fatal("provider client registry is unavailable")
	}
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
