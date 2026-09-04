package api

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/pkg/models"
)

const nativeLifecycleTransportTestTimeout = 40 * time.Millisecond

type nativeLifecycleTransportEndpoint struct {
	name           string
	method         string
	path           string
	body           string
	capabilities   config.ProviderCapabilities
	genericMessage string
}

func nativeLifecycleTransportEndpoints() []nativeLifecycleTransportEndpoint {
	return []nativeLifecycleTransportEndpoint{
		{
			name:           "responses",
			method:         http.MethodGet,
			path:           "/v1/responses/resp_transport",
			capabilities:   config.ProviderCapabilities{ResponsesLifecycle: true},
			genericMessage: "upstream response provider request failed",
		},
		{
			name:           "conversations",
			method:         http.MethodPost,
			path:           "/v1/conversations",
			body:           `{}`,
			capabilities:   config.ProviderCapabilities{Conversations: true},
			genericMessage: "upstream response provider request failed",
		},
		{
			name:           "chat_completions",
			method:         http.MethodGet,
			path:           "/v1/chat/completions/chatcmpl_transport",
			capabilities:   config.ProviderCapabilities{ChatCompletionsLifecycle: true},
			genericMessage: "upstream Chat Completions provider request failed",
		},
	}
}

func TestNativeLifecycleTransportPreservesParentCancellation(t *testing.T) {
	contextCases := []struct {
		name string
		make func() (context.Context, context.CancelFunc)
	}{
		{
			name: "canceled",
			make: func() (context.Context, context.CancelFunc) {
				ctx, cancel := context.WithCancel(context.Background())
				cancel()
				return ctx, cancel
			},
		},
		{
			name: "deadline_exceeded",
			make: func() (context.Context, context.CancelFunc) {
				ctx, cancel := context.WithTimeout(context.Background(), 0)
				<-ctx.Done()
				return ctx, cancel
			},
		},
	}

	for _, endpoint := range nativeLifecycleTransportEndpoints() {
		for _, contextCase := range contextCases {
			t.Run(endpoint.name+"/"+contextCase.name, func(t *testing.T) {
				router, handler := newNativeLifecycleTransportEndpoint(t, endpoint, 0, "")
				setProviderTransportForTest(t, handler, "native", providerURLRoundTripFunc(func(request *http.Request) (*http.Response, error) {
					<-request.Context().Done()
					return nil, request.Context().Err()
				}))

				ctx, cancel := contextCase.make()
				defer cancel()
				response := performNativeLifecycleTransportRequest(t, router, endpoint, ctx)
				assertNativeLifecycleTransportError(t, response, 499, "client_cancelled", "client disconnected")
			})
		}
	}
}

func TestNativeLifecycleTransportClassifiesProviderTimeout(t *testing.T) {
	timeoutCases := []struct {
		mode    string
		message string
	}{
		{mode: upstreamTimeoutModeTTFT, message: "provider timed out waiting for first byte"},
		{mode: upstreamTimeoutModeTotal, message: "provider timed out before full response completed"},
	}

	for _, endpoint := range nativeLifecycleTransportEndpoints() {
		for _, timeoutCase := range timeoutCases {
			t.Run(endpoint.name+"/"+timeoutCase.mode, func(t *testing.T) {
				router, handler := newNativeLifecycleTransportEndpoint(
					t,
					endpoint,
					nativeLifecycleTransportTestTimeout,
					timeoutCase.mode,
				)
				var calls atomic.Int32
				setProviderTransportForTest(t, handler, "native", blockedNativeLifecycleTransport(&calls))

				response := performNativeLifecycleTransportRequest(t, router, endpoint, context.Background())
				assertNativeLifecycleTransportError(t, response, http.StatusBadGateway, "upstream_timeout", timeoutCase.message)
				if got := calls.Load(); got != 1 {
					t.Fatalf("upstream calls = %d, want 1", got)
				}
			})
		}
	}
}

func TestNativeLifecycleTransportKeepsGenericProviderErrors(t *testing.T) {
	for _, endpoint := range nativeLifecycleTransportEndpoints() {
		t.Run(endpoint.name, func(t *testing.T) {
			router, handler := newNativeLifecycleTransportEndpoint(t, endpoint, time.Second, upstreamTimeoutModeTTFT)
			setProviderTransportForTest(t, handler, "native", providerURLRoundTripFunc(func(*http.Request) (*http.Response, error) {
				return nil, errors.New("transport unavailable")
			}))

			response := performNativeLifecycleTransportRequest(t, router, endpoint, context.Background())
			assertNativeLifecycleTransportError(t, response, http.StatusBadGateway, "provider_error", endpoint.genericMessage)
		})
	}
}

func TestNativeResponsesCreateClassifiesDialAndBodyTimeouts(t *testing.T) {
	timeoutCases := []struct {
		mode    string
		message string
	}{
		{mode: upstreamTimeoutModeTTFT, message: "provider timed out waiting for first byte"},
		{mode: upstreamTimeoutModeTotal, message: "provider timed out before full response completed"},
	}
	for _, timeoutCase := range timeoutCases {
		for _, phase := range []string{"dial", "body"} {
			t.Run(timeoutCase.mode+"/"+phase, func(t *testing.T) {
				providerConfig := config.ProviderConfig{
					Type:         "openai",
					APIKey:       "provider-secret",
					BaseURL:      "http://native.invalid/v1",
					DefaultModel: "gpt-native",
					Timeout:      nativeLifecycleTransportTestTimeout,
					TimeoutMode:  timeoutCase.mode,
					Capabilities: config.ProviderCapabilities{ResponsesLifecycle: true},
				}
				router, handler, cache := newNativeLifecycleRouterFromConfigs(t, map[string]config.ProviderConfig{"native": providerConfig})
				t.Cleanup(cache.Stop)

				var calls atomic.Int32
				var body *closeBlockedReadBody
				if phase == "dial" {
					setProviderTransportForTest(t, handler, "native", blockedNativeLifecycleTransport(&calls))
				} else {
					body = newCloseBlockedReadBody()
					setProviderTransportForTest(t, handler, "native", providerURLRoundTripFunc(func(request *http.Request) (*http.Response, error) {
						calls.Add(1)
						return &http.Response{
							StatusCode: http.StatusOK,
							Header:     http.Header{"Content-Type": []string{"application/json"}},
							Body:       body,
							Request:    request,
						}, nil
					}))
				}

				response := performLifecycleRequest(
					t,
					router,
					http.MethodPost,
					"/v1/responses",
					[]byte(`{"model":"native/gpt-native","input":"hello"}`),
				)
				assertNativeLifecycleTransportError(t, response, http.StatusBadGateway, "upstream_timeout", timeoutCase.message)
				if got := calls.Load(); got != 1 {
					t.Fatalf("upstream calls = %d, want 1", got)
				}
				if body != nil && body.closeCalls.Load() != 1 {
					t.Fatalf("body close calls = %d, want 1", body.closeCalls.Load())
				}
			})
		}
	}
}

func newNativeLifecycleTransportEndpoint(
	t *testing.T,
	endpoint nativeLifecycleTransportEndpoint,
	timeout time.Duration,
	timeoutMode string,
) (http.Handler, *Handler) {
	t.Helper()
	providerConfig := config.ProviderConfig{
		Type:         "openai",
		APIKey:       "provider-secret",
		BaseURL:      "http://native.invalid/v1",
		DefaultModel: "gpt-native",
		Timeout:      timeout,
		TimeoutMode:  timeoutMode,
		Capabilities: endpoint.capabilities,
	}
	providerConfigs := map[string]config.ProviderConfig{"native": providerConfig}
	if endpoint.name == "chat_completions" {
		router, handler, cache := newStoredChatLifecycleRouterFromConfigs(t, providerConfigs)
		t.Cleanup(cache.Stop)
		return router, handler
	}
	router, handler, cache := newNativeLifecycleRouterFromConfigs(t, providerConfigs)
	t.Cleanup(cache.Stop)
	return router, handler
}

func performNativeLifecycleTransportRequest(
	t *testing.T,
	router http.Handler,
	endpoint nativeLifecycleTransportEndpoint,
	ctx context.Context,
) *httptest.ResponseRecorder {
	t.Helper()
	request := httptest.NewRequest(endpoint.method, endpoint.path, strings.NewReader(endpoint.body)).WithContext(ctx)
	request.Header.Set("X-LunarGate-Provider", "native")
	if endpoint.body != "" {
		request.Header.Set("Content-Type", "application/json")
	}
	response := httptest.NewRecorder()
	router.ServeHTTP(response, request)
	return response
}

func blockedNativeLifecycleTransport(calls *atomic.Int32) http.RoundTripper {
	return providerURLRoundTripFunc(func(request *http.Request) (*http.Response, error) {
		calls.Add(1)
		<-request.Context().Done()
		return nil, request.Context().Err()
	})
}

func assertNativeLifecycleTransportError(
	t *testing.T,
	response *httptest.ResponseRecorder,
	wantStatus int,
	wantType string,
	wantMessage string,
) {
	t.Helper()
	if response.Code != wantStatus {
		t.Fatalf("status = %d, want %d; body=%s", response.Code, wantStatus, response.Body.String())
	}
	var payload models.ErrorResponse
	if err := json.Unmarshal(response.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode error response: %v; body=%s", err, response.Body.String())
	}
	if payload.Error.Type != wantType || payload.Error.Message != wantMessage {
		t.Fatalf("error = %#v, want type=%q message=%q", payload.Error, wantType, wantMessage)
	}
}
