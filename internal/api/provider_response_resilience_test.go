package api

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/resilience"
	"github.com/lunargate-ai/gateway/pkg/models"
)

const oversizedProviderResponseBytes = (16 << 20) + 1

type nonStreamingProtocolAPI struct {
	name          string
	path          string
	requestBody   string
	primaryModel  string
	fallbackModel string
	validPrimary  string
	validFallback string
	serve         func(*Handler, http.ResponseWriter, *http.Request)
}

func nonStreamingProtocolAPIs() []nonStreamingProtocolAPI {
	return []nonStreamingProtocolAPI{
		{
			name:          "chat_completions",
			path:          "/v1/chat/completions",
			requestBody:   `{"messages":[{"role":"user","content":"hello"}]}`,
			primaryModel:  "gpt-primary",
			fallbackModel: "gpt-fallback",
			validPrimary:  `{"id":"chatcmpl-primary","object":"chat.completion","created":1,"model":"gpt-primary","choices":[{"index":0,"message":{"role":"assistant","content":"primary"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2},"future":{"source":"primary"}}`,
			validFallback: `{"id":"chatcmpl-fallback","object":"chat.completion","created":2,"model":"gpt-fallback","choices":[{"index":0,"message":{"role":"assistant","content":"fallback"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3},"future":{"source":"fallback"}}`,
			serve:         (*Handler).ChatCompletions,
		},
		{
			name:          "embeddings",
			path:          "/v1/embeddings",
			requestBody:   `{"model":"embedding-primary","input":"hello"}`,
			primaryModel:  "embedding-primary",
			fallbackModel: "embedding-primary",
			validPrimary:  `{"object":"list","data":[{"object":"embedding","embedding":[0.1,0.2],"index":0}],"model":"embedding-primary","usage":{"prompt_tokens":1,"total_tokens":1},"future":{"source":"primary"}}`,
			validFallback: `{"object":"list","data":[{"object":"embedding","embedding":[0.3,0.4],"index":0}],"model":"embedding-fallback","usage":{"prompt_tokens":2,"total_tokens":2},"future":{"source":"fallback"}}`,
			serve:         (*Handler).Embeddings,
		},
	}
}

type protocolRoundTripFunc func(*http.Request) (*http.Response, error)

func (f protocolRoundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return f(request)
}

type trackedProtocolBody struct {
	body   io.ReadCloser
	closed *atomic.Int32
	done   atomic.Bool
}

func (b *trackedProtocolBody) Read(p []byte) (int, error) {
	return b.body.Read(p)
}

func (b *trackedProtocolBody) Close() error {
	if b.done.CompareAndSwap(false, true) {
		b.closed.Add(1)
	}
	return b.body.Close()
}

type unexpectedEOFBody struct {
	reader *strings.Reader
}

func newUnexpectedEOFBody(body string) io.ReadCloser {
	return &unexpectedEOFBody{reader: strings.NewReader(body)}
}

func (b *unexpectedEOFBody) Read(p []byte) (int, error) {
	n, err := b.reader.Read(p)
	if b.reader.Len() == 0 {
		return n, io.ErrUnexpectedEOF
	}
	return n, err
}

func (*unexpectedEOFBody) Close() error { return nil }

type repeatedProtocolBody struct {
	remaining int64
}

func (b *repeatedProtocolBody) Read(p []byte) (int, error) {
	if b.remaining <= 0 {
		return 0, io.EOF
	}
	n := len(p)
	if int64(n) > b.remaining {
		n = int(b.remaining)
	}
	for i := 0; i < n; i++ {
		p[i] = 'x'
	}
	b.remaining -= int64(n)
	return n, nil
}

func (*repeatedProtocolBody) Close() error { return nil }

func protocolResponseTransport(
	status int,
	bodyFactory func() io.ReadCloser,
	calls *atomic.Int32,
	closed *atomic.Int32,
	redirectCalls *atomic.Int32,
) http.RoundTripper {
	return protocolRoundTripFunc(func(request *http.Request) (*http.Response, error) {
		if request.URL.Path == "/must-not-follow" {
			redirectCalls.Add(1)
			return &http.Response{
				StatusCode: http.StatusTeapot,
				Header:     make(http.Header),
				Body:       http.NoBody,
			}, nil
		}
		calls.Add(1)
		var body io.ReadCloser = http.NoBody
		if bodyFactory != nil {
			body = bodyFactory()
		}
		headers := make(http.Header)
		headers.Set("Content-Type", "application/json")
		headers.Set("X-Protocol-Response", "preserved")
		if status >= http.StatusMultipleChoices && status < http.StatusBadRequest {
			headers.Set("Location", "/must-not-follow")
		}
		return &http.Response{
			StatusCode:    status,
			Header:        headers,
			Body:          &trackedProtocolBody{body: body, closed: closed},
			ContentLength: -1,
		}, nil
	})
}

func newNonStreamingProtocolHandler(
	t *testing.T,
	api nonStreamingProtocolAPI,
	withFallback bool,
	primaryTransport http.RoundTripper,
	fallbackTransport http.RoundTripper,
) (*Handler, *resilience.CircuitBreakerManager) {
	t.Helper()
	providerConfigs := map[string]config.ProviderConfig{
		"primary": {
			Type:    "openai",
			APIKey:  "test-primary",
			BaseURL: "https://primary.example/v1",
		},
	}
	var fallbackTargets []config.TargetConfig
	if withFallback {
		providerConfigs["fallback"] = config.ProviderConfig{
			Type:    "openai",
			APIKey:  "test-fallback",
			BaseURL: "https://fallback.example/v1",
		}
		fallbackTargets = []config.TargetConfig{{
			Provider: "fallback",
			Model:    api.fallbackModel,
			Weight:   1,
		}}
	}
	handler, cbm, _ := newResilienceClassificationHandler(
		t,
		providerConfigs,
		config.RouteConfig{
			Name:     api.name,
			Match:    config.MatchConfig{Path: api.path},
			Targets:  []config.TargetConfig{{Provider: "primary", Model: api.primaryModel, Weight: 1}},
			Fallback: fallbackTargets,
		},
		config.RetryConfig{
			Enabled:         true,
			MaxAttempts:     2,
			InitialDelay:    0,
			MaxDelay:        0,
			Multiplier:      1,
			RetryableErrors: []int{http.StatusTooManyRequests, http.StatusInternalServerError},
		},
	)
	handler.UpdateProviderConfigs(providerConfigs)
	setProviderTransportForTest(t, handler, "primary", primaryTransport)
	if withFallback {
		setProviderTransportForTest(t, handler, "fallback", fallbackTransport)
	}
	return handler, cbm
}

func performNonStreamingProtocolRequest(
	t *testing.T,
	api nonStreamingProtocolAPI,
	handler *Handler,
) *httptest.ResponseRecorder {
	t.Helper()
	request := httptest.NewRequest(http.MethodPost, api.path, strings.NewReader(api.requestBody))
	request.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()
	api.serve(handler, recorder, request)
	return recorder
}

func assertProtocolErrorType(t *testing.T, recorder *httptest.ResponseRecorder, want string) {
	t.Helper()
	if recorder.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502; body=%s", recorder.Code, recorder.Body.String())
	}
	var response models.ErrorResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode error response: %v; body=%s", err, recorder.Body.String())
	}
	if response.Error.Type != want {
		t.Fatalf("error type = %q, want %q; body=%s", response.Error.Type, want, recorder.Body.String())
	}
}

func TestNonStreamingProviderBodyFailuresUseRetryFallbackAndCircuitBreaker(t *testing.T) {
	bodyFailures := []struct {
		name          string
		bodyFactory   func() io.ReadCloser
		wantErrorType string
	}{
		{
			name:          "malformed",
			bodyFactory:   func() io.ReadCloser { return io.NopCloser(strings.NewReader(`{"id":`)) },
			wantErrorType: "provider_error",
		},
		{
			name:          "truncated",
			bodyFactory:   func() io.ReadCloser { return newUnexpectedEOFBody(`{"id":"partial"`) },
			wantErrorType: "provider_error",
		},
		{
			name: "oversize",
			bodyFactory: func() io.ReadCloser {
				return &repeatedProtocolBody{remaining: oversizedProviderResponseBytes}
			},
			wantErrorType: "upstream_response_too_large",
		},
	}

	for _, api := range nonStreamingProtocolAPIs() {
		for _, failure := range bodyFailures {
			t.Run(api.name+"/"+failure.name+"/fallback", func(t *testing.T) {
				var primaryCalls, primaryClosed, fallbackCalls, fallbackClosed, redirectCalls atomic.Int32
				primaryTransport := protocolResponseTransport(
					http.StatusOK,
					failure.bodyFactory,
					&primaryCalls,
					&primaryClosed,
					&redirectCalls,
				)
				fallbackTransport := protocolResponseTransport(
					http.StatusOK,
					func() io.ReadCloser { return io.NopCloser(strings.NewReader(api.validFallback)) },
					&fallbackCalls,
					&fallbackClosed,
					&redirectCalls,
				)
				handler, cbm := newNonStreamingProtocolHandler(t, api, true, primaryTransport, fallbackTransport)
				recorder := performNonStreamingProtocolRequest(t, api, handler)

				if recorder.Code != http.StatusOK {
					t.Fatalf("status = %d, want 200; body=%s", recorder.Code, recorder.Body.String())
				}
				if !bytes.Equal(recorder.Body.Bytes(), []byte(api.validFallback)) {
					t.Fatalf("fallback raw response changed:\n got: %s\nwant: %s", recorder.Body.Bytes(), api.validFallback)
				}
				if recorder.Header().Get("X-LunarGate-Provider") != "fallback" {
					t.Fatalf("provider header = %q, want fallback", recorder.Header().Get("X-LunarGate-Provider"))
				}
				if primaryCalls.Load() != 2 || primaryClosed.Load() != 2 {
					t.Fatalf("primary calls/closed = %d/%d, want 2/2", primaryCalls.Load(), primaryClosed.Load())
				}
				if fallbackCalls.Load() != 1 || fallbackClosed.Load() != 1 {
					t.Fatalf("fallback calls/closed = %d/%d, want 1/1", fallbackCalls.Load(), fallbackClosed.Load())
				}
				if redirectCalls.Load() != 0 {
					t.Fatalf("redirect calls = %d, want 0", redirectCalls.Load())
				}
				if failures := cbm.Get("primary").Counts().TotalFailures; failures != 1 {
					t.Fatalf("primary breaker failures = %d, want 1", failures)
				}
			})

			t.Run(api.name+"/"+failure.name+"/no_fallback", func(t *testing.T) {
				var primaryCalls, primaryClosed, redirectCalls atomic.Int32
				primaryTransport := protocolResponseTransport(
					http.StatusOK,
					failure.bodyFactory,
					&primaryCalls,
					&primaryClosed,
					&redirectCalls,
				)
				handler, cbm := newNonStreamingProtocolHandler(t, api, false, primaryTransport, nil)
				recorder := performNonStreamingProtocolRequest(t, api, handler)

				assertProtocolErrorType(t, recorder, failure.wantErrorType)
				if primaryCalls.Load() != 2 || primaryClosed.Load() != 2 {
					t.Fatalf("primary calls/closed = %d/%d, want 2/2", primaryCalls.Load(), primaryClosed.Load())
				}
				if failures := cbm.Get("primary").Counts().TotalFailures; failures != 1 {
					t.Fatalf("primary breaker failures = %d, want 1", failures)
				}
			})
		}
	}
}

func TestNonStreamingProviderInvalidStatusesUseRetryFallbackWithoutRedirect(t *testing.T) {
	statuses := []int{http.StatusCreated, http.StatusNoContent, http.StatusFound}
	for _, api := range nonStreamingProtocolAPIs() {
		for _, status := range statuses {
			statusName := http.StatusText(status)
			t.Run(api.name+"/"+statusName+"/fallback", func(t *testing.T) {
				var primaryCalls, primaryClosed, fallbackCalls, fallbackClosed, redirectCalls atomic.Int32
				primaryTransport := protocolResponseTransport(
					status,
					func() io.ReadCloser { return io.NopCloser(strings.NewReader(api.validPrimary)) },
					&primaryCalls,
					&primaryClosed,
					&redirectCalls,
				)
				fallbackTransport := protocolResponseTransport(
					http.StatusOK,
					func() io.ReadCloser { return io.NopCloser(strings.NewReader(api.validFallback)) },
					&fallbackCalls,
					&fallbackClosed,
					&redirectCalls,
				)
				handler, cbm := newNonStreamingProtocolHandler(t, api, true, primaryTransport, fallbackTransport)
				recorder := performNonStreamingProtocolRequest(t, api, handler)

				if recorder.Code != http.StatusOK || !bytes.Equal(recorder.Body.Bytes(), []byte(api.validFallback)) {
					t.Fatalf("fallback response = status %d body %s", recorder.Code, recorder.Body.String())
				}
				if primaryCalls.Load() != 2 || primaryClosed.Load() != 2 {
					t.Fatalf("primary calls/closed = %d/%d, want 2/2", primaryCalls.Load(), primaryClosed.Load())
				}
				if fallbackCalls.Load() != 1 || fallbackClosed.Load() != 1 {
					t.Fatalf("fallback calls/closed = %d/%d, want 1/1", fallbackCalls.Load(), fallbackClosed.Load())
				}
				if redirectCalls.Load() != 0 {
					t.Fatalf("redirect calls = %d, want 0", redirectCalls.Load())
				}
				if failures := cbm.Get("primary").Counts().TotalFailures; failures != 1 {
					t.Fatalf("primary breaker failures = %d, want 1", failures)
				}
			})

			t.Run(api.name+"/"+statusName+"/no_fallback", func(t *testing.T) {
				var primaryCalls, primaryClosed, redirectCalls atomic.Int32
				primaryTransport := protocolResponseTransport(
					status,
					func() io.ReadCloser { return io.NopCloser(strings.NewReader(api.validPrimary)) },
					&primaryCalls,
					&primaryClosed,
					&redirectCalls,
				)
				handler, cbm := newNonStreamingProtocolHandler(t, api, false, primaryTransport, nil)
				recorder := performNonStreamingProtocolRequest(t, api, handler)

				assertProtocolErrorType(t, recorder, "invalid_response_status")
				if primaryCalls.Load() != 2 || primaryClosed.Load() != 2 {
					t.Fatalf("primary calls/closed = %d/%d, want 2/2", primaryCalls.Load(), primaryClosed.Load())
				}
				if redirectCalls.Load() != 0 {
					t.Fatalf("redirect calls = %d, want 0", redirectCalls.Load())
				}
				if failures := cbm.Get("primary").Counts().TotalFailures; failures != 1 {
					t.Fatalf("primary breaker failures = %d, want 1", failures)
				}
			})
		}
	}
}

func TestNonStreamingProviderSuccessPreservesRawResponseAndContext(t *testing.T) {
	for _, api := range nonStreamingProtocolAPIs() {
		t.Run(api.name, func(t *testing.T) {
			var primaryCalls, primaryClosed, redirectCalls atomic.Int32
			primaryTransport := protocolResponseTransport(
				http.StatusOK,
				func() io.ReadCloser { return io.NopCloser(strings.NewReader(api.validPrimary)) },
				&primaryCalls,
				&primaryClosed,
				&redirectCalls,
			)
			handler, cbm := newNonStreamingProtocolHandler(t, api, false, primaryTransport, nil)
			recorder := performNonStreamingProtocolRequest(t, api, handler)

			if recorder.Code != http.StatusOK {
				t.Fatalf("status = %d, want 200; body=%s", recorder.Code, recorder.Body.String())
			}
			if !bytes.Equal(recorder.Body.Bytes(), []byte(api.validPrimary)) {
				t.Fatalf("raw response changed:\n got: %s\nwant: %s", recorder.Body.Bytes(), api.validPrimary)
			}
			if recorder.Header().Get("X-LunarGate-Provider") != "primary" {
				t.Fatalf("provider header = %q, want primary", recorder.Header().Get("X-LunarGate-Provider"))
			}
			if api.name == "embeddings" && recorder.Header().Get("X-Protocol-Response") != "preserved" {
				t.Fatalf("embeddings upstream header was not preserved")
			}
			if primaryCalls.Load() != 1 || primaryClosed.Load() != 1 {
				t.Fatalf("primary calls/closed = %d/%d, want 1/1", primaryCalls.Load(), primaryClosed.Load())
			}
			counts := cbm.Get("primary").Counts()
			if counts.TotalSuccesses != 1 || counts.TotalFailures != 0 {
				t.Fatalf("primary breaker successes/failures = %d/%d, want 1/0", counts.TotalSuccesses, counts.TotalFailures)
			}
		})
	}
}
