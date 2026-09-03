package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/streaming"
)

const (
	nativeSSEAPITimeout    = 40 * time.Millisecond
	nativeSSEAPIResponseID = "resp_sse_hardening"
)

type nativeSSEAPIEndpoint struct {
	name  string
	serve func(http.Handler, context.Context, http.ResponseWriter)
}

func nativeSSEAPIEndpoints() []nativeSSEAPIEndpoint {
	return []nativeSSEAPIEndpoint{
		{
			name: "create",
			serve: func(router http.Handler, ctx context.Context, writer http.ResponseWriter) {
				request := httptest.NewRequest(
					http.MethodPost,
					"/v1/responses",
					strings.NewReader(`{"model":"native/gpt-native","input":"hello","stream":true}`),
				).WithContext(ctx)
				request.Header.Set("Content-Type", "application/json")
				router.ServeHTTP(writer, request)
			},
		},
		{
			name: "lifecycle",
			serve: func(router http.Handler, ctx context.Context, writer http.ResponseWriter) {
				request := httptest.NewRequest(http.MethodGet, "/v1/responses/"+nativeSSEAPIResponseID, nil).WithContext(ctx)
				request.Header.Set("X-LunarGate-Provider", "native")
				router.ServeHTTP(writer, request)
			},
		},
	}
}

func TestNativeResponsesSSEPreflightFailuresReturnJSON502(t *testing.T) {
	oversized := "data: {\"padding\":\"" + strings.Repeat("x", streaming.MaxStreamRecordBytes) + "\"}\n\n"
	tests := []struct {
		name string
		body string
	}{
		{name: "empty"},
		{name: "empty frames", body: ": keepalive\n\n\n\n"},
		{name: "malformed", body: ": keepalive\n\ndata: {\n\n"},
		{name: "non object", body: "data: []\n\n"},
		{name: "missing sequence", body: "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_sequence\",\"status\":\"in_progress\"}}\n\n"},
		{name: "fractional sequence", body: "event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0.5,\"response\":{\"id\":\"resp_sequence\",\"status\":\"in_progress\"}}\n\n"},
		{name: "negative sequence", body: "event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":-1,\"response\":{\"id\":\"resp_sequence\",\"status\":\"in_progress\"}}\n\n"},
		{name: "created missing id", body: "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"status\":\"in_progress\"}}\n\n"},
		{name: "created invalid id", body: "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":7,\"status\":\"in_progress\"}}\n\n"},
		{name: "created padded id", body: "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\" resp_padded \",\"status\":\"in_progress\"}}\n\n"},
		{name: "queued missing id", body: "event: response.queued\ndata: {\"type\":\"response.queued\",\"response\":{\"status\":\"queued\"}}\n\n"},
		{name: "in progress missing id", body: "event: response.in_progress\ndata: {\"type\":\"response.in_progress\",\"response\":{\"status\":\"in_progress\"}}\n\n"},
		{name: "terminal missing response", body: "event: response.completed\ndata: {\"type\":\"response.completed\"}\n\n"},
		{name: "terminal missing id", body: "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n"},
		{name: "conflicting event ids", body: "event: response.completed\ndata: {\"type\":\"response.completed\",\"response_id\":\"resp_a\",\"response\":{\"id\":\"resp_b\",\"status\":\"completed\"}}\n\n"},
		{name: "terminal event type mismatch", body: "event: response.completed\ndata: {\"type\":\"response.failed\",\"response\":{\"id\":\"" + nativeSSEAPIResponseID + "\",\"status\":\"failed\"}}\n\n"},
		{name: "terminal status mismatch", body: "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"" + nativeSSEAPIResponseID + "\",\"status\":\"failed\"}}\n\n"},
		{name: "oversize", body: oversized},
		{name: "unterminated", body: "data: {\"type\":\"response.created\"}"},
	}

	for _, endpoint := range nativeSSEAPIEndpoints() {
		for _, test := range tests {
			t.Run(endpoint.name+"/"+test.name, func(t *testing.T) {
				router, handler := newNativeSSEAPIRouter(t, 0, "")
				setProviderTransportForTest(t, handler, "native", nativeSSEResponseTransport(
					http.StatusOK,
					func(*http.Request) io.ReadCloser { return io.NopCloser(strings.NewReader(test.body)) },
				))
				recorder := httptest.NewRecorder()

				endpoint.serve(router, context.Background(), recorder)

				assertNativeSSEJSON502(t, recorder)
			})
		}
	}
}

func TestNativeResponsesSSERejectsHTTP202BeforeStreaming(t *testing.T) {
	valid := "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_accepted\",\"status\":\"completed\"}}\n\n"
	for _, endpoint := range nativeSSEAPIEndpoints() {
		t.Run(endpoint.name, func(t *testing.T) {
			router, handler := newNativeSSEAPIRouter(t, 0, "")
			setProviderTransportForTest(t, handler, "native", nativeSSEResponseTransport(
				http.StatusAccepted,
				func(*http.Request) io.ReadCloser { return io.NopCloser(strings.NewReader(valid)) },
			))
			recorder := httptest.NewRecorder()

			endpoint.serve(router, context.Background(), recorder)

			assertNativeSSEJSON502(t, recorder)
		})
	}
}

func TestNativeResponsesLifecycleSSERejectsDifferentResourceID(t *testing.T) {
	for _, test := range []struct {
		name string
		body string
	}{
		{name: "terminal", body: "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_other\",\"status\":\"completed\"}}\n\n"},
		{name: "intermediate", body: "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"response_id\":\"resp_other\",\"delta\":\"must-not-leak\"}\n\n"},
	} {
		t.Run(test.name, func(t *testing.T) {
			router, handler := newNativeSSEAPIRouter(t, 0, "")
			setProviderTransportForTest(t, handler, "native", nativeSSEResponseTransport(
				http.StatusOK,
				func(*http.Request) io.ReadCloser { return io.NopCloser(strings.NewReader(test.body)) },
			))
			recorder := httptest.NewRecorder()
			request := httptest.NewRequest(http.MethodGet, "/v1/responses/"+nativeSSEAPIResponseID, nil)
			request.Header.Set("X-LunarGate-Provider", "native")

			router.ServeHTTP(recorder, request)

			assertNativeSSEJSON502(t, recorder)
			if strings.Contains(recorder.Body.String(), "resp_other") || strings.Contains(recorder.Body.String(), "must-not-leak") {
				t.Fatalf("mismatched lifecycle response leaked downstream: %q", recorder.Body.String())
			}
		})
	}
}

func TestNativeResponsesLifecycleRejectsNon200SSE(t *testing.T) {
	valid := "event: response.failed\ndata: {\"type\":\"response.failed\",\"provider_secret\":true}\n\n"
	for _, status := range []int{http.StatusFound, http.StatusBadRequest, http.StatusInternalServerError} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			router, handler := newNativeSSEAPIRouter(t, 0, "")
			body := &nativeSSEAPISteppedBody{steps: [][]byte{[]byte(valid)}}
			setProviderTransportForTest(t, handler, "native", nativeSSEResponseTransport(
				status,
				func(*http.Request) io.ReadCloser { return body },
			))
			recorder := httptest.NewRecorder()
			request := httptest.NewRequest(http.MethodGet, "/v1/responses/"+nativeSSEAPIResponseID, nil)
			request.Header.Set("X-LunarGate-Provider", "native")

			router.ServeHTTP(recorder, request)

			assertNativeSSEJSON502(t, recorder)
			if strings.Contains(recorder.Body.String(), "provider_secret") {
				t.Fatalf("upstream SSE leaked through Responses lifecycle: %q", recorder.Body.String())
			}
			if body.closeCalls.Load() != 1 {
				t.Fatalf("upstream body close calls = %d, want 1", body.closeCalls.Load())
			}
		})
	}
}

func TestNativeResponsesSSEPreflightTimeoutReturnsJSON502(t *testing.T) {
	for _, endpoint := range nativeSSEAPIEndpoints() {
		t.Run(endpoint.name, func(t *testing.T) {
			router, handler := newNativeSSEAPIRouter(t, nativeSSEAPITimeout, upstreamTimeoutModeTTFT)
			var body *nativeSSEPrefixBlockingBody
			setProviderTransportForTest(t, handler, "native", nativeSSEResponseTransport(
				http.StatusOK,
				func(*http.Request) io.ReadCloser {
					body = newNativeSSEPrefixBlockingBody("")
					return body
				},
			))
			recorder := httptest.NewRecorder()

			endpoint.serve(router, context.Background(), recorder)

			assertNativeSSEJSON502(t, recorder)
			if body == nil || body.closeCalls.Load() != 1 {
				t.Fatalf("upstream body close calls = %v, want 1", nativeSSECloseCalls(body))
			}
		})
	}
}

func TestNativeChatCompletionLifecycleRejectsResponsesSSE(t *testing.T) {
	responsesEvent := "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_must_not_leak\",\"status\":\"completed\"}}\n\n"
	for _, status := range []int{http.StatusOK, http.StatusFound, http.StatusBadRequest, http.StatusInternalServerError} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			providerConfig := config.ProviderConfig{
				Type:         "openai",
				APIKey:       "provider-secret",
				BaseURL:      "http://native-chat-sse.invalid/v1",
				DefaultModel: "gpt-native",
				Capabilities: config.ProviderCapabilities{ChatCompletionsLifecycle: true},
			}
			router, handler, cache := newStoredChatLifecycleRouterFromConfigs(t, map[string]config.ProviderConfig{"native": providerConfig})
			t.Cleanup(cache.Stop)
			body := &nativeSSEAPISteppedBody{steps: [][]byte{[]byte(responsesEvent)}}
			setProviderTransportForTest(t, handler, "native", nativeSSEResponseTransport(
				status,
				func(*http.Request) io.ReadCloser { return body },
			))
			request := httptest.NewRequest(http.MethodGet, "/v1/chat/completions/chatcmpl_protocol", nil)
			request.Header.Set("X-LunarGate-Provider", "native")
			recorder := httptest.NewRecorder()

			router.ServeHTTP(recorder, request)

			assertNativeSSEJSON502(t, recorder)
			if strings.Contains(recorder.Body.String(), "resp_must_not_leak") {
				t.Fatalf("Responses event leaked through Chat lifecycle: %q", recorder.Body.String())
			}
			if body.closeCalls.Load() != 1 {
				t.Fatalf("upstream body close calls = %d, want 1", body.closeCalls.Load())
			}
		})
	}
}

func TestNativeResponsesSSEMidstreamFailuresEmitOneTerminal(t *testing.T) {
	first := "event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":7,\"response\":{\"id\":\"" + nativeSSEAPIResponseID + "\",\"object\":\"response\",\"status\":\"in_progress\",\"model\":\"gpt-native\",\"output\":[]}}\n\n"
	oversized := "data: {\"padding\":\"" + strings.Repeat("x", streaming.MaxStreamRecordBytes) + "\"}\n\n"
	readFailure := errors.New("injected native SSE read failure")
	tests := []struct {
		name      string
		body      func(*http.Request) io.ReadCloser
		forbidden string
	}{
		{name: "eof", body: func(*http.Request) io.ReadCloser { return io.NopCloser(strings.NewReader(first)) }},
		{name: "malformed", body: func(*http.Request) io.ReadCloser { return io.NopCloser(strings.NewReader(first + "data: {\n\n")) }},
		{name: "oversize", body: func(*http.Request) io.ReadCloser { return io.NopCloser(strings.NewReader(first + oversized)) }},
		{
			name: "read error",
			body: func(*http.Request) io.ReadCloser {
				return &nativeSSEAPIScriptedBody{reader: strings.NewReader(first), err: readFailure}
			},
		},
		{
			name: "provider context cancellation",
			body: func(*http.Request) io.ReadCloser {
				return &nativeSSEAPIScriptedBody{reader: strings.NewReader(first), err: context.Canceled}
			},
		},
		{
			name: "provider deadline",
			body: func(*http.Request) io.ReadCloser {
				return &nativeSSEAPIScriptedBody{reader: strings.NewReader(first), err: context.DeadlineExceeded}
			},
		},
		{
			name: "missing sequence",
			body: func(*http.Request) io.ReadCloser {
				return io.NopCloser(strings.NewReader(first + "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"response_id\":\"" + nativeSSEAPIResponseID + "\",\"delta\":\"must-not-leak\"}\n\n"))
			},
			forbidden: "must-not-leak",
		},
		{
			name: "fractional sequence",
			body: func(*http.Request) io.ReadCloser {
				return io.NopCloser(strings.NewReader(first + "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":7.5,\"response_id\":\"" + nativeSSEAPIResponseID + "\",\"delta\":\"must-not-leak\"}\n\n"))
			},
			forbidden: "must-not-leak",
		},
		{
			name: "duplicate sequence",
			body: func(*http.Request) io.ReadCloser {
				return io.NopCloser(strings.NewReader(first + "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":7,\"response_id\":\"" + nativeSSEAPIResponseID + "\",\"delta\":\"must-not-leak\"}\n\n"))
			},
			forbidden: "must-not-leak",
		},
		{
			name: "decreasing sequence",
			body: func(*http.Request) io.ReadCloser {
				return io.NopCloser(strings.NewReader(first + "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":6,\"response_id\":\"" + nativeSSEAPIResponseID + "\",\"delta\":\"must-not-leak\"}\n\n"))
			},
			forbidden: "must-not-leak",
		},
		{
			name: "invalid terminal response",
			body: func(*http.Request) io.ReadCloser {
				return io.NopCloser(strings.NewReader(first + "event: response.completed\ndata: {\"type\":\"response.completed\"}\n\n"))
			},
			forbidden: "event: response.completed",
		},
		{
			name: "terminal event type mismatch",
			body: func(*http.Request) io.ReadCloser {
				return io.NopCloser(strings.NewReader(first + "event: response.completed\ndata: {\"type\":\"response.failed\",\"response\":{\"id\":\"" + nativeSSEAPIResponseID + "\",\"status\":\"failed\"}}\n\n"))
			},
			forbidden: "event: response.completed",
		},
		{
			name: "terminal status mismatch",
			body: func(*http.Request) io.ReadCloser {
				return io.NopCloser(strings.NewReader(first + "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"" + nativeSSEAPIResponseID + "\",\"status\":\"failed\"}}\n\n"))
			},
			forbidden: "event: response.completed",
		},
		{
			name: "terminal response id changed",
			body: func(*http.Request) io.ReadCloser {
				return io.NopCloser(strings.NewReader(first + "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_changed\",\"status\":\"completed\"}}\n\n"))
			},
			forbidden: "resp_changed",
		},
		{
			name: "terminal response id missing",
			body: func(*http.Request) io.ReadCloser {
				return io.NopCloser(strings.NewReader(first + "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n"))
			},
			forbidden: "event: response.completed",
		},
		{
			name: "intermediate response id changed",
			body: func(*http.Request) io.ReadCloser {
				return io.NopCloser(strings.NewReader(first + "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"response_id\":\"resp_changed\",\"delta\":\"must-not-leak\"}\n\n"))
			},
			forbidden: "must-not-leak",
		},
	}

	for _, endpoint := range nativeSSEAPIEndpoints() {
		for _, test := range tests {
			t.Run(endpoint.name+"/"+test.name, func(t *testing.T) {
				router, handler := newNativeSSEAPIRouter(t, 0, "")
				setProviderTransportForTest(t, handler, "native", nativeSSEResponseTransport(http.StatusOK, test.body))
				recorder := httptest.NewRecorder()

				endpoint.serve(router, context.Background(), recorder)

				assertNativeSSESingleFailure(t, recorder, first)
				if test.forbidden != "" && strings.Contains(recorder.Body.String(), test.forbidden) {
					t.Fatalf("invalid terminal frame was forwarded: %q", recorder.Body.String())
				}
			})
		}
	}
}

func TestNativeResponsesSSEMidstreamProviderTimeoutEmitsOneTerminal(t *testing.T) {
	first := "event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":2,\"response\":{\"id\":\"" + nativeSSEAPIResponseID + "\",\"object\":\"response\",\"status\":\"in_progress\",\"model\":\"gpt-native\",\"output\":[]}}\n\n"
	for _, endpoint := range nativeSSEAPIEndpoints() {
		t.Run(endpoint.name, func(t *testing.T) {
			router, handler := newNativeSSEAPIRouter(t, nativeSSEAPITimeout, upstreamTimeoutModeTotal)
			var body *nativeSSEPrefixBlockingBody
			setProviderTransportForTest(t, handler, "native", nativeSSEResponseTransport(
				http.StatusOK,
				func(*http.Request) io.ReadCloser {
					body = newNativeSSEPrefixBlockingBody(first)
					return body
				},
			))
			recorder := httptest.NewRecorder()

			endpoint.serve(router, context.Background(), recorder)

			assertNativeSSESingleFailure(t, recorder, first)
			if body == nil || body.closeCalls.Load() != 1 {
				t.Fatalf("upstream body close calls = %v, want 1", nativeSSECloseCalls(body))
			}
		})
	}
}

func TestNativeResponsesSSEStopsAtFirstTerminal(t *testing.T) {
	first := []byte("event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"" + nativeSSEAPIResponseID + "\",\"status\":\"in_progress\",\"model\":\"gpt-native\",\"output\":[]}}\n\n")
	terminal := []byte("event: response.completed\ndata: {\"type\":\"response.completed\",\"sequence_number\":1,\"response\":{\"id\":\"" + nativeSSEAPIResponseID + "\",\"status\":\"completed\",\"model\":\"gpt-native\",\"output\":[]}}\n\n")
	postTerminal := []byte("event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":2,\"delta\":\"must-not-leak\"}\n\n")
	duplicate := []byte("event: response.failed\ndata: {\"type\":\"response.failed\",\"sequence_number\":3,\"response\":{\"id\":\"resp_duplicate\",\"status\":\"failed\"}}\n\n")

	for _, endpoint := range nativeSSEAPIEndpoints() {
		t.Run(endpoint.name, func(t *testing.T) {
			router, handler := newNativeSSEAPIRouter(t, 0, "")
			body := &nativeSSEAPISteppedBody{steps: [][]byte{first, terminal, postTerminal, duplicate}}
			setProviderTransportForTest(t, handler, "native", nativeSSEResponseTransport(
				http.StatusOK,
				func(*http.Request) io.ReadCloser { return body },
			))
			recorder := httptest.NewRecorder()

			endpoint.serve(router, context.Background(), recorder)

			if recorder.Code != http.StatusOK {
				t.Fatalf("status = %d, want 200; body=%s", recorder.Code, recorder.Body.String())
			}
			if got, want := recorder.Body.String(), string(first)+string(terminal); got != want {
				t.Fatalf("stream after first terminal\n got: %q\nwant: %q", got, want)
			}
			if body.reads != 2 || body.closeCalls.Load() != 1 {
				t.Fatalf("upstream reads/closes = %d/%d, want 2/1", body.reads, body.closeCalls.Load())
			}
		})
	}
}

func TestNativeResponsesSSEParentTerminationDoesNotEmitTerminal(t *testing.T) {
	first := "event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"" + nativeSSEAPIResponseID + "\",\"status\":\"in_progress\",\"model\":\"gpt-native\",\"output\":[]}}\n\n"
	for _, endpoint := range nativeSSEAPIEndpoints() {
		for _, termination := range []string{"canceled", "deadline"} {
			t.Run(endpoint.name+"/"+termination, func(t *testing.T) {
				router, handler := newNativeSSEAPIRouter(t, 0, "")
				blocked := make(chan struct{})
				setProviderTransportForTest(t, handler, "native", nativeSSEResponseTransport(
					http.StatusOK,
					func(request *http.Request) io.ReadCloser {
						return &nativeSSEAPIContextBody{
							reader:  strings.NewReader(first),
							ctx:     request.Context(),
							blocked: blocked,
						}
					},
				))
				var ctx context.Context
				var cancel context.CancelFunc
				if termination == "deadline" {
					ctx, cancel = context.WithTimeout(context.Background(), nativeSSEAPITimeout)
				} else {
					ctx, cancel = context.WithCancel(context.Background())
				}
				defer cancel()
				recorder := httptest.NewRecorder()
				done := make(chan struct{})
				go func() {
					defer close(done)
					endpoint.serve(router, ctx, recorder)
				}()
				select {
				case <-blocked:
				case <-time.After(time.Second):
					t.Fatal("upstream body did not block after first frame")
				}
				if termination == "canceled" {
					cancel()
				}
				select {
				case <-done:
				case <-time.After(time.Second):
					t.Fatal("request did not stop after parent termination")
				}

				if strings.Contains(recorder.Body.String(), "response.failed") {
					t.Fatalf("terminated stream received synthetic terminal: %q", recorder.Body.String())
				}
				if got := recorder.Body.String(); got != first {
					t.Fatalf("terminated stream = %q, want first frame only", got)
				}
			})
		}
	}
}

func TestNativeResponsesSSEDownstreamFailureDoesNotEmitTerminal(t *testing.T) {
	first := "event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"" + nativeSSEAPIResponseID + "\",\"status\":\"in_progress\",\"model\":\"gpt-native\",\"output\":[]}}\n\n"
	second := "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":1,\"delta\":\"second\"}\n\n"
	terminal := "event: response.completed\ndata: {\"type\":\"response.completed\",\"sequence_number\":2,\"response\":{\"id\":\"" + nativeSSEAPIResponseID + "\",\"status\":\"completed\"}}\n\n"
	for _, endpoint := range nativeSSEAPIEndpoints() {
		t.Run(endpoint.name, func(t *testing.T) {
			router, handler := newNativeSSEAPIRouter(t, 0, "")
			setProviderTransportForTest(t, handler, "native", nativeSSEResponseTransport(
				http.StatusOK,
				func(*http.Request) io.ReadCloser { return io.NopCloser(strings.NewReader(first + second + terminal)) },
			))
			writer := &nativeSSEAPIFailingWriter{header: make(http.Header), failWriteAt: 2}

			endpoint.serve(router, context.Background(), writer)

			if got := writer.body.String(); got != first {
				t.Fatalf("downstream body = %q, want first frame only", got)
			}
			if strings.Contains(writer.body.String(), "response.failed") || writer.writes != 2 {
				t.Fatalf("downstream failure state = writes:%d body:%q", writer.writes, writer.body.String())
			}
		})
	}
}

func newNativeSSEAPIRouter(t *testing.T, timeout time.Duration, timeoutMode string) (http.Handler, *Handler) {
	t.Helper()
	providerConfig := config.ProviderConfig{
		Type:         "openai",
		APIKey:       "provider-secret",
		BaseURL:      "http://native-sse.invalid/v1",
		DefaultModel: "gpt-native",
		Timeout:      timeout,
		TimeoutMode:  timeoutMode,
		Capabilities: config.ProviderCapabilities{ResponsesLifecycle: true},
	}
	router, handler, cache := newNativeLifecycleRouterFromConfigs(t, map[string]config.ProviderConfig{"native": providerConfig})
	t.Cleanup(cache.Stop)
	return router, handler
}

func nativeSSEResponseTransport(status int, body func(*http.Request) io.ReadCloser) http.RoundTripper {
	return providerURLRoundTripFunc(func(request *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: status,
			Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
			Body:       body(request),
			Request:    request,
		}, nil
	})
}

func assertNativeSSEJSON502(t *testing.T, recorder *httptest.ResponseRecorder) {
	t.Helper()
	if recorder.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502; body=%s", recorder.Code, recorder.Body.String())
	}
	if contentType := recorder.Header().Get("Content-Type"); !strings.HasPrefix(contentType, "application/json") {
		t.Fatalf("Content-Type = %q, want application/json", contentType)
	}
	if !json.Valid(recorder.Body.Bytes()) {
		t.Fatalf("body is not JSON: %q", recorder.Body.String())
	}
	if strings.Contains(recorder.Body.String(), "event: response.") || strings.Contains(recorder.Body.String(), "data:") {
		t.Fatalf("preflight failure emitted SSE: %q", recorder.Body.String())
	}
}

func assertNativeSSESingleFailure(t *testing.T, recorder *httptest.ResponseRecorder, first string) {
	t.Helper()
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want committed 200; body=%s", recorder.Code, recorder.Body.String())
	}
	body := recorder.Body.String()
	if !strings.HasPrefix(body, first) {
		t.Fatalf("stream prefix = %q, want %q", body, first)
	}
	if got := strings.Count(body, "event: response.failed\n"); got != 1 {
		t.Fatalf("response.failed count = %d, want 1; body=%q", got, body)
	}
	created := decodeNativeResponsesSSEEvent(t, first, "response.created")
	failed := decodeNativeResponsesSSEEvent(t, body, "response.failed")
	var createdSequence, failedSequence int64
	if err := json.Unmarshal(created["sequence_number"], &createdSequence); err != nil {
		t.Fatalf("decode created sequence_number: %v", err)
	}
	if err := json.Unmarshal(failed["sequence_number"], &failedSequence); err != nil {
		t.Fatalf("decode failed sequence_number: %v", err)
	}
	if failedSequence != createdSequence+1 {
		t.Fatalf("synthetic failure sequence_number = %d, want %d", failedSequence, createdSequence+1)
	}
	var createdResponse, failedResponse map[string]json.RawMessage
	if err := json.Unmarshal(created["response"], &createdResponse); err != nil {
		t.Fatalf("decode created response: %v", err)
	}
	if err := json.Unmarshal(failed["response"], &failedResponse); err != nil {
		t.Fatalf("decode failed response: %v", err)
	}
	if got, want := parseJSONStringRaw(failedResponse["id"]), parseJSONStringRaw(createdResponse["id"]); got != want {
		t.Fatalf("synthetic failure id = %q, want locked id %q", got, want)
	}
}

type nativeSSEAPIScriptedBody struct {
	reader *strings.Reader
	err    error
}

func (b *nativeSSEAPIScriptedBody) Read(p []byte) (int, error) {
	if b.reader.Len() > 0 {
		return b.reader.Read(p)
	}
	return 0, b.err
}

func (*nativeSSEAPIScriptedBody) Close() error { return nil }

type nativeSSEPrefixBlockingBody struct {
	reader     *strings.Reader
	closed     chan struct{}
	closeOnce  sync.Once
	closeCalls atomic.Int32
}

func newNativeSSEPrefixBlockingBody(prefix string) *nativeSSEPrefixBlockingBody {
	return &nativeSSEPrefixBlockingBody{reader: strings.NewReader(prefix), closed: make(chan struct{})}
}

func (b *nativeSSEPrefixBlockingBody) Read(p []byte) (int, error) {
	if b.reader.Len() > 0 {
		return b.reader.Read(p)
	}
	<-b.closed
	return 0, errors.New("blocked native SSE body closed")
}

func (b *nativeSSEPrefixBlockingBody) Close() error {
	b.closeOnce.Do(func() {
		b.closeCalls.Add(1)
		close(b.closed)
	})
	return nil
}

func nativeSSECloseCalls(body *nativeSSEPrefixBlockingBody) int32 {
	if body == nil {
		return 0
	}
	return body.closeCalls.Load()
}

type nativeSSEAPISteppedBody struct {
	steps      [][]byte
	reads      int
	closeCalls atomic.Int32
}

func (b *nativeSSEAPISteppedBody) Read(p []byte) (int, error) {
	if b.reads >= len(b.steps) {
		return 0, io.EOF
	}
	step := b.steps[b.reads]
	b.reads++
	return copy(p, step), nil
}

func (b *nativeSSEAPISteppedBody) Close() error {
	b.closeCalls.Add(1)
	return nil
}

type nativeSSEAPIContextBody struct {
	reader      *strings.Reader
	ctx         context.Context
	blocked     chan<- struct{}
	blockedOnce sync.Once
}

func (b *nativeSSEAPIContextBody) Read(p []byte) (int, error) {
	if b.reader.Len() > 0 {
		return b.reader.Read(p)
	}
	b.blockedOnce.Do(func() { close(b.blocked) })
	<-b.ctx.Done()
	return 0, b.ctx.Err()
}

func (*nativeSSEAPIContextBody) Close() error { return nil }

type nativeSSEAPIFailingWriter struct {
	header      http.Header
	body        bytes.Buffer
	status      int
	writes      int
	flushes     int
	failWriteAt int
}

func (w *nativeSSEAPIFailingWriter) Header() http.Header { return w.header }

func (w *nativeSSEAPIFailingWriter) WriteHeader(status int) { w.status = status }

func (w *nativeSSEAPIFailingWriter) Write(p []byte) (int, error) {
	w.writes++
	if w.writes == w.failWriteAt {
		return 0, errors.New("injected downstream write failure")
	}
	return w.body.Write(p)
}

func (w *nativeSSEAPIFailingWriter) FlushError() error {
	w.flushes++
	return nil
}
