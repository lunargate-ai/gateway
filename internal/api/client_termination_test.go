package api

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/observability"
	"github.com/lunargate-ai/gateway/pkg/models"
	"github.com/prometheus/client_golang/prometheus/testutil"
)

var clientTerminationCases = []struct {
	name string
	err  error
}{
	{name: "cancelled", err: context.Canceled},
	{name: "deadline", err: context.DeadlineExceeded},
}

func TestClientRequestTerminationDoesNotClaimUpstreamDeadline(t *testing.T) {
	if isClientRequestTermination(context.Background(), context.DeadlineExceeded) {
		t.Fatal("live parent context classified an upstream deadline as client termination")
	}
}

func TestChatNonStreamClassifiesParentTerminationAsClientCancellation(t *testing.T) {
	for _, testCase := range clientTerminationCases {
		t.Run(testCase.name, func(t *testing.T) {
			runNonStreamClientTerminationTest(
				t,
				testCase.err,
				"/v1/chat/completions",
				`{"model":"gpt-test","messages":[{"role":"user","content":"hello"}]}`,
				func(handler *Handler, recorder http.ResponseWriter, request *http.Request) {
					handler.ChatCompletions(recorder, request)
				},
			)
		})
	}
}

func TestEmbeddingsClassifiesParentTerminationAsClientCancellation(t *testing.T) {
	for _, testCase := range clientTerminationCases {
		t.Run(testCase.name, func(t *testing.T) {
			runNonStreamClientTerminationTest(
				t,
				testCase.err,
				"/v1/embeddings",
				`{"model":"gpt-test","input":"hello"}`,
				func(handler *Handler, recorder http.ResponseWriter, request *http.Request) {
					handler.Embeddings(recorder, request)
				},
			)
		})
	}
}

func TestChatStreamClassifiesParentTerminationWithoutAppendingJSON(t *testing.T) {
	const upstreamChunk = `data: {"id":"chatcmpl-client-termination","object":"chat.completion.chunk","created":1,"model":"gpt-test","choices":[{"index":0,"delta":{"role":"assistant","content":"hello"},"finish_reason":null}]}

`

	for _, testCase := range clientTerminationCases {
		t.Run(testCase.name, func(t *testing.T) {
			capture := newCollectorCapture(t, true, false)
			handler, metrics := newClientTerminationHandler(t, capture.client)
			requestContext, trigger := newTriggeredRequestContext(testCase.err)
			t.Cleanup(trigger)

			setProviderTransportForTest(t, handler, "openai", providerURLRoundTripFunc(func(request *http.Request) (*http.Response, error) {
				return &http.Response{
					StatusCode: http.StatusOK,
					Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
					Body: newTerminateAfterFirstReadBody(
						request.Context(),
						trigger,
						upstreamChunk,
					),
					Request: request,
				}, nil
			}))

			recorder := httptest.NewRecorder()
			request := httptest.NewRequest(
				http.MethodPost,
				"/v1/chat/completions",
				strings.NewReader(`{"model":"gpt-test","stream":true,"messages":[{"role":"user","content":"hello"}]}`),
			).WithContext(requestContext)
			handler.ChatCompletions(recorder, request)

			if recorder.Code != http.StatusOK {
				t.Fatalf("stream status = %d, want committed 200; body=%s", recorder.Code, recorder.Body.String())
			}
			body := recorder.Body.String()
			if !strings.Contains(body, `"content":"hello"`) {
				t.Fatalf("stream body = %q, want first content chunk", body)
			}
			for _, forbidden := range []string{`{"error":`, "client_cancelled", "client disconnected", "[DONE]"} {
				if strings.Contains(body, forbidden) {
					t.Fatalf("stream body appended termination output %q: %q", forbidden, body)
				}
			}
			if !recorder.Flushed {
				t.Fatal("stream did not flush its first chunk")
			}
			if got := testutil.ToFloat64(metrics.ProviderErrors.WithLabelValues("openai", "all_failed")); got != 0 {
				t.Fatalf("provider all_failed metric = %v, want 0", got)
			}
			if got := testutil.ToFloat64(metrics.RequestsTotal.WithLabelValues("openai", "gpt-test", "499", "observed-route")); got != 1 {
				t.Fatalf("stream client-cancelled request metric = %v, want 1", got)
			}
			assertClientTerminationCollector(t, capture)
		})
	}
}

func runNonStreamClientTerminationTest(
	t *testing.T,
	parentErr error,
	path string,
	payload string,
	serve func(*Handler, http.ResponseWriter, *http.Request),
) {
	t.Helper()
	capture := newCollectorCapture(t, true, false)
	handler, metrics := newClientTerminationHandler(t, capture.client)
	requestContext, trigger := newTriggeredRequestContext(parentErr)
	t.Cleanup(trigger)

	var providerCalls atomic.Int32
	setProviderTransportForTest(t, handler, "openai", providerURLRoundTripFunc(func(request *http.Request) (*http.Response, error) {
		providerCalls.Add(1)
		trigger()
		<-request.Context().Done()
		return nil, request.Context().Err()
	}))

	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, path, strings.NewReader(payload)).WithContext(requestContext)
	serve(handler, recorder, request)

	if recorder.Code != 499 {
		t.Fatalf("status = %d, want 499; body=%s", recorder.Code, recorder.Body.String())
	}
	var response models.ErrorResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode error response: %v; body=%s", err, recorder.Body.String())
	}
	if response.Error.Type != "client_cancelled" || response.Error.Message != "client disconnected" {
		t.Fatalf("error = %#v, want sanitized client cancellation", response.Error)
	}
	if got := providerCalls.Load(); got != 1 {
		t.Fatalf("provider calls = %d, want 1", got)
	}
	if got := testutil.ToFloat64(metrics.ProviderErrors.WithLabelValues("openai", "all_failed")); got != 0 {
		t.Fatalf("provider all_failed metric = %v, want 0", got)
	}
	assertClientTerminationCollector(t, capture)
}

func newClientTerminationHandler(t *testing.T, collector *observability.CollectorClient) (*Handler, *observability.Metrics) {
	t.Helper()
	return newObservedOpenAIHandler(
		t,
		"http://client-termination.invalid",
		config.TargetConfig{Provider: "openai", Model: "gpt-test", Weight: 1},
		collector,
		config.CacheConfig{Enabled: false},
	)
}

func assertClientTerminationCollector(t *testing.T, capture *collectorCapture) {
	t.Helper()
	_, metric, requestLog := capture.waitForRequestEvents(t)
	if got := metric["status_code"]; got != float64(499) {
		t.Fatalf("collector metric status = %#v, want 499", got)
	}
	if got := metric["error_code"]; got != "client_cancelled" {
		t.Fatalf("collector metric error = %#v, want client_cancelled", got)
	}
	if got := requestLog["status_code"]; got != float64(499) {
		t.Fatalf("collector request log status = %#v, want 499", got)
	}
	if got := requestLog["error_code"]; got != "client_cancelled" {
		t.Fatalf("collector request log error = %#v, want client_cancelled", got)
	}
	if got := requestLog["error_message"]; got != "client disconnected" {
		t.Fatalf("collector request log message = %#v, want sanitized client message", got)
	}
}

type triggeredRequestContext struct {
	context.Context
	done chan struct{}
	err  error
	once sync.Once
}

func newTriggeredRequestContext(err error) (*triggeredRequestContext, func()) {
	ctx := &triggeredRequestContext{
		Context: context.Background(),
		done:    make(chan struct{}),
		err:     err,
	}
	return ctx, func() {
		ctx.once.Do(func() {
			close(ctx.done)
		})
	}
}

func (c *triggeredRequestContext) Done() <-chan struct{} {
	return c.done
}

func (c *triggeredRequestContext) Err() error {
	select {
	case <-c.done:
		return c.err
	default:
		return nil
	}
}

type terminateAfterFirstReadBody struct {
	ctx     context.Context
	trigger func()
	first   *strings.Reader
}

func newTerminateAfterFirstReadBody(ctx context.Context, trigger func(), first string) io.ReadCloser {
	return &terminateAfterFirstReadBody{
		ctx:     ctx,
		trigger: trigger,
		first:   strings.NewReader(first),
	}
}

func (b *terminateAfterFirstReadBody) Read(p []byte) (int, error) {
	if b.first.Len() > 0 {
		return b.first.Read(p)
	}
	b.trigger()
	<-b.ctx.Done()
	return 0, b.ctx.Err()
}

func (b *terminateAfterFirstReadBody) Close() error {
	return nil
}
