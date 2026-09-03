package streaming

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"
)

func TestProxySSEReservesHeadersUntilValidJSONRecord(t *testing.T) {
	tests := []struct {
		name    string
		body    string
		wantErr error
	}{
		{name: "empty", wantErr: ErrUpstreamStreamEmpty},
		{name: "comments only", body: ": keepalive\n\n\n\n", wantErr: ErrUpstreamStreamEmpty},
		{name: "malformed", body: "event: response.created\ndata: {\n\n", wantErr: ErrNativeSSEInvalidData},
		{name: "array", body: "data: []\n\n", wantErr: ErrNativeSSEInvalidData},
		{name: "null", body: "data: null\n\n", wantErr: ErrNativeSSEInvalidData},
		{name: "multiple values", body: "data: {} {}\n\n", wantErr: ErrNativeSSEInvalidData},
		{name: "unterminated", body: "data: {}", wantErr: ErrUpstreamStreamIncomplete},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			writer := &failingSSEWriter{}
			response := &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(strings.NewReader(test.body)),
			}

			err := NewHandler().ProxySSE(context.Background(), writer, response, "openai", nil)
			if !errors.Is(err, test.wantErr) {
				t.Fatalf("error = %v, want %v", err, test.wantErr)
			}
			if writer.status != 0 || writer.body.Len() != 0 || writer.flushes != 0 {
				t.Fatalf("preflight committed downstream: status=%d body=%q flushes=%d", writer.status, writer.body.String(), writer.flushes)
			}
		})
	}
}

func TestProxySSERejectsNon200SuccessWithoutCommitting(t *testing.T) {
	body := &steppedSSEBody{steps: [][]byte{[]byte("data: {}\n\n")}}
	writer := &failingSSEWriter{}
	response := &http.Response{StatusCode: http.StatusAccepted, Body: body}

	err := NewHandler().ProxySSE(context.Background(), writer, response, "openai", func(SSEEvent) bool { return true })
	if !errors.Is(err, ErrNativeSSEInvalidStatus) {
		t.Fatalf("error = %v, want ErrNativeSSEInvalidStatus", err)
	}
	if writer.status != 0 || writer.body.Len() != 0 || writer.flushes != 0 {
		t.Fatalf("invalid status committed downstream: status=%d body=%q flushes=%d", writer.status, writer.body.String(), writer.flushes)
	}
	if body.reads != 0 || !body.closed {
		t.Fatalf("invalid-status body reads/closed = %d/%t, want 0/true", body.reads, body.closed)
	}
}

func TestProxySSEPreservesBoundedLeadingFramesAfterPreflight(t *testing.T) {
	stream := ": keepalive\r\n\r\n\r\ndata: {\"type\":\"response.created\"}\r\n\r\n"
	writer := &failingSSEWriter{}
	response := &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(stream))}

	err := NewHandler().ProxySSE(context.Background(), writer, response, "openai", func(event SSEEvent) bool {
		return strings.Contains(string(event.Data), "response.created")
	})
	if err != nil {
		t.Fatalf("ProxySSE: %v", err)
	}
	if writer.status != http.StatusOK || writer.body.String() != stream {
		t.Fatalf("forwarded stream = status %d body %q, want byte-exact 200 stream", writer.status, writer.body.String())
	}
}

func TestProxySSECoalescesManyEmptyPreflightFrames(t *testing.T) {
	const emptyFrames = maxNativeSSEPreflightFrames / 2
	stream := strings.Repeat("\n", emptyFrames) + "data: {\"type\":\"response.created\"}\n\n"
	writer := &failingSSEWriter{}
	response := &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(stream))}
	observerCalls := 0

	err := NewHandler().ProxySSE(context.Background(), writer, response, "openai", func(event SSEEvent) bool {
		observerCalls++
		return strings.Contains(string(event.Data), "response.created")
	})
	if err != nil {
		t.Fatalf("ProxySSE: %v", err)
	}
	if observerCalls != 1 {
		t.Fatalf("observer calls = %d, want only the first useful event", observerCalls)
	}
	if writer.status != http.StatusOK || writer.body.String() != stream {
		t.Fatalf("forwarded preflight changed: status=%d bytes=%d, want 200/%d", writer.status, writer.body.Len(), len(stream))
	}
}

func TestProxySSEBoundsPreflightFrameCount(t *testing.T) {
	t.Run("boundary", func(t *testing.T) {
		stream := strings.Repeat("\n", maxNativeSSEPreflightFrames-1) + "data: {}\n\n"
		writer := &failingSSEWriter{}
		response := &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(stream))}

		err := NewHandler().ProxySSE(context.Background(), writer, response, "openai", func(SSEEvent) bool { return true })
		if err != nil {
			t.Fatalf("ProxySSE: %v", err)
		}
		if writer.status != http.StatusOK || writer.body.String() != stream {
			t.Fatalf("boundary preflight changed: status=%d bytes=%d, want 200/%d", writer.status, writer.body.Len(), len(stream))
		}
	})

	t.Run("over limit", func(t *testing.T) {
		stream := strings.Repeat("\n", maxNativeSSEPreflightFrames) + "data: {}\n\n"
		body := &steppedSSEBody{steps: [][]byte{[]byte(stream)}}
		writer := &failingSSEWriter{}
		response := &http.Response{StatusCode: http.StatusOK, Body: body}

		err := NewHandler().ProxySSE(context.Background(), writer, response, "openai", nil)
		if !errors.Is(err, ErrNativeSSEPreflightTooLarge) {
			t.Fatalf("error = %v, want ErrNativeSSEPreflightTooLarge", err)
		}
		if writer.status != 0 || writer.body.Len() != 0 || writer.flushes != 0 {
			t.Fatalf("frame-limited preflight committed downstream: status=%d body=%d flushes=%d", writer.status, writer.body.Len(), writer.flushes)
		}
		if !body.closed {
			t.Fatal("frame-limited upstream body was not closed")
		}
	})
}

func TestProxySSEBoundsLeadingPreflightFrames(t *testing.T) {
	comment := ":" + strings.Repeat("x", MaxStreamRecordBytes-2) + "\n\n"
	response := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(comment + "data: {}\n\n")),
	}
	writer := &failingSSEWriter{}

	err := NewHandler().ProxySSE(context.Background(), writer, response, "openai", nil)
	if !errors.Is(err, ErrNativeSSEPreflightTooLarge) {
		t.Fatalf("error = %v, want ErrNativeSSEPreflightTooLarge", err)
	}
	if writer.status != 0 || writer.body.Len() != 0 {
		t.Fatalf("oversized preflight committed downstream: status=%d body=%d", writer.status, writer.body.Len())
	}
}

func TestProxySSEClassifiesMidstreamFailures(t *testing.T) {
	first := "event: response.created\ndata: {\"type\":\"response.created\"}\n\n"
	readFailure := errors.New("injected upstream read failure")
	tests := []struct {
		name    string
		body    io.ReadCloser
		wantErr error
	}{
		{name: "eof", body: io.NopCloser(strings.NewReader(first)), wantErr: ErrUpstreamStreamIncomplete},
		{name: "malformed", body: io.NopCloser(strings.NewReader(first + "data: {\n\n")), wantErr: ErrNativeSSEInvalidData},
		{name: "oversize", body: io.NopCloser(strings.NewReader(first + nativeStreamLimitJSONEvent(t, MaxStreamRecordBytes+1))), wantErr: ErrStreamRecordTooLarge},
		{name: "read", body: &scriptedSSEReadCloser{payload: strings.NewReader(first), err: readFailure}, wantErr: ErrNativeSSEUpstreamRead},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			writer := &failingSSEWriter{}
			response := &http.Response{StatusCode: http.StatusOK, Body: test.body}

			err := NewHandler().ProxySSE(context.Background(), writer, response, "openai", func(SSEEvent) bool { return false })
			if !errors.Is(err, test.wantErr) {
				t.Fatalf("error = %v, want %v", err, test.wantErr)
			}
			if writer.status != http.StatusOK || writer.body.String() != first {
				t.Fatalf("forwarded prefix = status %d body %q, want exact first frame", writer.status, writer.body.String())
			}
		})
	}
}

func TestProxySSEClassifiesTransformFailureBeforeWrite(t *testing.T) {
	first := "data: {\"type\":\"response.created\"}\n\n"
	second := "data: {\"type\":\"response.delta\"}\n\n"
	writer := &failingSSEWriter{}
	response := &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(first + second))}

	err := NewHandler().ProxySSEWithDataTransformer(
		context.Background(),
		writer,
		response,
		"openai",
		func(SSEEvent) bool { return false },
		func(event SSEEvent) ([]byte, error) {
			if strings.Contains(string(event.Data), "response.delta") {
				return nil, errors.New("injected transform failure")
			}
			return nil, nil
		},
	)
	if !errors.Is(err, ErrNativeSSETransform) {
		t.Fatalf("error = %v, want ErrNativeSSETransform", err)
	}
	if writer.body.String() != first {
		t.Fatalf("transform failure forwarded invalid frame: %q", writer.body.String())
	}
}

func TestProxySSEStopsImmediatelyAfterFirstTerminal(t *testing.T) {
	first := []byte("data: {\"type\":\"response.created\"}\n\n")
	terminal := []byte("data: {\"type\":\"response.completed\"}\n\n")
	postTerminal := []byte("data: {\"type\":\"response.delta\"}\n\n")
	body := &steppedSSEBody{steps: [][]byte{first, terminal, postTerminal}}
	writer := &failingSSEWriter{}
	response := &http.Response{StatusCode: http.StatusOK, Body: body}

	err := NewHandler().ProxySSE(context.Background(), writer, response, "openai", func(event SSEEvent) bool {
		return strings.Contains(string(event.Data), "response.completed")
	})
	if err != nil {
		t.Fatalf("ProxySSE: %v", err)
	}
	if body.reads != 2 || !body.closed {
		t.Fatalf("upstream reads/closed = %d/%t, want 2/true", body.reads, body.closed)
	}
	if got, want := writer.body.String(), string(first)+string(terminal); got != want {
		t.Fatalf("forwarded stream = %q, want %q", got, want)
	}
}

func TestProxySSEParentTerminationDoesNotCommit(t *testing.T) {
	tests := []struct {
		name string
		ctx  func() context.Context
	}{
		{
			name: "canceled",
			ctx: func() context.Context {
				ctx, cancel := context.WithCancel(context.Background())
				cancel()
				return ctx
			},
		},
		{
			name: "deadline",
			ctx: func() context.Context {
				ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
				t.Cleanup(cancel)
				return ctx
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			body := &steppedSSEBody{steps: [][]byte{[]byte("data: {}\n\n")}}
			writer := &failingSSEWriter{}
			err := NewHandler().ProxySSE(test.ctx(), writer, &http.Response{StatusCode: http.StatusOK, Body: body}, "openai", nil)
			if err == nil || (!errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded)) {
				t.Fatalf("error = %v, want parent termination", err)
			}
			if writer.status != 0 || writer.body.Len() != 0 || body.reads != 0 || !body.closed {
				t.Fatalf("termination state = status:%d body:%q reads:%d closed:%t", writer.status, writer.body.String(), body.reads, body.closed)
			}
		})
	}
}

type scriptedSSEReadCloser struct {
	payload *strings.Reader
	err     error
}

func (b *scriptedSSEReadCloser) Read(p []byte) (int, error) {
	if b.payload.Len() > 0 {
		return b.payload.Read(p)
	}
	return 0, b.err
}

func (*scriptedSSEReadCloser) Close() error { return nil }
