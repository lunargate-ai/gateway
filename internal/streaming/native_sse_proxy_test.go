package streaming

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
)

type steppedSSEBody struct {
	steps  [][]byte
	reads  int
	closed bool
}

func (b *steppedSSEBody) Read(dst []byte) (int, error) {
	if b.reads >= len(b.steps) {
		return 0, io.EOF
	}
	step := b.steps[b.reads]
	b.reads++
	return copy(dst, step), nil
}

func (b *steppedSSEBody) Close() error {
	b.closed = true
	return nil
}

type failingSSEWriter struct {
	header      http.Header
	body        bytes.Buffer
	status      int
	writes      int
	flushes     int
	failWriteAt int
	failFlushAt int
}

func (w *failingSSEWriter) Header() http.Header {
	if w.header == nil {
		w.header = make(http.Header)
	}
	return w.header
}

func (w *failingSSEWriter) WriteHeader(status int) {
	w.status = status
}

func (w *failingSSEWriter) Write(payload []byte) (int, error) {
	w.writes++
	if w.failWriteAt > 0 && w.writes == w.failWriteAt {
		return 0, errors.New("downstream write failed")
	}
	return w.body.Write(payload)
}

func (w *failingSSEWriter) FlushError() error {
	w.flushes++
	if w.failFlushAt > 0 && w.flushes == w.failFlushAt {
		return errors.New("downstream flush failed")
	}
	return nil
}

func TestProxySSETransformsDataBeforeForwardingAndObservation(t *testing.T) {
	const transformedData = `{"type":"response.completed","response":{"id":"resp_upstream","status":"completed","conversation":{"id":"conv_local"}}}`
	rawStream := strings.Join([]string{
		": stream comment\r\n\r\n",
		"event: response.completed\r\n",
		"id: upstream-event\r\n",
		"retry: 1250\r\n",
		`data:{"type":"response.completed",` + "\r\n",
		": terminal comment\r\n",
		`data: "response":{"id":"resp_upstream","status":"completed"}}` + "\r\n\r\n",
	}, "")
	wantStream := strings.Join([]string{
		": stream comment\r\n\r\n",
		"event: response.completed\r\n",
		"id: upstream-event\r\n",
		"retry: 1250\r\n",
		"data:" + transformedData + "\r\n",
		": terminal comment\r\n\r\n",
	}, "")
	response := &http.Response{
		StatusCode: http.StatusAccepted,
		Body:       io.NopCloser(strings.NewReader(rawStream)),
	}
	writer := &failingSSEWriter{}
	var observedData string

	err := NewHandler().ProxySSEWithDataTransformer(
		context.Background(),
		writer,
		response,
		"openai",
		func(event SSEEvent) bool {
			if event.Event != "response.completed" {
				return false
			}
			observedData = string(event.Data)
			return true
		},
		func(event SSEEvent) ([]byte, error) {
			if event.Event != "response.completed" {
				return nil, nil
			}
			return []byte(transformedData), nil
		},
	)

	if err != nil {
		t.Fatalf("ProxySSEWithDataTransformer returned error: %v", err)
	}
	if writer.status != http.StatusAccepted {
		t.Fatalf("status = %d, want %d", writer.status, http.StatusAccepted)
	}
	if got := writer.body.String(); got != wantStream {
		t.Fatalf("transformed stream\n got: %q\nwant: %q", got, wantStream)
	}
	if observedData != transformedData {
		t.Fatalf("observer data = %q, want %q", observedData, transformedData)
	}
}

func TestProxySSEStopsReadingAfterDownstreamWriteFailure(t *testing.T) {
	body := &steppedSSEBody{steps: [][]byte{
		[]byte("event: response.created\ndata: {\"type\":\"response.created\"}\n\n"),
		[]byte("event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"two\"}\n\n"),
		[]byte("event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n"),
	}}
	writer := &failingSSEWriter{failWriteAt: 2}
	response := &http.Response{StatusCode: http.StatusOK, Body: body}

	err := NewHandler().ProxySSE(context.Background(), writer, response, "openai", func(event SSEEvent) bool {
		return strings.Contains(string(event.Data), `"type":"response.completed"`)
	})

	if err == nil || !strings.Contains(err.Error(), "downstream write failed") {
		t.Fatalf("error = %v, want downstream write failure", err)
	}
	if body.reads != 2 {
		t.Fatalf("upstream reads = %d, want 2 (no read after failed write)", body.reads)
	}
	if !body.closed {
		t.Fatal("upstream body was not closed")
	}
	if got := writer.body.String(); strings.Contains(got, "response.completed") {
		t.Fatalf("stream continued after downstream failure: %q", got)
	}
}

func TestProxySSEStopsReadingAfterDownstreamFlushFailure(t *testing.T) {
	body := &steppedSSEBody{steps: [][]byte{
		[]byte(": keepalive\n\n"),
		[]byte("event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n"),
	}}
	// The first flush publishes headers; the second follows the first frame.
	writer := &failingSSEWriter{failFlushAt: 2}
	response := &http.Response{StatusCode: http.StatusAccepted, Body: body}

	err := NewHandler().ProxySSE(context.Background(), writer, response, "openai", nil)

	if err == nil || !strings.Contains(err.Error(), "downstream flush failed") {
		t.Fatalf("error = %v, want downstream flush failure", err)
	}
	if body.reads != 1 {
		t.Fatalf("upstream reads = %d, want 1 (no read after failed flush)", body.reads)
	}
	if !body.closed {
		t.Fatal("upstream body was not closed")
	}
}
