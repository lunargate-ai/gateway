package api

import (
	"bytes"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/lunargate-ai/gateway/internal/streaming"
)

func TestNextResponsesSSEFrameEnforcesSharedRecordLimit(t *testing.T) {
	boundary := responsesWSLimitSSEFrame(t, streaming.MaxStreamRecordBytes) + "next"
	frame, remaining, ok, err := nextResponsesSSEFrame([]byte(boundary))
	if err != nil {
		t.Fatalf("decode boundary frame: %v", err)
	}
	if !ok {
		t.Fatal("boundary frame was not recognized as complete")
	}
	if len(frame) != streaming.MaxStreamRecordBytes {
		t.Fatalf("boundary frame size = %d, want %d", len(frame), streaming.MaxStreamRecordBytes)
	}
	if string(remaining) != "next" {
		t.Fatalf("remaining = %q, want next", remaining)
	}

	oversized := responsesWSLimitSSEFrame(t, streaming.MaxStreamRecordBytes+1)
	frame, remaining, ok, err = nextResponsesSSEFrame([]byte(oversized))
	if !errors.Is(err, streaming.ErrStreamRecordTooLarge) {
		t.Fatalf("oversize error = %v, want ErrStreamRecordTooLarge", err)
	}
	if ok || frame != nil || remaining != nil {
		t.Fatalf("oversized result = frame:%d remaining:%d ok:%v", len(frame), len(remaining), ok)
	}
}

func TestResponsesWebSocketProxyDropsOversizedBufferedFrame(t *testing.T) {
	proxy := newResponsesWebSocketProxy(nil)
	payload := bytes.Repeat([]byte{'x'}, streaming.MaxStreamRecordBytes+1)

	n, err := proxy.Write(payload)
	if !errors.Is(err, streaming.ErrStreamRecordTooLarge) {
		t.Fatalf("Write error = %v, want ErrStreamRecordTooLarge", err)
	}
	if n != len(payload) {
		t.Fatalf("Write consumed %d bytes, want %d", n, len(payload))
	}
	if proxy.buffer.Len() != 0 {
		t.Fatalf("oversized buffer retained %d bytes", proxy.buffer.Len())
	}
}

func TestResponsesWebSocketProxyBoundsLargeMultiFrameWrite(t *testing.T) {
	frame := []byte(":" + strings.Repeat("x", 1<<20) + "\n\n")
	payload := append(bytes.Repeat(frame, 10), []byte("partial")...)
	proxy := newResponsesWebSocketProxy(nil)

	n, err := proxy.Write(payload)
	if err != nil {
		t.Fatalf("Write returned error: %v", err)
	}
	if n != len(payload) {
		t.Fatalf("Write consumed %d bytes, want %d", n, len(payload))
	}
	if got := proxy.buffer.String(); got != "partial" {
		t.Fatalf("buffer = %q, want partial", got)
	}
	if proxy.buffer.Cap() >= 1<<20 {
		t.Fatalf("buffer retained %d bytes after draining complete frames", proxy.buffer.Cap())
	}
}

func TestResponsesWebSocketProxyBoundsHTTPErrorBody(t *testing.T) {
	proxy := newResponsesWebSocketProxy(nil)
	proxy.WriteHeader(http.StatusInternalServerError)
	payload := bytes.Repeat([]byte{'s'}, upstreamErrorBodyLimit+1)

	n, err := proxy.Write(payload)
	if err != nil {
		t.Fatalf("Write returned error: %v", err)
	}
	if n != len(payload) {
		t.Fatalf("Write consumed %d bytes, want %d", n, len(payload))
	}
	if proxy.buffer.Len() != upstreamErrorBodyLimit {
		t.Fatalf("buffered error body = %d bytes, want %d", proxy.buffer.Len(), upstreamErrorBodyLimit)
	}
	if !proxy.errorBodyTruncated {
		t.Fatal("oversized HTTP error body was not marked truncated")
	}

	_ = proxy.finalize()
	if proxy.terminalError == nil || proxy.terminalError.code != "upstream_response_too_large" {
		t.Fatalf("terminal error = %#v, want upstream_response_too_large", proxy.terminalError)
	}
	if strings.Contains(proxy.terminalError.message, strings.Repeat("s", 32)) {
		t.Fatalf("terminal error leaked upstream body: %q", proxy.terminalError.message)
	}
}

func TestResponsesWebSocketProxyAcceptsHTTPErrorBodyAtBoundary(t *testing.T) {
	proxy := newResponsesWebSocketProxy(nil)
	proxy.WriteHeader(http.StatusBadRequest)
	payload := bytes.Repeat([]byte{'x'}, upstreamErrorBodyLimit)

	if _, err := proxy.Write(payload); err != nil {
		t.Fatalf("Write returned error: %v", err)
	}
	if proxy.errorBodyTruncated {
		t.Fatal("boundary HTTP error body was marked truncated")
	}
	if proxy.buffer.Len() != upstreamErrorBodyLimit {
		t.Fatalf("buffered error body = %d bytes, want %d", proxy.buffer.Len(), upstreamErrorBodyLimit)
	}
}

func TestResponsesWebSocketProxyRejectsUnterminatedCompleteLookingEvent(t *testing.T) {
	type proxyResult struct {
		writeErr          error
		finalizeErr       error
		terminalSeen      bool
		completedResponse map[string]interface{}
	}
	result := make(chan proxyResult, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := responsesWebSocketUpgrader.Upgrade(w, r, nil)
		if err != nil {
			result <- proxyResult{writeErr: err}
			return
		}
		defer conn.Close()

		proxy := newResponsesWebSocketProxy(&responsesWebSocketSession{conn: conn})
		payload := []byte(`data: {"type":"response.completed","response":{"id":"resp_unterminated","status":"completed"}}` + "\n")
		_, writeErr := proxy.Write(payload)
		finalizeErr := proxy.finalize()
		result <- proxyResult{
			writeErr:          writeErr,
			finalizeErr:       finalizeErr,
			terminalSeen:      proxy.terminalSeen,
			completedResponse: proxy.completedResponse,
		}
	}))
	defer server.Close()

	conn := mustDialResponsesWebSocket(t, server.URL)
	defer conn.Close()
	event := readResponsesWebSocketEvent(t, conn)
	if got, _ := event["type"].(string); got != "error" {
		t.Fatalf("first event type = %q, want error; event=%#v", got, event)
	}
	errObj, _ := event["error"].(map[string]interface{})
	if code, _ := errObj["code"].(string); code != "upstream_stream_incomplete" {
		t.Fatalf("error code = %q, want upstream_stream_incomplete", code)
	}

	got := <-result
	if got.writeErr != nil || got.finalizeErr != nil {
		t.Fatalf("proxy errors = write:%v finalize:%v", got.writeErr, got.finalizeErr)
	}
	if got.terminalSeen || got.completedResponse != nil {
		t.Fatalf("unterminated event affected terminal state: seen=%t response=%#v", got.terminalSeen, got.completedResponse)
	}
}

func responsesWSLimitSSEFrame(t *testing.T, recordSize int) string {
	t.Helper()
	prefix := "event: response.output_text.delta\ndata: first\ndata: "
	fixedSize := len(prefix) + 1
	if recordSize < fixedSize {
		t.Fatalf("record size %d is smaller than fixture overhead %d", recordSize, fixedSize)
	}
	return prefix + strings.Repeat("x", recordSize-fixedSize) + "\n\n"
}
