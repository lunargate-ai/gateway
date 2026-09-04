package api

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/lunargate-ai/gateway/internal/streaming"
	"github.com/lunargate-ai/gateway/pkg/models"
)

var errStartupStreamRead = errors.New("injected startup stream read failure")

type startupStreamErrorBody struct{}

func (startupStreamErrorBody) Read([]byte) (int, error) { return 0, errStartupStreamRead }
func (startupStreamErrorBody) Close() error             { return nil }

type startupPayloadThenErrorBody struct {
	payload *strings.Reader
}

func (b *startupPayloadThenErrorBody) Read(p []byte) (int, error) {
	if b.payload.Len() > 0 {
		return b.payload.Read(p)
	}
	return 0, errStartupStreamRead
}

func (*startupPayloadThenErrorBody) Close() error { return nil }

func startupFailureBodies(protocol streamingStatusProtocol) []struct {
	name    string
	factory func() io.ReadCloser
} {
	malformed := "data: {\n\n"
	truncated := "data: {"
	oversized := func() string {
		return "data: " + strings.Repeat("x", streaming.MaxStreamRecordBytes) + "\n\n"
	}
	if protocol.name == "anthropic_sse" {
		malformed = "event: content_block_delta\ndata: {\n\n"
		truncated = "event: content_block_delta\ndata: {"
		oversized = func() string {
			return "event: content_block_delta\ndata: " + strings.Repeat("x", streaming.MaxStreamRecordBytes) + "\n\n"
		}
	}
	if protocol.name == "ollama_ndjson" {
		malformed = "{\n"
		truncated = "{"
		oversized = func() string {
			return strings.Repeat("x", streaming.MaxStreamRecordBytes+1) + "\n"
		}
	}

	return []struct {
		name    string
		factory func() io.ReadCloser
	}{
		{name: "empty", factory: func() io.ReadCloser { return http.NoBody }},
		{name: "empty_record", factory: func() io.ReadCloser {
			if protocol.name == "ollama_ndjson" {
				return io.NopCloser(strings.NewReader("  \n"))
			}
			return io.NopCloser(strings.NewReader("data:  \n\n"))
		}},
		{name: "no_output_record", factory: func() io.ReadCloser {
			switch protocol.name {
			case "anthropic_sse":
				return io.NopCloser(strings.NewReader("event: ping\ndata: {\"type\":\"ping\"}\n\n"))
			case "ollama_ndjson":
				return io.NopCloser(strings.NewReader("{}\n"))
			default:
				return io.NopCloser(strings.NewReader("data: {}\n\n"))
			}
		}},
		{name: "truncated", factory: func() io.ReadCloser { return io.NopCloser(strings.NewReader(truncated)) }},
		{name: "generic_read_error", factory: func() io.ReadCloser { return startupStreamErrorBody{} }},
		{name: "malformed", factory: func() io.ReadCloser { return io.NopCloser(strings.NewReader(malformed)) }},
		{name: "oversized", factory: func() io.ReadCloser { return io.NopCloser(strings.NewReader(oversized())) }},
	}
}

func TestChatStreamingFailuresBeforeFirstRecordReturnJSON502(t *testing.T) {
	for _, protocol := range streamingStatusProtocols() {
		for _, failure := range startupFailureBodies(protocol) {
			t.Run(protocol.name+"/"+failure.name, func(t *testing.T) {
				var calls, closed, redirectCalls atomic.Int32
				transport := protocolResponseTransport(
					http.StatusOK,
					failure.factory,
					&calls,
					&closed,
					&redirectCalls,
				)
				handler, _ := newStreamingStatusHandler(t, protocol, false, transport, nil)
				recorder := performStreamingStatusRequest(handler)

				if recorder.Code != http.StatusBadGateway {
					t.Fatalf("status = %d, want 502; body=%s", recorder.Code, recorder.Body.String())
				}
				if contentType := recorder.Header().Get("Content-Type"); !strings.HasPrefix(contentType, "application/json") {
					t.Fatalf("Content-Type = %q, want application/json", contentType)
				}
				if strings.Contains(recorder.Body.String(), "data:") || strings.Contains(recorder.Body.String(), "[DONE]") {
					t.Fatalf("startup failure was emitted as an SSE stream: %q", recorder.Body.String())
				}
				var response models.ErrorResponse
				if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
					t.Fatalf("decode JSON error: %v", err)
				}
				if response.Error.Type != "streaming_error" {
					t.Fatalf("error type = %q, want streaming_error", response.Error.Type)
				}
				if response.Error.Message != streaming.ChatStreamErrorMessage {
					t.Fatalf("error message = %q, want safe canonical message", response.Error.Message)
				}
				if strings.Contains(recorder.Body.String(), errStartupStreamRead.Error()) {
					t.Fatalf("upstream read diagnostic leaked to client: %s", recorder.Body.String())
				}
				if calls.Load() != 1 || closed.Load() != 1 {
					t.Fatalf("upstream calls/closed = %d/%d, want 1/1", calls.Load(), closed.Load())
				}
				if redirectCalls.Load() != 0 {
					t.Fatalf("redirect calls = %d, want 0", redirectCalls.Load())
				}
			})
		}
	}
}

func TestChatStreamingFailuresAfterFirstRecordUseOneSSETerminal(t *testing.T) {
	for _, protocol := range streamingStatusProtocols() {
		t.Run(protocol.name, func(t *testing.T) {
			var calls, closed, redirectCalls atomic.Int32
			transport := protocolResponseTransport(
				http.StatusOK,
				func() io.ReadCloser {
					return &startupPayloadThenErrorBody{payload: strings.NewReader(streamingValidPrefix(protocol.name))}
				},
				&calls,
				&closed,
				&redirectCalls,
			)
			handler, _ := newStreamingStatusHandler(t, protocol, false, transport, nil)
			recorder := performStreamingStatusRequest(handler)

			if recorder.Code != http.StatusOK {
				t.Fatalf("status = %d, want 200; body=%s", recorder.Code, recorder.Body.String())
			}
			if contentType := recorder.Header().Get("Content-Type"); !strings.HasPrefix(contentType, "text/event-stream") {
				t.Fatalf("Content-Type = %q, want text/event-stream", contentType)
			}
			if got := strings.Count(recorder.Body.String(), streaming.ChatStreamErrorCode); got != 1 {
				t.Fatalf("canonical error count = %d, want 1; body=%q", got, recorder.Body.String())
			}
			if got := strings.Count(recorder.Body.String(), "data: [DONE]\n\n"); got != 1 {
				t.Fatalf("[DONE] count = %d, want 1; body=%q", got, recorder.Body.String())
			}
			if calls.Load() != 1 || closed.Load() != 1 {
				t.Fatalf("upstream calls/closed = %d/%d, want 1/1", calls.Load(), closed.Load())
			}
		})
	}
}

func streamingValidPrefix(protocol string) string {
	switch protocol {
	case "anthropic_sse":
		return "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg-first\",\"model\":\"claude-test\",\"usage\":{\"input_tokens\":1}}}\n\n"
	case "ollama_ndjson":
		return "{\"model\":\"qwen\",\"message\":{\"role\":\"assistant\",\"content\":\"first\"},\"done\":false}\n"
	default:
		return "data: {\"id\":\"chatcmpl-first\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-test\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"first\"}}]}\n\n"
	}
}
