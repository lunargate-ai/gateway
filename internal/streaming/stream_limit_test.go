package streaming

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/pkg/models"
)

type streamLimitTerminalTranslator struct {
	models.ProviderTranslator
	calls int
}

func (t *streamLimitTerminalTranslator) ParseStreamChunk([]byte) (*models.StreamChunk, error) {
	t.calls++
	return nil, providers.ErrStreamDone
}

func TestReadSSEFrameEnforcesWholeMultilineEventLimit(t *testing.T) {
	boundary := streamLimitMultilineSSEEvent(t, MaxStreamRecordBytes)
	if got := sseFrameRecordSize([]byte(boundary)); got != MaxStreamRecordBytes {
		t.Fatalf("boundary event size = %d, want %d", got, MaxStreamRecordBytes)
	}
	frame, err := readSSEFrame(bufio.NewReader(strings.NewReader(boundary)))
	if err != nil {
		t.Fatalf("read boundary event: %v", err)
	}
	if frame == nil || string(frame) != boundary {
		t.Fatal("boundary event was not preserved byte-for-byte")
	}

	oversized := streamLimitMultilineSSEEvent(t, MaxStreamRecordBytes+1)
	frame, err = readSSEFrame(bufio.NewReader(strings.NewReader(oversized)))
	if !errors.Is(err, ErrStreamRecordTooLarge) {
		t.Fatalf("oversize error = %v, want ErrStreamRecordTooLarge", err)
	}
	if frame != nil {
		t.Fatalf("oversized frame = %d bytes, want nil", len(frame))
	}
}

func TestTranslatedSSEHandlersAcceptBoundaryEvent(t *testing.T) {
	tests := []struct {
		name string
		run  func(http.ResponseWriter, *http.Response, models.ProviderTranslator) error
	}{
		{
			name: "openai compatible",
			run: func(w http.ResponseWriter, response *http.Response, translator models.ProviderTranslator) error {
				return NewHandler().StreamResponse(context.Background(), w, response, translator)
			},
		},
		{
			name: "anthropic",
			run: func(w http.ResponseWriter, response *http.Response, translator models.ProviderTranslator) error {
				return NewHandler().StreamAnthropicResponse(context.Background(), w, response, translator)
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			base := providers.NewOpenAITranslator(config.ProviderConfig{APIKey: "dummy"})
			translator := &streamLimitTerminalTranslator{ProviderTranslator: base}
			response := &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(strings.NewReader(streamLimitMultilineSSEEvent(t, MaxStreamRecordBytes))),
			}
			recorder := httptest.NewRecorder()

			if err := test.run(recorder, response, translator); err != nil {
				t.Fatalf("boundary event failed: %v", err)
			}
			if translator.calls != 1 {
				t.Fatalf("translator calls = %d, want 1", translator.calls)
			}
			if got := recorder.Body.String(); got != "data: [DONE]\n\n" {
				t.Fatalf("client stream = %q, want done frame", got)
			}
		})
	}
}

func TestOversizedFirstChatRecordsDoNotCommitResponse(t *testing.T) {
	tests := []struct {
		name string
		body func(*testing.T) string
		run  func(http.ResponseWriter, *http.Response, models.ProviderTranslator) error
	}{
		{
			name: "openai compatible sse",
			body: func(t *testing.T) string {
				return streamLimitMultilineSSEEvent(t, MaxStreamRecordBytes+1)
			},
			run: func(w http.ResponseWriter, response *http.Response, translator models.ProviderTranslator) error {
				return NewHandler().StreamResponse(context.Background(), w, response, translator)
			},
		},
		{
			name: "anthropic sse",
			body: func(t *testing.T) string {
				return streamLimitMultilineSSEEvent(t, MaxStreamRecordBytes+1)
			},
			run: func(w http.ResponseWriter, response *http.Response, translator models.ProviderTranslator) error {
				return NewHandler().StreamAnthropicResponse(context.Background(), w, response, translator)
			},
		},
		{
			name: "ollama ndjson",
			body: func(*testing.T) string {
				return strings.Repeat("x", MaxStreamRecordBytes+1) + "\n"
			},
			run: func(w http.ResponseWriter, response *http.Response, translator models.ProviderTranslator) error {
				return NewHandler().StreamNDJSONResponse(context.Background(), w, response, translator)
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			base := providers.NewOpenAITranslator(config.ProviderConfig{APIKey: "dummy"})
			translator := &streamLimitTerminalTranslator{ProviderTranslator: base}
			response := &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(strings.NewReader(test.body(t))),
			}
			writer := &failingSSEWriter{}

			err := test.run(writer, response, translator)
			if !errors.Is(err, ErrStreamRecordTooLarge) {
				t.Fatalf("oversize error = %v, want ErrStreamRecordTooLarge", err)
			}
			if translator.calls != 0 {
				t.Fatalf("translator called %d times for oversized record", translator.calls)
			}
			if writer.status != 0 || writer.body.Len() != 0 {
				t.Fatalf("downstream was committed before validation: status=%d body=%q", writer.status, writer.body.String())
			}
		})
	}
}

func TestReadNDJSONRecordEnforcesBoundary(t *testing.T) {
	boundary := strings.Repeat("x", MaxStreamRecordBytes) + "\n"
	record, err := readNDJSONRecord(bufio.NewReader(strings.NewReader(boundary)))
	if err != nil {
		t.Fatalf("read boundary record: %v", err)
	}
	if len(record) != MaxStreamRecordBytes {
		t.Fatalf("boundary record size = %d, want %d", len(record), MaxStreamRecordBytes)
	}

	oversized := strings.Repeat("x", MaxStreamRecordBytes+1) + "\n"
	record, err = readNDJSONRecord(bufio.NewReader(strings.NewReader(oversized)))
	if !errors.Is(err, ErrStreamRecordTooLarge) {
		t.Fatalf("oversize error = %v, want ErrStreamRecordTooLarge", err)
	}
	if record != nil {
		t.Fatalf("oversized record = %d bytes, want nil", len(record))
	}
}

func TestNDJSONHandlerAcceptsBoundaryRecord(t *testing.T) {
	base := providers.NewOpenAITranslator(config.ProviderConfig{APIKey: "dummy"})
	translator := &streamLimitTerminalTranslator{ProviderTranslator: base}
	response := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(strings.Repeat("x", MaxStreamRecordBytes) + "\n")),
	}
	recorder := httptest.NewRecorder()

	if err := NewHandler().StreamNDJSONResponse(context.Background(), recorder, response, translator); err != nil {
		t.Fatalf("boundary NDJSON record failed: %v", err)
	}
	if translator.calls != 1 {
		t.Fatalf("translator calls = %d, want 1", translator.calls)
	}
	if got := recorder.Body.String(); got != "data: [DONE]\n\n" {
		t.Fatalf("client stream = %q, want done frame", got)
	}
}

func TestProxySSEEnforcesLimitWithoutForwardingPartialFrame(t *testing.T) {
	t.Run("boundary", func(t *testing.T) {
		body := nativeStreamLimitJSONEvent(t, MaxStreamRecordBytes)
		response := &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(body))}
		writer := &failingSSEWriter{}

		err := NewHandler().ProxySSE(context.Background(), writer, response, "openai", func(SSEEvent) bool { return true })
		if err != nil {
			t.Fatalf("boundary native event failed: %v", err)
		}
		if writer.body.String() != body {
			t.Fatal("boundary native event was not forwarded byte-for-byte")
		}
	})

	t.Run("oversize", func(t *testing.T) {
		response := &http.Response{
			StatusCode: http.StatusOK,
			Body:       io.NopCloser(strings.NewReader(nativeStreamLimitJSONEvent(t, MaxStreamRecordBytes+1))),
		}
		writer := &failingSSEWriter{}
		observed := false

		err := NewHandler().ProxySSE(context.Background(), writer, response, "openai", func(SSEEvent) bool {
			observed = true
			return true
		})
		if !errors.Is(err, ErrStreamRecordTooLarge) {
			t.Fatalf("oversize error = %v, want ErrStreamRecordTooLarge", err)
		}
		if writer.body.Len() != 0 {
			t.Fatalf("native proxy forwarded %d partial bytes", writer.body.Len())
		}
		if observed {
			t.Fatal("observer saw an oversized incomplete event")
		}
	})

	t.Run("unterminated", func(t *testing.T) {
		response := &http.Response{
			StatusCode: http.StatusOK,
			Body:       io.NopCloser(strings.NewReader("event: response.completed\ndata: {\"type\":\"response.completed\"}")),
		}
		writer := &failingSSEWriter{}

		err := NewHandler().ProxySSE(context.Background(), writer, response, "openai", func(SSEEvent) bool { return true })
		if !errors.Is(err, ErrUpstreamStreamIncomplete) {
			t.Fatalf("unterminated error = %v, want ErrUpstreamStreamIncomplete", err)
		}
		if writer.body.Len() != 0 {
			t.Fatalf("native proxy forwarded %d unterminated bytes", writer.body.Len())
		}
	})
}

func TestProxySSERejectsOversizedTransformedEventBeforeWrite(t *testing.T) {
	response := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader("event: response.completed\ndata: {}\n\n")),
	}
	writer := &failingSSEWriter{}

	err := NewHandler().ProxySSEWithDataTransformer(
		context.Background(),
		writer,
		response,
		"openai",
		func(SSEEvent) bool { return true },
		func(SSEEvent) ([]byte, error) {
			return bytes.Repeat([]byte{'x'}, MaxStreamRecordBytes), nil
		},
	)
	if !errors.Is(err, ErrStreamRecordTooLarge) {
		t.Fatalf("transformed oversize error = %v, want ErrStreamRecordTooLarge", err)
	}
	if writer.body.Len() != 0 {
		t.Fatalf("native proxy forwarded %d oversized transformed bytes", writer.body.Len())
	}
}

func streamLimitMultilineSSEEvent(t *testing.T, recordSize int) string {
	t.Helper()
	prefix := "event: limit\ndata: first\ndata: "
	const recordLineEnding = "\n"
	fixedSize := len(prefix) + len(recordLineEnding)
	if recordSize < fixedSize {
		t.Fatalf("record size %d is smaller than fixture overhead %d", recordSize, fixedSize)
	}
	return prefix + strings.Repeat("x", recordSize-fixedSize) + recordLineEnding + "\n"
}

func nativeStreamLimitJSONEvent(t *testing.T, recordSize int) string {
	t.Helper()
	const prefix = "event: limit\ndata: {\"padding\":\""
	const suffix = "\"}\n"
	fixedSize := len(prefix) + len(suffix)
	if recordSize < fixedSize {
		t.Fatalf("record size %d is smaller than native JSON fixture overhead %d", recordSize, fixedSize)
	}
	return prefix + strings.Repeat("x", recordSize-fixedSize) + suffix + "\n"
}

func assertSafeOversizeChatFailure(t *testing.T, body string) {
	t.Helper()
	frames := chatSSEDataFrames(body)
	if len(frames) != 2 || frames[1] != "[DONE]" {
		t.Fatalf("Chat failure frames = %#v, want error then done", frames)
	}
	var errorResponse models.ErrorResponse
	if err := json.Unmarshal([]byte(frames[0]), &errorResponse); err != nil {
		t.Fatalf("decode Chat error frame: %v", err)
	}
	if errorResponse.Error.Message != ChatStreamErrorMessage || errorResponse.Error.Type != ChatStreamErrorType {
		t.Fatalf("Chat error = %#v", errorResponse.Error)
	}
	if errorResponse.Error.Code == nil || *errorResponse.Error.Code != ChatStreamErrorCode {
		t.Fatalf("Chat error code = %#v, want %q", errorResponse.Error.Code, ChatStreamErrorCode)
	}
}
