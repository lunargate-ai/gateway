package streaming

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/pkg/models"
)

var errInjectedUpstreamRead = errors.New("injected upstream read failure")

type chatStreamProtocolCase struct {
	name          string
	validFirst    string
	malformed     string
	truncated     string
	oversized     func() string
	streamHandler func(context.Context, http.ResponseWriter, *http.Response) error
}

func chatStreamProtocolCases() []chatStreamProtocolCase {
	return []chatStreamProtocolCase{
		{
			name:       "openai_sse",
			validFirst: "data: {\"id\":\"chatcmpl-first\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-test\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"first\"}}]}\n\n",
			malformed:  "data: {\n\n",
			truncated:  "data: {",
			oversized: func() string {
				return "data: " + strings.Repeat("x", MaxStreamRecordBytes) + "\n\n"
			},
			streamHandler: func(ctx context.Context, w http.ResponseWriter, response *http.Response) error {
				translator := providers.NewOpenAITranslator(config.ProviderConfig{APIKey: "dummy"})
				return NewHandler().StreamResponse(ctx, w, response, translator)
			},
		},
		{
			name:       "anthropic_sse",
			validFirst: "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg-first\",\"model\":\"claude-test\",\"usage\":{\"input_tokens\":1}}}\n\n",
			malformed:  "event: content_block_delta\ndata: {\n\n",
			truncated:  "event: content_block_delta\ndata: {",
			oversized: func() string {
				return "event: content_block_delta\ndata: " + strings.Repeat("x", MaxStreamRecordBytes) + "\n\n"
			},
			streamHandler: func(ctx context.Context, w http.ResponseWriter, response *http.Response) error {
				base := providers.NewAnthropicTranslator(config.ProviderConfig{APIKey: "dummy"})
				return NewHandler().StreamAnthropicResponse(ctx, w, response, providers.NewAnthropicStreamTranslator(base))
			},
		},
		{
			name:       "ollama_ndjson",
			validFirst: "{\"model\":\"qwen\",\"message\":{\"role\":\"assistant\",\"content\":\"first\"},\"done\":false}\n",
			malformed:  "{\n",
			truncated:  "{",
			oversized: func() string {
				return strings.Repeat("x", MaxStreamRecordBytes+1) + "\n"
			},
			streamHandler: func(ctx context.Context, w http.ResponseWriter, response *http.Response) error {
				base := providers.NewOllamaTranslator(config.ProviderConfig{BaseURL: "http://localhost:11434"})
				return NewHandler().StreamNDJSONResponse(ctx, w, response, providers.NewOllamaStreamTranslator(base))
			},
		},
	}
}

type trackedReadCloser struct {
	reader io.Reader
	closed bool
}

func (r *trackedReadCloser) Read(p []byte) (int, error) { return r.reader.Read(p) }

func (r *trackedReadCloser) Close() error {
	r.closed = true
	return nil
}

type payloadThenErrorReader struct {
	payload *strings.Reader
	err     error
}

func newPayloadThenErrorReader(payload string, err error) *payloadThenErrorReader {
	return &payloadThenErrorReader{payload: strings.NewReader(payload), err: err}
}

func (r *payloadThenErrorReader) Read(p []byte) (int, error) {
	if r.payload.Len() > 0 {
		return r.payload.Read(p)
	}
	return 0, r.err
}

func TestTranslatedChatStreamsRejectFailuresBeforeFirstRecord(t *testing.T) {
	for _, protocol := range chatStreamProtocolCases() {
		failures := []struct {
			name   string
			reader io.Reader
		}{
			{name: "empty", reader: strings.NewReader("")},
			{name: "empty_record", reader: strings.NewReader(emptyChatStreamRecord(protocol.name))},
			{name: "no_output_record", reader: strings.NewReader(noOutputChatStreamRecord(protocol.name))},
			{name: "truncated", reader: strings.NewReader(protocol.truncated)},
			{name: "generic_read_error", reader: newPayloadThenErrorReader("", errInjectedUpstreamRead)},
			{name: "malformed", reader: strings.NewReader(protocol.malformed)},
			{name: "oversized", reader: strings.NewReader(protocol.oversized())},
		}
		for _, failure := range failures {
			t.Run(protocol.name+"/"+failure.name, func(t *testing.T) {
				body := &trackedReadCloser{reader: failure.reader}
				writer := &failingSSEWriter{}
				err := protocol.streamHandler(context.Background(), writer, &http.Response{
					StatusCode: http.StatusOK,
					Body:       body,
				})

				if err == nil {
					t.Fatal("failure before first record returned nil")
				}
				if writer.status != 0 || writer.body.Len() != 0 || writer.flushes != 0 {
					t.Fatalf("downstream committed before validation: status=%d flushes=%d body=%q", writer.status, writer.flushes, writer.body.String())
				}
				if !body.closed {
					t.Fatal("upstream body was not closed")
				}
			})
		}
	}
}

func emptyChatStreamRecord(protocol string) string {
	if protocol == "ollama_ndjson" {
		return "  \n"
	}
	return "data:  \n\n"
}

func noOutputChatStreamRecord(protocol string) string {
	switch protocol {
	case "anthropic_sse":
		return "event: ping\ndata: {\"type\":\"ping\"}\n\n"
	case "ollama_ndjson":
		return "{}\n"
	default:
		return "data: {}\n\n"
	}
}

func TestTranslatedChatStreamsTerminateFailuresAfterFirstRecordExactlyOnce(t *testing.T) {
	for _, protocol := range chatStreamProtocolCases() {
		failures := []struct {
			name   string
			reader io.Reader
		}{
			{name: "premature_eof", reader: strings.NewReader(protocol.validFirst)},
			{name: "truncated", reader: strings.NewReader(protocol.validFirst + protocol.truncated)},
			{name: "generic_read_error", reader: newPayloadThenErrorReader(protocol.validFirst, errInjectedUpstreamRead)},
			{name: "malformed", reader: strings.NewReader(protocol.validFirst + protocol.malformed)},
			{name: "oversized", reader: strings.NewReader(protocol.validFirst + protocol.oversized())},
		}
		for _, failure := range failures {
			t.Run(protocol.name+"/"+failure.name, func(t *testing.T) {
				body := &trackedReadCloser{reader: failure.reader}
				writer := &failingSSEWriter{}
				err := protocol.streamHandler(context.Background(), writer, &http.Response{
					StatusCode: http.StatusOK,
					Body:       body,
				})

				if err == nil {
					t.Fatal("failure after first record returned nil")
				}
				if writer.status != http.StatusOK {
					t.Fatalf("downstream status = %d, want 200", writer.status)
				}
				assertOneCanonicalChatFailure(t, writer.body.String())
				if !body.closed {
					t.Fatal("upstream body was not closed")
				}
			})
		}
	}
}

type cancelAfterPayloadReader struct {
	payload *strings.Reader
	cancel  context.CancelFunc
}

func (r *cancelAfterPayloadReader) Read(p []byte) (int, error) {
	if r.payload.Len() == 0 {
		return 0, io.EOF
	}
	n, err := r.payload.Read(p)
	r.cancel()
	return n, err
}

func TestTranslatedChatStreamsDoNotWriteTerminalAfterCancellation(t *testing.T) {
	for _, protocol := range chatStreamProtocolCases() {
		t.Run(protocol.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			body := &trackedReadCloser{reader: &cancelAfterPayloadReader{
				payload: strings.NewReader(protocol.validFirst),
				cancel:  cancel,
			}}
			writer := &failingSSEWriter{}
			err := protocol.streamHandler(ctx, writer, &http.Response{
				StatusCode: http.StatusOK,
				Body:       body,
			})

			if !errors.Is(err, context.Canceled) {
				t.Fatalf("error = %v, want context.Canceled", err)
			}
			if strings.Contains(writer.body.String(), ChatStreamErrorCode) || strings.Contains(writer.body.String(), "[DONE]") {
				t.Fatalf("canceled stream gained synthetic terminal: %q", writer.body.String())
			}
			if !body.closed {
				t.Fatal("upstream body was not closed")
			}
		})
	}
}

func TestOpenAIExtensionOnlyRecordIsPreservedBeforeFailure(t *testing.T) {
	translator := providers.NewOpenAITranslator(config.ProviderConfig{APIKey: "dummy"})
	writer := &failingSSEWriter{}
	err := NewHandler().StreamResponse(context.Background(), writer, &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader("data: {\"vendor_extension\":true}\n\n")),
	}, translator)

	if !errors.Is(err, ErrUpstreamStreamIncomplete) {
		t.Fatalf("error = %v, want ErrUpstreamStreamIncomplete", err)
	}
	if !strings.Contains(writer.body.String(), `"vendor_extension":true`) {
		t.Fatalf("extension-only frame was dropped: %q", writer.body.String())
	}
	assertOneCanonicalChatFailure(t, writer.body.String())
}

func assertOneCanonicalChatFailure(t *testing.T, body string) {
	t.Helper()
	frames := chatSSEDataFrames(body)
	if len(frames) < 3 || frames[len(frames)-1] != "[DONE]" {
		t.Fatalf("frames = %#v, want data then error then [DONE]", frames)
	}
	if got := strings.Count(body, "data: [DONE]\n\n"); got != 1 {
		t.Fatalf("[DONE] count = %d, want 1; body=%q", got, body)
	}
	errorFrames := 0
	for _, frame := range frames {
		var response models.ErrorResponse
		if json.Unmarshal([]byte(frame), &response) != nil || response.Error.Code == nil {
			continue
		}
		if *response.Error.Code == ChatStreamErrorCode {
			errorFrames++
		}
	}
	if errorFrames != 1 {
		t.Fatalf("canonical error frames = %d, want 1; body=%q", errorFrames, body)
	}
}
