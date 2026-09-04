package api

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/internal/streaming"
)

var errTranslatedResponsesRead = errors.New("injected translated Responses read failure")

type translatedResponsesReadErrorBody struct {
	payload *strings.Reader
}

func (b *translatedResponsesReadErrorBody) Read(p []byte) (int, error) {
	if b.payload.Len() > 0 {
		return b.payload.Read(p)
	}
	return 0, errTranslatedResponsesRead
}

func (*translatedResponsesReadErrorBody) Close() error { return nil }

func TestTranslatedResponsesStreamEmitsOneFailureAfterUpstreamBreak(t *testing.T) {
	tests := []struct {
		name    string
		payload string
		run     func(http.ResponseWriter, *http.Response) error
	}{
		{
			name:    "openai_sse",
			payload: "data: {\"id\":\"chatcmpl-first\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-test\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"first\"}}]}\n\n",
			run: func(w http.ResponseWriter, response *http.Response) error {
				translator := providers.NewOpenAITranslator(config.ProviderConfig{APIKey: "dummy"})
				return streaming.NewHandler().StreamResponse(context.Background(), w, response, translator)
			},
		},
		{
			name:    "anthropic_sse",
			payload: "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg-first\",\"model\":\"claude-test\",\"usage\":{\"input_tokens\":1}}}\n\n",
			run: func(w http.ResponseWriter, response *http.Response) error {
				base := providers.NewAnthropicTranslator(config.ProviderConfig{APIKey: "dummy"})
				return streaming.NewHandler().StreamAnthropicResponse(context.Background(), w, response, providers.NewAnthropicStreamTranslator(base))
			},
		},
		{
			name:    "ollama_ndjson",
			payload: "{\"model\":\"qwen\",\"message\":{\"role\":\"assistant\",\"content\":\"first\"},\"done\":false}\n",
			run: func(w http.ResponseWriter, response *http.Response) error {
				base := providers.NewOllamaTranslator(config.ProviderConfig{BaseURL: "http://localhost:11434"})
				return streaming.NewHandler().StreamNDJSONResponse(context.Background(), w, response, providers.NewOllamaStreamTranslator(base))
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			recorder := httptest.NewRecorder()
			proxy := newResponsesStreamProxy(recorder)
			streamErr := test.run(proxy, &http.Response{
				StatusCode: http.StatusOK,
				Body: &translatedResponsesReadErrorBody{
					payload: strings.NewReader(test.payload),
				},
			})
			if !errors.Is(streamErr, errTranslatedResponsesRead) {
				t.Fatalf("stream error = %v, want injected read failure", streamErr)
			}
			proxy.RecordStreamError(streamErr)
			if err := proxy.finalize(); err != nil {
				t.Fatalf("finalize: %v", err)
			}

			body := recorder.Body.String()
			if got := strings.Count(body, "event: response.failed\n"); got != 1 {
				t.Fatalf("response.failed count = %d, want 1; body=%q", got, body)
			}
			if strings.Contains(body, "event: response.completed\n") || strings.Contains(body, "event: response.incomplete\n") {
				t.Fatalf("broken translated stream emitted successful terminal: %q", body)
			}
		})
	}
}

func TestTranslatedResponsesFailureBeforeFirstRecordReturnsJSON502(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {")
	}))
	defer upstream.Close()

	handler := newResponsesWebSocketTestHandlerWithUpstreamType(upstream.URL, requestTypeChatCompletions)
	defer handler.cache.Stop()
	recorder := httptest.NewRecorder()
	handler.Responses(recorder, httptest.NewRequest(
		http.MethodPost,
		"/v1/responses",
		strings.NewReader(`{"model":"lunargate/auto","input":"hello","stream":true}`),
	))

	if recorder.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502; body=%s", recorder.Code, recorder.Body.String())
	}
	if contentType := recorder.Header().Get("Content-Type"); !strings.HasPrefix(contentType, "application/json") {
		t.Fatalf("Content-Type = %q, want application/json", contentType)
	}
	if strings.Contains(recorder.Body.String(), "event: response.") || strings.Contains(recorder.Body.String(), "data:") {
		t.Fatalf("startup failure was emitted as Responses SSE: %q", recorder.Body.String())
	}
}
