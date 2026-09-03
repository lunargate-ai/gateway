package streaming

import (
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

func TestProviderStreamErrorsReachChatClientSafely(t *testing.T) {
	tests := []struct {
		name         string
		body         string
		secret       string
		wantProvider string
		wantType     string
		run          func(*httptest.ResponseRecorder, *http.Response) error
	}{
		{
			name:         "openai compatible sse",
			body:         "data: {\"error\":{\"message\":\"openai diagnostic secret\",\"type\":\"server_error\"}}\n\n",
			secret:       "openai diagnostic secret",
			wantProvider: "openai",
			wantType:     "server_error",
			run: func(w *httptest.ResponseRecorder, resp *http.Response) error {
				translator := providers.NewOpenAITranslator(config.ProviderConfig{APIKey: "dummy"})
				return NewHandler().StreamResponse(context.Background(), w, resp, translator)
			},
		},
		{
			name:         "anthropic error event",
			body:         "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"anthropic diagnostic secret\"}}\n\n",
			secret:       "anthropic diagnostic secret",
			wantProvider: "anthropic",
			wantType:     "overloaded_error",
			run: func(w *httptest.ResponseRecorder, resp *http.Response) error {
				base := providers.NewAnthropicTranslator(config.ProviderConfig{APIKey: "dummy"})
				translator := providers.NewAnthropicStreamTranslator(base)
				return NewHandler().StreamAnthropicResponse(context.Background(), w, resp, translator)
			},
		},
		{
			name:         "ollama ndjson",
			body:         "{\"error\":\"ollama diagnostic secret\"}\n",
			secret:       "ollama diagnostic secret",
			wantProvider: "ollama",
			wantType:     "upstream_error",
			run: func(w *httptest.ResponseRecorder, resp *http.Response) error {
				base := providers.NewOllamaTranslator(config.ProviderConfig{BaseURL: "http://localhost:11434"})
				translator := providers.NewOllamaStreamTranslator(base)
				return NewHandler().StreamNDJSONResponse(context.Background(), w, resp, translator)
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			providerResp := &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(strings.NewReader(test.body)),
			}
			recorder := httptest.NewRecorder()

			err := test.run(recorder, providerResp)
			var providerErr *providers.ProviderError
			if !errors.As(err, &providerErr) {
				t.Fatalf("error = %v, want wrapped ProviderError", err)
			}
			if providerErr.Provider != test.wantProvider || providerErr.Type != test.wantType || providerErr.Message != test.secret {
				t.Fatalf("ProviderError = %#v, want provider=%q type=%q message=%q", providerErr, test.wantProvider, test.wantType, test.secret)
			}

			if strings.Contains(recorder.Body.String(), test.secret) {
				t.Fatalf("provider diagnostic leaked to Chat client: %s", recorder.Body.String())
			}
			frames := chatSSEDataFrames(recorder.Body.String())
			if len(frames) != 2 {
				t.Fatalf("frames = %#v, want error then done", frames)
			}
			var clientErr models.ErrorResponse
			if err := json.Unmarshal([]byte(frames[0]), &clientErr); err != nil {
				t.Fatalf("decode client error: %v; payload=%s", err, frames[0])
			}
			if clientErr.Error.Message != ChatStreamErrorMessage || clientErr.Error.Type != ChatStreamErrorType {
				t.Fatalf("client error = %#v", clientErr.Error)
			}
			if clientErr.Error.Code == nil || *clientErr.Error.Code != ChatStreamErrorCode {
				t.Fatalf("client error code = %#v, want %q", clientErr.Error.Code, ChatStreamErrorCode)
			}
			if frames[1] != "[DONE]" {
				t.Fatalf("terminal frame = %q, want [DONE]", frames[1])
			}
		})
	}
}

func chatSSEDataFrames(body string) []string {
	frames := make([]string, 0, 2)
	for _, rawFrame := range strings.Split(body, "\n\n") {
		for _, line := range strings.Split(rawFrame, "\n") {
			if strings.HasPrefix(line, "data:") {
				frames = append(frames, strings.TrimSpace(strings.TrimPrefix(line, "data:")))
			}
		}
	}
	return frames
}
