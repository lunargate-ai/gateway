package streaming

import (
	"bytes"
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/providers"
)

func TestStreamEntryPointsBoundUpstreamErrorBodies(t *testing.T) {
	const secret = "must-not-be-read-or-returned"
	body := strings.Repeat("x", maxUpstreamErrorBodyBytes+1) + secret

	for _, test := range upstreamErrorEntryPoints() {
		t.Run(test.name, func(t *testing.T) {
			tracked := newTrackedErrorBody(body)
			response := &http.Response{StatusCode: http.StatusInternalServerError, Body: tracked}

			err := test.run(httptest.NewRecorder(), response)
			var providerErr *providers.ProviderError
			if !errors.As(err, &providerErr) {
				t.Fatalf("error = %v, want ProviderError", err)
			}
			if providerErr.StatusCode != http.StatusInternalServerError || providerErr.Provider != test.provider {
				t.Fatalf("ProviderError = %#v, want status=%d provider=%q", providerErr, http.StatusInternalServerError, test.provider)
			}
			if providerErr.Type != "upstream_response_too_large" || providerErr.Message != "upstream error response exceeds the 1 MiB limit" {
				t.Fatalf("ProviderError = %#v, want neutral size error", providerErr)
			}
			if strings.Contains(providerErr.Error(), secret) {
				t.Fatalf("ProviderError leaked unread suffix: %v", providerErr)
			}
			if tracked.read != maxUpstreamErrorBodyBytes+1 {
				t.Fatalf("upstream bytes read = %d, want %d", tracked.read, maxUpstreamErrorBodyBytes+1)
			}
			if !tracked.closed {
				t.Fatal("upstream body was not closed")
			}
		})
	}
}

func TestStreamEntryPointsAcceptUpstreamErrorBodyAtBoundary(t *testing.T) {
	prefix := `{"error":{"type":"boundary_error","message":"`
	suffix := `"}}`
	message := strings.Repeat("x", maxUpstreamErrorBodyBytes-len(prefix)-len(suffix))
	body := prefix + message + suffix
	if len(body) != maxUpstreamErrorBodyBytes {
		t.Fatalf("fixture size = %d, want %d", len(body), maxUpstreamErrorBodyBytes)
	}

	for _, test := range upstreamErrorEntryPoints() {
		t.Run(test.name, func(t *testing.T) {
			tracked := newTrackedErrorBody(body)
			response := &http.Response{StatusCode: http.StatusBadRequest, Body: tracked}

			err := test.run(httptest.NewRecorder(), response)
			var providerErr *providers.ProviderError
			if !errors.As(err, &providerErr) {
				t.Fatalf("error = %v, want ProviderError", err)
			}
			if providerErr.Type != "boundary_error" || providerErr.Message != message {
				t.Fatalf("ProviderError type=%q message-bytes=%d, want boundary error with %d-byte message", providerErr.Type, len(providerErr.Message), len(message))
			}
			if tracked.read != maxUpstreamErrorBodyBytes {
				t.Fatalf("upstream bytes read = %d, want %d", tracked.read, maxUpstreamErrorBodyBytes)
			}
			if !tracked.closed {
				t.Fatal("upstream body was not closed")
			}
		})
	}
}

type upstreamErrorEntryPoint struct {
	name     string
	provider string
	run      func(http.ResponseWriter, *http.Response) error
}

func upstreamErrorEntryPoints() []upstreamErrorEntryPoint {
	openAI := providers.NewOpenAITranslator(config.ProviderConfig{APIKey: "dummy"})
	anthropic := providers.NewAnthropicStreamTranslator(providers.NewAnthropicTranslator(config.ProviderConfig{APIKey: "dummy"}))
	ollama := providers.NewOllamaStreamTranslator(providers.NewOllamaTranslator(config.ProviderConfig{BaseURL: "http://localhost:11434"}))
	return []upstreamErrorEntryPoint{
		{
			name:     "native sse proxy",
			provider: "openai",
			run: func(w http.ResponseWriter, response *http.Response) error {
				return NewHandler().ProxySSE(context.Background(), w, response, "openai", nil)
			},
		},
		{
			name:     "translated sse",
			provider: openAI.Name(),
			run: func(w http.ResponseWriter, response *http.Response) error {
				return NewHandler().StreamResponse(context.Background(), w, response, openAI)
			},
		},
		{
			name:     "anthropic sse",
			provider: anthropic.Name(),
			run: func(w http.ResponseWriter, response *http.Response) error {
				return NewHandler().StreamAnthropicResponse(context.Background(), w, response, anthropic)
			},
		},
		{
			name:     "ndjson",
			provider: ollama.Name(),
			run: func(w http.ResponseWriter, response *http.Response) error {
				return NewHandler().StreamNDJSONResponse(context.Background(), w, response, ollama)
			},
		},
	}
}

type trackedErrorBody struct {
	reader *bytes.Reader
	read   int
	closed bool
}

func newTrackedErrorBody(body string) *trackedErrorBody {
	return &trackedErrorBody{reader: bytes.NewReader([]byte(body))}
}

func (b *trackedErrorBody) Read(p []byte) (int, error) {
	n, err := b.reader.Read(p)
	b.read += n
	return n, err
}

func (b *trackedErrorBody) Close() error {
	b.closed = true
	return nil
}
