package providers

import (
	"errors"
	"net/http"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestStreamTranslatorsSurfaceProviderErrors(t *testing.T) {
	tests := []struct {
		name         string
		translator   func() models.ProviderTranslator
		payload      string
		wantProvider string
		wantType     string
		wantMessage  string
	}{
		{
			name: "openai compatible error envelope",
			translator: func() models.ProviderTranslator {
				return NewOpenAITranslator(config.ProviderConfig{APIKey: "dummy"})
			},
			payload:      `{"error":{"message":"openai diagnostic secret","type":"server_error","code":"internal_error"}}`,
			wantProvider: "openai",
			wantType:     "server_error",
			wantMessage:  "openai diagnostic secret",
		},
		{
			name: "openai responses error event",
			translator: func() models.ProviderTranslator {
				return NewOpenAITranslator(config.ProviderConfig{APIKey: "dummy"})
			},
			payload:      `{"type":"error","code":"server_error","message":"responses diagnostic secret"}`,
			wantProvider: "openai",
			wantType:     "server_error",
			wantMessage:  "responses diagnostic secret",
		},
		{
			name: "anthropic error event",
			translator: func() models.ProviderTranslator {
				base := NewAnthropicTranslator(config.ProviderConfig{APIKey: "dummy"})
				return NewAnthropicStreamTranslator(base)
			},
			payload:      `{"type":"error","error":{"type":"overloaded_error","message":"anthropic diagnostic secret"}}`,
			wantProvider: "anthropic",
			wantType:     "overloaded_error",
			wantMessage:  "anthropic diagnostic secret",
		},
		{
			name: "ollama error",
			translator: func() models.ProviderTranslator {
				base := NewOllamaTranslator(config.ProviderConfig{BaseURL: "http://localhost:11434"})
				return NewOllamaStreamTranslator(base)
			},
			payload:      `{"error":"ollama diagnostic secret"}`,
			wantProvider: "ollama",
			wantType:     "upstream_error",
			wantMessage:  "ollama diagnostic secret",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			chunk, err := test.translator().ParseStreamChunk([]byte(test.payload))
			if chunk != nil {
				t.Fatalf("chunk = %#v, want nil", chunk)
			}
			var providerErr *ProviderError
			if !errors.As(err, &providerErr) {
				t.Fatalf("error = %v, want ProviderError", err)
			}
			if providerErr.StatusCode != http.StatusBadGateway {
				t.Errorf("status = %d, want %d", providerErr.StatusCode, http.StatusBadGateway)
			}
			if providerErr.Provider != test.wantProvider {
				t.Errorf("provider = %q, want %q", providerErr.Provider, test.wantProvider)
			}
			if providerErr.Type != test.wantType {
				t.Errorf("type = %q, want %q", providerErr.Type, test.wantType)
			}
			if providerErr.Message != test.wantMessage {
				t.Errorf("message = %q, want %q", providerErr.Message, test.wantMessage)
			}
		})
	}
}

func TestOpenAIStreamTranslatorIgnoresNonErrorExtensionValues(t *testing.T) {
	translator := NewOpenAITranslator(config.ProviderConfig{APIKey: "dummy"})

	for _, errorValue := range []string{"null", "false", "0", "[]"} {
		t.Run(errorValue, func(t *testing.T) {
			payload := `{"id":"chatcmpl_1","object":"chat.completion.chunk","created":1,"model":"gpt","choices":[],"error":` + errorValue + `}`

			chunk, err := translator.ParseStreamChunk([]byte(payload))
			if err != nil {
				t.Fatalf("ParseStreamChunk returned error: %v", err)
			}
			if chunk == nil || chunk.ID != "chatcmpl_1" {
				t.Fatalf("chunk = %#v, want regular Chat chunk", chunk)
			}
		})
	}
}
