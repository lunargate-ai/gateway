package api

import (
	"errors"
	"net/http"
	"strings"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/pkg/models"
)

type nilChatResponseTranslator struct {
	models.ProviderTranslator
}

func (nilChatResponseTranslator) ParseResponse(*http.Response) (*models.UnifiedResponse, error) {
	return nil, nil
}

type nilEmbeddingsResponseTranslator struct {
	embeddingsTranslator
}

func (nilEmbeddingsResponseTranslator) ParseEmbeddingsResponse(*http.Response) (*models.EmbeddingsResponse, error) {
	return nil, nil
}

func TestProviderResponseParserRejectsNilTranslatorResult(t *testing.T) {
	base := providers.NewOpenAITranslator(config.ProviderConfig{})
	tests := []struct {
		name string
		call func() error
		want string
	}{
		{
			name: "chat",
			call: func() error {
				_, err := parseChatProviderResponse(
					nilChatResponseTranslator{ProviderTranslator: base},
					&http.Response{},
				)
				return err
			},
			want: "no chat response",
		},
		{
			name: "embeddings",
			call: func() error {
				_, err := parseEmbeddingsProviderResponse(
					nilEmbeddingsResponseTranslator{embeddingsTranslator: base},
					&http.Response{},
				)
				return err
			},
			want: "no embeddings response",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := test.call()
			var parseErr *providerResponseParseError
			if !errors.As(err, &parseErr) {
				t.Fatalf("error = %v, want providerResponseParseError", err)
			}
			if !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error = %q, want %q", err, test.want)
			}
		})
	}
}
