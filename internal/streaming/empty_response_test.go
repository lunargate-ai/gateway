package streaming

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestTranslatedStreamHandlersRejectEmptyProviderResponse(t *testing.T) {
	translator := providers.NewOpenAITranslator(config.ProviderConfig{
		APIKey: "dummy",
	})
	tests := []struct {
		name string
		run  func(http.ResponseWriter, *http.Response, models.ProviderTranslator) error
	}{
		{
			name: "sse",
			run: func(w http.ResponseWriter, response *http.Response, translator models.ProviderTranslator) error {
				return NewHandler().StreamResponse(context.Background(), w, response, translator)
			},
		},
		{
			name: "anthropic sse",
			run: func(w http.ResponseWriter, response *http.Response, translator models.ProviderTranslator) error {
				return NewHandler().StreamAnthropicResponse(context.Background(), w, response, translator)
			},
		},
		{
			name: "ndjson",
			run: func(w http.ResponseWriter, response *http.Response, translator models.ProviderTranslator) error {
				return NewHandler().StreamNDJSONResponse(context.Background(), w, response, translator)
			},
		},
	}
	responses := []struct {
		name     string
		response *http.Response
	}{
		{name: "nil response"},
		{name: "nil body", response: &http.Response{StatusCode: http.StatusOK}},
	}

	for _, test := range tests {
		for _, response := range responses {
			t.Run(test.name+"/"+response.name, func(t *testing.T) {
				recorder := httptest.NewRecorder()
				err := test.run(recorder, response.response, translator)

				if !errors.Is(err, ErrUpstreamStreamEmpty) {
					t.Fatalf("error = %v, want ErrUpstreamStreamEmpty", err)
				}
				if recorder.Code != http.StatusOK || recorder.Body.Len() != 0 {
					t.Fatalf("downstream response = status %d body %q, want untouched", recorder.Code, recorder.Body.String())
				}
			})
		}
	}
}
