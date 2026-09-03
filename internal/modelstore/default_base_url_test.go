package modelstore

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/providers"
)

func TestFetchModelsUsesProviderEffectiveDefaultBaseURL(t *testing.T) {
	tests := []struct {
		name         string
		providerID   string
		wantURL      string
		responseBody string
		wantModel    string
	}{
		{
			name:         "OpenAI",
			providerID:   "openai",
			wantURL:      "https://api.openai.com/v1/models",
			responseBody: `{"object":"list","data":[{"id":"remote-openai"}]}`,
			wantModel:    "openai/remote-openai",
		},
		{
			name:         "Ollama",
			providerID:   "ollama",
			wantURL:      "http://localhost:11434/api/tags",
			responseBody: `{"models":[{"name":"remote-ollama"}]}`,
			wantModel:    "ollama/remote-ollama",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			providerConfigs := map[string]config.ProviderConfig{
				test.providerID: {
					Models: config.ProviderModelsConfig{Mode: "fetch"},
				},
			}
			store := NewStore(providers.NewRegistry(providerConfigs), providerConfigs)
			store.client.Transport = modelstoreURLRoundTripFunc(func(request *http.Request) (*http.Response, error) {
				if got := request.URL.String(); got != test.wantURL {
					t.Fatalf("model discovery URL = %q, want %q", got, test.wantURL)
				}
				return &http.Response{
					StatusCode: http.StatusOK,
					Header:     make(http.Header),
					Body:       io.NopCloser(strings.NewReader(test.responseBody)),
					Request:    request,
				}, nil
			})

			models := store.AllModels(context.Background())
			if !hasModel(models, test.wantModel) {
				t.Fatalf("models = %#v, want %q", models, test.wantModel)
			}
		})
	}
}
