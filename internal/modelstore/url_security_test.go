package modelstore

import (
	"bytes"
	"context"
	"errors"
	"net/http"
	"strings"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

func TestFetchModelsPreservesBaseQueryButRedactsTransportError(t *testing.T) {
	const baseURL = "https://url-user:url-password@private-provider.example/root/v1?api_key=query-secret#fragment-secret"
	providerConfigs := map[string]config.ProviderConfig{
		"custom": {
			Type:         "openai",
			BaseURL:      baseURL,
			DefaultModel: "local-default",
			Models: config.ProviderModelsConfig{
				Mode: "fetch",
			},
		},
	}
	store := NewStore(providers.NewRegistry(providerConfigs), providerConfigs)
	store.client.Transport = modelstoreURLRoundTripFunc(func(request *http.Request) (*http.Response, error) {
		if got, want := request.URL.Path, "/root/v1/models"; got != want {
			t.Fatalf("models path = %q, want %q", got, want)
		}
		if got, want := request.URL.RawQuery, "api_key=query-secret"; got != want {
			t.Fatalf("models query = %q, want %q", got, want)
		}
		return nil, errors.New("connection-refused-category")
	})

	var logs bytes.Buffer
	previousLogger := log.Logger
	log.Logger = zerolog.New(&logs)
	t.Cleanup(func() { log.Logger = previousLogger })

	models := store.AllModels(context.Background())
	if !hasModel(models, "custom/local-default") {
		t.Fatalf("models = %#v, want local fallback", models)
	}
	logged := logs.String()
	for _, secret := range []string{"url-user", "url-password", "query-secret", "fragment-secret"} {
		if strings.Contains(logged, secret) {
			t.Fatalf("model fetch log leaked %q: %s", secret, logged)
		}
	}
	for _, useful := range []string{"connection-refused-category", "private-provider.example", "/root/v1/models"} {
		if !strings.Contains(logged, useful) {
			t.Fatalf("model fetch log lost %q: %s", useful, logged)
		}
	}
}

type modelstoreURLRoundTripFunc func(*http.Request) (*http.Response, error)

func (f modelstoreURLRoundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return f(request)
}
