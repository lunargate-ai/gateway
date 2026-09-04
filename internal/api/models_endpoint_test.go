package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/health"
	"github.com/lunargate-ai/gateway/internal/middleware"
	"github.com/lunargate-ai/gateway/internal/modelstore"
	"github.com/lunargate-ai/gateway/internal/observability"
	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/internal/resilience"
	"github.com/lunargate-ai/gateway/internal/routing"
	"github.com/lunargate-ai/gateway/internal/streaming"
	"github.com/lunargate-ai/gateway/pkg/models"
	"github.com/prometheus/client_golang/prometheus"
)

func TestGetModelReturnsListedCanonicalAndNestedIDs(t *testing.T) {
	providerConfigs := map[string]config.ProviderConfig{
		"abacus": {
			Type:         "openai",
			APIKey:       "dummy",
			DefaultModel: "route-llm-code",
			Models: config.ProviderModelsConfig{
				Mode:   "static",
				Static: []string{"route-llm-code", "meta-llama/Meta-Llama-3.3-70B-Instruct"},
			},
		},
	}
	registry := providers.NewRegistry(providerConfigs)
	handler := NewHandler(
		registry,
		routing.NewEngine(config.RoutingConfig{}),
		resilience.NewFallbackExecutor(resilience.NewRetrier(config.RetryConfig{}), resilience.NewCircuitBreakerManager()),
		middleware.NewCache(config.CacheConfig{}),
		streaming.NewHandler(),
		observability.NewMetricsWithRegisterer(prometheus.NewRegistry()),
		nil,
		nil,
		modelstore.NewStore(registry, providerConfigs),
	)
	router := NewRouter(handler, nil, nil, health.NewChecker("test"))

	listRecorder := httptest.NewRecorder()
	router.ServeHTTP(listRecorder, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	if listRecorder.Code != http.StatusOK {
		t.Fatalf("list status = %d; body=%s", listRecorder.Code, listRecorder.Body.String())
	}
	var list models.ModelList
	if err := json.Unmarshal(listRecorder.Body.Bytes(), &list); err != nil {
		t.Fatalf("decode model list: %v", err)
	}
	if len(list.Data) != 3 {
		t.Fatalf("listed models = %d, want 3: %#v", len(list.Data), list.Data)
	}

	for _, listed := range list.Data {
		t.Run(listed.ID, func(t *testing.T) {
			retrieveRecorder := httptest.NewRecorder()
			path := "/v1/models/" + url.PathEscape(listed.ID)
			router.ServeHTTP(retrieveRecorder, httptest.NewRequest(http.MethodGet, path, nil))
			if retrieveRecorder.Code != http.StatusOK {
				t.Fatalf("retrieve status = %d; path=%s body=%s", retrieveRecorder.Code, path, retrieveRecorder.Body.String())
			}
			var retrieved models.ModelInfo
			if err := json.Unmarshal(retrieveRecorder.Body.Bytes(), &retrieved); err != nil {
				t.Fatalf("decode retrieved model: %v", err)
			}
			if retrieved.ID != listed.ID || retrieved.Object != "model" || retrieved.OwnedBy != listed.OwnedBy {
				t.Fatalf("retrieved model = %#v, listed = %#v", retrieved, listed)
			}
		})
	}

	missingRecorder := httptest.NewRecorder()
	router.ServeHTTP(missingRecorder, httptest.NewRequest(http.MethodGet, "/v1/models/abacus%2Fmissing", nil))
	if missingRecorder.Code != http.StatusNotFound {
		t.Fatalf("missing status = %d; body=%s", missingRecorder.Code, missingRecorder.Body.String())
	}
}
