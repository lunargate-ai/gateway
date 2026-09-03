package api

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/middleware"
	"github.com/lunargate-ai/gateway/internal/observability"
	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/internal/resilience"
	"github.com/lunargate-ai/gateway/internal/routing"
	"github.com/lunargate-ai/gateway/internal/streaming"
	"github.com/lunargate-ai/gateway/pkg/models"
	"github.com/prometheus/client_golang/prometheus"
)

func TestValidateEmbeddingsCompatibilityRequiresExplicitBase64Capability(t *testing.T) {
	handler := &Handler{registry: providers.NewRegistry(map[string]config.ProviderConfig{
		"disabled": {Type: "openai"},
		"enabled":  {Type: "openai", Capabilities: config.ProviderCapabilities{EmbeddingsBase64: true}},
		"ollama":   {Type: "ollama", Capabilities: config.ProviderCapabilities{EmbeddingsBase64: true}},
	})}
	req := &models.EmbeddingsRequest{EncodingFormat: "base64"}

	for _, providerID := range []string{"disabled", "ollama"} {
		err := handler.validateEmbeddingsCompatibility(routing.Target{Provider: providerID}, req)
		var compatibilityErr *models.CompatibilityError
		if !errors.As(err, &compatibilityErr) {
			t.Fatalf("provider %s error = %v, want CompatibilityError", providerID, err)
		}
		if compatibilityErr.Field != "encoding_format" {
			t.Fatalf("provider %s field = %q, want encoding_format", providerID, compatibilityErr.Field)
		}
	}
	if err := handler.validateEmbeddingsCompatibility(routing.Target{Provider: "enabled"}, req); err != nil {
		t.Fatalf("enabled OpenAI-compatible provider rejected base64: %v", err)
	}
}

func TestCompatibleEmbeddingsFallbacksDropsBase64IncompatibleTargets(t *testing.T) {
	handler := &Handler{registry: providers.NewRegistry(map[string]config.ProviderConfig{
		"disabled": {Type: "openai"},
		"enabled":  {Type: "openai", Capabilities: config.ProviderCapabilities{EmbeddingsBase64: true}},
	})}
	fallbacks := []routing.Target{{Provider: "disabled"}, {Provider: "enabled"}}
	got := handler.compatibleEmbeddingsFallbacks(fallbacks, &models.EmbeddingsRequest{EncodingFormat: "base64"})
	if len(got) != 1 || got[0].Provider != "enabled" {
		t.Fatalf("compatible fallbacks = %#v, want only enabled", got)
	}
}

func TestEmbeddingsBase64RoundTripsWhenCapabilityEnabled(t *testing.T) {
	upstreamCalls := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamCalls++
		var payload map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatalf("decode upstream request: %v", err)
		}
		if payload["encoding_format"] != "base64" {
			t.Fatalf("encoding_format = %#v, want base64", payload["encoding_format"])
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"object":"list","data":[{"object":"embedding","embedding":"AQIDBA==","index":0}],"model":"text-embedding-3-small","usage":{"prompt_tokens":1,"total_tokens":1}}`))
	}))
	defer upstream.Close()

	reg := providers.NewRegistry(map[string]config.ProviderConfig{
		"openai": {
			Type:         "openai",
			APIKey:       "dummy",
			BaseURL:      upstream.URL,
			Capabilities: config.ProviderCapabilities{EmbeddingsBase64: true},
		},
	})
	router := routing.NewEngine(config.RoutingConfig{
		DefaultStrategy: "weighted",
		Routes: []config.RouteConfig{{
			Name:    "embeddings",
			Match:   config.MatchConfig{Path: "/v1/embeddings"},
			Targets: []config.TargetConfig{{Provider: "openai", Model: "text-embedding-3-small", Weight: 1}},
		}},
	})
	cache := middleware.NewCache(config.CacheConfig{Enabled: false})
	defer cache.Stop()
	handler := NewHandler(
		reg,
		router,
		resilience.NewFallbackExecutor(
			resilience.NewRetrier(config.RetryConfig{Enabled: false}),
			resilience.NewCircuitBreakerManager(),
		),
		cache,
		streaming.NewHandler(),
		observability.NewMetricsWithRegisterer(prometheus.NewRegistry()),
		nil,
		nil,
		nil,
	)

	payload := []byte(`{"model":"text-embedding-3-small","input":"hello","encoding_format":"base64"}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewReader(payload))
	rec := httptest.NewRecorder()
	handler.Embeddings(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	if upstreamCalls != 1 {
		t.Fatalf("upstream calls = %d, want 1", upstreamCalls)
	}
	var response map[string]interface{}
	if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	data, _ := response["data"].([]interface{})
	if len(data) != 1 {
		t.Fatalf("data = %#v, want one embedding", response["data"])
	}
	embedding, _ := data[0].(map[string]interface{})["embedding"].(string)
	if embedding != "AQIDBA==" {
		t.Fatalf("embedding = %q, want preserved base64", embedding)
	}
}
