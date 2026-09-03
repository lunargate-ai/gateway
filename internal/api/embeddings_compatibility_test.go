package api

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

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

func TestValidateEmbeddingsCompatibilityRejectsUnmappedOllamaFields(t *testing.T) {
	handler := &Handler{registry: providers.NewRegistry(map[string]config.ProviderConfig{
		"ollama-local": {Type: "ollama"},
	})}
	dimensions := 256
	tests := []struct {
		name      string
		request   models.EmbeddingsRequest
		wantField string
	}{
		{name: "dimensions", request: models.EmbeddingsRequest{Input: "hello", Dimensions: &dimensions}, wantField: "dimensions"},
		{name: "user", request: models.EmbeddingsRequest{Input: "hello", User: "customer-123"}, wantField: "user"},
		{name: "token array", request: models.EmbeddingsRequest{Input: []interface{}{float64(1), float64(2)}}, wantField: "input"},
		{name: "token batches", request: models.EmbeddingsRequest{Input: []interface{}{[]interface{}{float64(1), float64(2)}}}, wantField: "input"},
	}

	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			err := handler.validateEmbeddingsCompatibility(
				routing.Target{Provider: "ollama-local"},
				&testCase.request,
			)
			var compatibilityErr *models.CompatibilityError
			if !errors.As(err, &compatibilityErr) {
				t.Fatalf("error = %v, want CompatibilityError", err)
			}
			if compatibilityErr.Field != testCase.wantField || compatibilityErr.Provider != "ollama-local" {
				t.Fatalf("compatibility error = %#v", compatibilityErr)
			}
		})
	}
}

func TestCompatibleEmbeddingsFallbacksDropsOllamaForUnmappedFields(t *testing.T) {
	handler := &Handler{registry: providers.NewRegistry(map[string]config.ProviderConfig{
		"ollama-backup": {Type: "ollama"},
		"openai-backup": {Type: "openai"},
	})}
	dimensions := 256
	fallbacks := []routing.Target{
		{Provider: "ollama-backup", Model: "nomic-embed-text"},
		{Provider: "openai-backup", Model: "text-embedding-3-small"},
	}

	got := handler.compatibleEmbeddingsFallbacks(fallbacks, &models.EmbeddingsRequest{
		Input:      "hello",
		Dimensions: &dimensions,
	})
	if len(got) != 1 || got[0].Provider != "openai-backup" {
		t.Fatalf("compatible fallbacks = %#v, want only openai-backup", got)
	}
}

func TestValidateEmbeddingsCompatibilityAllowsOllamaTextInputs(t *testing.T) {
	handler := &Handler{registry: providers.NewRegistry(map[string]config.ProviderConfig{
		"ollama-local": {Type: "ollama"},
	})}
	for _, input := range []interface{}{
		"hello",
		[]string{"hello", "world"},
		[]interface{}{"hello", "world"},
	} {
		if err := handler.validateEmbeddingsCompatibility(
			routing.Target{Provider: "ollama-local"},
			&models.EmbeddingsRequest{Input: input},
		); err != nil {
			t.Fatalf("input %#v rejected: %v", input, err)
		}
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

func TestEmbeddingsPreservesNativeResponseEnvelopeAcrossCache(t *testing.T) {
	upstreamCalls := 0
	rawResponse := `{"object":"list","data":[{"object":"embedding","embedding":[0.1,0.2],"index":0,"future_item":"kept"}],"model":"text-embedding-3-small","usage":{"prompt_tokens":1,"total_tokens":1,"future_usage":7},"future_top_level":{"kept":true}}`
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		upstreamCalls++
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Upstream-Request-ID", "req_embeddings_raw")
		w.Header().Set("Set-Cookie", "must-not-leak=true")
		_, _ = w.Write([]byte(rawResponse))
	}))
	defer upstream.Close()

	reg := providers.NewRegistry(map[string]config.ProviderConfig{
		"openai": {Type: "openai", APIKey: "dummy", BaseURL: upstream.URL},
	})
	router := routing.NewEngine(config.RoutingConfig{
		DefaultStrategy: "weighted",
		Routes: []config.RouteConfig{{
			Name:    "embeddings",
			Match:   config.MatchConfig{Path: "/v1/embeddings"},
			Targets: []config.TargetConfig{{Provider: "openai", Model: "text-embedding-3-small", Weight: 1}},
		}},
	})
	cache := middleware.NewCache(config.CacheConfig{Enabled: true, TTL: time.Minute, MaxSize: 10})
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
	payload := []byte(`{"model":"text-embedding-3-small","input":"hello"}`)

	for attempt := 0; attempt < 2; attempt++ {
		recorder := httptest.NewRecorder()
		handler.Embeddings(recorder, httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewReader(payload)))
		if recorder.Code != http.StatusOK {
			t.Fatalf("attempt %d status = %d; body=%s", attempt, recorder.Code, recorder.Body.String())
		}
		if got := recorder.Body.String(); got != rawResponse {
			t.Fatalf("attempt %d response envelope changed:\n got %s\nwant %s", attempt, got, rawResponse)
		}
		if attempt == 0 {
			if got := recorder.Header().Get("X-Upstream-Request-ID"); got != "req_embeddings_raw" {
				t.Fatalf("safe upstream header = %q", got)
			}
			if got := recorder.Header().Get("Set-Cookie"); got != "" {
				t.Fatalf("unsafe upstream header leaked: %q", got)
			}
		}
	}
	if upstreamCalls != 1 {
		t.Fatalf("upstream calls = %d, want 1", upstreamCalls)
	}

	collectorPayload, ok := embeddingsResponseForCollector(&models.EmbeddingsResponse{RawJSON: json.RawMessage(rawResponse)}).(map[string]interface{})
	if !ok || collectorPayload["future_top_level"] == nil {
		t.Fatalf("collector response lost additive field: %#v", collectorPayload)
	}
}
