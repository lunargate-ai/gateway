package api

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/middleware"
	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/internal/routing"
	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestEffectiveTargetModelPrecedence(t *testing.T) {
	handler := &Handler{registry: providers.NewRegistry(map[string]config.ProviderConfig{
		"shared": {
			Type:         "openai",
			APIKey:       "test-key",
			DefaultModel: "provider-default",
		},
	})}

	tests := []struct {
		name         string
		target       routing.Target
		requestModel string
		want         string
	}{
		{
			name:         "target model",
			target:       routing.Target{Provider: "shared", Model: "target-model"},
			requestModel: "shared/request-model",
			want:         "target-model",
		},
		{
			name:         "request model",
			target:       routing.Target{Provider: "shared"},
			requestModel: "shared/request-model",
			want:         "request-model",
		},
		{
			name:   "provider default",
			target: routing.Target{Provider: "shared"},
			want:   "provider-default",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := handler.effectiveTargetModel(test.target, test.requestModel); got != test.want {
				t.Fatalf("effective target model = %q, want %q", got, test.want)
			}
		})
	}
}

func TestChatFallbackResponseDoesNotPoisonPrimaryModelCache(t *testing.T) {
	const (
		primaryModel  = "model-one"
		fallbackModel = "model-two"
	)
	var primaryReady atomic.Bool
	var primaryCalls, fallbackCalls atomic.Int32

	handler := newChatTargetModelFallbackCacheHandler(t, primaryModel, fallbackModel)
	setProviderTransportForTest(t, handler, "shared", providerURLRoundTripFunc(func(request *http.Request) (*http.Response, error) {
		model := requestModelFromBody(t, request)
		switch model {
		case primaryModel:
			primaryCalls.Add(1)
			if !primaryReady.Load() {
				return cacheTargetModelResponse(request, http.StatusInternalServerError, `{"error":{"message":"retry elsewhere"}}`), nil
			}
			return cacheTargetModelResponse(request, http.StatusOK, `{"id":"chat-primary","object":"chat.completion","created":1,"model":"model-one","choices":[{"index":0,"message":{"role":"assistant","content":"primary"},"finish_reason":"stop"}]}`), nil
		case fallbackModel:
			fallbackCalls.Add(1)
			return cacheTargetModelResponse(request, http.StatusOK, `{"id":"chat-fallback","object":"chat.completion","created":1,"model":"model-two","choices":[{"index":0,"message":{"role":"assistant","content":"fallback"},"finish_reason":"stop"}]}`), nil
		default:
			t.Fatalf("upstream model = %q, want %q or %q", model, primaryModel, fallbackModel)
			return nil, nil
		}
	}))

	payload := `{"messages":[{"role":"user","content":"hello"}]}`
	first := performTargetModelCacheRequest(t, handler.ChatCompletions, "/v1/chat/completions", payload)
	assertTargetModelCacheResponse(t, first, "MISS", "shared/model-two", `"content":"fallback"`)

	primaryReady.Store(true)
	second := performTargetModelCacheRequest(t, handler.ChatCompletions, "/v1/chat/completions", payload)
	assertTargetModelCacheResponse(t, second, "MISS", "shared/model-one", `"content":"primary"`)

	third := performTargetModelCacheRequest(t, handler.ChatCompletions, "/v1/chat/completions", payload)
	assertTargetModelCacheResponse(t, third, "HIT", "shared/model-one", `"content":"primary"`)
	if got := primaryCalls.Load(); got != 2 {
		t.Fatalf("primary model calls = %d, want 2", got)
	}
	if got := fallbackCalls.Load(); got != 1 {
		t.Fatalf("fallback model calls = %d, want 1", got)
	}
}

func TestEmbeddingsCacheSeparatesManuallyResolvedFallbackModel(t *testing.T) {
	// The public Embeddings API intentionally rejects an explicit cross-model
	// fallback. Exercise the lookup/store key contract directly with models
	// that have already been resolved by an allowed upstream workflow.
	request := &models.EmbeddingsRequest{
		RawJSON: json.RawMessage(`{"model":"embedding-one","input":"hello"}`),
		Model:   "shared/embedding-one",
		Input:   "hello",
	}
	primaryKey := middleware.GenerateEmbeddingsKeyForResolvedTargetWithHeaders(
		request,
		"shared",
		"embedding-one",
		"embeddings",
		nil,
	)
	fallbackKey := middleware.GenerateEmbeddingsKeyForResolvedTargetWithHeaders(
		request,
		"shared",
		"embedding-two",
		"embeddings",
		nil,
	)
	cache := middleware.NewCache(enabledTestCacheConfig())
	t.Cleanup(cache.Stop)
	cache.Set(fallbackKey, &models.EmbeddingsResponse{
		Object: "list",
		Model:  "embedding-two",
		Data: []models.EmbeddingData{{
			Object:    "embedding",
			Embedding: models.NewFloatEmbeddingValue([]float64{0.2}),
		}},
	})

	if cached := cache.Get(primaryKey); cached != nil {
		t.Fatalf("primary lookup returned fallback response: %#v", cached)
	}
	if cached := cache.Get(fallbackKey); cached == nil {
		t.Fatal("fallback response was not stored under its effective model")
	}
}

func newChatTargetModelFallbackCacheHandler(t *testing.T, primaryModel string, fallbackModel string) *Handler {
	t.Helper()
	handler, _, _ := newResilienceClassificationHandler(
		t,
		map[string]config.ProviderConfig{
			"shared": {
				Type:         "openai",
				APIKey:       "test-key",
				BaseURL:      "http://shared.invalid/v1",
				DefaultModel: primaryModel,
			},
		},
		config.RouteConfig{
			Name:     "target-model-cache",
			Match:    config.MatchConfig{Path: "/v1/chat/completions"},
			Targets:  []config.TargetConfig{{Provider: "shared", Model: primaryModel, Weight: 1}},
			Fallback: []config.TargetConfig{{Provider: "shared", Model: fallbackModel, Weight: 1}},
		},
		config.RetryConfig{Enabled: false},
	)
	cache := middleware.NewCache(enabledTestCacheConfig())
	t.Cleanup(cache.Stop)
	handler.cache = cache
	return handler
}

func requestModelFromBody(t *testing.T, request *http.Request) string {
	t.Helper()
	body, err := io.ReadAll(request.Body)
	if err != nil {
		t.Fatalf("read upstream request: %v", err)
	}
	_ = request.Body.Close()
	var payload struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("decode upstream request: %v", err)
	}
	return payload.Model
}

func cacheTargetModelResponse(request *http.Request, status int, body string) *http.Response {
	return &http.Response{
		StatusCode: status,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(strings.NewReader(body)),
		Request:    request,
	}
}

func performTargetModelCacheRequest(
	t *testing.T,
	handler func(http.ResponseWriter, *http.Request),
	path string,
	payload string,
) *httptest.ResponseRecorder {
	t.Helper()
	recorder := httptest.NewRecorder()
	handler(recorder, httptest.NewRequest(http.MethodPost, path, strings.NewReader(payload)))
	if recorder.Code != http.StatusOK {
		t.Fatalf("response status = %d, want 200; body=%s", recorder.Code, recorder.Body.String())
	}
	return recorder
}

func assertTargetModelCacheResponse(t *testing.T, recorder *httptest.ResponseRecorder, cacheStatus string, model string, bodyFragment string) {
	t.Helper()
	if got := recorder.Header().Get("X-LunarGate-Cache-Status"); got != cacheStatus {
		t.Fatalf("cache status = %q, want %q", got, cacheStatus)
	}
	if got := recorder.Header().Get("X-LunarGate-Model"); got != model {
		t.Fatalf("model header = %q, want %q", got, model)
	}
	if !strings.Contains(recorder.Body.String(), bodyFragment) {
		t.Fatalf("response body = %s, want fragment %s", recorder.Body.String(), bodyFragment)
	}
}
