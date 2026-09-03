package api

import (
	"bytes"
	"encoding/json"
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
	"github.com/prometheus/client_golang/prometheus"
)

func TestResponsesPassesUnknownPreviousResponseIDToNativeTarget(t *testing.T) {
	upstreamCalls := 0
	previousResponseID := ""
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamCalls++
		if r.URL.Path != "/v1/responses" {
			t.Fatalf("upstream path = %q, want /v1/responses", r.URL.Path)
		}
		var payload map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatalf("decode upstream request: %v", err)
		}
		previousResponseID, _ = payload["previous_response_id"].(string)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"resp_new","object":"response","created_at":1,"status":"completed","model":"gpt-5.4","output":[{"type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"ok"}]}],"output_text":"ok"}`))
	}))
	defer upstream.Close()

	handler, cache := newNativeContinuationTestHandler(t, upstream.URL+"/v1", "responses")
	defer cache.Stop()
	payload := []byte(`{"model":"gpt-5.4","previous_response_id":"resp_external","input":"continue","store":false}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(payload))
	rec := httptest.NewRecorder()
	handler.Responses(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	if upstreamCalls != 1 {
		t.Fatalf("upstream calls = %d, want 1", upstreamCalls)
	}
	if previousResponseID != "resp_external" {
		t.Fatalf("previous_response_id = %q, want resp_external", previousResponseID)
	}
}

func TestResponsesPassesPromptOnlyRequestToNativeTarget(t *testing.T) {
	var upstreamPayload map[string]interface{}
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/responses" {
			t.Fatalf("upstream path = %q, want /v1/responses", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&upstreamPayload); err != nil {
			t.Fatalf("decode upstream request: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"resp_prompt","object":"response","created_at":1,"status":"completed","model":"gpt-5.4","output":[],"output_text":"ok"}`))
	}))
	defer upstream.Close()

	handler, cache := newNativeContinuationTestHandler(t, upstream.URL+"/v1", "responses")
	defer cache.Stop()
	recorder := httptest.NewRecorder()
	handler.Responses(recorder, httptest.NewRequest(
		http.MethodPost,
		"/v1/responses",
		bytes.NewBufferString(`{"prompt":{"id":"pmpt_1","version":"2"},"store":false}`),
	))

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", recorder.Code, recorder.Body.String())
	}
	prompt, ok := upstreamPayload["prompt"].(map[string]interface{})
	if !ok || prompt["id"] != "pmpt_1" || prompt["version"] != "2" {
		t.Fatalf("prompt = %#v, want preserved prompt reference", upstreamPayload["prompt"])
	}
	if _, exists := upstreamPayload["input"]; exists {
		t.Fatalf("native request gained input: %#v", upstreamPayload)
	}
	if upstreamPayload["model"] != "gpt-5.4" {
		t.Fatalf("route-selected model = %#v, want gpt-5.4", upstreamPayload["model"])
	}
}

func TestResponsesRejectsUnknownPreviousResponseIDForTranslatedTarget(t *testing.T) {
	upstreamCalls := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamCalls++
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer upstream.Close()

	handler, cache := newNativeContinuationTestHandler(t, upstream.URL+"/v1", "chat_completions")
	defer cache.Stop()
	payload := []byte(`{"model":"gpt-5.4","previous_response_id":"resp_external","input":"continue"}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(payload))
	rec := httptest.NewRecorder()
	handler.Responses(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400; body=%s", rec.Code, rec.Body.String())
	}
	if upstreamCalls != 0 {
		t.Fatalf("upstream calls = %d, want 0", upstreamCalls)
	}
	var response struct {
		Error struct {
			Param *string `json:"param"`
			Code  *string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode error response: %v", err)
	}
	if response.Error.Param == nil || *response.Error.Param != "previous_response_id" {
		t.Fatalf("error param = %#v, want previous_response_id", response.Error.Param)
	}
	if response.Error.Code == nil || *response.Error.Code != "unsupported_feature" {
		t.Fatalf("error code = %#v, want unsupported_feature", response.Error.Code)
	}
}

func newNativeContinuationTestHandler(t *testing.T, baseURL string, upstreamRequestType string) (*Handler, *middleware.Cache) {
	t.Helper()
	registry := providers.NewRegistry(map[string]config.ProviderConfig{
		"openai": {Type: "openai", APIKey: "dummy", BaseURL: baseURL},
	})
	router := routing.NewEngine(config.RoutingConfig{
		DefaultStrategy: "weighted",
		Routes: []config.RouteConfig{{
			Name:  "responses",
			Match: config.MatchConfig{Path: "/v1/responses"},
			Targets: []config.TargetConfig{{
				Provider:            "openai",
				Model:               "gpt-5.4",
				Weight:              1,
				UpstreamRequestType: upstreamRequestType,
			}},
		}},
	})
	cache := middleware.NewCache(config.CacheConfig{Enabled: false})
	return NewHandler(
		registry,
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
	), cache
}
