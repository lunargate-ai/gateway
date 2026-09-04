package api

import (
	"bytes"
	"encoding/json"
	"fmt"
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

func TestChatCompletionsStoreFalseBypassesCache(t *testing.T) {
	h, calls, closeTest := newStoreSemanticsTestHandler(t)
	defer closeTest()

	payload := []byte(`{"model":"mock-gpt","store":false,"messages":[{"role":"user","content":"hello"}]}`)
	first := performJSONRequest(t, h.ChatCompletions, "/v1/chat/completions", payload)
	second := performJSONRequest(t, h.ChatCompletions, "/v1/chat/completions", payload)

	if *calls != 2 {
		t.Fatalf("upstream calls = %d, want 2", *calls)
	}
	var firstResp, secondResp models.UnifiedResponse
	if err := json.Unmarshal(first.Body.Bytes(), &firstResp); err != nil {
		t.Fatalf("decode first response: %v", err)
	}
	if err := json.Unmarshal(second.Body.Bytes(), &secondResp); err != nil {
		t.Fatalf("decode second response: %v", err)
	}
	if firstResp.ID == secondResp.ID {
		t.Fatalf("store:false reused cached response ID %q", firstResp.ID)
	}
}

func TestChatCompletionsStoreTrueBypassesCache(t *testing.T) {
	h, calls, closeTest := newStoreSemanticsTestHandler(t)
	defer closeTest()

	payload := []byte(`{"model":"mock-gpt","store":true,"messages":[{"role":"user","content":"hello"}]}`)
	first := performJSONRequest(t, h.ChatCompletions, "/v1/chat/completions", payload)
	second := performJSONRequest(t, h.ChatCompletions, "/v1/chat/completions", payload)

	if *calls != 2 {
		t.Fatalf("upstream calls = %d, want 2", *calls)
	}
	var firstResp, secondResp models.UnifiedResponse
	if err := json.Unmarshal(first.Body.Bytes(), &firstResp); err != nil {
		t.Fatalf("decode first response: %v", err)
	}
	if err := json.Unmarshal(second.Body.Bytes(), &secondResp); err != nil {
		t.Fatalf("decode second response: %v", err)
	}
	if firstResp.ID == secondResp.ID {
		t.Fatalf("store:true reused cached response ID %q", firstResp.ID)
	}
}

func TestResponsesCreateBypassesCacheAndStoreFalseSkipsLocalState(t *testing.T) {
	h, calls, closeTest := newStoreSemanticsTestHandler(t)
	defer closeTest()

	payload := []byte(`{"model":"mock-gpt","store":false,"input":"hello"}`)
	first := performJSONRequest(t, h.Responses, "/v1/responses", payload)
	second := performJSONRequest(t, h.Responses, "/v1/responses", payload)

	if *calls != 2 {
		t.Fatalf("upstream calls = %d, want 2", *calls)
	}
	var firstResp, secondResp models.ResponsesResponse
	if err := json.Unmarshal(first.Body.Bytes(), &firstResp); err != nil {
		t.Fatalf("decode first response: %v", err)
	}
	if err := json.Unmarshal(second.Body.Bytes(), &secondResp); err != nil {
		t.Fatalf("decode second response: %v", err)
	}
	if firstResp.ID == secondResp.ID {
		t.Fatalf("Responses create reused cached response ID %q", firstResp.ID)
	}
	if _, ok := h.responsesState.get(firstResp.ID); ok {
		t.Fatalf("store:false response %q was retained locally", firstResp.ID)
	}
	if _, ok := h.responsesState.get(secondResp.ID); ok {
		t.Fatalf("store:false response %q was retained locally", secondResp.ID)
	}
}

func TestMakeResponsesChatRequestDisablesReplayPolicies(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
	chatReq, err := makeResponsesChatRequest(req, &models.UnifiedRequest{Model: "mock-gpt"})
	if err != nil {
		t.Fatalf("makeResponsesChatRequest: %v", err)
	}
	for _, header := range []string{"X-LunarGate-No-Cache", "X-LunarGate-No-Retry", "X-LunarGate-No-Fallback"} {
		if got := chatReq.Header.Get(header); got != "true" {
			t.Fatalf("%s = %q, want true", header, got)
		}
	}
}

func TestMakeResponsesChatRequestPreservesResponsesEnvelope(t *testing.T) {
	raw := json.RawMessage(`{"model":"gpt-5","input":"hello","metadata":{"trace":"abc"}}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
	chatReq, err := makeResponsesChatRequest(req, &models.UnifiedRequest{
		RawJSON:           raw,
		SourceRequestType: "responses",
		Model:             "gpt-5",
		Messages:          []models.Message{{Role: "user", Content: "hello"}},
	})
	if err != nil {
		t.Fatalf("makeResponsesChatRequest: %v", err)
	}

	recorder := httptest.NewRecorder()
	_, parsed, ok := parseUnifiedRequest(recorder, chatReq, true)
	if !ok {
		t.Fatalf("parseUnifiedRequest failed: status=%d body=%s", recorder.Code, recorder.Body.String())
	}
	if parsed.SourceRequestType != "responses" {
		t.Fatalf("source request type = %q, want responses", parsed.SourceRequestType)
	}
	if string(parsed.RawJSON) != string(raw) {
		t.Fatalf("raw envelope = %s, want %s", parsed.RawJSON, raw)
	}
}

func newStoreSemanticsTestHandler(t *testing.T) (*Handler, *int, func()) {
	t.Helper()
	calls := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(w, `{"id":"chatcmpl-%d","object":"chat.completion","created":1,"model":"mock-gpt","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`, calls)
	}))

	reg := providers.NewRegistry(map[string]config.ProviderConfig{
		"openai": {Type: "openai", APIKey: "dummy", BaseURL: upstream.URL},
	})
	router := routing.NewEngine(config.RoutingConfig{
		DefaultStrategy: "weighted",
		Routes: []config.RouteConfig{{
			Name:    "default",
			Match:   config.MatchConfig{Path: "*"},
			Targets: []config.TargetConfig{{Provider: "openai", Model: "mock-gpt", Weight: 1}},
		}},
	})
	cache := middleware.NewCache(config.CacheConfig{Enabled: true, TTL: time.Hour, MaxSize: 10})
	h := NewHandler(
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
	return h, &calls, func() {
		cache.Stop()
		upstream.Close()
	}
}

func performJSONRequest(t *testing.T, handler http.HandlerFunc, path string, payload []byte) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, path, bytes.NewReader(payload))
	rec := httptest.NewRecorder()
	handler(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d; body=%s", rec.Code, http.StatusOK, rec.Body.String())
	}
	return rec
}
