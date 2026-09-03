package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestResponsesLegacyPathFallbackStillUsesInternalChatRoute(t *testing.T) {
	var upstreamCalls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		upstreamCalls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-legacy","object":"chat.completion","created":1,"model":"gpt-test","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	t.Cleanup(upstream.Close)

	handler, _, _ := newResilienceClassificationHandler(
		t,
		map[string]config.ProviderConfig{
			"openai": {Type: "openai", APIKey: "test", BaseURL: upstream.URL},
		},
		config.RouteConfig{
			Name:    "legacy-chat",
			Match:   config.MatchConfig{Path: "/v1/chat/completions"},
			Targets: []config.TargetConfig{{Provider: "openai", Model: "gpt-test", Weight: 1}},
		},
		config.RetryConfig{Enabled: false},
	)
	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/responses",
		strings.NewReader(`{"model":"lunargate/auto","input":"hello","store":false}`),
	)
	recorder := httptest.NewRecorder()

	handler.Responses(recorder, request)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", recorder.Code, recorder.Body.String())
	}
	if got := recorder.Header().Get("X-LunarGate-Route"); got != "legacy-chat" {
		t.Fatalf("route = %q, want legacy-chat", got)
	}
	if got := upstreamCalls.Load(); got != 1 {
		t.Fatalf("upstream calls = %d, want 1", got)
	}
}

func TestResponsesNativeConversationUnknownModelPreservesModelNotFound(t *testing.T) {
	handler, capture, _ := newNativeConversationProtocolTestHandler(t)
	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/responses",
		strings.NewReader(`{"model":"gpt-unknown","conversation":{"id":"conv_native_protocol"},"input":"hello","store":false}`),
	)
	recorder := httptest.NewRecorder()

	handler.Responses(recorder, request)

	assertModelNotFoundError(t, recorder)
	assertNoNativeConversationUpstreamCalls(t, capture)
}

func TestResponsesUnavailableRequestedProviderPreservesProviderNotFound(t *testing.T) {
	handler, capture, _ := newNativeConversationProtocolTestHandler(t)
	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/responses",
		strings.NewReader(`{"model":"lunargate/auto","input":"hello","store":false}`),
	)
	request.Header.Set("X-LunarGate-Provider", "missing")
	recorder := httptest.NewRecorder()

	handler.Responses(recorder, request)

	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400; body=%s", recorder.Code, recorder.Body.String())
	}
	var response models.ErrorResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if response.Error.Param == nil || *response.Error.Param != "provider" {
		t.Fatalf("param = %#v, want provider", response.Error.Param)
	}
	if response.Error.Code == nil || *response.Error.Code != "provider_not_found" {
		t.Fatalf("code = %#v, want provider_not_found", response.Error.Code)
	}
	assertNoNativeConversationUpstreamCalls(t, capture)
}

func assertNoNativeConversationUpstreamCalls(t *testing.T, capture *nativeConversationProtocolCapture) {
	t.Helper()
	if got := capture.chatCalls.Load(); got != 0 {
		t.Fatalf("chat-completions upstream calls = %d, want 0", got)
	}
	if got := capture.responsesCalls.Load(); got != 0 {
		t.Fatalf("Responses upstream calls = %d, want 0", got)
	}
	if requests := capture.snapshot(); len(requests) != 0 {
		t.Fatalf("upstream requests = %#v, want none", requests)
	}
}
