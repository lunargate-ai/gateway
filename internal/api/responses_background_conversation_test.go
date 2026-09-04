package api

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
)

func TestResponsesRejectBackgroundWithLocalConversation(t *testing.T) {
	router, handler, calls, closeTest := newBackgroundConversationRouter(t)
	defer closeTest()
	conversation, err := handler.conversationsState.create(nil, nil)
	if err != nil {
		t.Fatalf("create local conversation: %v", err)
	}

	response := performLifecycleRequest(t, router, http.MethodPost, "/v1/responses", []byte(`{
		"model":"native/gpt-native",
		"conversation":"`+conversation.ID+`",
		"input":"hello",
		"background":true,
		"store":true
	}`))
	if response.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400; body=%s", response.Code, response.Body.String())
	}
	assertLifecycleError(t, response.Body.Bytes(), "background", "unsupported_feature")
	if calls.Load() != 0 {
		t.Fatalf("local background request made %d upstream calls", calls.Load())
	}
	items, ok := handler.conversationsState.getItems(conversation.ID)
	if !ok || len(items) != 0 {
		t.Fatalf("local conversation mutated: items=%#v ok=%t", items, ok)
	}
}

func TestResponsesWebSocketRejectsBackgroundWithLocalConversation(t *testing.T) {
	router, handler, calls, closeTest := newBackgroundConversationRouter(t)
	defer closeTest()
	server := httptest.NewServer(router)
	defer server.Close()
	conversation, err := handler.conversationsState.create(nil, nil)
	if err != nil {
		t.Fatalf("create local conversation: %v", err)
	}

	conn := mustDialResponsesWebSocket(t, server.URL)
	defer conn.Close()
	sendResponsesWebSocketJSON(t, conn, map[string]interface{}{
		"type":         "response.create",
		"model":        "native/gpt-native",
		"conversation": conversation.ID,
		"input":        "hello",
		"background":   true,
		"store":        true,
	})
	event := readResponsesWebSocketEvent(t, conn)
	if event["type"] != "error" || event["status"] != float64(http.StatusBadRequest) {
		t.Fatalf("websocket event = %#v, want HTTP 400 error", event)
	}
	errorObject, _ := event["error"].(map[string]interface{})
	if errorObject["param"] != "background" || errorObject["code"] != "unsupported_feature" {
		t.Fatalf("websocket error = %#v", errorObject)
	}
	if calls.Load() != 0 {
		t.Fatalf("local background websocket made %d upstream calls", calls.Load())
	}
	items, ok := handler.conversationsState.getItems(conversation.ID)
	if !ok || len(items) != 0 {
		t.Fatalf("local conversation mutated: items=%#v ok=%t", items, ok)
	}
}

func TestResponsesAllowBackgroundWithNativeConversation(t *testing.T) {
	var calls atomic.Int32
	var upstreamPayload map[string]json.RawMessage
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read upstream request: %v", err)
			return
		}
		if err := json.Unmarshal(body, &upstreamPayload); err != nil {
			t.Errorf("decode upstream request: %v", err)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"resp_background_native",
			"object":"response",
			"created_at":1,
			"status":"queued",
			"model":"gpt-native",
			"output":[]
		}`)
	}))
	defer upstream.Close()
	router, handler, cache := newNativeLifecycleRouter(t, upstream.URL+"/v1", map[string]config.ProviderCapabilities{
		"native": backgroundConversationCapabilities(),
	})
	defer cache.Stop()
	binding, err := handler.validateConversationProvider("native")
	if err != nil {
		t.Fatalf("create native binding: %v", err)
	}
	handler.conversationBindings.put("conv_background_native", binding)

	response := performLifecycleRequest(t, router, http.MethodPost, "/v1/responses", []byte(`{
		"model":"native/gpt-native",
		"conversation":"conv_background_native",
		"input":"hello",
		"background":true,
		"store":true
	}`))
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", response.Code, response.Body.String())
	}
	if calls.Load() != 1 {
		t.Fatalf("native background request made %d upstream calls", calls.Load())
	}
	if string(upstreamPayload["background"]) != "true" || parseJSONStringRaw(upstreamPayload["conversation"]) != "conv_background_native" {
		t.Fatalf("upstream payload = %#v", upstreamPayload)
	}
	if _, bound := handler.responseBindings.get("resp_background_native"); !bound {
		t.Fatal("native background response binding was not retained")
	}
}

func newBackgroundConversationRouter(
	t *testing.T,
) (http.Handler, *Handler, *atomic.Int32, func()) {
	t.Helper()
	var calls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"resp_background_local",
			"object":"response",
			"created_at":1,
			"status":"queued",
			"model":"gpt-native",
			"output":[]
		}`)
	}))
	router, handler, cache := newNativeLifecycleRouter(t, upstream.URL+"/v1", map[string]config.ProviderCapabilities{
		"native": backgroundConversationCapabilities(),
	})
	return router, handler, &calls, func() {
		cache.Stop()
		upstream.Close()
	}
}

func backgroundConversationCapabilities() config.ProviderCapabilities {
	return config.ProviderCapabilities{
		ResponsesLifecycle:  true,
		Conversations:       true,
		BackgroundResponses: true,
	}
}
