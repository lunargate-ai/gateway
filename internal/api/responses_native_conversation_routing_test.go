package api

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
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

const nativeConversationProtocolTestID = "conv_native_protocol"

type nativeConversationProtocolRequest struct {
	path         string
	model        string
	conversation json.RawMessage
	stream       bool
}

type nativeConversationProtocolCapture struct {
	chatCalls      atomic.Int32
	responsesCalls atomic.Int32
	mu             sync.Mutex
	requests       []nativeConversationProtocolRequest
}

func (c *nativeConversationProtocolCapture) append(request nativeConversationProtocolRequest) {
	c.mu.Lock()
	defer c.mu.Unlock()
	request.conversation = append(json.RawMessage(nil), request.conversation...)
	c.requests = append(c.requests, request)
}

func (c *nativeConversationProtocolCapture) snapshot() []nativeConversationProtocolRequest {
	c.mu.Lock()
	defer c.mu.Unlock()
	requests := make([]nativeConversationProtocolRequest, len(c.requests))
	copy(requests, c.requests)
	for index := range requests {
		requests[index].conversation = append(json.RawMessage(nil), requests[index].conversation...)
	}
	return requests
}

func TestResponsesNativeConversationPinsResponsesProtocol(t *testing.T) {
	tests := []struct {
		name        string
		bodyModel   string
		headerModel string
		wantModel   string
	}{
		{name: "payload model", bodyModel: "gpt-native", wantModel: "gpt-native"},
		{name: "canonical payload model", bodyModel: "native/gpt-native", wantModel: "gpt-native"},
		{name: "canonical model header", bodyModel: "gpt-other", headerModel: "native/gpt-native", wantModel: "gpt-native"},
		{name: "automatic model", bodyModel: "lunargate/auto", wantModel: "gpt-other"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			handler, capture, binding := newNativeConversationProtocolTestHandler(t)
			requestBody := fmt.Sprintf(
				`{"model":%q,"conversation":{"id":%q,"future_association":{"keep":true}},"input":"hello","store":false}`,
				test.bodyModel,
				nativeConversationProtocolTestID,
			)
			request := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(requestBody))
			if test.headerModel != "" {
				request.Header.Set("X-LunarGate-Model", test.headerModel)
			}
			recorder := httptest.NewRecorder()

			handler.Responses(recorder, request)

			if recorder.Code != http.StatusOK {
				t.Fatalf("status = %d, want 200; body=%s", recorder.Code, recorder.Body.String())
			}
			assertNativeConversationProtocolRouting(t, handler, capture, binding, test.wantModel, false)
			var response map[string]interface{}
			if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
				t.Fatalf("decode response: %v", err)
			}
			assertNativeConversationProtocolAssociation(t, response["conversation"])
		})
	}
}

func TestResponsesWebSocketNativeConversationPinsResponsesProtocol(t *testing.T) {
	handler, capture, binding := newNativeConversationProtocolTestHandler(t)
	server := httptest.NewServer(http.HandlerFunc(handler.ResponsesWebSocket))
	t.Cleanup(server.Close)

	connection := mustDialResponsesWebSocket(t, server.URL)
	t.Cleanup(func() { _ = connection.Close() })
	sendResponsesWebSocketJSON(t, connection, map[string]interface{}{
		"type":  "response.create",
		"model": "native/gpt-native",
		"conversation": map[string]interface{}{
			"id": nativeConversationProtocolTestID,
			"future_association": map[string]interface{}{
				"keep": true,
			},
		},
		"input": "hello",
		"store": false,
	})

	events := readResponsesWebSocketEventsUntilTerminal(t, connection)
	if hasResponsesWebSocketEventType(events, "error") || !hasResponsesWebSocketEventType(events, "response.completed") {
		t.Fatalf("websocket events = %v, want response.completed without error", eventTypes(events))
	}
	assertNativeConversationProtocolRouting(t, handler, capture, binding, "gpt-native", true)
	for _, event := range events {
		if event["type"] != "response.completed" {
			continue
		}
		response, _ := event["response"].(map[string]interface{})
		assertNativeConversationProtocolAssociation(t, response["conversation"])
		return
	}
	t.Fatal("response.completed event not found")
}

func newNativeConversationProtocolTestHandler(
	t *testing.T,
) (*Handler, *nativeConversationProtocolCapture, conversationBinding) {
	t.Helper()
	capture := &nativeConversationProtocolCapture{}
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read upstream request: %v", err)
			http.Error(w, "read failed", http.StatusInternalServerError)
			return
		}
		var payload map[string]json.RawMessage
		if err := json.Unmarshal(body, &payload); err != nil {
			t.Errorf("decode upstream request: %v", err)
			http.Error(w, "decode failed", http.StatusInternalServerError)
			return
		}
		var stream bool
		_ = json.Unmarshal(payload["stream"], &stream)
		capture.append(nativeConversationProtocolRequest{
			path:         r.URL.Path,
			model:        parseJSONStringRaw(payload["model"]),
			conversation: payload["conversation"],
			stream:       stream,
		})

		switch r.URL.Path {
		case "/v1/chat/completions":
			capture.chatCalls.Add(1)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusTeapot)
			_, _ = io.WriteString(w, `{"error":{"message":"wrong upstream protocol","type":"invalid_request_error"}}`)
		case "/v1/responses":
			capture.responsesCalls.Add(1)
			writeNativeConversationProtocolResponse(t, w, payload, stream)
		default:
			t.Errorf("unexpected upstream path %q", r.URL.Path)
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(upstream.Close)

	providerConfigs := map[string]config.ProviderConfig{
		"native": {
			Type:         "openai",
			APIKey:       "native-secret",
			BaseURL:      upstream.URL + "/v1",
			DefaultModel: "gpt-native",
			Capabilities: config.ProviderCapabilities{Conversations: true, ResponsesLifecycle: true},
		},
	}
	registry := providers.NewRegistry(providerConfigs)
	routingEngine := routing.NewEngine(config.RoutingConfig{
		DefaultStrategy: "round-robin",
		Routes: []config.RouteConfig{{
			Name:  "native-conversation",
			Match: config.MatchConfig{Path: "/v1/responses"},
			Targets: []config.TargetConfig{
				{Provider: "native", Model: "gpt-native", Weight: 1, UpstreamRequestType: requestTypeChatCompletions},
				{Provider: "native", Model: "gpt-other", Weight: 1, UpstreamRequestType: requestTypeResponses},
				{Provider: "native", Model: "gpt-native", Weight: 1, UpstreamRequestType: requestTypeResponses},
			},
		}},
	})
	cache := middleware.NewCache(config.CacheConfig{Enabled: false})
	t.Cleanup(cache.Stop)
	handler := NewHandler(
		registry,
		routingEngine,
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
	handler.UpdateProviderConfigs(providerConfigs)
	binding, err := handler.validateConversationProvider("native")
	if err != nil {
		t.Fatalf("validate native conversation provider: %v", err)
	}
	if !handler.conversationBindings.put(nativeConversationProtocolTestID, binding) {
		t.Fatal("failed to retain native conversation binding")
	}
	return handler, capture, binding
}

func writeNativeConversationProtocolResponse(
	t *testing.T,
	w http.ResponseWriter,
	payload map[string]json.RawMessage,
	stream bool,
) {
	t.Helper()
	var conversation interface{}
	if err := json.Unmarshal(payload["conversation"], &conversation); err != nil {
		t.Errorf("decode upstream conversation: %v", err)
		http.Error(w, "invalid conversation", http.StatusInternalServerError)
		return
	}
	model := parseJSONStringRaw(payload["model"])
	response := map[string]interface{}{
		"id":           "resp_native_conversation_protocol",
		"object":       "response",
		"created_at":   1,
		"status":       "completed",
		"model":        model,
		"conversation": conversation,
		"output":       []interface{}{},
		"output_text":  "",
	}
	if !stream {
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(response); err != nil {
			t.Errorf("encode upstream response: %v", err)
		}
		return
	}
	event, err := json.Marshal(map[string]interface{}{
		"type":            "response.completed",
		"sequence_number": 0,
		"response":        response,
	})
	if err != nil {
		t.Errorf("encode upstream stream event: %v", err)
		http.Error(w, "encode failed", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	_, _ = fmt.Fprintf(w, "event: response.completed\ndata: %s\n\n", event)
}

func assertNativeConversationProtocolRouting(
	t *testing.T,
	handler *Handler,
	capture *nativeConversationProtocolCapture,
	wantBinding conversationBinding,
	wantModel string,
	wantStream bool,
) {
	t.Helper()
	if got := capture.chatCalls.Load(); got != 0 {
		t.Fatalf("chat-completions upstream calls = %d, want 0", got)
	}
	if got := capture.responsesCalls.Load(); got != 1 {
		t.Fatalf("Responses upstream calls = %d, want 1", got)
	}
	requests := capture.snapshot()
	if len(requests) != 1 {
		t.Fatalf("upstream requests = %#v, want one", requests)
	}
	if requests[0].path != "/v1/responses" {
		t.Fatalf("upstream path = %q, want /v1/responses", requests[0].path)
	}
	if requests[0].model != wantModel {
		t.Fatalf("upstream model = %q, want %q", requests[0].model, wantModel)
	}
	if requests[0].stream != wantStream {
		t.Fatalf("upstream stream = %t, want %t", requests[0].stream, wantStream)
	}
	assertNativeConversationProtocolAssociation(t, decodeNativeConversationProtocolJSON(t, requests[0].conversation))
	gotBinding, ok := handler.conversationBindings.get(nativeConversationProtocolTestID)
	if !ok || !sameConversationBindingOwner(gotBinding, wantBinding) {
		t.Fatalf("conversation binding = %#v, %t; want exact owner %#v", gotBinding, ok, wantBinding)
	}
	if _, local := handler.conversationsState.get(nativeConversationProtocolTestID); local {
		t.Fatal("native conversation was copied into local conversation state")
	}
}

func decodeNativeConversationProtocolJSON(t *testing.T, raw json.RawMessage) interface{} {
	t.Helper()
	var decoded interface{}
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("decode captured conversation: %v; raw=%s", err, raw)
	}
	return decoded
}

func assertNativeConversationProtocolAssociation(t *testing.T, raw interface{}) {
	t.Helper()
	conversation, ok := raw.(map[string]interface{})
	if !ok {
		t.Fatalf("conversation = %#v, want object", raw)
	}
	if conversation["id"] != nativeConversationProtocolTestID {
		t.Fatalf("conversation id = %#v, want %q", conversation["id"], nativeConversationProtocolTestID)
	}
	future, ok := conversation["future_association"].(map[string]interface{})
	if !ok || future["keep"] != true {
		t.Fatalf("conversation association = %#v, want future fields preserved", conversation)
	}
}
