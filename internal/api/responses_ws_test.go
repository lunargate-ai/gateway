package api

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/health"
	"github.com/lunargate-ai/gateway/internal/middleware"
	"github.com/lunargate-ai/gateway/internal/observability"
	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/internal/resilience"
	"github.com/lunargate-ai/gateway/internal/routing"
	"github.com/lunargate-ai/gateway/internal/security"
	"github.com/lunargate-ai/gateway/internal/streaming"
	"github.com/prometheus/client_golang/prometheus"
)

func TestResponsesWebSocket_ResponseCreateStreamsEvents(t *testing.T) {
	var calls int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		call := atomic.AddInt32(&calls, 1)
		chunkID := fmt.Sprintf("chatcmpl-%d", call)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, fmt.Sprintf(
			"data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"mock-gpt\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"},\"finish_reason\":null}]}\n\n",
			chunkID,
		))
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()

	h := newResponsesWebSocketTestHandler(upstream.URL)
	server := httptest.NewServer(http.HandlerFunc(h.ResponsesWebSocket))
	defer server.Close()

	conn := mustDialResponsesWebSocket(t, server.URL)
	defer conn.Close()

	sendResponsesWebSocketJSON(t, conn, map[string]interface{}{
		"type":  "response.create",
		"model": "lunargate/auto",
		"input": "Say hi",
	})
	events := readResponsesWebSocketEventsUntilTerminal(t, conn)
	if !hasResponsesWebSocketEventType(events, "response.created") {
		t.Fatalf("expected response.created event, got %v", eventTypes(events))
	}
	if !hasResponsesWebSocketEventType(events, "response.completed") {
		t.Fatalf("expected response.completed event, got %v", eventTypes(events))
	}
	firstResponseID := extractCompletedResponseID(events)
	if firstResponseID == "" {
		t.Fatalf("expected completed response id in websocket events")
	}

	sendResponsesWebSocketJSON(t, conn, map[string]interface{}{
		"type":                 "response.create",
		"model":                "lunargate/auto",
		"input":                "Continue",
		"previous_response_id": firstResponseID,
	})
	followUp := readResponsesWebSocketEventsUntilTerminal(t, conn)
	if hasResponsesWebSocketEventType(followUp, "error") {
		t.Fatalf("expected follow-up response to succeed, got %v", eventTypes(followUp))
	}
	if !hasResponsesWebSocketEventType(followUp, "response.completed") {
		t.Fatalf("expected follow-up response.completed event, got %v", eventTypes(followUp))
	}
}

func TestResponsesWebSocket_OriginPolicy(t *testing.T) {
	h := &Handler{}
	server := httptest.NewServer(http.HandlerFunc(h.ResponsesWebSocket))
	defer server.Close()

	host := strings.TrimPrefix(server.URL, "http://")
	tests := []struct {
		name       string
		origin     string
		wantStatus int
	}{
		{name: "cli without origin", wantStatus: http.StatusSwitchingProtocols},
		{name: "same origin http", origin: "http://" + host, wantStatus: http.StatusSwitchingProtocols},
		{name: "same origin https", origin: "https://" + host, wantStatus: http.StatusSwitchingProtocols},
		{name: "same origin ws", origin: "ws://" + host, wantStatus: http.StatusSwitchingProtocols},
		{name: "same origin wss", origin: "wss://" + host, wantStatus: http.StatusSwitchingProtocols},
		{name: "cross origin host", origin: "https://attacker.example", wantStatus: http.StatusForbidden},
		{name: "cross origin port", origin: "http://127.0.0.1:1", wantStatus: http.StatusForbidden},
		{name: "opaque browser origin", origin: "null", wantStatus: http.StatusForbidden},
		{name: "unsupported scheme", origin: "file://" + host, wantStatus: http.StatusForbidden},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			headers := make(http.Header)
			if tc.origin != "" {
				headers.Set("Origin", tc.origin)
			}

			wsURL := "ws" + strings.TrimPrefix(server.URL, "http")
			conn, resp, err := websocket.DefaultDialer.Dial(wsURL, headers)
			if conn != nil {
				_ = conn.Close()
			}
			status := 0
			if resp != nil {
				status = resp.StatusCode
				_ = resp.Body.Close()
			}

			if tc.wantStatus == http.StatusSwitchingProtocols {
				if err != nil {
					t.Fatalf("same-origin websocket handshake failed: %v", err)
				}
				if status != tc.wantStatus {
					t.Fatalf("handshake status = %d, want %d", status, tc.wantStatus)
				}
				return
			}

			if err == nil {
				t.Fatal("cross-origin websocket handshake unexpectedly succeeded")
			}
			if status != tc.wantStatus {
				t.Fatalf("handshake status = %d, want %d", status, tc.wantStatus)
			}
		})
	}
}

func TestCheckResponsesWebSocketOrigin_NormalizesDefaultPorts(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "http://example.com:80/v1/responses", nil)
	req.Host = "example.com:80"
	req.Header.Set("Origin", "http://EXAMPLE.COM")

	if !checkResponsesWebSocketOrigin(req) {
		t.Fatal("expected equivalent default port and case-insensitive host to be accepted")
	}
}

func TestResponsesWebSocket_RechecksAuthenticationAfterConfigReload(t *testing.T) {
	h := &Handler{}
	authManager, err := security.NewManager(responsesWebSocketSecurityConfig("old-key"))
	if err != nil {
		t.Fatalf("failed to create auth manager: %v", err)
	}
	rateLimiter := middleware.NewRateLimiter(config.RateLimitConfig{Enabled: false})
	router := NewRouter(h, authManager, rateLimiter, health.NewChecker("test"))
	server := httptest.NewServer(router)
	defer server.Close()

	headers := make(http.Header)
	headers.Set("Authorization", "Bearer old-key")
	conn := mustDialResponsesWebSocketWithHeaders(t, server.URL, headers)
	defer conn.Close()

	sendResponsesWebSocketJSON(t, conn, responsesWebSocketWarmupPayload())
	firstEvents := readResponsesWebSocketEventsUntilTerminal(t, conn)
	if !hasResponsesWebSocketEventType(firstEvents, "response.completed") {
		t.Fatalf("first request did not complete: %v", eventTypes(firstEvents))
	}

	if err := authManager.UpdateConfig(responsesWebSocketSecurityConfig("new-key")); err != nil {
		t.Fatalf("failed to reload auth config: %v", err)
	}
	sendResponsesWebSocketJSON(t, conn, responsesWebSocketWarmupPayload())
	event := readResponsesWebSocketEvent(t, conn)
	if got, _ := event["type"].(string); got != "error" {
		t.Fatalf("revoked credential event type = %q, want error", got)
	}
	if status, _ := event["status"].(float64); int(status) != http.StatusUnauthorized {
		t.Fatalf("revoked credential status = %v, want %d", event["status"], http.StatusUnauthorized)
	}
	errObj, _ := event["error"].(map[string]interface{})
	if got, _ := errObj["type"].(string); got != "invalid_api_key" {
		t.Fatalf("revoked credential error type = %q, want invalid_api_key", got)
	}
}

func TestResponsesWebSocket_AppliesReloadedRateLimitPerMessage(t *testing.T) {
	h := &Handler{}
	authManager, err := security.NewManager(config.SecurityConfig{Enabled: false})
	if err != nil {
		t.Fatalf("failed to create auth manager: %v", err)
	}
	rateLimiter := middleware.NewRateLimiter(config.RateLimitConfig{Enabled: false})
	router := NewRouter(h, authManager, rateLimiter, health.NewChecker("test"))
	server := httptest.NewServer(router)
	defer server.Close()

	conn := mustDialResponsesWebSocket(t, server.URL)
	defer conn.Close()

	rateLimiter.UpdateConfig(config.RateLimitConfig{
		Enabled:           true,
		RequestsPerMinute: 1,
		BurstSize:         1,
	})
	sendResponsesWebSocketJSON(t, conn, responsesWebSocketWarmupPayload())
	firstEvents := readResponsesWebSocketEventsUntilTerminal(t, conn)
	if !hasResponsesWebSocketEventType(firstEvents, "response.completed") {
		t.Fatalf("first rate-limited request did not complete: %v", eventTypes(firstEvents))
	}

	sendResponsesWebSocketJSON(t, conn, responsesWebSocketWarmupPayload())
	event := readResponsesWebSocketEvent(t, conn)
	if got, _ := event["type"].(string); got != "error" {
		t.Fatalf("rate-limited event type = %q, want error", got)
	}
	if status, _ := event["status"].(float64); int(status) != http.StatusTooManyRequests {
		t.Fatalf("rate-limited status = %v, want %d", event["status"], http.StatusTooManyRequests)
	}
	errObj, _ := event["error"].(map[string]interface{})
	if got, _ := errObj["code"].(string); got != "rate_limit_exceeded" {
		t.Fatalf("rate-limited error code = %q, want rate_limit_exceeded", got)
	}
}

func responsesWebSocketSecurityConfig(key string) config.SecurityConfig {
	return config.SecurityConfig{
		Enabled:  true,
		Provider: "api_key",
		APIKey: config.APIKeyAuthConfig{
			Header: "Authorization",
			Prefix: "Bearer",
			Keys: []config.APIKeyCredential{
				{Name: "websocket-test", Value: key},
			},
		},
	}
}

func responsesWebSocketWarmupPayload() map[string]interface{} {
	return map[string]interface{}{
		"type":     "response.create",
		"model":    "mock-gpt",
		"generate": false,
	}
}

func TestResponsesWebSocket_RejectsUnknownEventType(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"mock-gpt","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	defer upstream.Close()

	h := newResponsesWebSocketTestHandler(upstream.URL)
	server := httptest.NewServer(http.HandlerFunc(h.ResponsesWebSocket))
	defer server.Close()

	conn := mustDialResponsesWebSocket(t, server.URL)
	defer conn.Close()

	sendResponsesWebSocketJSON(t, conn, map[string]interface{}{
		"type": "session.ping",
	})
	event := readResponsesWebSocketEvent(t, conn)
	if got, _ := event["type"].(string); got != "error" {
		t.Fatalf("expected error event, got %q", got)
	}
	errObj, _ := event["error"].(map[string]interface{})
	if errObj == nil {
		t.Fatalf("expected error payload in websocket response")
	}
	errType, _ := errObj["type"].(string)
	if errType != "invalid_request_error" {
		t.Fatalf("expected invalid_request_error, got %q", errType)
	}
}

func TestResponsesWebSocket_RejectsUnknownPreviousResponseID(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"mock-gpt\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"},\"finish_reason\":null}]}\n\n")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()

	h := newResponsesWebSocketTestHandler(upstream.URL)
	server := httptest.NewServer(http.HandlerFunc(h.ResponsesWebSocket))
	defer server.Close()

	conn := mustDialResponsesWebSocket(t, server.URL)
	defer conn.Close()

	sendResponsesWebSocketJSON(t, conn, map[string]interface{}{
		"type":                 "response.create",
		"model":                "lunargate/auto",
		"input":                "Continue",
		"previous_response_id": "resp_missing",
	})
	event := readResponsesWebSocketEvent(t, conn)
	if got, _ := event["type"].(string); got != "error" {
		t.Fatalf("expected error event, got %q", got)
	}
	errObj, _ := event["error"].(map[string]interface{})
	if errObj == nil {
		t.Fatalf("expected error payload in websocket response")
	}
	msg, _ := errObj["message"].(string)
	if !strings.Contains(msg, "Previous response with id 'resp_missing' not found.") {
		t.Fatalf("expected previous_response_not_found error, got %q", msg)
	}
	code, _ := errObj["code"].(string)
	if code != "previous_response_not_found" {
		t.Fatalf("expected previous_response_not_found code, got %q", code)
	}
}

func TestResponsesWebSocket_MapsUpstreamError(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"message":"bad request from provider","type":"invalid_request_error"}}`))
	}))
	defer upstream.Close()

	h := newResponsesWebSocketTestHandler(upstream.URL)
	server := httptest.NewServer(http.HandlerFunc(h.ResponsesWebSocket))
	defer server.Close()

	conn := mustDialResponsesWebSocket(t, server.URL)
	defer conn.Close()

	sendResponsesWebSocketJSON(t, conn, map[string]interface{}{
		"type":  "response.create",
		"model": "lunargate/auto",
		"input": "Say hi",
	})
	event := readResponsesWebSocketEvent(t, conn)
	if got, _ := event["type"].(string); got != "error" {
		t.Fatalf("expected error event, got %q", got)
	}
	errObj, _ := event["error"].(map[string]interface{})
	if errObj == nil {
		t.Fatalf("expected error payload in websocket response")
	}
	errType, _ := errObj["type"].(string)
	if errType != "invalid_request_error" {
		t.Fatalf("expected invalid_request_error, got %q", errType)
	}
	msg, _ := errObj["message"].(string)
	if msg != "bad request from provider" {
		t.Fatalf("expected provider error message passthrough, got %q", msg)
	}
	if status, _ := event["status"].(float64); int(status) != http.StatusBadRequest {
		t.Fatalf("expected websocket error status 400, got %v", event["status"])
	}
}

func TestResponsesWebSocket_ContinuesFromNonCompletedTerminal(t *testing.T) {
	tests := []struct {
		name              string
		firstTerminalType string
		firstStream       string
		wantMessageCount  int
	}{
		{
			name:              "incomplete",
			firstTerminalType: "response.incomplete",
			firstStream: "data: {\"id\":\"chatcmpl-incomplete\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"mock-gpt\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"}}]}\n\n" +
				"data: {\"id\":\"chatcmpl-incomplete\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"mock-gpt\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"length\"}]}\n\n" +
				"data: [DONE]\n\n",
			wantMessageCount: 3,
		},
		{
			name:              "failed",
			firstTerminalType: "response.failed",
			firstStream:       "data: {\"id\":\"chatcmpl-failed\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"mock-gpt\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"}}]}\n\n",
			wantMessageCount:  2,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var upstreamCalls int32
			var secondBody []byte
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				call := atomic.AddInt32(&upstreamCalls, 1)
				if call == 2 {
					var err error
					secondBody, err = io.ReadAll(r.Body)
					if err != nil {
						t.Fatalf("read continuation body: %v", err)
					}
				}
				w.Header().Set("Content-Type", "text/event-stream")
				if call == 1 {
					_, _ = io.WriteString(w, test.firstStream)
					return
				}
				_, _ = io.WriteString(w, "data: {\"id\":\"chatcmpl-completed\",\"object\":\"chat.completion.chunk\",\"created\":2,\"model\":\"mock-gpt\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"continued\"}}]}\n\n")
				_, _ = io.WriteString(w, "data: [DONE]\n\n")
			}))
			defer upstream.Close()

			h := newResponsesWebSocketTestHandler(upstream.URL)
			server := httptest.NewServer(http.HandlerFunc(h.ResponsesWebSocket))
			defer server.Close()
			conn := mustDialResponsesWebSocket(t, server.URL)
			defer conn.Close()

			sendResponsesWebSocketJSON(t, conn, map[string]interface{}{
				"type":  "response.create",
				"model": "lunargate/auto",
				"input": "Say hi",
			})
			firstEvents := readResponsesWebSocketEventsUntilTerminal(t, conn)
			firstResponseID := extractTerminalResponseID(firstEvents, test.firstTerminalType)
			if firstResponseID == "" {
				t.Fatalf("expected %s response ID; events=%v", test.firstTerminalType, eventTypes(firstEvents))
			}

			sendResponsesWebSocketJSON(t, conn, map[string]interface{}{
				"type":                 "response.create",
				"model":                "lunargate/auto",
				"input":                "Continue",
				"previous_response_id": firstResponseID,
			})
			continued := readResponsesWebSocketEventsUntilTerminal(t, conn)
			if !hasResponsesWebSocketEventType(continued, "response.completed") {
				t.Fatalf("continuation events = %v, want response.completed", eventTypes(continued))
			}
			if got := atomic.LoadInt32(&upstreamCalls); got != 2 {
				t.Fatalf("upstream calls = %d, want 2", got)
			}

			var body map[string]interface{}
			if err := json.Unmarshal(secondBody, &body); err != nil {
				t.Fatalf("decode continuation body: %v", err)
			}
			messages, _ := body["messages"].([]interface{})
			if len(messages) != test.wantMessageCount {
				t.Fatalf("continuation messages = %#v, want %d", messages, test.wantMessageCount)
			}
			last, _ := messages[len(messages)-1].(map[string]interface{})
			if last["role"] != "user" || last["content"] != "Continue" {
				t.Fatalf("last continuation message = %#v", last)
			}
		})
	}
}

func TestResponsesWebSocket_NativeStreamRequiresTerminalEvent(t *testing.T) {
	var upstreamCalls int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&upstreamCalls, 1)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "event: response.output_text.delta\r\ndata: {\"type\":\"response.output_text.delta\",\"response_id\":\"resp_partial\",\"delta\":\"partial\"}\r\n\r\n")
		_, _ = io.WriteString(w, "data: [DONE]\r\n\r\n")
	}))
	defer upstream.Close()

	h := newResponsesWebSocketTestHandlerWithUpstreamType(upstream.URL, "responses")
	server := httptest.NewServer(http.HandlerFunc(h.ResponsesWebSocket))
	defer server.Close()
	conn := mustDialResponsesWebSocket(t, server.URL)
	defer conn.Close()

	sendResponsesWebSocketJSON(t, conn, map[string]interface{}{
		"type":  "response.create",
		"model": "lunargate/auto",
		"input": "Say hi",
	})
	events := readResponsesWebSocketEventsUntilTerminal(t, conn)
	if !hasResponsesWebSocketEventType(events, "error") {
		t.Fatalf("expected terminal error for truncated native stream, got %v", eventTypes(events))
	}
	last := events[len(events)-1]
	errObj, _ := last["error"].(map[string]interface{})
	if code, _ := errObj["code"].(string); code != "upstream_stream_incomplete" {
		t.Fatalf("error code = %q, want upstream_stream_incomplete", code)
	}

	sendResponsesWebSocketJSON(t, conn, map[string]interface{}{
		"type":                 "response.create",
		"previous_response_id": "resp_partial",
		"input":                "Continue",
	})
	rejected := readResponsesWebSocketEvent(t, conn)
	if got, _ := rejected["type"].(string); got != "error" {
		t.Fatalf("expected uncached partial response to be rejected, got %q", got)
	}
	if got := atomic.LoadInt32(&upstreamCalls); got != 1 {
		t.Fatalf("upstream calls = %d, want 1", got)
	}
}

func TestResponsesWebSocket_CancelledTerminalIsNotCached(t *testing.T) {
	proxy := &responsesWebSocketProxy{}
	proxy.captureEventState([]byte(`{"type":"response.cancelled","response":{"id":"resp_cancelled","status":"cancelled"}}`))

	if !proxy.done || !proxy.terminalSeen {
		t.Fatal("cancelled response must be treated as terminal")
	}
	if proxy.terminalError == nil {
		t.Fatal("cancelled response must not be eligible for continuation caching")
	}
	if proxy.completedResponse != nil {
		t.Fatal("cancelled response must not be captured as completed")
	}
}

func TestResponsesWebSocket_GenerateFalseWarmupCachesStateForFollowUp(t *testing.T) {
	var upstreamCalls int32
	var capturedBody []byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&upstreamCalls, 1)
		var err error
		capturedBody, err = io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("failed to read upstream body: %v", err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"id\":\"chatcmpl-warm\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"mock-gpt\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"}}]}\n\n")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()

	h := newResponsesWebSocketTestHandler(upstream.URL)
	server := httptest.NewServer(http.HandlerFunc(h.ResponsesWebSocket))
	defer server.Close()

	conn := mustDialResponsesWebSocket(t, server.URL)
	defer conn.Close()

	sendResponsesWebSocketJSON(t, conn, map[string]interface{}{
		"type":         "response.create",
		"model":        "lunargate/auto",
		"generate":     false,
		"instructions": "Be helpful",
		"tools": []interface{}{
			map[string]interface{}{
				"type": "function",
				"name": "exec_command",
				"parameters": map[string]interface{}{
					"type":       "object",
					"properties": map[string]interface{}{},
				},
			},
		},
	})
	warmupEvents := readResponsesWebSocketEventsUntilTerminal(t, conn)
	if got := atomic.LoadInt32(&upstreamCalls); got != 0 {
		t.Fatalf("expected warmup to stay local to websocket session, got %d upstream calls", got)
	}
	if !hasResponsesWebSocketEventType(warmupEvents, "response.completed") {
		t.Fatalf("expected warmup response.completed event, got %v", eventTypes(warmupEvents))
	}
	warmupResponseID := extractCompletedResponseID(warmupEvents)
	if warmupResponseID == "" {
		t.Fatalf("expected warmup response id")
	}

	sendResponsesWebSocketJSON(t, conn, map[string]interface{}{
		"type":                 "response.create",
		"previous_response_id": warmupResponseID,
		"input": []interface{}{
			map[string]interface{}{
				"type": "message",
				"role": "user",
				"content": []interface{}{
					map[string]interface{}{
						"type": "input_text",
						"text": "Say hi",
					},
				},
			},
		},
	})
	events := readResponsesWebSocketEventsUntilTerminal(t, conn)
	if !hasResponsesWebSocketEventType(events, "response.completed") {
		t.Fatalf("expected generated follow-up to complete, got %v", eventTypes(events))
	}

	if got := atomic.LoadInt32(&upstreamCalls); got != 1 {
		t.Fatalf("expected exactly one upstream call after follow-up, got %d", got)
	}

	var body map[string]interface{}
	if err := json.Unmarshal(capturedBody, &body); err != nil {
		t.Fatalf("failed to decode upstream body: %v", err)
	}
	if body["previous_response_id"] != nil {
		t.Fatalf("expected websocket layer to resolve previous_response_id locally, got %v", body["previous_response_id"])
	}
	if stream, _ := body["stream"].(bool); !stream {
		t.Fatalf("expected stream=true in upstream request")
	}
	messages, _ := body["messages"].([]interface{})
	if len(messages) != 2 {
		t.Fatalf("expected developer+user messages, got %d", len(messages))
	}
	developerMsg, _ := messages[0].(map[string]interface{})
	if role, _ := developerMsg["role"].(string); role != "developer" {
		t.Fatalf("expected first message role developer, got %q", role)
	}
	userMsg, _ := messages[1].(map[string]interface{})
	if role, _ := userMsg["role"].(string); role != "user" {
		t.Fatalf("expected second message role user, got %q", role)
	}
	if content, _ := userMsg["content"].(string); content != "Say hi" {
		t.Fatalf("expected user message content %q, got %q", "Say hi", content)
	}
	tools, _ := body["tools"].([]interface{})
	if len(tools) != 1 {
		t.Fatalf("expected warmup tools to be preserved, got %d", len(tools))
	}
}

func TestResponsesWebSocket_ContinuationIncludesPriorToolCallOutputHistory(t *testing.T) {
	var callCount int32
	var secondBody []byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		call := atomic.AddInt32(&callCount, 1)
		if call == 2 {
			var err error
			secondBody, err = io.ReadAll(r.Body)
			if err != nil {
				t.Fatalf("failed to read second upstream body: %v", err)
			}
		}
		w.Header().Set("Content-Type", "text/event-stream")
		switch call {
		case 1:
			_, _ = io.WriteString(w, "data: {\"id\":\"chatcmpl-tool\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"mock-gpt\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_assemble\",\"type\":\"function\",\"function\":{\"name\":\"exec_command\",\"arguments\":\"{\\\"cmd\\\":\\\"pwd\\\"}\"}}]}}]}\n\n")
			_, _ = io.WriteString(w, "data: [DONE]\n\n")
		default:
			_, _ = io.WriteString(w, "data: {\"id\":\"chatcmpl-final\",\"object\":\"chat.completion.chunk\",\"created\":2,\"model\":\"mock-gpt\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"done\"}}]}\n\n")
			_, _ = io.WriteString(w, "data: [DONE]\n\n")
		}
	}))
	defer upstream.Close()

	h := newResponsesWebSocketTestHandler(upstream.URL)
	server := httptest.NewServer(http.HandlerFunc(h.ResponsesWebSocket))
	defer server.Close()

	conn := mustDialResponsesWebSocket(t, server.URL)
	defer conn.Close()

	sendResponsesWebSocketJSON(t, conn, map[string]interface{}{
		"type":  "response.create",
		"model": "lunargate/auto",
		"input": "Run pwd",
		"tools": []interface{}{
			map[string]interface{}{
				"type": "function",
				"name": "exec_command",
				"parameters": map[string]interface{}{
					"type":       "object",
					"properties": map[string]interface{}{},
				},
			},
		},
	})
	firstEvents := readResponsesWebSocketEventsUntilTerminal(t, conn)
	firstResponseID := extractCompletedResponseID(firstEvents)
	if firstResponseID == "" {
		t.Fatalf("expected first response id")
	}

	sendResponsesWebSocketJSON(t, conn, map[string]interface{}{
		"type":                 "response.create",
		"previous_response_id": firstResponseID,
		"input": []interface{}{
			map[string]interface{}{
				"type":    "function_call_output",
				"call_id": "call_assemble",
				"output":  "/tmp",
			},
		},
	})
	secondEvents := readResponsesWebSocketEventsUntilTerminal(t, conn)
	if !hasResponsesWebSocketEventType(secondEvents, "response.completed") {
		t.Fatalf("expected second response.completed event, got %v", eventTypes(secondEvents))
	}

	var body map[string]interface{}
	if err := json.Unmarshal(secondBody, &body); err != nil {
		t.Fatalf("failed to decode second upstream body: %v", err)
	}
	messages, _ := body["messages"].([]interface{})
	if len(messages) != 3 {
		t.Fatalf("expected user + assistant tool call + tool output history, got %d", len(messages))
	}
	assistant, _ := messages[1].(map[string]interface{})
	if role, _ := assistant["role"].(string); role != "assistant" {
		t.Fatalf("expected assistant history message, got %q", role)
	}
	toolCalls, _ := assistant["tool_calls"].([]interface{})
	if len(toolCalls) != 1 {
		t.Fatalf("expected assistant tool call history, got %d tool calls", len(toolCalls))
	}
	toolResp, _ := messages[2].(map[string]interface{})
	if role, _ := toolResp["role"].(string); role != "tool" {
		t.Fatalf("expected tool output message, got %q", role)
	}
	if got, _ := toolResp["tool_call_id"].(string); got != "call_assemble" {
		t.Fatalf("expected tool_call_id call_assemble, got %q", got)
	}
}

func TestResponsesWebSocket_EvictsPreviousResponseCacheAfterFailedContinuation(t *testing.T) {
	var upstreamCalls int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		call := atomic.AddInt32(&upstreamCalls, 1)
		if call == 1 {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":{"message":"continuation failed","type":"invalid_request_error"}}`))
			return
		}
		t.Fatalf("unexpected second upstream call after cache eviction")
	}))
	defer upstream.Close()

	h := newResponsesWebSocketTestHandler(upstream.URL)
	server := httptest.NewServer(http.HandlerFunc(h.ResponsesWebSocket))
	defer server.Close()

	conn := mustDialResponsesWebSocket(t, server.URL)
	defer conn.Close()

	sendResponsesWebSocketJSON(t, conn, map[string]interface{}{
		"type":     "response.create",
		"model":    "lunargate/auto",
		"generate": false,
		"tools": []interface{}{
			map[string]interface{}{
				"type": "function",
				"name": "exec_command",
				"parameters": map[string]interface{}{
					"type":       "object",
					"properties": map[string]interface{}{},
				},
			},
		},
	})
	warmupEvents := readResponsesWebSocketEventsUntilTerminal(t, conn)
	warmupResponseID := extractCompletedResponseID(warmupEvents)
	if warmupResponseID == "" {
		t.Fatalf("expected warmup response id")
	}

	sendResponsesWebSocketJSON(t, conn, map[string]interface{}{
		"type":                 "response.create",
		"previous_response_id": warmupResponseID,
		"input":                "Run command",
	})
	failed := readResponsesWebSocketEvent(t, conn)
	if got, _ := failed["type"].(string); got != "error" {
		t.Fatalf("expected continuation failure error event, got %q", got)
	}

	sendResponsesWebSocketJSON(t, conn, map[string]interface{}{
		"type":                 "response.create",
		"previous_response_id": warmupResponseID,
		"input":                "Retry command",
	})
	evicted := readResponsesWebSocketEvent(t, conn)
	if got, _ := evicted["type"].(string); got != "error" {
		t.Fatalf("expected previous_response_not_found after eviction, got %q", got)
	}
	errObj, _ := evicted["error"].(map[string]interface{})
	if code, _ := errObj["code"].(string); code != "previous_response_not_found" {
		t.Fatalf("expected previous_response_not_found after eviction, got %q", code)
	}
}

func newResponsesWebSocketTestHandler(upstreamURL string) *Handler {
	return newResponsesWebSocketTestHandlerWithUpstreamType(upstreamURL, "")
}

func newResponsesWebSocketTestHandlerWithUpstreamType(upstreamURL, upstreamRequestType string) *Handler {
	providerID := "openai"
	cfgProviders := map[string]config.ProviderConfig{
		providerID: {Type: "openai", APIKey: "dummy", BaseURL: upstreamURL},
	}
	reg := providers.NewRegistry(cfgProviders)
	router := routing.NewEngine(config.RoutingConfig{
		DefaultStrategy: "weighted",
		Routes: []config.RouteConfig{
			{
				Name:    "responses-default",
				Match:   config.MatchConfig{Path: "/v1/responses"},
				Targets: []config.TargetConfig{{Provider: providerID, Model: "mock-gpt", Weight: 1, UpstreamRequestType: upstreamRequestType}},
			},
		},
	})
	retrier := resilience.NewRetrier(config.RetryConfig{Enabled: false})
	cbm := resilience.NewCircuitBreakerManager()
	fb := resilience.NewFallbackExecutor(retrier, cbm)
	cache := middleware.NewCache(config.CacheConfig{Enabled: false})
	streamer := streaming.NewHandler()
	metrics := observability.NewMetricsWithRegisterer(prometheus.NewRegistry())
	return NewHandler(reg, router, fb, cache, streamer, metrics, nil, nil, nil)
}

func mustDialResponsesWebSocket(t *testing.T, serverURL string) *websocket.Conn {
	return mustDialResponsesWebSocketWithHeaders(t, serverURL, nil)
}

func mustDialResponsesWebSocketWithHeaders(t *testing.T, serverURL string, headers http.Header) *websocket.Conn {
	t.Helper()
	wsURL := "ws" + strings.TrimPrefix(serverURL, "http") + "/v1/responses"
	conn, resp, err := websocket.DefaultDialer.Dial(wsURL, headers)
	if err != nil {
		statusCode := 0
		if resp != nil {
			statusCode = resp.StatusCode
		}
		t.Fatalf("failed to connect websocket (status=%d): %v", statusCode, err)
	}
	return conn
}

func sendResponsesWebSocketJSON(t *testing.T, conn *websocket.Conn, payload map[string]interface{}) {
	t.Helper()
	b, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("failed to marshal websocket payload: %v", err)
	}
	if err := conn.WriteMessage(websocket.TextMessage, b); err != nil {
		t.Fatalf("failed to write websocket payload: %v", err)
	}
}

func readResponsesWebSocketEvent(t *testing.T, conn *websocket.Conn) map[string]interface{} {
	t.Helper()
	_ = conn.SetReadDeadline(time.Now().Add(5 * time.Second))
	_, msg, err := conn.ReadMessage()
	if err != nil {
		t.Fatalf("failed to read websocket event: %v", err)
	}
	var payload map[string]interface{}
	if err := json.Unmarshal(msg, &payload); err != nil {
		t.Fatalf("failed to decode websocket payload %q: %v", string(msg), err)
	}
	return payload
}

func readResponsesWebSocketEventsUntilTerminal(t *testing.T, conn *websocket.Conn) []map[string]interface{} {
	t.Helper()
	events := make([]map[string]interface{}, 0, 8)
	for i := 0; i < 64; i++ {
		event := readResponsesWebSocketEvent(t, conn)
		events = append(events, event)
		typ, _ := event["type"].(string)
		if typ == "response.completed" || typ == "response.failed" || typ == "response.incomplete" || typ == "error" {
			break
		}
	}
	return events
}

func hasResponsesWebSocketEventType(events []map[string]interface{}, eventType string) bool {
	for _, event := range events {
		if typ, _ := event["type"].(string); typ == eventType {
			return true
		}
	}
	return false
}

func eventTypes(events []map[string]interface{}) []string {
	out := make([]string, 0, len(events))
	for _, event := range events {
		typ, _ := event["type"].(string)
		out = append(out, typ)
	}
	return out
}

func extractCompletedResponseID(events []map[string]interface{}) string {
	return extractTerminalResponseID(events, "response.completed")
}

func extractTerminalResponseID(events []map[string]interface{}, terminalType string) string {
	for _, event := range events {
		if typ, _ := event["type"].(string); typ != terminalType {
			continue
		}
		response, _ := event["response"].(map[string]interface{})
		if response == nil {
			continue
		}
		id, _ := response["id"].(string)
		if strings.TrimSpace(id) != "" {
			return strings.TrimSpace(id)
		}
	}
	return ""
}

func TestMakeResponsesWebSocketHTTPRequest_SetsSessionIDWhenMissing(t *testing.T) {
	baseReq := httptest.NewRequest(http.MethodGet, "http://example.com/v1/responses", nil)
	body := []byte(`{"type":"response.create","model":"lunargate/auto","input":"hello"}`)

	req := makeResponsesWebSocketHTTPRequest(baseReq, body, "wsresp_test_session")

	if got := req.Header.Get("x-lunargate-sessionid"); got != "wsresp_test_session" {
		t.Fatalf("expected injected session header, got %q", got)
	}
	if req.Method != http.MethodPost {
		t.Fatalf("expected POST method, got %s", req.Method)
	}
	if req.URL.Path != "/v1/responses" {
		t.Fatalf("expected /v1/responses path, got %q", req.URL.Path)
	}
	gotBody, err := io.ReadAll(req.Body)
	if err != nil {
		t.Fatalf("failed to read request body: %v", err)
	}
	if !bytes.Equal(gotBody, body) {
		t.Fatalf("expected request body to be preserved")
	}
}

func TestMakeResponsesWebSocketHTTPRequest_PreservesExistingSessionID(t *testing.T) {
	baseReq := httptest.NewRequest(http.MethodGet, "http://example.com/v1/responses", nil)
	baseReq.Header.Set("x-lunargate-sessionid", "client_session")
	body := []byte(`{"type":"response.create","model":"lunargate/auto","input":"hello"}`)

	req := makeResponsesWebSocketHTTPRequest(baseReq, body, "wsresp_generated")

	if got := req.Header.Get("x-lunargate-sessionid"); got != "client_session" {
		t.Fatalf("expected client session header to win, got %q", got)
	}
}

func TestResponsesWebSocketSSEDecoderHandlesCRLFAndMultilineData(t *testing.T) {
	raw := []byte(": keepalive\r\nevent: response.completed\r\ndata:{\"type\":\"response.completed\",\r\ndata: \"response\":{\"id\":\"resp_native\",\"status\":\"completed\"}}\r\n\r\nnext")

	frame, remaining, ok, err := nextResponsesSSEFrame(raw)
	if err != nil {
		t.Fatalf("decode SSE frame: %v", err)
	}
	if !ok {
		t.Fatal("expected one complete SSE frame")
	}
	if got, want := string(remaining), "next"; got != want {
		t.Fatalf("remaining bytes = %q, want %q", got, want)
	}
	payload, ok := responsesSSEData(frame)
	if !ok {
		t.Fatal("expected a data payload")
	}
	var event map[string]interface{}
	if err := json.Unmarshal(payload, &event); err != nil {
		t.Fatalf("multiline SSE payload is not valid JSON: %v; payload=%q", err, payload)
	}
	if got, _ := event["type"].(string); got != "response.completed" {
		t.Fatalf("event type = %q, want response.completed", got)
	}
	response, _ := event["response"].(map[string]interface{})
	if got, _ := response["id"].(string); got != "resp_native" {
		t.Fatalf("response id = %q, want resp_native", got)
	}
}

func TestResponsesWebSocketSSEDecoderWaitsForCompleteCRLFFrame(t *testing.T) {
	raw := []byte("event: response.completed\r\ndata: {\"type\":\"response.completed\"}\r\n")
	if _, _, ok, err := nextResponsesSSEFrame(raw); err != nil || ok {
		if err != nil {
			t.Fatalf("decode incomplete SSE frame: %v", err)
		}
		t.Fatal("unterminated SSE event must stay buffered")
	}
}

func TestIsBenignResponsesWebSocketClose(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want bool
	}{
		{
			name: "normal closure",
			err: &websocket.CloseError{
				Code: websocket.CloseNormalClosure,
				Text: "bye",
			},
			want: true,
		},
		{
			name: "going away",
			err: &websocket.CloseError{
				Code: websocket.CloseGoingAway,
				Text: "going away",
			},
			want: true,
		},
		{
			name: "no status received",
			err: &websocket.CloseError{
				Code: websocket.CloseNoStatusReceived,
				Text: "",
			},
			want: true,
		},
		{
			name: "abnormal closure",
			err: &websocket.CloseError{
				Code: websocket.CloseAbnormalClosure,
				Text: "unexpected EOF",
			},
			want: true,
		},
		{
			name: "wrapped unexpected eof",
			err:  errors.New("read tcp 127.0.0.1:1234->127.0.0.1:8080: unexpected EOF"),
			want: true,
		},
		{
			name: "other error",
			err:  errors.New("broken pipe"),
			want: false,
		},
		{
			name: "nil error",
			err:  nil,
			want: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := isBenignResponsesWebSocketClose(tc.err)
			if got != tc.want {
				t.Fatalf("expected %v, got %v", tc.want, got)
			}
		})
	}
}
