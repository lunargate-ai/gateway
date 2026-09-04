package api

import (
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/lunargate-ai/gateway/internal/health"
)

func TestTranslatedResponsesStoreIncompleteAndFailedTerminals(t *testing.T) {
	tests := []struct {
		name          string
		upstreamMode  string
		stream        bool
		wantEvent     string
		wantStatus    string
		wantItemRoles []string
	}{
		{
			name:          "non-stream incomplete",
			upstreamMode:  "nonstream_incomplete",
			wantStatus:    "incomplete",
			wantItemRoles: []string{"user", "assistant"},
		},
		{
			name:          "stream incomplete",
			upstreamMode:  "stream_incomplete",
			stream:        true,
			wantEvent:     "response.incomplete",
			wantStatus:    "incomplete",
			wantItemRoles: []string{"user", "assistant"},
		},
		{
			name:          "stream failed",
			upstreamMode:  "stream_failed",
			stream:        true,
			wantEvent:     "response.failed",
			wantStatus:    "failed",
			wantItemRoles: []string{"user"},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			router, handler, closeTest := newTranslatedTerminalLifecycleRouter(t, test.upstreamMode)
			defer closeTest()
			conversation, err := handler.conversationsState.create(nil, nil)
			if err != nil {
				t.Fatalf("create conversation: %v", err)
			}

			body := fmt.Sprintf(`{
				"model":"mock-gpt",
				"conversation":%q,
				"input":"terminal input",
				"stream":%t,
				"store":true
			}`, conversation.ID, test.stream)
			created := performLifecycleRequest(t, router, http.MethodPost, "/v1/responses", []byte(body))
			if created.Code != http.StatusOK {
				t.Fatalf("create status = %d, want 200; body=%s", created.Code, created.Body.String())
			}

			terminal := decodeTranslatedTerminalResponse(t, created, test.wantEvent)
			if terminal["status"] != test.wantStatus {
				t.Fatalf("terminal status = %#v, want %q", terminal["status"], test.wantStatus)
			}
			responseConversation, _ := terminal["conversation"].(map[string]interface{})
			if responseConversation["id"] != conversation.ID {
				t.Fatalf("terminal conversation = %#v, want %q", terminal["conversation"], conversation.ID)
			}

			responseID := lifecycleStringField(t, terminal, "id")
			retrievedRecorder := performLifecycleRequest(t, router, http.MethodGet, "/v1/responses/"+responseID, nil)
			if retrievedRecorder.Code != http.StatusOK {
				t.Fatalf("retrieve status = %d, want 200; body=%s", retrievedRecorder.Code, retrievedRecorder.Body.String())
			}
			retrieved := decodeLifecycleObject(t, retrievedRecorder.Body.Bytes())
			if !reflect.DeepEqual(retrieved, terminal) {
				t.Fatalf("retrieved response = %#v, want terminal %#v", retrieved, terminal)
			}

			items, ok := handler.conversationsState.getItems(conversation.ID)
			if !ok || len(items) != len(test.wantItemRoles) {
				t.Fatalf("conversation items = %#v, want %d items", items, len(test.wantItemRoles))
			}
			for index, wantRole := range test.wantItemRoles {
				if role := parseJSONStringRaw(items[index]["role"]); role != wantRole {
					t.Fatalf("conversation item %d role = %q, want %q", index, role, wantRole)
				}
			}
		})
	}
}

func TestResponsesWebSocketStoresTranslatedNonCompletedTerminals(t *testing.T) {
	tests := []struct {
		name         string
		upstreamMode string
		wantEvent    string
		wantStatus   string
	}{
		{name: "incomplete", upstreamMode: "stream_incomplete", wantEvent: "response.incomplete", wantStatus: "incomplete"},
		{name: "failed", upstreamMode: "stream_failed", wantEvent: "response.failed", wantStatus: "failed"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			router, handler, closeTest := newTranslatedTerminalLifecycleRouter(t, test.upstreamMode)
			defer closeTest()
			server := httptest.NewServer(router)
			defer server.Close()
			conversation, err := handler.conversationsState.create(nil, nil)
			if err != nil {
				t.Fatalf("create conversation: %v", err)
			}

			conn := mustDialResponsesWebSocket(t, server.URL)
			defer conn.Close()
			sendResponsesWebSocketJSON(t, conn, map[string]interface{}{
				"type":         "response.create",
				"model":        "mock-gpt",
				"conversation": conversation.ID,
				"input":        "websocket terminal input",
				"store":        true,
			})
			events := readResponsesWebSocketEventsUntilTerminal(t, conn)
			terminal := responsesWebSocketTerminalResponse(t, events, test.wantEvent)
			if terminal["status"] != test.wantStatus {
				t.Fatalf("terminal status = %#v, want %q", terminal["status"], test.wantStatus)
			}
			responseConversation, _ := terminal["conversation"].(map[string]interface{})
			if responseConversation["id"] != conversation.ID {
				t.Fatalf("terminal conversation = %#v, want %q", terminal["conversation"], conversation.ID)
			}
			responseID := lifecycleStringField(t, terminal, "id")

			// A following request proves the first synchronous HTTP bridge has
			// finished its post-stream conversation and lifecycle updates.
			sendResponsesWebSocketJSON(t, conn, map[string]interface{}{
				"type":     "response.create",
				"model":    "mock-gpt",
				"generate": false,
			})
			warmup := readResponsesWebSocketEventsUntilTerminal(t, conn)
			if !hasResponsesWebSocketEventType(warmup, "response.completed") {
				t.Fatalf("warmup events = %v, want response.completed", eventTypes(warmup))
			}

			retrieved := performLifecycleRequest(t, router, http.MethodGet, "/v1/responses/"+responseID, nil)
			if retrieved.Code != http.StatusOK {
				t.Fatalf("retrieve status = %d, want 200; body=%s", retrieved.Code, retrieved.Body.String())
			}
			if got := decodeLifecycleObject(t, retrieved.Body.Bytes())["status"]; got != test.wantStatus {
				t.Fatalf("retrieved status = %#v, want %q", got, test.wantStatus)
			}
		})
	}
}

func newTranslatedTerminalLifecycleRouter(
	t *testing.T,
	mode string,
) (http.Handler, *Handler, func()) {
	t.Helper()
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if _, err := io.Copy(io.Discard, r.Body); err != nil {
			t.Errorf("read upstream request: %v", err)
		}
		switch mode {
		case "nonstream_incomplete":
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{
				"id":"chatcmpl_terminal",
				"object":"chat.completion",
				"created":1,
				"model":"mock-gpt",
				"choices":[{"index":0,"message":{"role":"assistant","content":"partial answer"},"finish_reason":"length"}]
			}`)
		case "stream_incomplete":
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, "data: {\"id\":\"chatcmpl_terminal\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"mock-gpt\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"partial answer\"},\"finish_reason\":null}]}\n\n")
			_, _ = io.WriteString(w, "data: {\"id\":\"chatcmpl_terminal\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"mock-gpt\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"length\"}]}\n\n")
			_, _ = io.WriteString(w, "data: [DONE]\n\n")
		case "stream_failed":
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, "data: {\"error\":{\"type\":\"server_error\",\"message\":\"upstream private diagnostic\"}}\n\n")
		default:
			t.Errorf("unknown upstream mode %q", mode)
			w.WriteHeader(http.StatusInternalServerError)
		}
	}))

	handler := newResponsesWebSocketTestHandler(upstream.URL)
	router := NewRouter(handler, nil, nil, health.NewChecker("test"))
	return router, handler, func() {
		handler.cache.Stop()
		upstream.Close()
	}
}

func decodeTranslatedTerminalResponse(
	t *testing.T,
	response *httptest.ResponseRecorder,
	eventType string,
) map[string]interface{} {
	t.Helper()
	if eventType == "" {
		return decodeLifecycleObject(t, response.Body.Bytes())
	}
	for _, event := range decodeSSEEvents(t, response.Body.String()) {
		if event["type"] != eventType {
			continue
		}
		terminal, _ := event["response"].(map[string]interface{})
		if terminal == nil {
			t.Fatalf("%s response = %#v, want object", eventType, event["response"])
		}
		return terminal
	}
	t.Fatalf("terminal event %q not found in %s", eventType, response.Body.String())
	return nil
}

func responsesWebSocketTerminalResponse(
	t *testing.T,
	events []map[string]interface{},
	eventType string,
) map[string]interface{} {
	t.Helper()
	for _, event := range events {
		if event["type"] != eventType {
			continue
		}
		response, _ := event["response"].(map[string]interface{})
		if response == nil {
			t.Fatalf("%s response = %#v, want object", eventType, event["response"])
		}
		return response
	}
	t.Fatalf("terminal event %q not found in %v", eventType, eventTypes(events))
	return nil
}

func TestResponsesStreamFailureRecordsTerminalState(t *testing.T) {
	recorder := httptest.NewRecorder()
	proxy := newResponsesStreamProxy(recorder)
	callbackCalls := 0
	proxy.beforeTerminal = func(response map[string]interface{}) {
		callbackCalls++
		response["conversation"] = map[string]interface{}{"id": "conv_terminal"}
	}
	if _, err := proxy.Write([]byte("data: {\"error\":{\"type\":\"server_error\",\"message\":\"private\"}}\n\n")); err != nil {
		t.Fatalf("write stream error: %v", err)
	}
	if _, err := proxy.Write([]byte("data: [DONE]\n\n")); err != nil {
		t.Fatalf("write done: %v", err)
	}
	if err := proxy.finalize(); err != nil {
		t.Fatalf("finalize: %v", err)
	}

	if callbackCalls != 1 {
		t.Fatalf("beforeTerminal calls = %d, want 1", callbackCalls)
	}
	if proxy.terminalResponse == nil || proxy.terminalResponse["status"] != "failed" {
		t.Fatalf("terminal response = %#v, want failed", proxy.terminalResponse)
	}
	if proxy.completedResponse != nil {
		t.Fatalf("completed response = %#v, want nil", proxy.completedResponse)
	}
	conversation, _ := proxy.terminalResponse["conversation"].(map[string]interface{})
	if conversation["id"] != "conv_terminal" {
		t.Fatalf("terminal conversation = %#v", proxy.terminalResponse["conversation"])
	}
}
