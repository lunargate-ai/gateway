package api

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
)

func TestResponsesUsesAndUpdatesLocalConversation(t *testing.T) {
	var upstreamPayload map[string]interface{}
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("upstream path = %q", r.URL.Path)
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(body, &upstreamPayload); err != nil {
			t.Fatalf("decode upstream payload: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"chatcmpl_conversation",
			"object":"chat.completion",
			"created":123,
			"model":"gpt-test",
			"choices":[{"index":0,"message":{"role":"assistant","content":"answer"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}
		}`))
	}))
	defer upstream.Close()

	handler, cache := newNativeContinuationTestHandler(t, upstream.URL+"/v1", "chat_completions")
	defer cache.Stop()
	initial, err := prepareConversationItems([]json.RawMessage{
		json.RawMessage(`{"role":"user","content":"history","phase":"final_answer"}`),
	})
	if err != nil {
		t.Fatal(err)
	}
	conversation, err := handler.conversationsState.create(map[string]string{"test": "conversation"}, initial)
	if err != nil {
		t.Fatal(err)
	}

	payload := []byte(`{
		"model":"gpt-5.4",
		"conversation":{"id":"` + conversation.ID + `"},
		"input":"new input",
		"store":false
	}`)
	recorder := httptest.NewRecorder()
	handler.Responses(recorder, httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(payload)))
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", recorder.Code, recorder.Body.String())
	}

	messages, ok := upstreamPayload["messages"].([]interface{})
	if !ok || len(messages) != 2 {
		t.Fatalf("upstream messages = %#v", upstreamPayload["messages"])
	}
	if got := messages[0].(map[string]interface{})["content"]; got != "history" {
		t.Fatalf("first upstream content = %#v", got)
	}
	if got := messages[1].(map[string]interface{})["content"]; got != "new input" {
		t.Fatalf("second upstream content = %#v", got)
	}
	if _, leaked := upstreamPayload["conversation"]; leaked {
		t.Fatalf("local conversation ID leaked upstream: %#v", upstreamPayload)
	}

	var response map[string]interface{}
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	responseConversation, _ := response["conversation"].(map[string]interface{})
	if responseConversation["id"] != conversation.ID {
		t.Fatalf("response conversation = %#v", response["conversation"])
	}
	if response["status"] != "completed" || response["output_text"] != "answer" {
		t.Fatalf("response = %#v", response)
	}

	items, ok := handler.conversationsState.getItems(conversation.ID)
	if !ok || len(items) != 3 {
		t.Fatalf("conversation items = %#v, ok = %t", items, ok)
	}
	if parseJSONStringRaw(items[0]["phase"]) != "final_answer" {
		t.Fatalf("initial additive field was lost: %s", mustMarshalForTest(t, items[0]))
	}
	if parseJSONStringRaw(items[1]["role"]) != "user" || parseJSONStringRaw(items[2]["role"]) != "assistant" {
		t.Fatalf("conversation item roles = %q, %q", parseJSONStringRaw(items[1]["role"]), parseJSONStringRaw(items[2]["role"]))
	}
	responseID, _ := response["id"].(string)
	if _, _, stored := handler.responsesState.getCompleted(responseID); stored {
		t.Fatal("store:false response was retained outside the conversation")
	}
}

func TestStreamingResponsesUpdatesLocalConversationAndTerminalEvent(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"id\":\"chatcmpl_stream_conversation\",\"object\":\"chat.completion.chunk\",\"created\":123,\"model\":\"gpt-test\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"answer\"},\"finish_reason\":null}]}\n\n")
		_, _ = io.WriteString(w, "data: {\"id\":\"chatcmpl_stream_conversation\",\"object\":\"chat.completion.chunk\",\"created\":123,\"model\":\"gpt-test\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()

	handler, cache := newNativeContinuationTestHandler(t, upstream.URL+"/v1", "chat_completions")
	defer cache.Stop()
	initial, err := prepareConversationItems([]json.RawMessage{
		json.RawMessage(`{"role":"user","content":"history"}`),
	})
	if err != nil {
		t.Fatal(err)
	}
	conversation, err := handler.conversationsState.create(nil, initial)
	if err != nil {
		t.Fatal(err)
	}

	recorder := httptest.NewRecorder()
	handler.Responses(recorder, httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewBufferString(`{
		"model":"gpt-5.4",
		"conversation":"`+conversation.ID+`",
		"input":"new input",
		"stream":true
	}`)))
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", recorder.Code, recorder.Body.String())
	}
	events := decodeSSEEvents(t, recorder.Body.String())
	var terminal map[string]interface{}
	for _, event := range events {
		if event["type"] == "response.completed" {
			terminal, _ = event["response"].(map[string]interface{})
		}
	}
	if terminal == nil {
		t.Fatalf("missing terminal response: %s", recorder.Body.String())
	}
	responseConversation, _ := terminal["conversation"].(map[string]interface{})
	if responseConversation["id"] != conversation.ID {
		t.Fatalf("terminal conversation = %#v", terminal["conversation"])
	}
	items, ok := handler.conversationsState.getItems(conversation.ID)
	if !ok || len(items) != 3 {
		t.Fatalf("conversation items = %#v, ok = %t", items, ok)
	}
	if parseJSONStringRaw(items[2]["role"]) != "assistant" {
		t.Fatalf("last item = %s", mustMarshalForTest(t, items[2]))
	}
}

func TestResponsesRejectsConflictingConversationAndPreviousResponse(t *testing.T) {
	upstreamCalls := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamCalls++
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer upstream.Close()
	handler, cache := newNativeContinuationTestHandler(t, upstream.URL+"/v1", "responses")
	defer cache.Stop()

	recorder := httptest.NewRecorder()
	handler.Responses(recorder, httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewBufferString(`{
		"model":"gpt-5.4",
		"conversation":"conv_test",
		"previous_response_id":"resp_test",
		"input":"hello"
	}`)))
	assertConversationError(t, recorder, http.StatusBadRequest, "conversation", "invalid_parameter_combination")
	if upstreamCalls != 0 {
		t.Fatalf("upstream calls = %d", upstreamCalls)
	}
}

func TestResponsesRejectsInvalidConversationShape(t *testing.T) {
	handler, cache := newNativeContinuationTestHandler(t, "http://127.0.0.1:1/v1", "responses")
	defer cache.Stop()
	recorder := httptest.NewRecorder()
	handler.Responses(recorder, httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewBufferString(`{
		"model":"gpt-5.4",
		"conversation":42,
		"input":"hello"
	}`)))
	assertConversationError(t, recorder, http.StatusBadRequest, "conversation", "invalid_value")
}

func TestResponsesPassesRemoteConversationToNativeTarget(t *testing.T) {
	var upstreamConversation interface{}
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatal(err)
		}
		upstreamConversation = payload["conversation"]
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"resp_remote",
			"object":"response",
			"created_at":123,
			"status":"completed",
			"model":"gpt-test",
			"output":[{"id":"msg_remote","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"ok"}]}],
			"output_text":"ok"
		}`))
	}))
	defer upstream.Close()
	router, handler, cache := newNativeLifecycleRouter(t, upstream.URL+"/v1", map[string]config.ProviderCapabilities{
		"native": {Conversations: true, ResponsesLifecycle: true},
	})
	defer cache.Stop()
	binding, err := handler.validateConversationProvider("native")
	if err != nil {
		t.Fatal(err)
	}
	handler.conversationBindings.put("conv_remote", binding)

	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewBufferString(`{
		"model":"native/gpt-native",
		"conversation":{"id":"conv_remote"},
		"input":"hello"
	}`))
	router.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", recorder.Code, recorder.Body.String())
	}
	conversation, ok := upstreamConversation.(map[string]interface{})
	if !ok || conversation["id"] != "conv_remote" {
		t.Fatalf("upstream conversation = %#v", upstreamConversation)
	}
}

func TestResponsesRejectsUnknownRemoteConversationWithoutBinding(t *testing.T) {
	upstreamCalls := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamCalls++
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer upstream.Close()
	handler, cache := newNativeContinuationTestHandler(t, upstream.URL+"/v1", "chat_completions")
	defer cache.Stop()

	recorder := httptest.NewRecorder()
	handler.Responses(recorder, httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewBufferString(`{
		"model":"gpt-5.4",
		"conversation":"conv_remote",
		"input":"hello"
	}`)))
	assertConversationError(t, recorder, http.StatusNotFound, "conversation_id", "conversation_not_found")
	if upstreamCalls != 0 {
		t.Fatalf("upstream calls = %d", upstreamCalls)
	}
}
