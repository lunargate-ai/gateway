package api

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestResponsesPreservesTranslatedChatRefusal(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("upstream path = %q", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"chatcmpl_refusal",
			"object":"chat.completion",
			"created":123,
			"model":"gpt-test",
			"choices":[{"index":0,"message":{"role":"assistant","content":null,"refusal":"I can't help with that."},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}
		}`)
	}))
	defer upstream.Close()

	handler, cache := newNativeContinuationTestHandler(t, upstream.URL+"/v1", requestTypeChatCompletions)
	defer cache.Stop()
	recorder := httptest.NewRecorder()
	handler.Responses(recorder, httptest.NewRequest(
		http.MethodPost,
		"/v1/responses",
		strings.NewReader(`{"model":"gpt-5.4","input":"unsafe request"}`),
	))

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, body=%s", recorder.Code, recorder.Body.String())
	}
	var response map[string]interface{}
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if response["output_text"] != "" {
		t.Fatalf("output_text = %#v, want empty for refusal", response["output_text"])
	}
	output, _ := response["output"].([]interface{})
	if len(output) != 1 {
		t.Fatalf("output = %#v", response["output"])
	}
	message, _ := output[0].(map[string]interface{})
	content, _ := message["content"].([]interface{})
	if len(content) != 1 {
		t.Fatalf("message content = %#v", message["content"])
	}
	refusal, _ := content[0].(map[string]interface{})
	if refusal["type"] != "refusal" || refusal["refusal"] != "I can't help with that." {
		t.Fatalf("refusal content = %#v", refusal)
	}
	if _, disguised := refusal["text"]; disguised {
		t.Fatalf("refusal was exposed as text: %#v", refusal)
	}
}

func TestResponsesStreamPreservesTranslatedChatRefusal(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("upstream path = %q", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		frames := []string{
			`data: {"id":"chatcmpl_refusal_stream","object":"chat.completion.chunk","created":123,"model":"gpt-test","choices":[{"index":0,"delta":{"role":"assistant","refusal":"I can't "},"finish_reason":null}]}` + "\n\n",
			`data: {"id":"chatcmpl_refusal_stream","object":"chat.completion.chunk","created":123,"model":"gpt-test","choices":[{"index":0,"delta":{"refusal":"help with that."},"finish_reason":null}]}` + "\n\n",
			`data: {"id":"chatcmpl_refusal_stream","object":"chat.completion.chunk","created":123,"model":"gpt-test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\n",
			`data: {"id":"chatcmpl_refusal_stream","object":"chat.completion.chunk","created":123,"model":"gpt-test","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}}` + "\n\n",
			"data: [DONE]\n\n",
		}
		for _, frame := range frames {
			_, _ = io.WriteString(w, frame)
		}
	}))
	defer upstream.Close()

	handler, cache := newNativeContinuationTestHandler(t, upstream.URL+"/v1", requestTypeChatCompletions)
	defer cache.Stop()
	recorder := httptest.NewRecorder()
	handler.Responses(recorder, httptest.NewRequest(
		http.MethodPost,
		"/v1/responses",
		strings.NewReader(`{"model":"gpt-5.4","input":"unsafe request","stream":true}`),
	))

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, body=%s", recorder.Code, recorder.Body.String())
	}
	events := decodeSSEEvents(t, recorder.Body.String())
	assertSequenceNumbersMonotonic(t, events)
	types := make([]string, 0, len(events))
	var deltas strings.Builder
	var refusalDone map[string]interface{}
	var contentDone map[string]interface{}
	var itemDone map[string]interface{}
	var completed map[string]interface{}
	for _, event := range events {
		typeName, _ := event["type"].(string)
		types = append(types, typeName)
		switch typeName {
		case "response.output_text.delta", "response.output_text.done":
			t.Fatalf("refusal was emitted as output text: %#v", event)
		case "response.refusal.delta":
			delta, _ := event["delta"].(string)
			deltas.WriteString(delta)
		case "response.refusal.done":
			refusalDone = event
		case "response.content_part.done":
			part, _ := event["part"].(map[string]interface{})
			if part["type"] == "refusal" {
				contentDone = part
			}
		case "response.output_item.done":
			item, _ := event["item"].(map[string]interface{})
			if item["type"] == "message" {
				itemDone = item
			}
		case "response.completed":
			completed, _ = event["response"].(map[string]interface{})
		}
	}

	const wantRefusal = "I can't help with that."
	if deltas.String() != wantRefusal {
		t.Fatalf("refusal deltas = %q, want %q", deltas.String(), wantRefusal)
	}
	if refusalDone == nil || refusalDone["refusal"] != wantRefusal {
		t.Fatalf("refusal.done = %#v", refusalDone)
	}
	if contentDone == nil || contentDone["refusal"] != wantRefusal {
		t.Fatalf("content_part.done = %#v", contentDone)
	}
	assertCompletedRefusalMessage(t, itemDone, wantRefusal)
	if completed == nil || completed["output_text"] != "" {
		t.Fatalf("completed response = %#v", completed)
	}
	output, _ := completed["output"].([]interface{})
	if len(output) != 1 {
		t.Fatalf("completed output = %#v", completed["output"])
	}
	message, _ := output[0].(map[string]interface{})
	assertCompletedRefusalMessage(t, message, wantRefusal)

	wantOrder := []string{
		"response.output_item.added",
		"response.content_part.added",
		"response.refusal.delta",
		"response.refusal.done",
		"response.content_part.done",
		"response.output_item.done",
		"response.completed",
	}
	previous := -1
	for _, typeName := range wantOrder {
		index := firstIndex(types, typeName)
		if index <= previous {
			t.Fatalf("event %q out of order in %v", typeName, types)
		}
		previous = index
	}
}

func assertCompletedRefusalMessage(t *testing.T, message map[string]interface{}, want string) {
	t.Helper()
	if message == nil {
		t.Fatal("missing completed refusal message")
	}
	content, _ := message["content"].([]interface{})
	if len(content) != 1 {
		t.Fatalf("refusal message content = %#v", message["content"])
	}
	part, _ := content[0].(map[string]interface{})
	if part["type"] != "refusal" || part["refusal"] != want {
		t.Fatalf("refusal message part = %#v", part)
	}
	if _, disguised := part["text"]; disguised {
		t.Fatalf("refusal message contains text field: %#v", part)
	}
}
