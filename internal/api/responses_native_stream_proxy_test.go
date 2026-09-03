package api

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

func TestTrackedFlusherPropagatesNativeResponsesFlushError(t *testing.T) {
	target := &responsesFlushErrorWriter{header: make(http.Header)}
	proxy := newResponsesStreamProxy(target)
	proxy.enableNativePassthrough()
	tracked := &trackedFlusherResponseWriter{
		trackedResponseWriter: &trackedResponseWriter{ResponseWriter: proxy},
		flusher:               proxy,
	}

	if err := tracked.FlushError(); !errors.Is(err, errResponsesDownstreamFlush) {
		t.Fatalf("FlushError = %v, want downstream flush error", err)
	}
}

func TestChatCompletionsToResponsesStreamStillUsesChatSSE(t *testing.T) {
	nativeStream := strings.Join([]string{
		`event: response.created` + "\n",
		`data: {"type":"response.created","response":{"id":"resp_for_chat","object":"response","created_at":1,"status":"in_progress","model":"gpt-5.4","output":[]}}` + "\n\n",
		`event: response.output_text.delta` + "\n",
		`data: {"type":"response.output_text.delta","response_id":"resp_for_chat","output_index":0,"content_index":0,"delta":"hello"}` + "\n\n",
		`event: response.completed` + "\n",
		`data: {"type":"response.completed","response":{"id":"resp_for_chat","object":"response","created_at":1,"status":"completed","model":"gpt-5.4","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}` + "\n\n",
	}, "")
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(nativeStream))
	}))
	defer upstream.Close()

	handler, cache := newNativeContinuationTestHandler(t, upstream.URL+"/v1", requestTypeResponses)
	defer cache.Stop()
	recorder := httptest.NewRecorder()
	handler.ChatCompletions(recorder, httptest.NewRequest(
		http.MethodPost,
		"/v1/responses",
		strings.NewReader(`{"model":"gpt-5.4","messages":[{"role":"user","content":"hello"}],"stream":true}`),
	))

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", recorder.Code, recorder.Body.String())
	}
	body := recorder.Body.String()
	if strings.Contains(body, "event: response.") || strings.Contains(body, `"type":"response.created"`) {
		t.Fatalf("native Responses SSE leaked to Chat Completions client: %s", body)
	}
	if !strings.Contains(body, `"object":"chat.completion.chunk"`) || !strings.Contains(body, "data: [DONE]\n\n") {
		t.Fatalf("Chat Completions SSE was not emitted: %s", body)
	}
}

func TestResponsesNativeStreamProxiesEnvelopeAndStoresTerminalState(t *testing.T) {
	rawStream := strings.Join([]string{
		": upstream keepalive\r\n\r\n",
		"event: response.created\r\n",
		"id: upstream-event-1\r\n",
		"retry: 1250\r\n",
		`data: {"type":"response.created","event_id":"evt_upstream_1","sequence_number":41,"response":{"id":"resp_native_stream","object":"response","created_at":1788372000,"status":"in_progress","model":"gpt-5.4","output":[],"future_created_field":{"kept":true}}}` + "\r\n\r\n",
		"event: response.future_tool_progress\n",
		`data: {"type":"response.future_tool_progress","event_id":"evt_upstream_2",` + "\n",
		`data: "sequence_number":42,"progress":{"percent":50,"future":"kept"}}` + "\n\n",
		"event: response.completed\n",
		`data: {"type":"response.completed","event_id":"evt_upstream_3","sequence_number":43,"response":{"id":"resp_native_stream","object":"response","created_at":1788372000,"status":"completed","model":"gpt-5.4","output":[{"type":"hosted_tool_call","id":"tool_native","status":"completed","server_label":"web_search","result":{"sources":[{"url":"https://example.test"}]},"future_output_field":{"kept":true}}],"output_text":"done","usage":{"input_tokens":11,"output_tokens":7,"total_tokens":18,"input_tokens_details":{"cached_tokens":5},"output_tokens_details":{"reasoning_tokens":3}},"future_terminal_field":{"large_integer":9007199254740993}}}` + "\n\n",
		": trailing comment\n\n",
	}, "")

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/responses" {
			t.Fatalf("upstream path = %q, want /v1/responses", r.URL.Path)
		}
		w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
		w.Header().Set("Cache-Control", "no-store")
		w.Header().Set("X-OpenAI-Request-ID", "stream-request-id")
		w.Header().Set("X-RateLimit-Remaining-Requests", "17")
		w.Header().Set("Set-Cookie", "session=secret; HttpOnly")
		w.Header().Set("Connection", "X-Upstream-Hop")
		w.Header().Set("X-Upstream-Hop", "secret")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(rawStream))
	}))
	defer upstream.Close()

	handler, cache := newNativeContinuationTestHandler(t, upstream.URL+"/v1", requestTypeResponses)
	defer cache.Stop()
	recorder := httptest.NewRecorder()
	handler.Responses(recorder, httptest.NewRequest(
		http.MethodPost,
		"/v1/responses",
		strings.NewReader(`{"model":"gpt-5.4","input":"hello","stream":true}`),
	))

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", recorder.Code, recorder.Body.String())
	}
	if got := recorder.Body.String(); got != rawStream {
		t.Fatalf("native SSE changed\n got: %q\nwant: %q", got, rawStream)
	}
	if got := recorder.Header().Get("X-OpenAI-Request-ID"); got != "stream-request-id" {
		t.Fatalf("safe request header = %q", got)
	}
	if got := recorder.Header().Get("X-RateLimit-Remaining-Requests"); got != "17" {
		t.Fatalf("safe rate-limit header = %q", got)
	}
	if got := recorder.Header().Get("Content-Type"); got != "text/event-stream; charset=utf-8" {
		t.Fatalf("content type = %q, want upstream value", got)
	}
	for _, key := range []string{"Set-Cookie", "Connection", "X-Upstream-Hop", "Transfer-Encoding"} {
		if got := recorder.Header().Values(key); len(got) != 0 {
			t.Fatalf("unsafe header %s leaked: %q", key, got)
		}
	}

	stored, _, ok := handler.responsesState.getCompleted("resp_native_stream")
	if !ok {
		t.Fatal("completed native terminal response was not stored")
	}
	if !strings.Contains(string(stored), "9007199254740993") {
		t.Fatalf("stored terminal response changed a large additive integer: %s", stored)
	}
	var response map[string]json.RawMessage
	if err := json.Unmarshal(stored, &response); err != nil {
		t.Fatalf("decode stored response: %v", err)
	}
	for _, field := range []string{"output", "usage", "future_terminal_field"} {
		if len(response[field]) == 0 {
			t.Fatalf("stored terminal response lost %q: %s", field, stored)
		}
	}
	var output []map[string]interface{}
	if err := json.Unmarshal(response["output"], &output); err != nil {
		t.Fatalf("decode stored output: %v", err)
	}
	if len(output) != 1 || output[0]["type"] != "hosted_tool_call" || output[0]["future_output_field"] == nil {
		t.Fatalf("hosted/additive output was not retained: %#v", output)
	}
	var usage map[string]interface{}
	if err := json.Unmarshal(response["usage"], &usage); err != nil {
		t.Fatalf("decode stored usage: %v", err)
	}
	if usage["input_tokens_details"] == nil || usage["output_tokens_details"] == nil {
		t.Fatalf("usage details were not retained: %#v", usage)
	}
}

func TestResponsesNativeStreamAttachesLocalConversationBeforeStateUpdate(t *testing.T) {
	const responseID = "resp_native_local_conversation"
	createdFrame := strings.Join([]string{
		"event: response.created\r\n",
		"id: upstream-created-event\r\n",
		`data: {"type":"response.created","response":{"id":"` + responseID + `","object":"response","status":"in_progress","model":"gpt-5.4","output":[]}}` + "\r\n\r\n",
	}, "")
	terminalFrame := strings.Join([]string{
		"event: response.completed\r\n",
		"id: upstream-terminal-event\r\n",
		"retry: 1250\r\n",
		`data:{"type":"response.completed","event_id":"evt_terminal","sequence_number":2,"future_event":{"kept":true},` + "\r\n",
		": preserve terminal comment\r\n",
		`data: "response":{"id":"` + responseID + `","object":"response","created_at":1788372000,"status":"completed","model":"gpt-5.4","conversation":{"id":"conv_upstream_unrelated"},"output":[{"id":"msg_native_local","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"answer","annotations":[]}],"future_item":{"kept":true}}],"output_text":"answer","usage":{"input_tokens":2,"output_tokens":1,"total_tokens":3},"future_response":{"large_integer":9007199254740993}}}` + "\r\n\r\n",
	}, "")

	var upstreamCalls atomic.Int32
	upstreamPayloads := make(chan map[string]json.RawMessage, 1)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamCalls.Add(1)
		var payload map[string]json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Errorf("decode upstream request: %v", err)
			upstreamPayloads <- nil
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		upstreamPayloads <- payload
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(createdFrame + terminalFrame))
	}))
	defer upstream.Close()

	handler, cache := newNativeContinuationTestHandler(t, upstream.URL+"/v1", requestTypeResponses)
	defer cache.Stop()
	initial, err := prepareConversationItems([]json.RawMessage{
		json.RawMessage(`{"role":"user","content":"history","future_history":{"kept":true}}`),
	})
	if err != nil {
		t.Fatal(err)
	}
	conversation, err := handler.conversationsState.create(nil, initial)
	if err != nil {
		t.Fatal(err)
	}

	recorder := httptest.NewRecorder()
	handler.Responses(recorder, httptest.NewRequest(
		http.MethodPost,
		"/v1/responses",
		strings.NewReader(`{"model":"gpt-5.4","conversation":{"id":"`+conversation.ID+`"},"input":"new input","stream":true}`),
	))
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", recorder.Code, recorder.Body.String())
	}
	if upstreamCalls.Load() != 1 {
		t.Fatalf("upstream calls = %d, want exactly one", upstreamCalls.Load())
	}
	upstreamPayload := <-upstreamPayloads
	if upstreamPayload == nil {
		t.Fatal("upstream request could not be decoded")
	}
	if _, leaked := upstreamPayload["conversation"]; leaked {
		t.Fatalf("local conversation leaked upstream: %s", mustMarshalForTest(t, upstreamPayload))
	}
	var upstreamInput []json.RawMessage
	if err := json.Unmarshal(upstreamPayload["input"], &upstreamInput); err != nil || len(upstreamInput) != 2 {
		t.Fatalf("upstream input = %s, want history and new input; error=%v", upstreamPayload["input"], err)
	}

	body := recorder.Body.String()
	if !strings.HasPrefix(body, createdFrame) {
		t.Fatalf("non-terminal frame changed: %q", body)
	}
	for _, preserved := range []string{
		"id: upstream-terminal-event\r\n",
		"retry: 1250\r\n",
		": preserve terminal comment\r\n",
	} {
		if !strings.Contains(body, preserved) {
			t.Fatalf("terminal SSE metadata %q was not preserved: %q", preserved, body)
		}
	}
	terminalEvent := decodeNativeResponsesSSEEvent(t, body, "response.completed")
	if len(terminalEvent["future_event"]) == 0 {
		t.Fatalf("terminal event lost additive field: %s", mustMarshalForTest(t, terminalEvent))
	}
	var downstreamResponse map[string]json.RawMessage
	if err := json.Unmarshal(terminalEvent["response"], &downstreamResponse); err != nil {
		t.Fatalf("decode downstream terminal response: %v", err)
	}
	if got := parseJSONStringRaw(downstreamResponse["id"]); got != responseID {
		t.Fatalf("downstream response id = %q, want upstream id %q", got, responseID)
	}
	if got, err := parseResponsesConversationID(downstreamResponse["conversation"]); err != nil || got != conversation.ID {
		t.Fatalf("downstream conversation = %s, want %q; error=%v", downstreamResponse["conversation"], conversation.ID, err)
	}
	if bytes.Contains(terminalEvent["response"], []byte("conv_upstream_unrelated")) {
		t.Fatalf("unrelated upstream conversation leaked downstream: %s", terminalEvent["response"])
	}
	if !bytes.Contains(terminalEvent["response"], []byte("9007199254740993")) {
		t.Fatalf("terminal response changed additive large integer: %s", terminalEvent["response"])
	}

	stored, _, ok := handler.responsesState.getCompleted(responseID)
	if !ok {
		t.Fatal("completed response state was not retained")
	}
	if !bytes.Equal(stored, terminalEvent["response"]) {
		t.Fatalf("stored terminal differs from downstream terminal\nstored: %s\ndownstream: %s", stored, terminalEvent["response"])
	}
	items, ok := handler.conversationsState.getItems(conversation.ID)
	if !ok || len(items) != 3 {
		t.Fatalf("conversation items = %#v, ok=%t; want history, input, and output", items, ok)
	}
	if got := parseJSONStringRaw(items[2]["id"]); got != "msg_native_local" {
		t.Fatalf("appended output item id = %q, want native item id", got)
	}
	if len(items[2]["future_item"]) == 0 {
		t.Fatalf("appended output item lost additive field: %s", mustMarshalForTest(t, items[2]))
	}
}

func decodeNativeResponsesSSEEvent(t *testing.T, body string, eventType string) map[string]json.RawMessage {
	t.Helper()
	normalized := strings.ReplaceAll(body, "\r\n", "\n")
	for _, frame := range strings.Split(normalized, "\n\n") {
		var currentEvent string
		dataLines := make([]string, 0, 1)
		for _, line := range strings.Split(frame, "\n") {
			switch {
			case strings.HasPrefix(line, "event:"):
				currentEvent = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
			case strings.HasPrefix(line, "data:"):
				dataLines = append(dataLines, strings.TrimPrefix(strings.TrimPrefix(line, "data:"), " "))
			}
		}
		if currentEvent != eventType {
			continue
		}
		var event map[string]json.RawMessage
		if err := json.Unmarshal([]byte(strings.Join(dataLines, "\n")), &event); err != nil {
			t.Fatalf("decode %s event: %v", eventType, err)
		}
		return event
	}
	t.Fatalf("missing SSE event %q in %q", eventType, body)
	return nil
}

func TestResponsesNativeStreamDoesNotStoreNonCompletedTerminalAsCompleted(t *testing.T) {
	testCases := []struct {
		eventType string
		status    string
	}{
		{eventType: "response.failed", status: "failed"},
		{eventType: "response.incomplete", status: "incomplete"},
		{eventType: "response.cancelled", status: "cancelled"},
	}

	for _, testCase := range testCases {
		t.Run(testCase.status, func(t *testing.T) {
			responseID := "resp_" + testCase.status
			rawStream := fmt.Sprintf(
				"event: %s\ndata: {\"type\":%q,\"response\":{\"id\":%q,\"object\":\"response\",\"status\":%q,\"model\":\"gpt-5.4\",\"output\":[],\"future_terminal_field\":true}}\n\n",
				testCase.eventType,
				testCase.eventType,
				responseID,
				testCase.status,
			)
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = w.Write([]byte(rawStream))
			}))
			defer upstream.Close()

			handler, cache := newNativeContinuationTestHandler(t, upstream.URL+"/v1", requestTypeResponses)
			defer cache.Stop()
			recorder := httptest.NewRecorder()
			handler.Responses(recorder, httptest.NewRequest(
				http.MethodPost,
				"/v1/responses",
				strings.NewReader(`{"model":"gpt-5.4","input":"hello","stream":true}`),
			))

			if recorder.Code != http.StatusOK {
				t.Fatalf("status = %d, want 200; body=%s", recorder.Code, recorder.Body.String())
			}
			if got := recorder.Body.String(); got != rawStream {
				t.Fatalf("native terminal stream changed: %q", got)
			}
			if _, _, stored := handler.responsesState.getCompleted(responseID); stored {
				t.Fatalf("%s terminal was stored as completed", testCase.status)
			}
		})
	}
}

func TestResponsesNativeStreamPrematureEOFDoesNotAppendSyntheticEvents(t *testing.T) {
	rawStream := strings.Join([]string{
		`event: response.created` + "\n",
		`data: {"type":"response.created","response":{"id":"resp_early_eof","object":"response","status":"in_progress","model":"gpt-5.4","output":[]}}` + "\n\n",
		`event: response.output_text.delta` + "\n",
		`data: {"type":"response.output_text.delta","response_id":"resp_early_eof","delta":"partial"}` + "\n\n",
	}, "")
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(rawStream))
	}))
	defer upstream.Close()

	handler, cache := newNativeContinuationTestHandler(t, upstream.URL+"/v1", requestTypeResponses)
	defer cache.Stop()
	recorder := httptest.NewRecorder()
	handler.Responses(recorder, httptest.NewRequest(
		http.MethodPost,
		"/v1/responses",
		strings.NewReader(`{"model":"gpt-5.4","input":"hello","stream":true}`),
	))

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want already-sent 200; body=%s", recorder.Code, recorder.Body.String())
	}
	if got := recorder.Body.String(); got != rawStream {
		t.Fatalf("premature stream was rewritten or received a synthetic event\n got: %q\nwant: %q", got, rawStream)
	}
	if _, _, stored := handler.responsesState.getCompleted("resp_early_eof"); stored {
		t.Fatal("premature stream was stored as completed")
	}
}
