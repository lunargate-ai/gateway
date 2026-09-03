package api

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
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
		w.WriteHeader(http.StatusAccepted)
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

	if recorder.Code != http.StatusAccepted {
		t.Fatalf("status = %d, want 202; body=%s", recorder.Code, recorder.Body.String())
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
