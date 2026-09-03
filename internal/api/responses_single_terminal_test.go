package api

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

func TestResponsesNativeStreamKeepsOnlyFirstTerminalEvent(t *testing.T) {
	preTerminal, firstTerminal, afterTerminal, secondTerminal, done := duplicateNativeTerminalFrames()
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, preTerminal+firstTerminal+afterTerminal+secondTerminal+done)
	}))
	defer upstream.Close()

	handler, cache := newNativeContinuationTestHandler(t, upstream.URL+"/v1", requestTypeResponses)
	defer cache.Stop()
	recorder := httptest.NewRecorder()
	handler.Responses(recorder, httptest.NewRequest(
		http.MethodPost,
		"/v1/responses",
		strings.NewReader(`{"model":"gpt-5.4","input":"hello","stream":true,"store":true}`),
	))

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", recorder.Code, recorder.Body.String())
	}
	wantBody := preTerminal + firstTerminal + done
	if got := recorder.Body.String(); got != wantBody {
		t.Fatalf("native stream after first terminal\n got: %q\nwant: %q", got, wantBody)
	}
	stored, _, ok := handler.responsesState.getCompleted("resp_first")
	if !ok {
		t.Fatal("first terminal response was not stored")
	}
	var response map[string]json.RawMessage
	if err := json.Unmarshal(stored, &response); err != nil {
		t.Fatalf("decode stored first terminal: %v", err)
	}
	if got := parseJSONStringRaw(response["output_text"]); got != "first" {
		t.Fatalf("stored output_text = %q, want first", got)
	}
	if _, _, ok := handler.responsesState.getCompleted("resp_second"); ok {
		t.Fatal("second terminal response was stored")
	}
}

func TestResponsesWebSocketKeepsOnlyFirstNativeTerminalEvent(t *testing.T) {
	preTerminal, firstTerminal, afterTerminal, secondTerminal, done := duplicateNativeTerminalFrames()
	var upstreamCalls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		if upstreamCalls.Add(1) == 1 {
			_, _ = io.WriteString(w, preTerminal+firstTerminal+afterTerminal+secondTerminal+done)
			return
		}
		_, _ = io.WriteString(w,
			"event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_followup\",\"status\":\"in_progress\",\"model\":\"gpt-5.4\",\"output\":[]}}\n\n"+
				"event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_followup\",\"status\":\"completed\",\"model\":\"gpt-5.4\",\"output\":[],\"output_text\":\"followup\"}}\n\n"+
				done,
		)
	}))
	defer upstream.Close()

	handler := newResponsesWebSocketTestHandlerWithUpstreamType(upstream.URL, requestTypeResponses)
	defer handler.cache.Stop()
	server := httptest.NewServer(http.HandlerFunc(handler.ResponsesWebSocket))
	defer server.Close()
	conn := mustDialResponsesWebSocket(t, server.URL)
	defer conn.Close()

	sendResponsesWebSocketJSON(t, conn, map[string]interface{}{
		"type":  "response.create",
		"model": "lunargate/auto",
		"input": "first request",
	})
	firstEvents := readResponsesWebSocketEventsUntilTerminal(t, conn)
	if got := strings.Join(eventTypes(firstEvents), ","); got != "response.created,response.completed" {
		t.Fatalf("first event types = %s, want response.created,response.completed", got)
	}
	if got := extractCompletedResponseID(firstEvents); got != "resp_first" {
		t.Fatalf("first terminal response id = %q, want resp_first", got)
	}

	// A second request is also a synchronization point: any frame leaked after
	// the first terminal would already be queued ahead of its response.created.
	sendResponsesWebSocketJSON(t, conn, map[string]interface{}{
		"type":  "response.create",
		"model": "lunargate/auto",
		"input": "second request",
	})
	followUpEvents := readResponsesWebSocketEventsUntilTerminal(t, conn)
	if got := strings.Join(eventTypes(followUpEvents), ","); got != "response.created,response.completed" {
		t.Fatalf("follow-up event types = %s, want response.created,response.completed", got)
	}
	if got := extractCompletedResponseID(followUpEvents); got != "resp_followup" {
		t.Fatalf("follow-up terminal response id = %q, want resp_followup", got)
	}
	if got := upstreamCalls.Load(); got != 2 {
		t.Fatalf("upstream calls = %d, want 2", got)
	}
}

func TestResponsesWebSocketAdapterConsumesApplicationFramesAfterTerminal(t *testing.T) {
	_, firstTerminal, afterTerminal, secondTerminal, done := duplicateNativeTerminalFrames()
	proxy := &responsesWebSocketProxy{}
	firstPayload, ok := responsesSSEData([]byte(firstTerminal))
	if !ok {
		t.Fatal("first terminal fixture has no SSE data")
	}
	proxy.captureEventState(firstPayload)
	if !proxy.terminalSeen || proxy.responseID != "resp_first" {
		t.Fatalf("first terminal state = seen:%t id:%q", proxy.terminalSeen, proxy.responseID)
	}

	for _, frame := range []string{afterTerminal, secondTerminal} {
		forwarded, err := proxy.processSSEFrame([]byte(frame))
		if err != nil {
			t.Fatalf("consume post-terminal frame: %v", err)
		}
		if forwarded {
			t.Fatalf("post-terminal frame was forwarded: %q", frame)
		}
	}
	if proxy.responseID != "resp_first" {
		t.Fatalf("post-terminal frame overwrote response id with %q", proxy.responseID)
	}
	if output, _ := proxy.completedResponse["output_text"].(string); output != "first" {
		t.Fatalf("post-terminal frame overwrote output_text with %q", output)
	}

	forwarded, err := proxy.processSSEFrame([]byte(done))
	if err != nil || forwarded || !proxy.done {
		t.Fatalf("DONE result = forwarded:%t done:%t error:%v", forwarded, proxy.done, err)
	}
}

func duplicateNativeTerminalFrames() (preTerminal, firstTerminal, afterTerminal, secondTerminal, done string) {
	preTerminal = "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_first\",\"status\":\"in_progress\",\"model\":\"gpt-5.4\",\"output\":[]}}\n\n"
	firstTerminal = "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_first\",\"status\":\"completed\",\"model\":\"gpt-5.4\",\"output\":[],\"output_text\":\"first\"}}\n\n"
	afterTerminal = "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"response_id\":\"resp_first\",\"delta\":\"must-not-leak\"}\n\n"
	secondTerminal = "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_second\",\"status\":\"completed\",\"model\":\"gpt-5.4\",\"output\":[],\"output_text\":\"second\"}}\n\n"
	done = "data: [DONE]\n\n"
	return
}
