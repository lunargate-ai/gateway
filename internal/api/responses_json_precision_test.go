package api

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestResponsesResponseToMapPreservesLargeInteger(t *testing.T) {
	payload, err := responsesResponseToMap(&models.ResponsesResponse{
		ID:        "resp_precision",
		Object:    "response",
		CreatedAt: 9007199254740993,
		Status:    "completed",
		Model:     "gpt-test",
		Output:    []models.ResponsesOutput{},
	})
	if err != nil {
		t.Fatalf("responsesResponseToMap returned error: %v", err)
	}
	assertPrecisionNumber(t, payload["created_at"])
}

func TestResponsesTranslatedHTTPPreservesLargeInteger(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("upstream path = %q, want /v1/chat/completions", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"chatcmpl_precision","object":"chat.completion","created":`+precisionTestInteger+`,"model":"gpt-test","choices":[{"index":0,"message":{"role":"assistant","content":"answer"},"finish_reason":"stop"}]}`)
	}))
	t.Cleanup(upstream.Close)

	handler, cache := newNativeContinuationTestHandler(t, upstream.URL+"/v1", requestTypeChatCompletions)
	t.Cleanup(cache.Stop)
	recorder := httptest.NewRecorder()
	handler.Responses(recorder, httptest.NewRequest(
		http.MethodPost,
		"/v1/responses",
		strings.NewReader(`{"model":"gpt-5.4","input":"hello","store":false}`),
	))

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", recorder.Code, recorder.Body.String())
	}
	if !bytes.Contains(recorder.Body.Bytes(), []byte(precisionTestInteger)) {
		t.Fatalf("translated response changed large integer: %s", recorder.Body.String())
	}
	var payload map[string]interface{}
	if err := decodeJSONStrict(bytes.NewReader(recorder.Body.Bytes()), &payload); err != nil {
		t.Fatalf("decode translated response: %v", err)
	}
	assertPrecisionNumber(t, payload["created_at"])
}

func TestResponsesLocalConversationAppendAndReplayPreservesLargeInteger(t *testing.T) {
	requestBodies := make(chan []byte, 2)
	var calls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read upstream request: %v", err)
			http.Error(w, "read failed", http.StatusInternalServerError)
			return
		}
		requestBodies <- append([]byte(nil), body...)
		call := calls.Add(1)
		responseID := "resp_precision_followup"
		output := `[]`
		if call == 1 {
			responseID = "resp_precision_initial"
			output = `[{"type":"message","id":"msg_precision","status":"completed","role":"assistant","content":[{"type":"output_text","text":"answer","annotations":[]}],"future_item":{"large_integer":` + precisionTestInteger + `}}]`
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"`+responseID+`","object":"response","created_at":1,"status":"completed","model":"gpt-5.4","output":`+output+`,"output_text":"answer"}`)
	}))
	t.Cleanup(upstream.Close)

	handler, cache := newNativeContinuationTestHandler(t, upstream.URL+"/v1", requestTypeResponses)
	t.Cleanup(cache.Stop)
	conversation, err := handler.conversationsState.create(nil, nil)
	if err != nil {
		t.Fatalf("create local conversation: %v", err)
	}

	first := httptest.NewRecorder()
	handler.Responses(first, httptest.NewRequest(
		http.MethodPost,
		"/v1/responses",
		strings.NewReader(`{"model":"gpt-5.4","conversation":"`+conversation.ID+`","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"first"}],"future_input":{"large_integer":`+precisionTestInteger+`}}],"store":false}`),
	))
	if first.Code != http.StatusOK {
		t.Fatalf("first status = %d, want 200; body=%s", first.Code, first.Body.String())
	}
	assertPrecisionConversationInput(t, receiveContinuationBody(t, requestBodies), "future_input", 0)

	stored, ok := handler.conversationsState.getItems(conversation.ID)
	if !ok || len(stored) != 2 {
		t.Fatalf("stored conversation items = %#v, %t; want request and output", stored, ok)
	}
	assertPrecisionRawObject(t, stored[0]["future_input"])
	assertPrecisionRawObject(t, stored[1]["future_item"])

	second := httptest.NewRecorder()
	handler.Responses(second, httptest.NewRequest(
		http.MethodPost,
		"/v1/responses",
		strings.NewReader(`{"model":"gpt-5.4","conversation":"`+conversation.ID+`","input":"second","store":false}`),
	))
	if second.Code != http.StatusOK {
		t.Fatalf("second status = %d, want 200; body=%s", second.Code, second.Body.String())
	}
	secondBody := receiveContinuationBody(t, requestBodies)
	assertPrecisionConversationInput(t, secondBody, "future_input", 0)
	assertPrecisionConversationInput(t, secondBody, "future_item", 1)
	if got := calls.Load(); got != 2 {
		t.Fatalf("upstream calls = %d, want 2", got)
	}
}

func TestResponsesWebSocketTerminalCapturePreservesLargeInteger(t *testing.T) {
	for _, eventType := range []string{"response.completed", "response.failed", "response.incomplete"} {
		t.Run(eventType, func(t *testing.T) {
			proxy := &responsesWebSocketProxy{}
			proxy.captureEventState([]byte(`{"type":"` + eventType + `","response":{"id":"resp_precision","output":[{"future_item":{"large_integer":` + precisionTestInteger + `}}]}}`))

			response := proxy.completedResponse
			if eventType != "response.completed" {
				response = proxy.terminalResponse
			}
			if response == nil {
				t.Fatalf("captured response = nil; terminal error=%#v", proxy.terminalError)
			}
			item := precisionObject(t, precisionArray(t, response["output"])[0])
			future := precisionObject(t, item["future_item"])
			assertPrecisionNumber(t, future["large_integer"])
		})
	}
}

func TestCloneResponsesInterfaceSlicePreservesLargeInteger(t *testing.T) {
	var source []interface{}
	if err := decodeJSONStrict(strings.NewReader(`[{"future":{"large_integer":`+precisionTestInteger+`}}]`), &source); err != nil {
		t.Fatalf("decode source: %v", err)
	}
	cloned := cloneResponsesInterfaceSlice(source)
	future := precisionObject(t, precisionObject(t, cloned[0])["future"])
	assertPrecisionNumber(t, future["large_integer"])

	future["large_integer"] = json.Number("1")
	originalFuture := precisionObject(t, precisionObject(t, source[0])["future"])
	assertPrecisionNumber(t, originalFuture["large_integer"])
}

func assertPrecisionConversationInput(t *testing.T, body []byte, field string, index int) {
	t.Helper()
	var payload map[string]interface{}
	if err := decodeJSONStrict(bytes.NewReader(body), &payload); err != nil {
		t.Fatalf("decode upstream request: %v; body=%s", err, body)
	}
	input := precisionArray(t, payload["input"])
	if index < 0 || index >= len(input) {
		t.Fatalf("input index %d out of range for %#v", index, input)
	}
	item := precisionObject(t, input[index])
	future := precisionObject(t, item[field])
	assertPrecisionNumber(t, future["large_integer"])
}

func assertPrecisionRawObject(t *testing.T, raw json.RawMessage) {
	t.Helper()
	var value map[string]interface{}
	if err := decodeJSONStrict(bytes.NewReader(raw), &value); err != nil {
		t.Fatalf("decode stored JSON object: %v; raw=%s", err, raw)
	}
	assertPrecisionNumber(t, value["large_integer"])
}
