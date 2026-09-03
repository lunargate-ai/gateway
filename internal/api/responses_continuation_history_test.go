package api

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/pkg/models"
)

const completeContinuationOutput = `[
	{"type":"message","id":"msg_1","status":"completed","role":"assistant","content":[{"type":"output_text","text":"answer","annotations":[]}],"future_message":{"kept":true}},
	{"type":"reasoning","id":"rs_1","status":"completed","summary":[{"type":"summary_text","text":"summary"}],"encrypted_content":"encrypted-reasoning","phase":"analysis","future_reasoning":{"kept":true}},
	{"type":"computer_call","id":"computer_1","call_id":"call_computer_1","status":"completed","action":{"type":"click","x":17,"y":23},"pending_safety_checks":[],"future_computer":{"kept":true}},
	{"type":"future_output_item","id":"future_1","phase":"commentary","payload":{"nested":[true,"kept"],"large_integer":9007199254740993}},
	{"type":"function_call","id":"fc_1","call_id":"call_1","status":"completed","name":"lookup","arguments":"{\"q\":\"moon\"}","future_function":{"kept":true}}
]`

func TestResponsesCompletedResponseToInputItemsClonesCompleteOutput(t *testing.T) {
	var completed map[string]interface{}
	if err := decodeJSONStrict(strings.NewReader(`{"output":`+completeContinuationOutput+`}`), &completed); err != nil {
		t.Fatalf("decode completed response fixture: %v", err)
	}

	items := responsesCompletedResponseToInputItems(completed)
	if len(items) != 5 {
		t.Fatalf("cloned output items = %d, want 5", len(items))
	}
	original := completed["output"].([]interface{})
	originalReasoning := original[1].(map[string]interface{})
	clonedReasoning := items[1].(map[string]interface{})
	originalReasoning["phase"] = "mutated-original"
	if got := clonedReasoning["phase"]; got != "analysis" {
		t.Fatalf("clone changed after original mutation: phase=%v", got)
	}

	clonedMessage := items[0].(map[string]interface{})
	clonedFuture := clonedMessage["future_message"].(map[string]interface{})
	clonedFuture["kept"] = false
	originalMessage := original[0].(map[string]interface{})
	originalFuture := originalMessage["future_message"].(map[string]interface{})
	if got := originalFuture["kept"]; got != true {
		t.Fatalf("original changed after clone mutation: future_message.kept=%v", got)
	}
}

func TestTranslatedResponsesContinuationHistoryLifecycleCompatibility(t *testing.T) {
	accepted := []struct {
		name string
		raw  string
	}{
		{name: "assistant content omitted", raw: `[{"type":"message","role":"assistant","status":"completed"}]`},
		{name: "assistant content null", raw: `[{"type":"message","role":"assistant","content":null,"status":"completed"}]`},
		{name: "assistant content empty string", raw: `[{"type":"message","role":"assistant","content":"","status":"completed"}]`},
		{name: "assistant content empty array", raw: `[{"type":"message","role":"assistant","content":[],"status":"completed"}]`},
	}
	for _, test := range accepted {
		t.Run("allows "+test.name, func(t *testing.T) {
			if err := validateTranslatedResponsesInput(json.RawMessage(test.raw), "input", "openai-chat", "openai"); err != nil {
				t.Fatalf("compatible continuation history was rejected: %v", err)
			}
		})
	}

	rejected := []struct {
		name      string
		raw       string
		wantField string
	}{
		{name: "empty user", raw: `[{"type":"message","role":"user","content":[]}]`, wantField: "input[0].content"},
		{name: "empty system", raw: `[{"type":"message","role":"system"}]`, wantField: "input[0].content"},
		{name: "empty developer", raw: `[{"type":"message","role":"developer","content":null}]`, wantField: "input[0].content"},
		{name: "incomplete message", raw: `[{"type":"message","role":"assistant","content":"partial","status":"incomplete"}]`, wantField: "input[0].status"},
		{name: "unknown message status", raw: `[{"type":"message","role":"assistant","content":"partial","status":"paused"}]`, wantField: "input[0].status"},
		{name: "incomplete function call", raw: `[{"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{}","status":"incomplete"}]`, wantField: "input[0].status"},
		{name: "unknown function call status", raw: `[{"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{}","status":"paused"}]`, wantField: "input[0].status"},
		{name: "incomplete function output", raw: `[{"type":"function_call_output","call_id":"call_1","output":"partial","status":"incomplete"}]`, wantField: "input[0].status"},
		{name: "unknown function output status", raw: `[{"type":"function_call_output","call_id":"call_1","output":"partial","status":"paused"}]`, wantField: "input[0].status"},
		{name: "future output item", raw: `[{"type":"future_output_item","id":"future_1","payload":{"kept":true}}]`, wantField: "input[0].type"},
	}
	for _, test := range rejected {
		t.Run("rejects "+test.name, func(t *testing.T) {
			err := validateTranslatedResponsesInput(json.RawMessage(test.raw), "input", "openai-chat", "openai")
			var compatibilityErr *models.CompatibilityError
			if !errors.As(err, &compatibilityErr) {
				t.Fatalf("error = %v, want CompatibilityError", err)
			}
			if compatibilityErr.Field != test.wantField || compatibilityErr.Provider != "openai-chat" {
				t.Fatalf("compatibility error = %#v, want field=%q provider=openai-chat", compatibilityErr, test.wantField)
			}
		})
	}
}

func TestResponsesHTTPRejectsIncompleteTranslatedContinuationBeforeUpstream(t *testing.T) {
	var calls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"chatcmpl_partial","object":"chat.completion","created":1,"model":"gpt-5.4","choices":[{"index":0,"message":{"role":"assistant","content":"partial answer"},"finish_reason":"length"}]}`)
	}))
	defer upstream.Close()

	handler, cache := newNativeContinuationTestHandler(t, upstream.URL+"/v1", requestTypeChatCompletions)
	defer cache.Stop()
	first := httptest.NewRecorder()
	handler.Responses(first, httptest.NewRequest(
		http.MethodPost,
		"/v1/responses",
		bytes.NewBufferString(`{"model":"gpt-5.4","input":"first","store":true}`),
	))
	if first.Code != http.StatusOK {
		t.Fatalf("initial status = %d, want 200; body=%s", first.Code, first.Body.String())
	}
	var partial map[string]interface{}
	if err := decodeJSONStrict(bytes.NewReader(first.Body.Bytes()), &partial); err != nil {
		t.Fatalf("decode initial response: %v", err)
	}
	responseID, _ := partial["id"].(string)
	output, _ := partial["output"].([]interface{})
	if responseID == "" || len(output) != 1 {
		t.Fatalf("initial response = %#v, want stored partial output", partial)
	}
	outputItem, _ := output[0].(map[string]interface{})
	if partial["status"] != "incomplete" || outputItem["status"] != "incomplete" {
		t.Fatalf("initial response = %#v, want incomplete response and output item", partial)
	}

	calls.Store(0)
	followUpPayload, err := json.Marshal(map[string]interface{}{
		"model":                "gpt-5.4",
		"previous_response_id": responseID,
		"input":                "next",
		"store":                false,
	})
	if err != nil {
		t.Fatalf("marshal follow-up request: %v", err)
	}
	followUp := httptest.NewRecorder()
	handler.Responses(followUp, httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(followUpPayload)))

	if followUp.Code != http.StatusBadRequest {
		t.Fatalf("follow-up status = %d, want 400; body=%s", followUp.Code, followUp.Body.String())
	}
	if got := calls.Load(); got != 0 {
		t.Fatalf("rejected continuation made %d upstream calls, want 0", got)
	}
	var response models.ErrorResponse
	if err := json.Unmarshal(followUp.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode follow-up error: %v", err)
	}
	if response.Error.Type != "invalid_request_error" || response.Error.Param == nil || *response.Error.Param != "input[1].status" ||
		response.Error.Code == nil || *response.Error.Code != "unsupported_feature" {
		t.Fatalf("follow-up error = %#v, want compatibility error for input[1].status", response.Error)
	}
}

func TestResponsesHTTPLocalContinuationPreservesCompleteOutput(t *testing.T) {
	var calls atomic.Int32
	followUpBodies := make(chan []byte, 1)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read upstream request: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		calls.Add(1)
		followUpBodies <- append([]byte(nil), body...)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, completedContinuationResponse("resp_complete_followup", `[]`))
	}))
	defer upstream.Close()

	providerConfigs := map[string]config.ProviderConfig{
		"native": {
			Type:         "openai",
			APIKey:       "provider-secret",
			BaseURL:      upstream.URL + "/v1",
			DefaultModel: "gpt-native",
		},
	}
	router, handler, cache := newNativeLifecycleRouterFromConfigs(t, providerConfigs)
	defer cache.Stop()

	fingerprint, ok := handler.responseAccountFingerprint("native")
	if !ok {
		t.Fatal("native provider fingerprint is unavailable")
	}
	requestPayload := map[string]json.RawMessage{
		"model": json.RawMessage(`"native/gpt-native"`),
		"input": json.RawMessage(`"first"`),
	}
	var completed map[string]interface{}
	if err := decodeJSONStrict(strings.NewReader(completedContinuationResponse("resp_complete_history", completeContinuationOutput)), &completed); err != nil {
		t.Fatalf("decode completed response fixture: %v", err)
	}
	responseHeaders := make(http.Header)
	responseHeaders.Set("X-LunarGate-Provider", "native")
	responseHeaders.Set("X-LunarGate-Route", "responses")
	responseHeaders.Set("X-LunarGate-Model", "native/gpt-native")
	claim := handler.retainLocalResponseSnapshot(
		"resp_complete_history",
		responseHeaders,
		responseExecutionOwner{
			Provider:            "native",
			Route:               "responses",
			Model:               "native/gpt-native",
			UpstreamRequestType: requestTypeResponses,
			AccountFingerprint:  fingerprint,
		},
		requestPayload,
		completed,
	)
	if !claim.retained() {
		t.Fatalf("retain local response snapshot = %v", claim)
	}

	followUp := performLifecycleRequest(t, router, http.MethodPost, "/v1/responses", []byte(`{"model":"native/gpt-native","previous_response_id":"resp_complete_history","input":"next","store":false}`))
	if followUp.Code != http.StatusOK {
		t.Fatalf("follow-up status = %d, want 200; body=%s", followUp.Code, followUp.Body.String())
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("upstream calls = %d, want 1", got)
	}
	assertCompleteContinuationOutput(t, receiveContinuationBody(t, followUpBodies))
}

func TestResponsesWebSocketContinuationPreservesCompleteOutput(t *testing.T) {
	var calls atomic.Int32
	followUpBodies := make(chan []byte, 1)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read upstream request: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		call := calls.Add(1)
		responseID := "resp_ws_complete_history"
		output := completeContinuationOutput
		if call > 1 {
			followUpBodies <- append([]byte(nil), body...)
			responseID = "resp_ws_complete_followup"
			output = `[]`
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, completedContinuationStream(responseID, output))
	}))
	defer upstream.Close()

	handler := newResponsesWebSocketTestHandlerWithUpstreamType(upstream.URL, requestTypeResponses)
	defer handler.cache.Stop()
	server := httptest.NewServer(http.HandlerFunc(handler.ResponsesWebSocket))
	defer server.Close()
	connection := mustDialResponsesWebSocket(t, server.URL)
	defer connection.Close()

	sendResponsesWebSocketJSON(t, connection, map[string]interface{}{
		"type":  "response.create",
		"model": "lunargate/auto",
		"input": "first",
	})
	firstEvents := readResponsesWebSocketEventsUntilTerminal(t, connection)
	if responseID := extractCompletedResponseID(firstEvents); responseID != "resp_ws_complete_history" {
		t.Fatalf("first response ID = %q, want resp_ws_complete_history; events=%v", responseID, eventTypes(firstEvents))
	}

	sendResponsesWebSocketJSON(t, connection, map[string]interface{}{
		"type":                 "response.create",
		"model":                "lunargate/auto",
		"previous_response_id": "resp_ws_complete_history",
		"input":                "next",
	})
	followUpEvents := readResponsesWebSocketEventsUntilTerminal(t, connection)
	if responseID := extractCompletedResponseID(followUpEvents); responseID != "resp_ws_complete_followup" {
		t.Fatalf("follow-up response ID = %q, want resp_ws_complete_followup; events=%v", responseID, eventTypes(followUpEvents))
	}
	if got := calls.Load(); got != 2 {
		t.Fatalf("upstream calls = %d, want 2", got)
	}
	assertCompleteContinuationOutput(t, receiveContinuationBody(t, followUpBodies))
}

func completedContinuationResponse(responseID string, output string) string {
	var compact bytes.Buffer
	if err := json.Compact(&compact, []byte(output)); err == nil {
		output = compact.String()
	}
	return `{"id":"` + responseID + `","object":"response","created_at":1,"status":"completed","model":"gpt-native","output":` + output + `,"output_text":"answer"}`
}

func completedContinuationStream(responseID string, output string) string {
	created := `{"type":"response.created","sequence_number":0,"response":{"id":"` + responseID + `","object":"response","created_at":1,"status":"in_progress","model":"gpt-native","output":[]}}`
	completed := `{"type":"response.completed","sequence_number":1,"response":` + completedContinuationResponse(responseID, output) + `}`
	return "event: response.created\ndata: " + created + "\n\n" +
		"event: response.completed\ndata: " + completed + "\n\n" +
		"data: [DONE]\n\n"
}

func receiveContinuationBody(t *testing.T, bodies <-chan []byte) []byte {
	t.Helper()
	select {
	case body := <-bodies:
		return body
	case <-time.After(5 * time.Second):
		t.Fatal("follow-up request did not reach upstream")
		return nil
	}
}

func assertCompleteContinuationOutput(t *testing.T, body []byte) {
	t.Helper()
	var payload map[string]json.RawMessage
	if err := decodeJSONStrict(bytes.NewReader(body), &payload); err != nil {
		t.Fatalf("decode follow-up request: %v; body=%s", err, body)
	}
	if _, exists := payload["previous_response_id"]; exists {
		t.Fatalf("locally resolved previous_response_id reached upstream: %s", body)
	}

	var input []json.RawMessage
	if err := decodeJSONStrict(bytes.NewReader(payload["input"]), &input); err != nil {
		t.Fatalf("decode follow-up input: %v; input=%s", err, payload["input"])
	}
	var wantOutput []json.RawMessage
	if err := decodeJSONStrict(strings.NewReader(completeContinuationOutput), &wantOutput); err != nil {
		t.Fatalf("decode expected output: %v", err)
	}
	if len(input) != len(wantOutput)+2 {
		t.Fatalf("follow-up input has %d items, want initial + %d output + current; input=%s", len(input), len(wantOutput), payload["input"])
	}
	for index := range wantOutput {
		assertSameJSONValue(t, input[index+1], wantOutput[index])
	}
}

func assertSameJSONValue(t *testing.T, got json.RawMessage, want json.RawMessage) {
	t.Helper()
	var gotValue interface{}
	if err := decodeJSONStrict(bytes.NewReader(got), &gotValue); err != nil {
		t.Fatalf("decode actual JSON %s: %v", got, err)
	}
	var wantValue interface{}
	if err := decodeJSONStrict(bytes.NewReader(want), &wantValue); err != nil {
		t.Fatalf("decode expected JSON %s: %v", want, err)
	}
	if !reflect.DeepEqual(gotValue, wantValue) {
		t.Fatalf("continuation item changed:\n got: %s\nwant: %s", got, want)
	}
}
