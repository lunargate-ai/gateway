package api

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestTranslatedResponsesHTTPStableWireContract(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"chatcmpl_contract",
			"object":"chat.completion",
			"created":123,
			"model":"gpt-contract",
			"choices":[{"index":0,"message":{"role":"assistant","content":"answer"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":7,"completion_tokens":5,"total_tokens":12,"prompt_tokens_details":{"cached_tokens":2,"cache_write_tokens":1},"completion_tokens_details":{"reasoning_tokens":3,"accepted_prediction_tokens":1}}
		}`)
	}))
	defer upstream.Close()

	handler, cache := newNativeContinuationTestHandler(t, upstream.URL+"/v1", requestTypeChatCompletions)
	defer cache.Stop()
	payload := []byte(`{
		"model":"gpt-5.4",
		"input":"hello",
		"instructions":"be concise",
		"max_output_tokens":42,
		"reasoning":{"effort":"low"},
		"store":false,
		"temperature":0.25,
		"text":{"format":{"type":"json_object"}},
		"tool_choice":"auto",
		"tools":[{"type":"function","name":"lookup","parameters":{"type":"object","properties":{}}}],
		"top_p":0.75,
		"user":"contract-user"
	}`)
	recorder := httptest.NewRecorder()
	handler.Responses(recorder, httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(payload)))
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, body=%s", recorder.Code, recorder.Body.String())
	}

	response := decodeResponsesJSONForTest(t, recorder.Body.Bytes())
	assertSyntheticResponsesEnvelopeForTest(t, response, "completed")
	if got := responsesTextFromMapForTest(response); got != "answer" {
		t.Fatalf("output text = %q, want answer", got)
	}
	if response["instructions"] != "be concise" || response["store"] != false || response["user"] != "contract-user" {
		t.Fatalf("request controls were not reflected in response: %#v", response)
	}
	if response["max_output_tokens"] != float64(42) || response["temperature"] != float64(0.25) || response["top_p"] != float64(0.75) {
		t.Fatalf("numeric controls were not reflected in response: %#v", response)
	}
	reasoning, _ := response["reasoning"].(map[string]interface{})
	if reasoning["effort"] != "low" || reasoning["summary"] != nil {
		t.Fatalf("reasoning config = %#v", reasoning)
	}
	assertSyntheticResponsesUsageForTest(t, response["usage"], 7, 5, 12, 2, 1, 3)
	assertSyntheticResponsesTextAnnotationsForTest(t, response)
}

func TestTranslatedResponsesSSEStableWireContract(t *testing.T) {
	recorder := httptest.NewRecorder()
	proxy := newResponsesStreamProxy(recorder)
	proxy.requestPayload = map[string]json.RawMessage{
		"instructions": json.RawMessage(`"stream instructions"`),
		"reasoning":    json.RawMessage(`{"effort":"medium"}`),
		"store":        json.RawMessage(`false`),
		"temperature":  json.RawMessage(`0.4`),
		"top_p":        json.RawMessage(`0.8`),
	}
	frames := []string{
		`data: {"id":"chatcmpl_sse_contract","object":"chat.completion.chunk","created":123,"model":"gpt-contract","choices":[{"index":0,"delta":{"content":"answer"},"finish_reason":null}]}` + "\n\n",
		`data: {"id":"chatcmpl_sse_contract","object":"chat.completion.chunk","created":123,"model":"gpt-contract","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":5,"total_tokens":12,"prompt_tokens_details":{"cached_tokens":2,"cache_write_tokens":1},"completion_tokens_details":{"reasoning_tokens":3,"rejected_prediction_tokens":1}}}` + "\n\n",
		"data: [DONE]\n\n",
	}
	for _, frame := range frames {
		if _, err := proxy.Write([]byte(frame)); err != nil {
			t.Fatalf("write stream frame: %v", err)
		}
	}
	if err := proxy.finalize(); err != nil {
		t.Fatalf("finalize: %v", err)
	}

	events := decodeSSEEvents(t, recorder.Body.String())
	assertSyntheticResponsesSequenceForTest(t, events)
	var created, inProgress, delta, partDone, completed map[string]interface{}
	for _, event := range events {
		switch event["type"] {
		case "response.created":
			created, _ = event["response"].(map[string]interface{})
		case "response.in_progress":
			inProgress, _ = event["response"].(map[string]interface{})
		case "response.output_text.delta":
			delta = event
		case "response.content_part.done":
			part, _ := event["part"].(map[string]interface{})
			if part["type"] == "output_text" {
				partDone = part
			}
		case "response.completed":
			completed, _ = event["response"].(map[string]interface{})
		}
	}
	assertSyntheticResponsesEnvelopeForTest(t, created, "in_progress")
	assertSyntheticResponsesEnvelopeForTest(t, inProgress, "in_progress")
	assertSyntheticResponsesEnvelopeForTest(t, completed, "completed")
	if completed["instructions"] != "stream instructions" || completed["store"] != false {
		t.Fatalf("terminal request controls = %#v", completed)
	}
	if logprobs, ok := delta["logprobs"].([]interface{}); !ok || len(logprobs) != 0 {
		t.Fatalf("delta logprobs = %#v, want empty array", delta["logprobs"])
	}
	if annotations, ok := partDone["annotations"].([]interface{}); !ok || len(annotations) != 0 {
		t.Fatalf("output annotations = %#v, want empty array", partDone["annotations"])
	}
	if got := responsesTextFromMapForTest(completed); got != "answer" {
		t.Fatalf("terminal output text = %q, want answer", got)
	}
	assertSyntheticResponsesUsageForTest(t, completed["usage"], 7, 5, 12, 2, 1, 3)
}

func TestTranslatedResponsesWebSocketStableWireContract(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, `data: {"id":"chatcmpl_ws_contract","object":"chat.completion.chunk","created":123,"model":"gpt-contract","choices":[{"index":0,"delta":{"content":"answer"},"finish_reason":null}]}`+"\n\n")
		_, _ = io.WriteString(w, `data: {"id":"chatcmpl_ws_contract","object":"chat.completion.chunk","created":123,"model":"gpt-contract","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":5,"total_tokens":12}}`+"\n\n")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()

	handler := newResponsesWebSocketTestHandler(upstream.URL)
	defer handler.cache.Stop()
	server := httptest.NewServer(http.HandlerFunc(handler.ResponsesWebSocket))
	defer server.Close()
	conn := mustDialResponsesWebSocket(t, server.URL)
	defer conn.Close()

	sendResponsesWebSocketJSON(t, conn, map[string]interface{}{
		"type":         "response.create",
		"model":        "lunargate/auto",
		"input":        "hello",
		"instructions": "ws instructions",
		"store":        false,
	})
	events := readResponsesWebSocketEventsUntilTerminal(t, conn)
	assertSyntheticResponsesSequenceForTest(t, events)
	var completed map[string]interface{}
	for _, event := range events {
		if event["type"] == "response.completed" {
			completed, _ = event["response"].(map[string]interface{})
		}
	}
	assertSyntheticResponsesEnvelopeForTest(t, completed, "completed")
	if completed["instructions"] != "ws instructions" || completed["store"] != false {
		t.Fatalf("websocket terminal request controls = %#v", completed)
	}
	assertSyntheticResponsesUsageForTest(t, completed["usage"], 7, 5, 12, 0, 0, 0)
}

func TestResponsesWebSocketWarmupAndErrorSequenceContract(t *testing.T) {
	handler := &Handler{}
	server := httptest.NewServer(http.HandlerFunc(handler.ResponsesWebSocket))
	defer server.Close()
	conn := mustDialResponsesWebSocket(t, server.URL)
	defer conn.Close()

	sendResponsesWebSocketJSON(t, conn, map[string]interface{}{
		"type":         "response.create",
		"model":        "gpt-contract",
		"generate":     false,
		"instructions": "warmup instructions",
		"store":        false,
	})
	warmup := readResponsesWebSocketEventsUntilTerminal(t, conn)
	assertSyntheticResponsesSequenceForTest(t, warmup)
	if len(warmup) != 2 {
		t.Fatalf("warmup event count = %d, want 2", len(warmup))
	}
	for index, status := range []string{"in_progress", "completed"} {
		response, _ := warmup[index]["response"].(map[string]interface{})
		assertSyntheticResponsesEnvelopeForTest(t, response, status)
		if response["instructions"] != "warmup instructions" || response["store"] != false {
			t.Fatalf("warmup response controls = %#v", response)
		}
	}

	sendResponsesWebSocketJSON(t, conn, map[string]interface{}{"type": "unknown.event"})
	errorEvent := readResponsesWebSocketEvent(t, conn)
	assertSyntheticResponsesSequenceForTest(t, []map[string]interface{}{errorEvent})
	if errorEvent["type"] != "error" || errorEvent["message"] == "" {
		t.Fatalf("error event = %#v", errorEvent)
	}
	for _, field := range []string{"code", "message", "param", "sequence_number"} {
		if _, exists := errorEvent[field]; !exists {
			t.Fatalf("error event missing official field %q: %#v", field, errorEvent)
		}
	}
}

func TestResponsesWebSocketSyntheticFailureFollowsNativeSequence(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, `event: response.created
data: {"type":"response.created","sequence_number":7,"response":{"id":"resp_ws_native","object":"response","created_at":123,"status":"in_progress","model":"gpt-native","output":[]}}

event: response.output_text.delta
data: {"type":"response.output_text.delta","sequence_number":8,"response_id":"resp_changed","item_id":"msg_1","output_index":0,"content_index":0,"delta":"must-not-leak","logprobs":[]}

`)
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
		"input": "hello",
	})
	created := readResponsesWebSocketEvent(t, conn)
	failureEvent := readResponsesWebSocketEvent(t, conn)
	if created["type"] != "response.created" || created["sequence_number"] != float64(7) {
		t.Fatalf("native created event = %#v", created)
	}
	if failureEvent["type"] != "response.failed" || failureEvent["sequence_number"] != float64(8) {
		t.Fatalf("synthetic failure did not follow native event sequence: %#v", failureEvent)
	}
	if _, leaked := failureEvent["event_id"]; leaked {
		t.Fatalf("synthetic failure exposed non-contract event_id: %#v", failureEvent)
	}
	failedResponse, _ := failureEvent["response"].(map[string]interface{})
	if failedResponse["status"] != "failed" || failedResponse["incomplete_details"] != nil || failedResponse["usage"] != nil {
		t.Fatalf("synthetic failure envelope = %#v", failedResponse)
	}
}

func TestResponsesWebSocketRejectsInvalidNativeSequences(t *testing.T) {
	tests := []struct {
		name  string
		event string
	}{
		{
			name:  "missing",
			event: `{"type":"response.output_text.delta","response_id":"resp_ws_sequence","delta":"must-not-leak"}`,
		},
		{
			name:  "fractional",
			event: `{"type":"response.output_text.delta","sequence_number":7.5,"response_id":"resp_ws_sequence","delta":"must-not-leak"}`,
		},
		{
			name:  "duplicate",
			event: `{"type":"response.output_text.delta","sequence_number":7,"response_id":"resp_ws_sequence","delta":"must-not-leak"}`,
		},
		{
			name:  "decreasing",
			event: `{"type":"response.output_text.delta","sequence_number":6,"response_id":"resp_ws_sequence","delta":"must-not-leak"}`,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = io.WriteString(w, `event: response.created
data: {"type":"response.created","sequence_number":7,"response":{"id":"resp_ws_sequence","object":"response","created_at":123,"status":"in_progress","model":"gpt-native","output":[]}}

event: response.output_text.delta
data: `+test.event+`

`)
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
				"input": "hello",
			})
			created := readResponsesWebSocketEvent(t, conn)
			failure := readResponsesWebSocketEvent(t, conn)
			if created["sequence_number"] != float64(7) {
				t.Fatalf("created sequence_number = %#v, want 7", created["sequence_number"])
			}
			if failure["type"] != "response.failed" || failure["sequence_number"] != float64(8) {
				t.Fatalf("synthetic failure = %#v, want response.failed sequence 8", failure)
			}
		})
	}
}

func responsesTextFromTypedResponseForTest(response *models.ResponsesResponse) string {
	if response == nil {
		return ""
	}
	parts := make([]string, 0, 1)
	for _, item := range response.Output {
		if item.Type != "message" {
			continue
		}
		for _, content := range item.Content {
			if content.Type == "output_text" {
				parts = append(parts, content.Text)
			}
		}
	}
	return strings.Join(parts, "")
}

func responsesTextFromMapForTest(response map[string]interface{}) string {
	if response == nil {
		return ""
	}
	parts := make([]string, 0, 1)
	output, _ := response["output"].([]interface{})
	for _, rawItem := range output {
		item, _ := rawItem.(map[string]interface{})
		if item["type"] != "message" {
			continue
		}
		content, _ := item["content"].([]interface{})
		for _, rawPart := range content {
			part, _ := rawPart.(map[string]interface{})
			if part["type"] == "output_text" {
				text, _ := part["text"].(string)
				parts = append(parts, text)
			}
		}
	}
	return strings.Join(parts, "")
}

func assertSyntheticResponsesSequenceForTest(t *testing.T, events []map[string]interface{}) {
	t.Helper()
	for index, event := range events {
		sequence, ok := event["sequence_number"].(float64)
		if !ok || int(sequence) != index {
			t.Fatalf("event %d sequence_number = %#v, want %d", index, event["sequence_number"], index)
		}
		if _, exists := event["event_id"]; exists {
			t.Fatalf("event %d exposed non-contract event_id: %#v", index, event)
		}
	}
}

func assertSyntheticResponsesEnvelopeForTest(
	t *testing.T,
	response map[string]interface{},
	status string,
) {
	t.Helper()
	if response == nil {
		t.Fatal("missing response envelope")
	}
	if response["object"] != "response" || response["status"] != status {
		t.Fatalf("response identity/status = %#v", response)
	}
	for _, field := range []string{
		"id", "created_at", "error", "incomplete_details", "instructions",
		"max_output_tokens", "metadata", "model", "output",
		"parallel_tool_calls", "previous_response_id", "reasoning", "store",
		"temperature", "text", "tool_choice", "tools", "top_p",
		"truncation", "usage", "user",
	} {
		if _, exists := response[field]; !exists {
			t.Fatalf("response missing stable field %q: %#v", field, response)
		}
	}
	if _, exists := response["output_text"]; exists {
		t.Fatalf("response exposed SDK-only output_text: %#v", response)
	}
	if status == "completed" {
		if _, exists := response["completed_at"]; !exists {
			t.Fatalf("completed response missing completed_at: %#v", response)
		}
	}
}

func assertSyntheticResponsesUsageForTest(
	t *testing.T,
	rawUsage interface{},
	inputTokens int,
	outputTokens int,
	totalTokens int,
	cachedTokens int,
	cacheWriteTokens int,
	reasoningTokens int,
) {
	t.Helper()
	usage, _ := rawUsage.(map[string]interface{})
	if usage == nil {
		t.Fatalf("usage = %#v, want object", rawUsage)
	}
	if usage["input_tokens"] != float64(inputTokens) ||
		usage["output_tokens"] != float64(outputTokens) ||
		usage["total_tokens"] != float64(totalTokens) {
		t.Fatalf("usage counters = %#v", usage)
	}
	inputDetails, _ := usage["input_tokens_details"].(map[string]interface{})
	if inputDetails["cached_tokens"] != float64(cachedTokens) ||
		inputDetails["cache_write_tokens"] != float64(cacheWriteTokens) {
		t.Fatalf("input token details = %#v", inputDetails)
	}
	outputDetails, _ := usage["output_tokens_details"].(map[string]interface{})
	if outputDetails["reasoning_tokens"] != float64(reasoningTokens) {
		t.Fatalf("output token details = %#v", outputDetails)
	}
}

func assertSyntheticResponsesTextAnnotationsForTest(t *testing.T, response map[string]interface{}) {
	t.Helper()
	output, _ := response["output"].([]interface{})
	if len(output) == 0 {
		t.Fatalf("response output = %#v", response["output"])
	}
	message, _ := output[0].(map[string]interface{})
	content, _ := message["content"].([]interface{})
	if len(content) == 0 {
		t.Fatalf("message content = %#v", message["content"])
	}
	part, _ := content[0].(map[string]interface{})
	annotations, ok := part["annotations"].([]interface{})
	if !ok || len(annotations) != 0 {
		t.Fatalf("output text annotations = %#v, want empty array", part["annotations"])
	}
}

func decodeResponsesJSONForTest(t *testing.T, payload []byte) map[string]interface{} {
	t.Helper()
	var decoded map[string]interface{}
	if err := json.Unmarshal(payload, &decoded); err != nil {
		t.Fatalf("decode response JSON: %v", err)
	}
	return decoded
}
