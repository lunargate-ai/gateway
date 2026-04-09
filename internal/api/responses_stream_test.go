package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestResponsesStreamProxy_ToolCallIDsStayStableAcrossCallAndFC(t *testing.T) {
	rec := httptest.NewRecorder()
	proxy := newResponsesStreamProxy(rec)
	proxy.responseID = "resp_test_1"
	proxy.model = "gpt-5.3-codex"
	proxy.created = 123

	if err := proxy.ensureStarted(); err != nil {
		t.Fatalf("ensureStarted error: %v", err)
	}

	if err := proxy.processToolCallDelta(models.ToolCall{
		ID:   "call_abc123",
		Type: "function",
		Function: models.ToolCallFunction{
			Name:      "exec_command",
			Arguments: "{\"command\":",
		},
	}); err != nil {
		t.Fatalf("processToolCallDelta call_* error: %v", err)
	}

	if err := proxy.processToolCallDelta(models.ToolCall{
		ID:   "fc_abc123",
		Type: "function",
		Function: models.ToolCallFunction{
			Arguments: "\"pwd\"}",
		},
	}); err != nil {
		t.Fatalf("processToolCallDelta fc_* error: %v", err)
	}

	if err := proxy.emitCompleted(); err != nil {
		t.Fatalf("emitCompleted error: %v", err)
	}

	events := decodeSSEEvents(t, rec.Body.String())

	addedFound := false
	for _, evt := range events {
		typeName, _ := evt["type"].(string)
		switch typeName {
		case "response.output_item.added":
			item, _ := evt["item"].(map[string]interface{})
			if item == nil {
				continue
			}
			itemType, _ := item["type"].(string)
			if itemType != "function_call" {
				continue
			}
			addedFound = true
			if got, _ := item["id"].(string); got != "fc_abc123" {
				t.Fatalf("expected function_call item id fc_abc123, got %q", got)
			}
			if got, _ := item["call_id"].(string); got != "call_abc123" {
				t.Fatalf("expected function_call call_id call_abc123, got %q", got)
			}
		case "response.function_call_arguments.delta", "response.function_call_arguments.done":
			if got, _ := evt["item_id"].(string); got != "fc_abc123" {
				t.Fatalf("expected %s item_id fc_abc123, got %q", typeName, got)
			}
		}
	}

	if !addedFound {
		t.Fatalf("expected function_call response.output_item.added event")
	}
}

func TestResponsesStreamProxy_MergeTextDelta_DeduplicatesDoneSnapshots(t *testing.T) {
	proxy := newResponsesStreamProxy(httptest.NewRecorder())

	if got := proxy.mergeTextDelta("Hello"); got != "Hello" {
		t.Fatalf("expected first delta to pass through, got %q", got)
	}
	if got := proxy.mergeTextDelta("Hello"); got != "" {
		t.Fatalf("expected exact duplicate to be dropped, got %q", got)
	}
	if got := proxy.mergeTextDelta("Hello world"); got != " world" {
		t.Fatalf("expected snapshot delta tail, got %q", got)
	}
	if got := proxy.mergeTextDelta(" world"); got != "" {
		t.Fatalf("expected overlapping suffix to be dropped, got %q", got)
	}
	if final := proxy.text.String(); final != "Hello world" {
		t.Fatalf("expected merged text to be %q, got %q", "Hello world", final)
	}
}

func TestResponsesStreamProxy_ToolOnlyTurnDoesNotEmitEmptyAssistantMessage(t *testing.T) {
	rec := httptest.NewRecorder()
	proxy := newResponsesStreamProxy(rec)
	proxy.responseID = "resp_tool_only"
	proxy.model = "gpt-5.3-codex"
	proxy.created = 123

	if err := proxy.ensureStarted(); err != nil {
		t.Fatalf("ensureStarted error: %v", err)
	}

	if err := proxy.processToolCallDelta(models.ToolCall{
		ID:   "call_xyz",
		Type: "function",
		Function: models.ToolCallFunction{
			Name:      "exec_command",
			Arguments: `{"command":"pwd"}`,
		},
	}); err != nil {
		t.Fatalf("processToolCallDelta error: %v", err)
	}

	if err := proxy.emitCompleted(); err != nil {
		t.Fatalf("emitCompleted error: %v", err)
	}

	events := decodeSSEEvents(t, rec.Body.String())
	for _, evt := range events {
		typeName, _ := evt["type"].(string)
		if typeName != "response.output_item.done" {
			continue
		}
		item, _ := evt["item"].(map[string]interface{})
		if item == nil {
			continue
		}
		if itemType, _ := item["type"].(string); itemType == "message" {
			t.Fatalf("unexpected empty assistant message completion in tool-only turn")
		}
	}

	var completed map[string]interface{}
	for _, evt := range events {
		if evtType, _ := evt["type"].(string); evtType == "response.completed" {
			responseObj, _ := evt["response"].(map[string]interface{})
			if responseObj != nil {
				completed = responseObj
			}
		}
	}
	if completed == nil {
		t.Fatalf("expected response.completed event")
	}
	output, _ := completed["output"].([]interface{})
	if len(output) != 1 {
		t.Fatalf("expected only tool output item in tool-only turn, got %d items", len(output))
	}
}

func TestResponsesStreamProxy_ErrorPassthroughKeepsContentType(t *testing.T) {
	rec := httptest.NewRecorder()
	proxy := newResponsesStreamProxy(rec)

	proxy.Header().Set("Content-Type", "application/json")
	proxy.WriteHeader(http.StatusBadRequest)
	_, err := proxy.Write([]byte(`{"error":{"message":"bad request","type":"invalid_request_error"}}`))
	if err != nil {
		t.Fatalf("write error: %v", err)
	}
	if err := proxy.finalize(); err != nil {
		t.Fatalf("finalize error: %v", err)
	}

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected status %d, got %d", http.StatusBadRequest, rec.Code)
	}
	if got := rec.Header().Get("Content-Type"); got != "application/json" {
		t.Fatalf("expected content-type application/json, got %q", got)
	}
	body := strings.TrimSpace(rec.Body.String())
	if !strings.Contains(body, `"invalid_request_error"`) {
		t.Fatalf("expected passthrough error payload, got %q", body)
	}
}

func TestResponsesStreamProxy_EventOrderingWithTextAndToolCall(t *testing.T) {
	rec := httptest.NewRecorder()
	proxy := newResponsesStreamProxy(rec)

	firstChunk := `data: {"id":"chatcmpl-order","object":"chat.completion.chunk","created":1,"model":"mock-gpt","choices":[{"index":0,"delta":{"content":"Hello"}}]}

`
	if _, err := proxy.Write([]byte(firstChunk)); err != nil {
		t.Fatalf("write first chunk error: %v", err)
	}

	secondChunk := `data: {"id":"chatcmpl-order","object":"chat.completion.chunk","created":1,"model":"mock-gpt","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_order","type":"function","function":{"name":"exec_command","arguments":"{\"cmd\":\"p"}}]}}]}

`
	if _, err := proxy.Write([]byte(secondChunk)); err != nil {
		t.Fatalf("write second chunk error: %v", err)
	}

	thirdChunk := `data: {"id":"chatcmpl-order","object":"chat.completion.chunk","created":1,"model":"mock-gpt","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_order","type":"function","function":{"arguments":"wd\"}"}}]},"finish_reason":"tool_calls"}]}

`
	if _, err := proxy.Write([]byte(thirdChunk)); err != nil {
		t.Fatalf("write third chunk error: %v", err)
	}

	if _, err := proxy.Write([]byte("data: [DONE]\n\n")); err != nil {
		t.Fatalf("write done chunk error: %v", err)
	}
	if err := proxy.finalize(); err != nil {
		t.Fatalf("finalize error: %v", err)
	}

	events := decodeSSEEvents(t, rec.Body.String())
	typeOrder := make([]string, 0, len(events))
	for _, evt := range events {
		if typeName, _ := evt["type"].(string); typeName != "" {
			typeOrder = append(typeOrder, typeName)
		}
	}

	idxCreated := firstIndex(typeOrder, "response.created")
	idxMsgAdded := firstIndex(typeOrder, "response.output_item.added")
	idxPartAdded := firstIndex(typeOrder, "response.content_part.added")
	idxTextDelta := firstIndex(typeOrder, "response.output_text.delta")
	idxToolArgsDelta := firstIndex(typeOrder, "response.function_call_arguments.delta")
	idxTextDone := firstIndex(typeOrder, "response.output_text.done")
	idxToolArgsDone := firstIndex(typeOrder, "response.function_call_arguments.done")
	idxCompleted := firstIndex(typeOrder, "response.completed")

	if idxCreated < 0 || idxMsgAdded < 0 || idxPartAdded < 0 || idxTextDelta < 0 || idxToolArgsDelta < 0 || idxTextDone < 0 || idxToolArgsDone < 0 || idxCompleted < 0 {
		t.Fatalf("expected lifecycle events missing, order=%v", typeOrder)
	}
	if !(idxCreated < idxMsgAdded && idxMsgAdded < idxPartAdded && idxPartAdded < idxTextDelta) {
		t.Fatalf("expected message/text start ordering, order=%v", typeOrder)
	}
	if !(idxTextDelta < idxTextDone && idxToolArgsDelta < idxToolArgsDone && idxToolArgsDone < idxCompleted) {
		t.Fatalf("expected completion ordering, order=%v", typeOrder)
	}
}

func TestResponsesStreamProxy_FunctionArgumentsAssembleAcrossChunks(t *testing.T) {
	rec := httptest.NewRecorder()
	proxy := newResponsesStreamProxy(rec)

	chunks := []string{
		`data: {"id":"chatcmpl-args","object":"chat.completion.chunk","created":1,"model":"mock-gpt","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_assemble","type":"function","function":{"name":"exec_command","arguments":"{\"cmd\":\"p"}}]}}]}

`,
		`data: {"id":"chatcmpl-args","object":"chat.completion.chunk","created":1,"model":"mock-gpt","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_assemble","type":"function","function":{"arguments":"wd\",\"cwd\":\"/"}}]}}]}

`,
		`data: {"id":"chatcmpl-args","object":"chat.completion.chunk","created":1,"model":"mock-gpt","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_assemble","type":"function","function":{"arguments":"tmp\"}"}}]},"finish_reason":"tool_calls"}]}

`,
	}

	for i, chunk := range chunks {
		if _, err := proxy.Write([]byte(chunk)); err != nil {
			t.Fatalf("write chunk %d error: %v", i+1, err)
		}
	}
	if _, err := proxy.Write([]byte("data: [DONE]\n\n")); err != nil {
		t.Fatalf("write done chunk error: %v", err)
	}
	if err := proxy.finalize(); err != nil {
		t.Fatalf("finalize error: %v", err)
	}

	events := decodeSSEEvents(t, rec.Body.String())
	expectedArgs := `{"cmd":"pwd","cwd":"/tmp"}`

	foundDoneArgs := false
	var completedResponse map[string]interface{}
	for _, evt := range events {
		typeName, _ := evt["type"].(string)
		if typeName == "response.function_call_arguments.done" {
			if got, _ := evt["arguments"].(string); got == expectedArgs {
				foundDoneArgs = true
			}
		}
		if typeName == "response.completed" {
			completedResponse, _ = evt["response"].(map[string]interface{})
		}
	}
	if !foundDoneArgs {
		t.Fatalf("expected function_call_arguments.done with assembled args %q", expectedArgs)
	}
	if completedResponse == nil {
		t.Fatalf("expected response.completed event")
	}

	output, _ := completedResponse["output"].([]interface{})
	if len(output) == 0 {
		t.Fatalf("expected completed output items")
	}
	foundOutputArgs := false
	for _, rawItem := range output {
		item, _ := rawItem.(map[string]interface{})
		if item == nil {
			continue
		}
		itemType, _ := item["type"].(string)
		if itemType != "function_call" {
			continue
		}
		if got, _ := item["arguments"].(string); got == expectedArgs {
			foundOutputArgs = true
			break
		}
	}
	if !foundOutputArgs {
		t.Fatalf("expected completed function_call output to include assembled args %q", expectedArgs)
	}
}

func TestResponsesStreamProxy_ReasoningLifecycleAndCompletedSummary(t *testing.T) {
	rec := httptest.NewRecorder()
	proxy := newResponsesStreamProxy(rec)

	chunks := []string{
		`data: {"id":"chatcmpl-reason","object":"chat.completion.chunk","created":1,"model":"mock-gpt","choices":[{"index":0,"delta":{"reasoning_content":"step 1: gather context. ","content":"Answer"}}]}

`,
		`data: {"id":"chatcmpl-reason","object":"chat.completion.chunk","created":1,"model":"mock-gpt","choices":[{"index":0,"delta":{"reasoning_content":"step 2: propose fix.","content":" done"}}]}

`,
	}

	for i, chunk := range chunks {
		if _, err := proxy.Write([]byte(chunk)); err != nil {
			t.Fatalf("write chunk %d error: %v", i+1, err)
		}
	}
	if _, err := proxy.Write([]byte("data: [DONE]\n\n")); err != nil {
		t.Fatalf("write done chunk error: %v", err)
	}
	if err := proxy.finalize(); err != nil {
		t.Fatalf("finalize error: %v", err)
	}

	events := decodeSSEEvents(t, rec.Body.String())
	assertSequenceNumbersMonotonic(t, events)

	if !containsEventType(events, "response.reasoning_summary_part.added") {
		t.Fatalf("expected response.reasoning_summary_part.added event")
	}
	if !containsEventType(events, "response.reasoning_summary_text.delta") {
		t.Fatalf("expected response.reasoning_summary_text.delta event")
	}
	if !containsEventType(events, "response.reasoning_summary_text.done") {
		t.Fatalf("expected response.reasoning_summary_text.done event")
	}
	if !containsEventType(events, "response.reasoning_summary_part.done") {
		t.Fatalf("expected response.reasoning_summary_part.done event")
	}

	var reasoningItemDone map[string]interface{}
	var completedResponse map[string]interface{}
	for _, evt := range events {
		evtType, _ := evt["type"].(string)
		if evtType == "response.output_item.done" {
			item, _ := evt["item"].(map[string]interface{})
			if item != nil {
				if itemType, _ := item["type"].(string); itemType == "reasoning" {
					reasoningItemDone = item
				}
			}
		}
		if evtType == "response.completed" {
			completedResponse, _ = evt["response"].(map[string]interface{})
		}
	}
	if reasoningItemDone == nil {
		t.Fatalf("expected completed reasoning output item")
	}
	summary, _ := reasoningItemDone["summary"].([]interface{})
	if len(summary) == 0 {
		t.Fatalf("expected reasoning summary in output item")
	}
	summaryPart, _ := summary[0].(map[string]interface{})
	if summaryPart == nil {
		t.Fatalf("expected reasoning summary object")
	}
	reasoningText, _ := summaryPart["text"].(string)
	if !strings.Contains(reasoningText, "step 1") || !strings.Contains(reasoningText, "step 2") {
		t.Fatalf("expected merged reasoning summary, got %q", reasoningText)
	}

	if completedResponse == nil {
		t.Fatalf("expected response.completed payload")
	}
	reasoningObj, _ := completedResponse["reasoning"].(map[string]interface{})
	if reasoningObj == nil {
		t.Fatalf("expected response.completed to include reasoning object")
	}
	completedSummary, _ := reasoningObj["summary"].([]interface{})
	if len(completedSummary) == 0 {
		t.Fatalf("expected response.completed reasoning summary")
	}
}

func TestResponsesStreamProxy_FunctionArgumentsDoneIncludesName(t *testing.T) {
	rec := httptest.NewRecorder()
	proxy := newResponsesStreamProxy(rec)
	proxy.responseID = "resp_tool_done_name"
	proxy.model = "gpt-5.3-codex"
	proxy.created = 123

	if err := proxy.ensureStarted(); err != nil {
		t.Fatalf("ensureStarted error: %v", err)
	}
	if err := proxy.processToolCallDelta(models.ToolCall{
		ID:   "call_done_name",
		Type: "function",
		Function: models.ToolCallFunction{
			Name:      "exec_command",
			Arguments: "{\"cmd\":\"pwd\"}",
		},
	}); err != nil {
		t.Fatalf("processToolCallDelta error: %v", err)
	}
	if err := proxy.emitCompleted(); err != nil {
		t.Fatalf("emitCompleted error: %v", err)
	}

	events := decodeSSEEvents(t, rec.Body.String())
	assertSequenceNumbersMonotonic(t, events)

	var doneEvent map[string]interface{}
	for _, evt := range events {
		if evtType, _ := evt["type"].(string); evtType == "response.function_call_arguments.done" {
			doneEvent = evt
			break
		}
	}
	if doneEvent == nil {
		t.Fatalf("expected response.function_call_arguments.done event")
	}
	if got, _ := doneEvent["name"].(string); got != "exec_command" {
		t.Fatalf("expected done event name %q, got %q", "exec_command", got)
	}
	if _, ok := doneEvent["sequence_number"]; !ok {
		t.Fatalf("expected sequence_number on done event")
	}
}

func decodeSSEEvents(t *testing.T, body string) []map[string]interface{} {
	t.Helper()
	frames := strings.Split(body, "\n\n")
	events := make([]map[string]interface{}, 0, len(frames))
	for _, frame := range frames {
		frame = strings.TrimSpace(frame)
		if frame == "" {
			continue
		}
		if !strings.HasPrefix(frame, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(frame, "data:"))
		if payload == "" || payload == "[DONE]" {
			continue
		}
		var event map[string]interface{}
		if err := json.Unmarshal([]byte(payload), &event); err != nil {
			t.Fatalf("failed to decode SSE payload %q: %v", payload, err)
		}
		events = append(events, event)
	}
	return events
}

func containsEventType(events []map[string]interface{}, targetType string) bool {
	for _, event := range events {
		if evtType, _ := event["type"].(string); evtType == targetType {
			return true
		}
	}
	return false
}

func assertSequenceNumbersMonotonic(t *testing.T, events []map[string]interface{}) {
	t.Helper()
	prev := 0
	for i, evt := range events {
		raw, ok := evt["sequence_number"]
		if !ok {
			t.Fatalf("event %d missing sequence_number", i)
		}
		nFloat, ok := raw.(float64)
		if !ok {
			t.Fatalf("event %d has non-numeric sequence_number type %T", i, raw)
		}
		n := int(nFloat)
		if n <= prev {
			t.Fatalf("sequence_number not monotonic at event %d: prev=%d current=%d", i, prev, n)
		}
		prev = n
	}
}

func firstIndex(items []string, target string) int {
	for i, item := range items {
		if item == target {
			return i
		}
	}
	return -1
}
