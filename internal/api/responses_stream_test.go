package api

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestResponsesStreamProxy_EmitsFailedInsteadOfCompletedAfterStreamError(t *testing.T) {
	rec := httptest.NewRecorder()
	proxy := newResponsesStreamProxy(rec)
	partial := "data: {\"id\":\"resp_partial\",\"object\":\"chat.completion.chunk\",\"created\":123,\"model\":\"gpt\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"}}]}\n\n"
	if _, err := proxy.Write([]byte(partial)); err != nil {
		t.Fatalf("write partial chunk: %v", err)
	}
	proxy.RecordStreamError(errors.New("upstream closed early"))
	if err := proxy.finalize(); err != nil {
		t.Fatalf("finalize: %v", err)
	}

	events := decodeSSEEvents(t, rec.Body.String())
	foundFailed := false
	for _, event := range events {
		switch event["type"] {
		case "response.completed":
			t.Fatal("truncated stream must not emit response.completed")
		case "response.failed":
			foundFailed = true
			response, _ := event["response"].(map[string]interface{})
			if response["status"] != "failed" {
				t.Fatalf("failed response status = %#v", response["status"])
			}
		}
	}
	if !foundFailed {
		t.Fatal("expected response.failed event")
	}
}

func TestResponsesStreamProxy_ReplacesChatCompletionID(t *testing.T) {
	recorder := httptest.NewRecorder()
	proxy := newResponsesStreamProxy(recorder)
	chunk := "data: {\"id\":\"chatcmpl-upstream\",\"object\":\"chat.completion.chunk\",\"created\":123,\"model\":\"gpt\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n"
	if _, err := proxy.Write([]byte(chunk)); err != nil {
		t.Fatalf("write stream chunk: %v", err)
	}
	if err := proxy.finalize(); err != nil {
		t.Fatalf("finalize: %v", err)
	}
	if !strings.HasPrefix(proxy.responseID, "resp_") {
		t.Fatalf("translated response ID = %q, want resp_ prefix", proxy.responseID)
	}
	if strings.Contains(recorder.Body.String(), "chatcmpl-upstream") {
		t.Fatalf("upstream Chat Completions ID leaked into Responses stream: %s", recorder.Body.String())
	}
	for _, event := range decodeSSEEvents(t, recorder.Body.String()) {
		if responseID, ok := event["response_id"].(string); ok && responseID != proxy.responseID {
			t.Fatalf("event response_id = %q, want stable %q", responseID, proxy.responseID)
		}
		if response, ok := event["response"].(map[string]interface{}); ok {
			if id, ok := response["id"].(string); ok && id != proxy.responseID {
				t.Fatalf("response id = %q, want stable %q", id, proxy.responseID)
			}
		}
	}
}

func TestResponsesStreamProxy_ReturnsDownstreamFlushError(t *testing.T) {
	writer := &responsesFlushErrorWriter{header: make(http.Header)}
	proxy := newResponsesStreamProxy(writer)

	err := proxy.ensureStarted()
	if !errors.Is(err, errResponsesDownstreamFlush) {
		t.Fatalf("ensureStarted error = %v, want downstream flush error", err)
	}
}

var errResponsesDownstreamFlush = errors.New("injected responses downstream flush failure")

type responsesFlushErrorWriter struct {
	header http.Header
}

func (w *responsesFlushErrorWriter) Header() http.Header {
	return w.header
}

func (w *responsesFlushErrorWriter) WriteHeader(int) {}

func (w *responsesFlushErrorWriter) Write(payload []byte) (int, error) {
	return len(payload), nil
}

func (w *responsesFlushErrorWriter) FlushError() error {
	return errResponsesDownstreamFlush
}

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

func TestResponsesStreamProxy_MergeTextDelta_PreservesRepeatedDeltas(t *testing.T) {
	proxy := newResponsesStreamProxy(httptest.NewRecorder())

	if got, err := proxy.mergeTextDelta("ha"); err != nil || got != "ha" {
		t.Fatalf("expected first delta to pass through, got %q", got)
	}
	if got, err := proxy.mergeTextDelta("ha"); err != nil || got != "ha" {
		t.Fatalf("expected repeated delta to pass through, got %q", got)
	}
	if got, err := proxy.mergeReasoningDelta("think"); err != nil || got != "think" {
		t.Fatalf("expected first reasoning delta to pass through, got %q", got)
	}
	if got, err := proxy.mergeReasoningDelta("think"); err != nil || got != "think" {
		t.Fatalf("expected repeated reasoning delta to pass through, got %q", got)
	}
	if final := proxy.text.String(); final != "haha" {
		t.Fatalf("expected merged text to be %q, got %q", "haha", final)
	}
	if final := proxy.reasoningText.String(); final != "thinkthink" {
		t.Fatalf("expected merged reasoning to be %q, got %q", "thinkthink", final)
	}
}

func TestResponsesStreamProxy_ToolCallIDIsExactAndOpaque(t *testing.T) {
	t.Run("internal whitespace is preserved", func(t *testing.T) {
		recorder := httptest.NewRecorder()
		proxy := newResponsesStreamProxy(recorder)
		index := 0
		if err := proxy.processToolCallDelta(models.ToolCall{
			Index: &index,
			ID:    "opaque call id",
			Type:  "function",
			Function: models.ToolCallFunction{
				Name:      "lookup",
				Arguments: "{}",
			},
		}); err != nil {
			t.Fatalf("process exact tool id: %v", err)
		}
		if err := proxy.emitCompleted(); err != nil {
			t.Fatalf("emit completed: %v", err)
		}
		for _, event := range decodeSSEEvents(t, recorder.Body.String()) {
			if event["type"] != "response.output_item.added" {
				continue
			}
			item, _ := event["item"].(map[string]interface{})
			if item != nil && item["type"] == "function_call" && item["call_id"] != "opaque call id" {
				t.Fatalf("call_id = %#v, want exact opaque value", item["call_id"])
			}
		}
	})

	t.Run("surrounding whitespace fails closed", func(t *testing.T) {
		proxy := newResponsesStreamProxy(httptest.NewRecorder())
		index := 0
		err := proxy.processToolCallDelta(models.ToolCall{Index: &index, ID: " call_1 "})
		if !errors.Is(err, errResponsesStreamInvalidToolID) {
			t.Fatalf("error = %v, want %v", err, errResponsesStreamInvalidToolID)
		}
		if len(proxy.toolCalls) != 0 {
			t.Fatalf("invalid id mutated tool state: %#v", proxy.toolCalls)
		}
	})
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

func TestResponsesStreamProxy_IncludesUsageInCompletedResponse(t *testing.T) {
	rec := httptest.NewRecorder()
	proxy := newResponsesStreamProxy(rec)

	chunk := "data: {\"id\":\"resp_usage\",\"object\":\"chat.completion.chunk\",\"created\":123,\"model\":\"gpt-5.3-codex\",\"choices\":[],\"usage\":{\"prompt_tokens\":17,\"completion_tokens\":9,\"total_tokens\":26}}\n\n"
	if _, err := proxy.Write([]byte(chunk)); err != nil {
		t.Fatalf("write usage chunk: %v", err)
	}
	if _, err := proxy.Write([]byte("data: [DONE]\n\n")); err != nil {
		t.Fatalf("write done chunk: %v", err)
	}
	if err := proxy.finalize(); err != nil {
		t.Fatalf("finalize error: %v", err)
	}

	events := decodeSSEEvents(t, rec.Body.String())
	for _, event := range events {
		if event["type"] != "response.completed" {
			continue
		}
		response, _ := event["response"].(map[string]interface{})
		usage, _ := response["usage"].(map[string]interface{})
		if usage == nil {
			t.Fatal("expected response.completed usage")
		}
		if usage["input_tokens"] != float64(17) || usage["output_tokens"] != float64(9) || usage["total_tokens"] != float64(26) {
			t.Fatalf("unexpected usage: %#v", usage)
		}
		return
	}
	t.Fatal("expected response.completed event")
}

func TestResponsesStreamProxy_MergesSplitUsageAcrossChunks(t *testing.T) {
	rec := httptest.NewRecorder()
	proxy := newResponsesStreamProxy(rec)

	chunks := []string{
		"data: {\"id\":\"resp_usage\",\"object\":\"chat.completion.chunk\",\"created\":123,\"model\":\"claude\",\"choices\":[],\"usage\":{\"prompt_tokens\":17,\"completion_tokens\":0,\"total_tokens\":17}}\n\n",
		"data: {\"id\":\"resp_usage\",\"object\":\"chat.completion.chunk\",\"created\":123,\"model\":\"claude\",\"choices\":[],\"usage\":{\"prompt_tokens\":0,\"completion_tokens\":9,\"total_tokens\":0}}\n\n",
		"data: [DONE]\n\n",
	}
	for _, chunk := range chunks {
		if _, err := proxy.Write([]byte(chunk)); err != nil {
			t.Fatalf("write chunk: %v", err)
		}
	}
	if err := proxy.finalize(); err != nil {
		t.Fatalf("finalize error: %v", err)
	}

	events := decodeSSEEvents(t, rec.Body.String())
	for _, event := range events {
		if event["type"] != "response.completed" {
			continue
		}
		response, _ := event["response"].(map[string]interface{})
		usage, _ := response["usage"].(map[string]interface{})
		if usage["input_tokens"] != float64(17) || usage["output_tokens"] != float64(9) || usage["total_tokens"] != float64(26) {
			t.Fatalf("unexpected merged usage: %#v", usage)
		}
		return
	}
	t.Fatal("expected response.completed event")
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
	if reasoningObj["summary"] != nil {
		t.Fatalf("response reasoning config must not duplicate generated summary: %#v", reasoningObj)
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

func TestResponsesStreamProxyWritesNamedSSEEvents(t *testing.T) {
	recorder := httptest.NewRecorder()
	proxy := newResponsesStreamProxy(recorder)

	if err := proxy.writeEvent(map[string]interface{}{
		"type": "response.created",
		"response": map[string]interface{}{
			"id":     "resp_named",
			"object": "response",
			"status": "in_progress",
		},
	}); err != nil {
		t.Fatalf("write event: %v", err)
	}

	body := recorder.Body.String()
	if !strings.HasPrefix(body, "event: response.created\ndata: ") {
		t.Fatalf("named SSE event missing: %q", body)
	}
	events := decodeSSEEvents(t, body)
	if len(events) != 1 || events[0]["type"] != "response.created" {
		t.Fatalf("decoded events = %#v", events)
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
		var dataLines []string
		for _, line := range strings.Split(frame, "\n") {
			line = strings.TrimSuffix(line, "\r")
			if strings.HasPrefix(line, "data:") {
				dataLines = append(dataLines, strings.TrimPrefix(strings.TrimPrefix(line, "data:"), " "))
			}
		}
		payload := strings.TrimSpace(strings.Join(dataLines, "\n"))
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
	prev := -1
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
