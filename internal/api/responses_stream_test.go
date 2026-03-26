package api

import (
	"encoding/json"
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
