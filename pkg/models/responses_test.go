package models

import (
	"strings"
	"testing"
)

func TestResponsesToUnifiedRequest_MapsTopLevelFunctionTool(t *testing.T) {
	req := &ResponsesRequest{
		Model: "lunargate/auto",
		Input: "hello",
		Tools: []ResponsesTool{
			{
				Type:        "function",
				Name:        "get_weather",
				Description: "Return weather",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"city": map[string]interface{}{"type": "string"},
					},
				},
			},
		},
		ToolChoice: map[string]interface{}{
			"type": "function",
			"name": "get_weather",
		},
	}

	unified, err := ResponsesToUnifiedRequest(req)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if len(unified.Tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(unified.Tools))
	}
	if unified.Tools[0].Function.Name != "get_weather" {
		t.Fatalf("expected tool function name %q, got %q", "get_weather", unified.Tools[0].Function.Name)
	}

	choiceObj, ok := unified.ToolChoice.(map[string]interface{})
	if !ok {
		t.Fatalf("expected normalized tool_choice map, got %T", unified.ToolChoice)
	}
	fnObj, ok := choiceObj["function"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected nested function object in tool_choice")
	}
	name, _ := fnObj["name"].(string)
	if name != "get_weather" {
		t.Fatalf("expected tool_choice function name %q, got %q", "get_weather", name)
	}
}

func TestResponsesToUnifiedRequest_RejectsFunctionToolWithoutName(t *testing.T) {
	req := &ResponsesRequest{
		Model: "lunargate/auto",
		Input: "hello",
		Tools: []ResponsesTool{{
			Type: "function",
		}},
	}

	_, err := ResponsesToUnifiedRequest(req)
	if err == nil {
		t.Fatalf("expected error for missing function name")
	}
}

func TestResponsesToUnifiedRequest_MapsWebSearchToolToFunctionCompat(t *testing.T) {
	req := &ResponsesRequest{
		Model: "lunargate/auto",
		Input: "hello",
		Tools: []ResponsesTool{
			{
				Type: "web_search",
			},
		},
		ToolChoice: map[string]interface{}{
			"type": "web_search",
		},
	}

	unified, err := ResponsesToUnifiedRequest(req)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if len(unified.Tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(unified.Tools))
	}
	if unified.Tools[0].Type != "function" {
		t.Fatalf("expected normalized tool type %q, got %q", "function", unified.Tools[0].Type)
	}
	if unified.Tools[0].Function.Name != "web_search" {
		t.Fatalf("expected normalized tool name %q, got %q", "web_search", unified.Tools[0].Function.Name)
	}
	schema, ok := unified.Tools[0].Function.Parameters.(map[string]interface{})
	if !ok {
		t.Fatalf("expected function schema map, got %T", unified.Tools[0].Function.Parameters)
	}
	typeName, _ := schema["type"].(string)
	if typeName != "object" {
		t.Fatalf("expected schema type object, got %q", typeName)
	}
	if _, ok := schema["properties"].(map[string]interface{}); !ok {
		t.Fatalf("expected object schema properties map to be present")
	}

	choiceObj, ok := unified.ToolChoice.(map[string]interface{})
	if !ok {
		t.Fatalf("expected normalized tool_choice map, got %T", unified.ToolChoice)
	}
	fnObj, ok := choiceObj["function"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected nested function object in tool_choice")
	}
	name, _ := fnObj["name"].(string)
	if name != "web_search" {
		t.Fatalf("expected normalized tool_choice function name %q, got %q", "web_search", name)
	}
}

func TestResponsesToUnifiedRequest_MapsFunctionCallOutputChain(t *testing.T) {
	req := &ResponsesRequest{
		Model:              "lunargate/auto",
		PreviousResponseID: "resp_prev_123",
		Input: []interface{}{
			map[string]interface{}{
				"type":      "function_call",
				"id":        "call_123",
				"name":      "exec_command",
				"arguments": "{\"cmd\":\"pwd\"}",
			},
			map[string]interface{}{
				"type":    "function_call_output",
				"call_id": "call_123",
				"output":  "/workspace",
			},
		},
	}

	unified, err := ResponsesToUnifiedRequest(req)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(unified.Messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(unified.Messages))
	}
	if unified.PreviousResponseID != "resp_prev_123" {
		t.Fatalf("expected previous_response_id to be preserved, got %q", unified.PreviousResponseID)
	}

	assistant := unified.Messages[0]
	if assistant.Role != "assistant" {
		t.Fatalf("expected first message role assistant, got %q", assistant.Role)
	}
	if len(assistant.ToolCalls) != 1 {
		t.Fatalf("expected one tool call, got %d", len(assistant.ToolCalls))
	}
	if assistant.ToolCalls[0].ID != "call_123" {
		t.Fatalf("expected tool call id %q, got %q", "call_123", assistant.ToolCalls[0].ID)
	}

	tool := unified.Messages[1]
	if tool.Role != "tool" {
		t.Fatalf("expected second message role tool, got %q", tool.Role)
	}
	if tool.ToolCallID != "call_123" {
		t.Fatalf("expected tool_call_id %q, got %q", "call_123", tool.ToolCallID)
	}
	if content, _ := tool.Content.(string); content != "/workspace" {
		t.Fatalf("expected tool content %q, got %q", "/workspace", content)
	}
}

func TestResponsesToUnifiedRequest_MapsToolCallIDField(t *testing.T) {
	req := &ResponsesRequest{
		Model: "lunargate/auto",
		Input: []interface{}{
			map[string]interface{}{
				"role":         "tool",
				"tool_call_id": "call_fNZA2mJYPmNxovT2TqmQDYkR",
				"content":      "ok",
			},
		},
	}

	unified, err := ResponsesToUnifiedRequest(req)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(unified.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(unified.Messages))
	}
	msg := unified.Messages[0]
	if msg.Role != "tool" {
		t.Fatalf("expected role %q, got %q", "tool", msg.Role)
	}
	if msg.ToolCallID != "call_fNZA2mJYPmNxovT2TqmQDYkR" {
		t.Fatalf("expected tool_call_id to be preserved")
	}
}

func TestResponsesToUnifiedRequest_MergesConsecutiveFunctionCallItems(t *testing.T) {
	req := &ResponsesRequest{
		Model: "lunargate/auto",
		Input: []interface{}{
			map[string]interface{}{
				"type":      "function_call",
				"id":        "call_1",
				"name":      "read_a",
				"arguments": "{}",
			},
			map[string]interface{}{
				"type":      "function_call",
				"id":        "call_2",
				"name":      "read_b",
				"arguments": "{}",
			},
			map[string]interface{}{
				"role":         "tool",
				"tool_call_id": "call_1",
				"content":      "A",
			},
			map[string]interface{}{
				"role":         "tool",
				"tool_call_id": "call_2",
				"content":      "B",
			},
		},
	}

	unified, err := ResponsesToUnifiedRequest(req)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(unified.Messages) != 3 {
		t.Fatalf("expected 3 messages (1 assistant+2 tool), got %d", len(unified.Messages))
	}
	assistant := unified.Messages[0]
	if assistant.Role != "assistant" {
		t.Fatalf("expected first message assistant, got %q", assistant.Role)
	}
	if len(assistant.ToolCalls) != 2 {
		t.Fatalf("expected assistant to contain 2 tool calls, got %d", len(assistant.ToolCalls))
	}
	if assistant.ToolCalls[0].ID != "call_1" || assistant.ToolCalls[1].ID != "call_2" {
		t.Fatalf("unexpected tool call ids order/content")
	}
	if unified.Messages[1].Role != "tool" || unified.Messages[1].ToolCallID != "call_1" {
		t.Fatalf("expected tool response for call_1")
	}
	if unified.Messages[2].Role != "tool" || unified.Messages[2].ToolCallID != "call_2" {
		t.Fatalf("expected tool response for call_2")
	}
}

func TestResponsesToUnifiedRequest_MapsReasoningEffort(t *testing.T) {
	req := &ResponsesRequest{
		Model: "lunargate/auto",
		Input: "hello",
		Reasoning: &Reasoning{
			Effort: "medium",
		},
	}

	unified, err := ResponsesToUnifiedRequest(req)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if unified.ReasoningEffort != "medium" {
		t.Fatalf("expected reasoning_effort=medium, got %q", unified.ReasoningEffort)
	}
}

func TestUnifiedResponseToResponses_SerializesNonStringContent(t *testing.T) {
	resp := &UnifiedResponse{
		ID:      "resp_1",
		Created: 123,
		Model:   "openai/gpt-5.4",
		Choices: []Choice{{
			Index: 0,
			Message: &Message{
				Role: "assistant",
				Content: []map[string]interface{}{
					{"type": "image", "url": "https://example.com/a.png"},
				},
			},
		}},
	}

	out := UnifiedResponseToResponses(resp)
	if out == nil {
		t.Fatalf("expected non-nil response")
	}
	if len(out.Output) != 1 {
		t.Fatalf("expected 1 output item, got %d", len(out.Output))
	}
	if len(out.Output[0].Content) != 1 {
		t.Fatalf("expected one output content part, got %d", len(out.Output[0].Content))
	}
	text := out.Output[0].Content[0].Text
	if !strings.Contains(text, "\"type\":\"image\"") {
		t.Fatalf("expected serialized non-string content in output text, got %q", text)
	}
	if !strings.Contains(out.OutputText, "\"url\":\"https://example.com/a.png\"") {
		t.Fatalf("expected output_text to include serialized content, got %q", out.OutputText)
	}
}

func TestUnifiedResponseToResponses_PreservesReasoningAsOutputItem(t *testing.T) {
	resp := &UnifiedResponse{
		ID:      "resp_reasoning_1",
		Created: 123,
		Model:   "openai/gpt-5.4",
		Choices: []Choice{{
			Index: 0,
			Message: &Message{
				Role:             "assistant",
				Content:          "final answer",
				ReasoningContent: "step 1 then step 2",
			},
		}},
	}

	out := UnifiedResponseToResponses(resp)
	if out == nil {
		t.Fatalf("expected non-nil response")
	}
	if len(out.Output) != 2 {
		t.Fatalf("expected message + reasoning output items, got %d", len(out.Output))
	}

	var reasoningItem *ResponsesOutput
	for i := range out.Output {
		if out.Output[i].Type == "reasoning" {
			reasoningItem = &out.Output[i]
			break
		}
	}
	if reasoningItem == nil {
		t.Fatalf("expected reasoning output item")
	}
	if reasoningItem.Status != "completed" {
		t.Fatalf("expected completed reasoning item status, got %q", reasoningItem.Status)
	}
	if len(reasoningItem.Summary) != 1 {
		t.Fatalf("expected one reasoning summary part, got %d", len(reasoningItem.Summary))
	}
	if reasoningItem.Summary[0].Type != "summary_text" {
		t.Fatalf("expected summary_text type, got %q", reasoningItem.Summary[0].Type)
	}
	if reasoningItem.Summary[0].Text != "step 1 then step 2" {
		t.Fatalf("expected preserved reasoning text, got %q", reasoningItem.Summary[0].Text)
	}
}
