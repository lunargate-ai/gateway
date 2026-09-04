package api

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"testing"
	"time"
)

func TestPrepareConversationItemsRejectsInvalidKnownItemShapes(t *testing.T) {
	tests := []struct {
		name  string
		raw   string
		param string
		code  string
	}{
		{name: "explicit null type", raw: `{"type":null,"role":"user","content":"hello"}`, param: "items[0].type", code: "invalid_value"},
		{name: "explicit numeric type", raw: `{"type":7,"role":"user","content":"hello"}`, param: "items[0].type", code: "invalid_value"},
		{name: "padded type", raw: `{"type":" message ","role":"user","content":"hello"}`, param: "items[0].type", code: "invalid_value"},
		{name: "non canonical known type", raw: `{"type":"MESSAGE","role":"user","content":"hello"}`, param: "items[0].type", code: "invalid_value"},
		{name: "message missing role", raw: `{"type":"message","content":"hello"}`, param: "items[0].role", code: "invalid_conversation_item"},
		{name: "inferred message numeric role", raw: `{"role":7,"content":"hello"}`, param: "items[0].role", code: "invalid_value"},
		{name: "message unknown role", raw: `{"type":"message","role":"critic","content":"hello"}`, param: "items[0].role", code: "invalid_value"},
		{name: "message padded id", raw: `{"type":"message","id":" msg_1 ","role":"user","content":"hello"}`, param: "items[0].id", code: "invalid_value"},
		{name: "message missing content", raw: `{"type":"message","role":"user"}`, param: "items[0].content", code: "invalid_conversation_item"},
		{name: "message null content", raw: `{"type":"message","role":"user","content":null}`, param: "items[0].content", code: "invalid_value"},
		{name: "message numeric content", raw: `{"type":"message","role":"user","content":7}`, param: "items[0].content", code: "invalid_value"},
		{name: "message scalar content part", raw: `{"type":"message","role":"user","content":["hello"]}`, param: "items[0].content[0]", code: "invalid_value"},
		{name: "message content part missing type", raw: `{"type":"message","role":"user","content":[{"text":"hello"}]}`, param: "items[0].content[0].type", code: "invalid_conversation_item"},
		{name: "message input text missing text", raw: `{"type":"message","role":"user","content":[{"type":"input_text"}]}`, param: "items[0].content[0].text", code: "invalid_conversation_item"},
		{name: "message output text non string text", raw: `{"type":"message","role":"assistant","content":[{"type":"output_text","text":7}]}`, param: "items[0].content[0].text", code: "invalid_value"},
		{name: "message refusal missing refusal", raw: `{"type":"message","role":"assistant","content":[{"type":"refusal"}]}`, param: "items[0].content[0].refusal", code: "invalid_conversation_item"},
		{name: "message unknown status", raw: `{"type":"message","role":"assistant","content":[],"status":"paused"}`, param: "items[0].status", code: "invalid_value"},
		{name: "message numeric status", raw: `{"type":"message","role":"assistant","content":[],"status":1}`, param: "items[0].status", code: "invalid_value"},
		{name: "function call missing call id", raw: `{"type":"function_call","name":"lookup","arguments":"{}"}`, param: "items[0].call_id", code: "invalid_conversation_item"},
		{name: "function call blank name", raw: `{"type":"function_call","call_id":"call_1","name":"  ","arguments":"{}"}`, param: "items[0].name", code: "invalid_value"},
		{name: "function call missing arguments", raw: `{"type":"function_call","call_id":"call_1","name":"lookup"}`, param: "items[0].arguments", code: "invalid_conversation_item"},
		{name: "function call object arguments", raw: `{"type":"function_call","call_id":"call_1","name":"lookup","arguments":{}}`, param: "items[0].arguments", code: "invalid_value"},
		{name: "function output missing call id", raw: `{"type":"function_call_output","output":"done"}`, param: "items[0].call_id", code: "invalid_conversation_item"},
		{name: "function output numeric call id", raw: `{"type":"function_call_output","call_id":7,"output":"done"}`, param: "items[0].call_id", code: "invalid_value"},
		{name: "function output missing output", raw: `{"type":"function_call_output","call_id":"call_1"}`, param: "items[0].output", code: "invalid_conversation_item"},
		{name: "function output boolean output", raw: `{"type":"function_call_output","call_id":"call_1","output":true}`, param: "items[0].output", code: "invalid_value"},
		{name: "function output malformed text part", raw: `{"type":"function_call_output","call_id":"call_1","output":[{"type":"input_text"}]}`, param: "items[0].output[0].text", code: "invalid_conversation_item"},
		{name: "reasoning missing id", raw: `{"type":"reasoning","summary":[]}`, param: "items[0].id", code: "invalid_conversation_item"},
		{name: "reasoning padded id", raw: `{"type":"reasoning","id":" rs_1 ","summary":[]}`, param: "items[0].id", code: "invalid_value"},
		{name: "reasoning missing summary", raw: `{"type":"reasoning","id":"rs_1"}`, param: "items[0].summary", code: "invalid_conversation_item"},
		{name: "reasoning string summary", raw: `{"type":"reasoning","id":"rs_1","summary":"summary"}`, param: "items[0].summary", code: "invalid_value"},
		{name: "reasoning scalar summary part", raw: `{"type":"reasoning","id":"rs_1","summary":[7]}`, param: "items[0].summary[0]", code: "invalid_value"},
		{name: "reasoning summary text missing text", raw: `{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text"}]}`, param: "items[0].summary[0].text", code: "invalid_conversation_item"},
		{name: "reasoning malformed optional content", raw: `{"type":"reasoning","id":"rs_1","summary":[],"content":[{"type":"reasoning_text","text":false}]}`, param: "items[0].content[0].text", code: "invalid_value"},
	}

	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			items, err := prepareConversationItems([]json.RawMessage{json.RawMessage(testCase.raw)})
			if err == nil {
				t.Fatalf("prepareConversationItems accepted invalid item: %s", testCase.raw)
			}
			if items != nil {
				t.Fatalf("items = %#v, want nil on validation failure", items)
			}
			var inputErr *conversationItemInputError
			if !errors.As(err, &inputErr) {
				t.Fatalf("error = %T %v, want conversationItemInputError", err, err)
			}
			if inputErr.param != testCase.param || inputErr.code != testCase.code {
				t.Fatalf("error = %#v, want param=%q code=%q", inputErr, testCase.param, testCase.code)
			}
		})
	}
}

func TestPrepareConversationItemsAcceptsKnownItemsAndPreservesSuppliedFields(t *testing.T) {
	rawItems := []json.RawMessage{
		json.RawMessage(`{"role":"user","content":[{"type":"input_text","text":"  hello\n","prompt_cache_breakpoint":{"mode":"explicit"},"future_part":{"large_integer":9007199254740993}}],"phase":"final_answer","future_item":{"keep":true}}`),
		json.RawMessage(`{"type":"message","id":"msg_existing","status":"incomplete","role":"assistant","content":[{"type":"output_text","text":"","annotations":[],"future_text":true},{"type":"refusal","refusal":"","future_refusal":true}],"phase":"commentary","future_message":7}`),
		json.RawMessage(`{"type":"function_call","call_id":"call_1","name":"lookup","arguments":"","caller":{"type":"direct"},"future_call":{"keep":true}}`),
		json.RawMessage(`{"type":"function_call_output","call_id":"call_1","output":[{"type":"input_text","text":"done","future_output_part":true}],"name":"lookup","namespace":"tools","future_output":{"keep":true}}`),
		json.RawMessage(`{"type":"reasoning","id":"rs_stable","summary":[{"type":"summary_text","text":"","future_summary":true}],"content":[{"type":"reasoning_text","text":"thought","future_reasoning":true}],"encrypted_content":"opaque","future_top":true}`),
	}

	items, err := prepareConversationItems(rawItems)
	if err != nil {
		t.Fatalf("prepareConversationItems: %v", err)
	}
	if len(items) != len(rawItems) {
		t.Fatalf("items = %d, want %d", len(items), len(rawItems))
	}

	for index, raw := range rawItems {
		var supplied map[string]json.RawMessage
		if err := json.Unmarshal(raw, &supplied); err != nil {
			t.Fatal(err)
		}
		for field, want := range supplied {
			if got := items[index][field]; !bytes.Equal(got, want) {
				t.Fatalf("items[%d].%s changed: got=%s want=%s", index, field, got, want)
			}
		}
	}

	if parseJSONStringRaw(items[0]["type"]) != "message" ||
		!strings.HasPrefix(conversationItemID(items[0]), "msg_") ||
		parseJSONStringRaw(items[0]["status"]) != "completed" {
		t.Fatalf("inferred message defaults = %s", mustMarshalForTest(t, items[0]))
	}
	if conversationItemID(items[1]) != "msg_existing" || parseJSONStringRaw(items[1]["status"]) != "incomplete" {
		t.Fatalf("supplied message lifecycle changed: %s", mustMarshalForTest(t, items[1]))
	}
	if !strings.HasPrefix(conversationItemID(items[2]), "fc_") || parseJSONStringRaw(items[2]["status"]) != "completed" {
		t.Fatalf("function call defaults = %s", mustMarshalForTest(t, items[2]))
	}
	if !strings.HasPrefix(conversationItemID(items[3]), "fc_") || parseJSONStringRaw(items[3]["status"]) != "completed" {
		t.Fatalf("function output defaults = %s", mustMarshalForTest(t, items[3]))
	}
	if conversationItemID(items[4]) != "rs_stable" || parseJSONStringRaw(items[4]["status"]) != "completed" {
		t.Fatalf("reasoning defaults = %s", mustMarshalForTest(t, items[4]))
	}
}

func TestPrepareConversationItemsPreservesOpaqueFutureItems(t *testing.T) {
	rawItems := []json.RawMessage{
		json.RawMessage(`{"type":"future_modal_item","payload":{"large_integer":9007199254740993,"nested":[true,"kept"]},"future_flag":true}`),
		json.RawMessage(`{"type":"future_hosted_output","id":"item_future_stable","status":{"state":"queued"},"payload":null}`),
	}
	items, err := prepareConversationItems(rawItems)
	if err != nil {
		t.Fatalf("prepareConversationItems: %v", err)
	}
	if !strings.HasPrefix(conversationItemID(items[0]), "item_") {
		t.Fatalf("generated future item ID = %q", conversationItemID(items[0]))
	}
	if _, hasStatus := items[0]["status"]; hasStatus {
		t.Fatalf("opaque future item received a synthesized status: %s", mustMarshalForTest(t, items[0]))
	}
	if conversationItemID(items[1]) != "item_future_stable" {
		t.Fatalf("future item ID changed: %s", mustMarshalForTest(t, items[1]))
	}

	for index, raw := range rawItems {
		var supplied map[string]json.RawMessage
		if err := json.Unmarshal(raw, &supplied); err != nil {
			t.Fatal(err)
		}
		for field, want := range supplied {
			if got := items[index][field]; !bytes.Equal(got, want) {
				t.Fatalf("items[%d].%s changed: got=%s want=%s", index, field, got, want)
			}
		}
	}
}

func TestLocalConversationOpaqueFutureItemRoundTrips(t *testing.T) {
	handler := &Handler{conversationsState: newConversationStateStore(time.Hour)}
	router := conversationTestRouter(handler)
	created := performConversationRequest(t, router, http.MethodPost, "/v1/conversations", `{
		"items":[{
			"type":"future_modal_item",
			"payload":{"large_integer":9007199254740993,"nested":[true,"kept"]},
			"future_flag":true
		}]
	}`)
	if created.Code != http.StatusOK {
		t.Fatalf("create status = %d, body = %s", created.Code, created.Body.String())
	}
	var conversation conversationObject
	decodeConversationResponse(t, created, &conversation)

	listed := performConversationRequest(t, router, http.MethodGet, "/v1/conversations/"+conversation.ID+"/items?order=asc", "")
	if listed.Code != http.StatusOK {
		t.Fatalf("list status = %d, body = %s", listed.Code, listed.Body.String())
	}
	var result conversationItemList
	decodeConversationResponse(t, listed, &result)
	if len(result.Data) != 1 {
		t.Fatalf("items = %d, want 1", len(result.Data))
	}
	item := result.Data[0]
	if parseJSONStringRaw(item["type"]) != "future_modal_item" || !strings.HasPrefix(conversationItemID(item), "item_") {
		t.Fatalf("future item identity changed: %s", mustMarshalForTest(t, item))
	}
	if _, hasStatus := item["status"]; hasStatus {
		t.Fatalf("future item status was synthesized: %s", mustMarshalForTest(t, item))
	}
	if !bytes.Contains(item["payload"], []byte("9007199254740993")) || string(item["future_flag"]) != "true" {
		t.Fatalf("future item fields changed: %s", mustMarshalForTest(t, item))
	}
}

func TestResponsesConversationItemsSatisfyLocalKnownItemValidation(t *testing.T) {
	items, err := responsesConversationItems(
		json.RawMessage(`[
			{"role":"user","content":[{"type":"input_text","text":"question"}]},
			{"type":"function_call_output","call_id":"call_1","output":"result"}
		]`),
		map[string]interface{}{
			"output": []interface{}{
				map[string]interface{}{
					"type": "message", "id": "msg_1", "status": "completed", "role": "assistant",
					"content": []interface{}{map[string]interface{}{"type": "output_text", "text": "answer"}},
				},
				map[string]interface{}{
					"type": "reasoning", "id": "rs_1", "status": "completed",
					"summary": []interface{}{map[string]interface{}{"type": "summary_text", "text": "summary"}},
				},
				map[string]interface{}{
					"type": "function_call", "id": "fc_1", "status": "completed",
					"call_id": "call_2", "name": "lookup", "arguments": "{}",
				},
			},
		},
	)
	if err != nil {
		t.Fatalf("responsesConversationItems: %v", err)
	}
	if len(items) != 5 {
		t.Fatalf("conversation items = %d, want 5", len(items))
	}
	for index, item := range items {
		if conversationItemID(item) == "" {
			t.Fatalf("items[%d] has no ID: %s", index, mustMarshalForTest(t, item))
		}
	}
}

func TestLocalConversationKnownItemValidationIsAtomic(t *testing.T) {
	handler := &Handler{conversationsState: newConversationStateStore(time.Hour)}
	router := conversationTestRouter(handler)
	created := performConversationRequest(t, router, http.MethodPost, "/v1/conversations", `{
		"items":[{"role":"user","content":"existing"}]
	}`)
	if created.Code != http.StatusOK {
		t.Fatalf("create status = %d, body = %s", created.Code, created.Body.String())
	}
	var conversation conversationObject
	decodeConversationResponse(t, created, &conversation)

	invalidAppend := performConversationRequest(t, router, http.MethodPost, "/v1/conversations/"+conversation.ID+"/items", `{
		"items":[
			{"type":"message","role":"user","content":"would otherwise be valid"},
			{"type":"function_call","call_id":"call_1","arguments":"{}"}
		]
	}`)
	assertConversationError(t, invalidAppend, http.StatusBadRequest, "items[1].name", "invalid_conversation_item")

	stored, ok := handler.conversationsState.getItems(conversation.ID)
	if !ok || len(stored) != 1 || parseJSONStringRaw(stored[0]["content"]) != "existing" {
		t.Fatalf("invalid batch mutated local conversation: ok=%t items=%s", ok, mustMarshalForTest(t, stored))
	}
}
