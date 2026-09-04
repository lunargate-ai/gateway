package providers

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestOpenAITranslator_ChatToResponsesRejectsUnmappedControls(t *testing.T) {
	one := 1
	zero := 0.0
	seed := 42
	tests := []struct {
		name      string
		request   models.UnifiedRequest
		wantField string
	}{
		{name: "n", request: models.UnifiedRequest{N: &one}, wantField: "n"},
		{name: "stop", request: models.UnifiedRequest{Stop: "END"}, wantField: "stop"},
		{name: "frequency penalty", request: models.UnifiedRequest{FrequencyPenalty: &zero}, wantField: "frequency_penalty"},
		{name: "presence penalty", request: models.UnifiedRequest{PresencePenalty: &zero}, wantField: "presence_penalty"},
		{name: "seed", request: models.UnifiedRequest{Seed: &seed}, wantField: "seed"},
		{name: "response format", request: models.UnifiedRequest{ResponseFormat: &models.ResponseFormat{Type: "json_object"}}, wantField: "response_format"},
		{name: "logit bias", request: models.UnifiedRequest{LogitBias: map[string]int{"42": 1}}, wantField: "logit_bias"},
		{name: "both token limits", request: models.UnifiedRequest{RawJSON: json.RawMessage(`{"model":"gpt-5.4","messages":[],"max_tokens":64,"max_completion_tokens":128}`)}, wantField: "max_completion_tokens"},
		{name: "unknown control", request: models.UnifiedRequest{RawJSON: json.RawMessage(`{"model":"gpt-5.4","messages":[],"future_chat_control":true}`)}, wantField: "future_chat_control"},
	}

	translator := NewOpenAITranslator(config.ProviderConfig{APIKey: "dummy"})
	ctx := WithUpstreamRequestType(context.Background(), "responses")
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.request.SourceRequestType = "chat_completions"
			tt.request.Model = "gpt-5.4"
			_, err := translator.TranslateRequest(ctx, &tt.request)
			var compatibilityErr *models.CompatibilityError
			if !errors.As(err, &compatibilityErr) {
				t.Fatalf("error = %v, want CompatibilityError", err)
			}
			if compatibilityErr.Field != tt.wantField || compatibilityErr.Provider != "openai" {
				t.Fatalf("compatibility error = %#v, want field=%q provider=openai", compatibilityErr, tt.wantField)
			}
		})
	}
}

func TestOpenAITranslator_ChatToResponsesMapsFaithfulControls(t *testing.T) {
	temperature := 0.2
	topP := 0.8
	maxTokens := 128
	store := false
	strict := true
	translator := NewOpenAITranslator(config.ProviderConfig{
		APIKey:  "dummy",
		BaseURL: "https://api.openai.com/v1",
	})
	ctx := WithUpstreamRequestType(context.Background(), "responses")
	request, err := translator.TranslateRequest(ctx, &models.UnifiedRequest{
		SourceRequestType: "chat_completions",
		Model:             "gpt-5.4",
		Messages:          []models.Message{{Role: "user", Content: "hello"}},
		Temperature:       &temperature,
		TopP:              &topP,
		MaxTokens:         &maxTokens,
		Store:             &store,
		User:              "customer-123",
		ReasoningEffort:   "high",
		Tools: []models.Tool{{
			Type: "function",
			Function: models.ToolFunction{
				Name:       "lookup",
				Parameters: map[string]interface{}{"type": "object"},
				Strict:     &strict,
			},
		}},
		ToolChoice: "auto",
	})
	if err != nil {
		t.Fatalf("TranslateRequest returned error: %v", err)
	}
	body, err := io.ReadAll(request.Body)
	if err != nil {
		t.Fatalf("read request body: %v", err)
	}
	var payload map[string]interface{}
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("decode request body: %v", err)
	}
	if payload["temperature"] != temperature || payload["top_p"] != topP || payload["max_output_tokens"] != float64(maxTokens) {
		t.Fatalf("sampling/output controls = %#v", payload)
	}
	if payload["store"] != false || payload["user"] != "customer-123" || payload["tool_choice"] != "auto" {
		t.Fatalf("mapped controls = %#v", payload)
	}
	reasoning, _ := payload["reasoning"].(map[string]interface{})
	if reasoning["effort"] != "high" {
		t.Fatalf("reasoning = %#v", payload["reasoning"])
	}
	tools, _ := payload["tools"].([]interface{})
	if len(tools) != 1 || tools[0].(map[string]interface{})["name"] != "lookup" || tools[0].(map[string]interface{})["strict"] != true {
		t.Fatalf("tools = %#v", payload["tools"])
	}
}

func TestOpenAITranslator_ChatToResponsesAlwaysDisablesHiddenStorage(t *testing.T) {
	falseValue := false
	tests := []struct {
		name   string
		stream bool
		store  *bool
		raw    json.RawMessage
	}{
		{name: "non-stream absent"},
		{name: "stream absent", stream: true},
		{name: "non-stream false", store: &falseValue, raw: json.RawMessage(`{"model":"gpt-5.4","messages":[{"role":"user","content":"hello"}],"store":false}`)},
		{name: "stream false", stream: true, store: &falseValue, raw: json.RawMessage(`{"model":"gpt-5.4","messages":[{"role":"user","content":"hello"}],"stream":true,"store":false}`)},
		{name: "explicit null", raw: json.RawMessage(`{"model":"gpt-5.4","messages":[{"role":"user","content":"hello"}],"store":null}`)},
	}

	translator := NewOpenAITranslator(config.ProviderConfig{APIKey: "dummy"})
	ctx := WithUpstreamRequestType(context.Background(), "responses")
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request, err := translator.TranslateRequest(ctx, &models.UnifiedRequest{
				RawJSON:           test.raw,
				SourceRequestType: "chat_completions",
				Model:             "gpt-5.4",
				Messages:          []models.Message{{Role: "user", Content: "hello"}},
				Stream:            test.stream,
				Store:             test.store,
			})
			if err != nil {
				t.Fatalf("TranslateRequest: %v", err)
			}
			body, err := io.ReadAll(request.Body)
			if err != nil {
				t.Fatalf("read request body: %v", err)
			}
			var payload map[string]interface{}
			if err := json.Unmarshal(body, &payload); err != nil {
				t.Fatalf("decode request body: %v", err)
			}
			if store, present := payload["store"].(bool); !present || store {
				t.Fatalf("upstream store = %#v, want explicit false", payload["store"])
			}
		})
	}
}

func TestOpenAITranslator_ChatToResponsesRejectsHiddenStorage(t *testing.T) {
	trueValue := true
	tests := []struct {
		name  string
		store *bool
		raw   json.RawMessage
	}{
		{name: "typed true", store: &trueValue},
		{name: "raw true", raw: json.RawMessage(`{"model":"gpt-5.4","messages":[],"store":true}`)},
		{name: "raw invalid", raw: json.RawMessage(`{"model":"gpt-5.4","messages":[],"store":"yes"}`)},
	}

	translator := NewOpenAITranslator(config.ProviderConfig{APIKey: "dummy"})
	ctx := WithUpstreamRequestType(context.Background(), "responses")
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request, err := translator.TranslateRequest(ctx, &models.UnifiedRequest{
				RawJSON:           test.raw,
				SourceRequestType: "chat_completions",
				Model:             "gpt-5.4",
				Messages:          []models.Message{{Role: "user", Content: "hello"}},
				Store:             test.store,
			})
			if request != nil {
				t.Fatalf("request = %#v, want no upstream request", request)
			}
			var compatibilityErr *models.CompatibilityError
			if !errors.As(err, &compatibilityErr) || compatibilityErr.Field != "store" {
				t.Fatalf("error = %#v, want store CompatibilityError", err)
			}
		})
	}
}

func TestOpenAITranslator_ChatToResponsesRejectsLossyMessages(t *testing.T) {
	zero := 0
	tests := []struct {
		name      string
		raw       string
		messages  []models.Message
		wantField string
	}{
		{
			name:      "unknown message field",
			raw:       `{"model":"gpt-5.4","messages":[{"role":"user","content":"hello","vendor_hint":true}]}`,
			messages:  []models.Message{{Role: "user", Content: "hello"}},
			wantField: "messages[0].vendor_hint",
		},
		{
			name:      "audio content",
			raw:       `{"model":"gpt-5.4","messages":[{"role":"user","content":[{"type":"input_audio","input_audio":{"data":"abc","format":"wav"}}]}]}`,
			messages:  []models.Message{{Role: "user", Content: []interface{}{map[string]interface{}{"type": "input_audio", "input_audio": map[string]interface{}{"data": "abc", "format": "wav"}}}}},
			wantField: "messages[0].content[0].type",
		},
		{
			name:      "file content",
			raw:       `{"model":"gpt-5.4","messages":[{"role":"user","content":[{"type":"file","file":{"file_id":"file_1"}}]}]}`,
			messages:  []models.Message{{Role: "user", Content: []interface{}{map[string]interface{}{"type": "file", "file": map[string]interface{}{"file_id": "file_1"}}}}},
			wantField: "messages[0].content[0].type",
		},
		{
			name:      "assistant refusal",
			raw:       `{"model":"gpt-5.4","messages":[{"role":"assistant","content":null,"refusal":"no"}]}`,
			messages:  []models.Message{{Role: "assistant", Refusal: "no"}},
			wantField: "messages[0].refusal",
		},
		{
			name:      "user name",
			raw:       `{"model":"gpt-5.4","messages":[{"role":"user","content":"hello","name":"alice"}]}`,
			messages:  []models.Message{{Role: "user", Content: "hello", Name: "alice"}},
			wantField: "messages[0].name",
		},
		{
			name:      "assistant reasoning",
			raw:       `{"model":"gpt-5.4","messages":[{"role":"assistant","content":"answer","reasoning_content":"private"}]}`,
			messages:  []models.Message{{Role: "assistant", Content: "answer", ReasoningContent: "private"}},
			wantField: "messages[0].reasoning_content",
		},
		{
			name:      "tool name",
			raw:       `{"model":"gpt-5.4","messages":[{"role":"tool","name":"lookup","tool_call_id":"call_1","content":"ok"}]}`,
			messages:  []models.Message{{Role: "tool", Name: "lookup", ToolCallID: "call_1", Content: "ok"}},
			wantField: "messages[0].name",
		},
		{
			name:      "legacy function call",
			raw:       `{"model":"gpt-5.4","messages":[{"role":"assistant","content":null,"function_call":{"name":"lookup","arguments":"{}"}}]}`,
			messages:  []models.Message{{Role: "assistant", FunctionCall: &models.FunctionCall{Name: "lookup", Arguments: "{}"}}},
			wantField: "messages[0].function_call",
		},
		{
			name: "stream tool call index",
			raw:  `{"model":"gpt-5.4","messages":[{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}"}}]}]}`,
			messages: []models.Message{{Role: "assistant", ToolCalls: []models.ToolCall{{
				Index: &zero, ID: "call_1", Type: "function", Function: models.ToolCallFunction{Name: "lookup", Arguments: "{}"},
			}}}},
			wantField: "messages[0].tool_calls[0].index",
		},
		{
			name:      "unsupported tool call type",
			raw:       `{"model":"gpt-5.4","messages":[{"role":"assistant","tool_calls":[{"id":"call_1","type":"custom","function":{"name":"lookup","arguments":"{}"}}]}]}`,
			messages:  []models.Message{{Role: "assistant", ToolCalls: []models.ToolCall{{ID: "call_1", Type: "custom", Function: models.ToolCallFunction{Name: "lookup", Arguments: "{}"}}}}},
			wantField: "messages[0].tool_calls[0].type",
		},
		{
			name:      "missing tool result call id",
			raw:       `{"model":"gpt-5.4","messages":[{"role":"tool","content":"ok"}]}`,
			messages:  []models.Message{{Role: "tool", Content: "ok"}},
			wantField: "messages[0].tool_call_id",
		},
		{
			name:      "responses text part in chat input",
			raw:       `{"model":"gpt-5.4","messages":[{"role":"user","content":[{"type":"input_text","text":"hello"}]}]}`,
			messages:  []models.Message{{Role: "user", Content: []interface{}{map[string]interface{}{"type": "input_text", "text": "hello"}}}},
			wantField: "messages[0].content[0].type",
		},
		{
			name:      "assistant image",
			raw:       `{"model":"gpt-5.4","messages":[{"role":"assistant","content":[{"type":"image_url","image_url":{"url":"https://example.com/image.png"}}]}]}`,
			messages:  []models.Message{{Role: "assistant", Content: []interface{}{map[string]interface{}{"type": "image_url", "image_url": map[string]interface{}{"url": "https://example.com/image.png"}}}}},
			wantField: "messages[0].content[0].type",
		},
		{
			name:      "empty content part list",
			raw:       `{"model":"gpt-5.4","messages":[{"role":"user","content":[]}]}`,
			messages:  []models.Message{{Role: "user", Content: []interface{}{}}},
			wantField: "messages[0].content",
		},
	}

	translator := NewOpenAITranslator(config.ProviderConfig{APIKey: "dummy"})
	ctx := WithUpstreamRequestType(context.Background(), "responses")
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request, err := translator.TranslateRequest(ctx, &models.UnifiedRequest{
				RawJSON:           json.RawMessage(test.raw),
				SourceRequestType: "chat_completions",
				Model:             "gpt-5.4",
				Messages:          test.messages,
			})
			if request != nil {
				t.Fatalf("request = %#v, want no upstream request", request)
			}
			var compatibilityErr *models.CompatibilityError
			if !errors.As(err, &compatibilityErr) || compatibilityErr.Field != test.wantField {
				t.Fatalf("error = %#v, want field %q", err, test.wantField)
			}
		})
	}
}

func TestOpenAITranslator_ChatToResponsesPreservesTextImageAndToolHistory(t *testing.T) {
	imageURL := "data:image/png;base64,aQ=="
	messages := []models.Message{
		{
			Role: "user",
			Content: []interface{}{
				map[string]interface{}{"type": "text", "text": "look"},
				map[string]interface{}{"type": "image_url", "image_url": map[string]interface{}{"url": imageURL, "detail": "high"}},
			},
		},
		{
			Role: "assistant",
			ToolCalls: []models.ToolCall{{
				ID:   "call_1",
				Type: "function",
				Function: models.ToolCallFunction{
					Name:      "inspect",
					Arguments: `{"id":1}`,
				},
			}},
		},
		{Role: "tool", ToolCallID: "call_1", Content: `{"ok":true}`},
	}
	raw := json.RawMessage(`{
		"model":"gpt-5.4",
		"messages":[
			{"role":"user","content":[{"type":"text","text":"look"},{"type":"image_url","image_url":{"url":"` + imageURL + `","detail":"high"}}]},
			{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"inspect","arguments":"{\"id\":1}"}}]},
			{"role":"tool","tool_call_id":"call_1","content":"{\"ok\":true}"}
		]
	}`)

	translator := NewOpenAITranslator(config.ProviderConfig{APIKey: "dummy"})
	request, err := translator.TranslateRequest(WithUpstreamRequestType(context.Background(), "responses"), &models.UnifiedRequest{
		RawJSON:           raw,
		SourceRequestType: "chat_completions",
		Model:             "gpt-5.4",
		Messages:          messages,
	})
	if err != nil {
		t.Fatalf("TranslateRequest: %v", err)
	}
	body, err := io.ReadAll(request.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	var payload map[string]interface{}
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("decode body: %v", err)
	}
	input, _ := payload["input"].([]interface{})
	if len(input) != 3 {
		t.Fatalf("input = %#v, want message, function call, and function output", input)
	}
	userMessage, _ := input[0].(map[string]interface{})
	content, _ := userMessage["content"].([]interface{})
	if len(content) != 2 {
		t.Fatalf("user content = %#v", userMessage["content"])
	}
	image, _ := content[1].(map[string]interface{})
	if image["type"] != "input_image" || image["image_url"] != imageURL || image["detail"] != "high" {
		t.Fatalf("mapped image = %#v", image)
	}
	functionCall, _ := input[1].(map[string]interface{})
	if functionCall["type"] != "function_call" || functionCall["call_id"] != "call_1" || functionCall["name"] != "inspect" {
		t.Fatalf("function call = %#v", functionCall)
	}
	functionOutput, _ := input[2].(map[string]interface{})
	if functionOutput["type"] != "function_call_output" || functionOutput["call_id"] != "call_1" || functionOutput["output"] != `{"ok":true}` {
		t.Fatalf("function output = %#v", functionOutput)
	}
}

func TestOpenAITranslator_ChatToResponsesRejectsInvalidToolHistory(t *testing.T) {
	tests := []struct {
		name      string
		raw       string
		wantField string
	}{
		{
			name: "tool output before call",
			raw: `{
				"model":"gpt-5.4",
				"messages":[
					{"role":"tool","tool_call_id":"call_1","content":"early"},
					{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}"}}]}
				]
			}`,
			wantField: "messages[0].tool_call_id",
		},
		{
			name: "unknown call id",
			raw: `{
				"model":"gpt-5.4",
				"messages":[
					{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}"}}]},
					{"role":"tool","tool_call_id":"call_missing","content":"wrong"}
				]
			}`,
			wantField: "messages[1].tool_call_id",
		},
		{
			name: "duplicate tool output",
			raw: `{
				"model":"gpt-5.4",
				"messages":[
					{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}"}}]},
					{"role":"tool","tool_call_id":"call_1","content":"first"},
					{"role":"tool","tool_call_id":"call_1","content":"second"}
				]
			}`,
			wantField: "messages[2].tool_call_id",
		},
		{
			name: "globally duplicate call id after output",
			raw: `{
				"model":"gpt-5.4",
				"messages":[
					{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"first","arguments":"{}"}}]},
					{"role":"tool","tool_call_id":"call_1","content":"done"},
					{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"second","arguments":"{}"}}]}
				]
			}`,
			wantField: "messages[2].tool_calls[0].id",
		},
	}

	translator := NewOpenAITranslator(config.ProviderConfig{APIKey: "dummy"})
	ctx := WithUpstreamRequestType(context.Background(), "responses")
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request := decodeOpenAIChatToResponsesTestRequest(t, test.raw)
			upstreamRequest, err := translator.TranslateRequest(ctx, request)
			if upstreamRequest != nil {
				t.Fatalf("request = %#v, want no upstream request", upstreamRequest)
			}
			var compatibilityErr *models.CompatibilityError
			if !errors.As(err, &compatibilityErr) || compatibilityErr.Field != test.wantField {
				t.Fatalf("error = %#v, want field %q", err, test.wantField)
			}
		})
	}
}

func TestOpenAITranslator_ChatToResponsesPreservesMultiToolInterleavedHistory(t *testing.T) {
	request := decodeOpenAIChatToResponsesTestRequest(t, `{
		"model":"gpt-5.4",
		"messages":[
			{"role":"user","content":"inspect both"},
			{"role":"assistant","content":"I will inspect both.","tool_calls":[
				{"id":"call_x","type":"function","function":{"name":"first","arguments":"{\"id\":1}"}},
				{"id":"fc_x","type":"function","function":{"name":"second","arguments":"{\"id\":2}"}}
			]},
			{"role":"system","content":"Keep the original tool order."},
			{"role":"tool","tool_call_id":"fc_x","content":"second result"},
			{"role":"developer","content":"Continue after both results."},
			{"role":"tool","tool_call_id":"call_x","content":"first result"},
			{"role":"assistant","tool_calls":[
				{"id":"call_pending","type":"function","function":{"name":"finalize","arguments":"{}"}}
			]}
		]
	}`)
	// Match the real Chat handler path, which adds internal streaming indexes
	// while retaining the original request JSON for exact compatibility checks.
	if err := models.NormalizeUnifiedRequest(request); err != nil {
		t.Fatalf("NormalizeUnifiedRequest: %v", err)
	}

	translator := NewOpenAITranslator(config.ProviderConfig{APIKey: "dummy"})
	upstreamRequest, err := translator.TranslateRequest(
		WithUpstreamRequestType(context.Background(), "responses"),
		request,
	)
	if err != nil {
		t.Fatalf("TranslateRequest: %v", err)
	}
	body, err := io.ReadAll(upstreamRequest.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	var payload map[string]interface{}
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("decode body: %v", err)
	}
	input, _ := payload["input"].([]interface{})
	if len(input) != 9 {
		t.Fatalf("input = %#v, want nine ordered items", input)
	}

	wantTypes := []string{"message", "function_call", "function_call", "message", "message", "function_call_output", "message", "function_call_output", "function_call"}
	wantCallIDs := map[int]string{1: "call_x", 2: "fc_x", 5: "fc_x", 7: "call_x", 8: "call_pending"}
	for index, rawItem := range input {
		item, _ := rawItem.(map[string]interface{})
		itemType, _ := item["type"].(string)
		if itemType == "" {
			itemType = "message"
		}
		if itemType != wantTypes[index] {
			t.Fatalf("input[%d].type = %q, want %q; input=%#v", index, itemType, wantTypes[index], input)
		}
		if wantCallID, hasCallID := wantCallIDs[index]; hasCallID {
			if item["call_id"] != wantCallID {
				t.Fatalf("input[%d].call_id = %#v, want %q", index, item["call_id"], wantCallID)
			}
		}
		if itemType == "function_call" {
			if _, hasSyntheticID := item["id"]; hasSyntheticID {
				t.Fatalf("input[%d] synthesized a colliding item id: %#v", index, item)
			}
		}
	}
}

func TestOpenAITranslator_ChatToResponsesDeepSeekRejectsUnresolvedToolCall(t *testing.T) {
	request := decodeOpenAIChatToResponsesTestRequest(t, `{
		"model":"deepseek-chat",
		"messages":[
			{"role":"assistant","tool_calls":[{"id":"call_pending","type":"function","function":{"name":"lookup","arguments":"{}"}}]}
		]
	}`)
	translator := NewOpenAITranslator(config.ProviderConfig{
		APIKey:               "dummy",
		CompatibilityProfile: "deepseek",
	})
	upstreamRequest, err := translator.TranslateRequest(
		WithUpstreamRequestType(context.Background(), "responses"),
		request,
	)
	if upstreamRequest != nil {
		t.Fatalf("request = %#v, want no upstream request", upstreamRequest)
	}
	var compatibilityErr *models.CompatibilityError
	if !errors.As(err, &compatibilityErr) || compatibilityErr.Field != "messages[0].tool_calls[0].id" {
		t.Fatalf("error = %#v, want unresolved call CompatibilityError", err)
	}
}

func decodeOpenAIChatToResponsesTestRequest(t *testing.T, raw string) *models.UnifiedRequest {
	t.Helper()
	var request models.UnifiedRequest
	if err := json.Unmarshal([]byte(raw), &request); err != nil {
		t.Fatalf("decode request: %v", err)
	}
	request.RawJSON = json.RawMessage(raw)
	request.SourceRequestType = "chat_completions"
	return &request
}

func TestOpenAITranslator_ChatToResponsesPreservesStringWhitespace(t *testing.T) {
	translator := NewOpenAITranslator(config.ProviderConfig{APIKey: "dummy"})
	request, err := translator.TranslateRequest(WithUpstreamRequestType(context.Background(), "responses"), &models.UnifiedRequest{
		RawJSON:           json.RawMessage(`{"model":"gpt-5.4","messages":[{"role":"user","content":"  hello  "}]}`),
		SourceRequestType: "chat_completions",
		Model:             "gpt-5.4",
		Messages:          []models.Message{{Role: "user", Content: "  hello  "}},
	})
	if err != nil {
		t.Fatalf("TranslateRequest: %v", err)
	}
	body, err := io.ReadAll(request.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	var payload map[string]interface{}
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("decode body: %v", err)
	}
	input := payload["input"].([]interface{})
	content := input[0].(map[string]interface{})["content"].([]interface{})
	if got := content[0].(map[string]interface{})["text"]; got != "  hello  " {
		t.Fatalf("text = %#v, want whitespace preserved", got)
	}
}

func TestOpenAITranslator_ChatToResponsesPreservesInterleavedInstructionOrder(t *testing.T) {
	messages := []models.Message{
		{Role: "system", Content: "first system"},
		{Role: "user", Content: "first user"},
		{Role: "system", Content: "second system"},
		{Role: "developer", Content: "developer"},
		{Role: "user", Content: "second user"},
	}
	translator := NewOpenAITranslator(config.ProviderConfig{APIKey: "dummy"})
	request, err := translator.TranslateRequest(WithUpstreamRequestType(context.Background(), "responses"), &models.UnifiedRequest{
		RawJSON: json.RawMessage(`{
			"model":"gpt-5.4",
			"messages":[
				{"role":"system","content":"first system"},
				{"role":"user","content":"first user"},
				{"role":"system","content":"second system"},
				{"role":"developer","content":"developer"},
				{"role":"user","content":"second user"}
			]
		}`),
		SourceRequestType: "chat_completions",
		Model:             "gpt-5.4",
		Messages:          messages,
	})
	if err != nil {
		t.Fatalf("TranslateRequest: %v", err)
	}
	body, err := io.ReadAll(request.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	var payload map[string]interface{}
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("decode body: %v", err)
	}
	if _, exists := payload["instructions"]; exists {
		t.Fatalf("interleaved guidance moved into instructions: %#v", payload["instructions"])
	}
	input, _ := payload["input"].([]interface{})
	wantRoles := []string{"system", "user", "system", "developer", "user"}
	if len(input) != len(wantRoles) {
		t.Fatalf("input = %#v, want %d ordered messages", input, len(wantRoles))
	}
	for index, wantRole := range wantRoles {
		item, _ := input[index].(map[string]interface{})
		if item["role"] != wantRole {
			t.Fatalf("input[%d].role = %#v, want %q; input=%#v", index, item["role"], wantRole, input)
		}
	}
}

func TestOpenAITranslator_ChatToResponsesMapsMaxCompletionTokens(t *testing.T) {
	maxCompletionTokens := 257
	req := &models.UnifiedRequest{
		SourceRequestType:   "chat_completions",
		Model:               "gpt-5.4",
		Messages:            []models.Message{{Role: "user", Content: "hello"}},
		MaxCompletionTokens: &maxCompletionTokens,
		RawJSON:             json.RawMessage(`{"model":"gpt-5.4","messages":[{"role":"user","content":"hello"}],"max_completion_tokens":257}`),
	}
	if err := models.NormalizeUnifiedRequest(req); err != nil {
		t.Fatalf("NormalizeUnifiedRequest returned error: %v", err)
	}

	translator := NewOpenAITranslator(config.ProviderConfig{APIKey: "dummy"})
	request, err := translator.TranslateRequest(WithUpstreamRequestType(context.Background(), "responses"), req)
	if err != nil {
		t.Fatalf("TranslateRequest returned error: %v", err)
	}
	body, err := io.ReadAll(request.Body)
	if err != nil {
		t.Fatalf("read request body: %v", err)
	}
	var payload map[string]interface{}
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("decode request body: %v", err)
	}
	if payload["max_output_tokens"] != float64(maxCompletionTokens) {
		t.Fatalf("max_output_tokens = %#v, want %d", payload["max_output_tokens"], maxCompletionTokens)
	}
}

func TestOpenAITranslator_ChatToResponsesPreservesToolStrictness(t *testing.T) {
	falseValue := false
	trueValue := true
	tests := []struct {
		name       string
		rawStrict  string
		typed      *bool
		want       bool
		wantReject bool
	}{
		{name: "absent defaults false", want: false},
		{name: "explicit false", rawStrict: `,"strict":false`, typed: &falseValue, want: false},
		{name: "explicit true", rawStrict: `,"strict":true`, typed: &trueValue, want: true},
		{name: "null defaults false", rawStrict: `,"strict":null`, want: false},
		{name: "string", rawStrict: `,"strict":"yes"`, wantReject: true},
	}

	translator := NewOpenAITranslator(config.ProviderConfig{APIKey: "dummy"})
	ctx := WithUpstreamRequestType(context.Background(), "responses")
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			raw := json.RawMessage(`{
				"model":"gpt-5.4",
				"messages":[{"role":"user","content":"hello"}],
				"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}` + test.rawStrict + `}}]
			}`)
			request, err := translator.TranslateRequest(ctx, &models.UnifiedRequest{
				RawJSON:           raw,
				SourceRequestType: "chat_completions",
				Model:             "gpt-5.4",
				Messages:          []models.Message{{Role: "user", Content: "hello"}},
				Tools: []models.Tool{{
					Type: "function",
					Function: models.ToolFunction{
						Name:       "lookup",
						Parameters: map[string]interface{}{"type": "object"},
						Strict:     test.typed,
					},
				}},
			})
			if test.wantReject {
				var compatibilityErr *models.CompatibilityError
				if !errors.As(err, &compatibilityErr) {
					t.Fatalf("error = %v, want CompatibilityError", err)
				}
				if compatibilityErr.Field != "tools[0].function.strict" {
					t.Fatalf("CompatibilityError field = %q", compatibilityErr.Field)
				}
				return
			}
			if err != nil {
				t.Fatalf("TranslateRequest: %v", err)
			}
			body, err := io.ReadAll(request.Body)
			if err != nil {
				t.Fatalf("read request body: %v", err)
			}
			var payload map[string]interface{}
			if err := json.Unmarshal(body, &payload); err != nil {
				t.Fatalf("decode request body: %v", err)
			}
			tools, _ := payload["tools"].([]interface{})
			if len(tools) != 1 {
				t.Fatalf("tools = %#v", payload["tools"])
			}
			tool, _ := tools[0].(map[string]interface{})
			if strict, ok := tool["strict"].(bool); !ok || strict != test.want {
				t.Fatalf("strict = %#v, want %t", tool["strict"], test.want)
			}
		})
	}
}
