package api

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/internal/routing"
	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestTranslatedResponsesRejectsUnmappedFieldsPerTarget(t *testing.T) {
	handler := &Handler{registry: providers.NewRegistry(map[string]config.ProviderConfig{
		"openai-chat": {Type: "openai"},
	})}
	tests := []struct {
		name      string
		raw       string
		wantField string
	}{
		{name: "metadata", raw: `{"model":"gpt-5","input":"hi","metadata":{"trace":"x"}}`, wantField: "metadata"},
		{name: "include", raw: `{"model":"gpt-5","input":"hi","include":["reasoning.encrypted_content"]}`, wantField: "include"},
		{name: "text verbosity", raw: `{"model":"gpt-5","input":"hi","text":{"verbosity":"low"}}`, wantField: "text.verbosity"},
		{name: "parallel tool calls", raw: `{"model":"gpt-5","input":"hi","parallel_tool_calls":false}`, wantField: "parallel_tool_calls"},
		{name: "prompt", raw: `{"model":"gpt-5","input":"hi","prompt":{"id":"pmpt_1"}}`, wantField: "prompt"},
		{name: "prompt cache", raw: `{"model":"gpt-5","input":"hi","prompt_cache_key":"tenant"}`, wantField: "prompt_cache_key"},
		{name: "prompt cache options", raw: `{"model":"gpt-5","input":"hi","prompt_cache_options":{"mode":"explicit","ttl":"30m"}}`, wantField: "prompt_cache_options"},
		{name: "prompt cache retention", raw: `{"model":"gpt-5","input":"hi","prompt_cache_retention":"24h"}`, wantField: "prompt_cache_retention"},
		{name: "context management", raw: `{"model":"gpt-5","input":"hi","context_management":[{"type":"compact","compact_threshold":1000}]}`, wantField: "context_management"},
		{name: "safety", raw: `{"model":"gpt-5","input":"hi","safety_identifier":"user_hash"}`, wantField: "safety_identifier"},
		{name: "service tier", raw: `{"model":"gpt-5","input":"hi","service_tier":"priority"}`, wantField: "service_tier"},
		{name: "stream options", raw: `{"model":"gpt-5","input":"hi","stream":true,"stream_options":{"include_obfuscation":false}}`, wantField: "stream_options"},
		{name: "background", raw: `{"model":"gpt-5","input":"hi","background":true}`, wantField: "background"},
		{name: "max tool calls", raw: `{"model":"gpt-5","input":"hi","max_tool_calls":2}`, wantField: "max_tool_calls"},
		{name: "truncation", raw: `{"model":"gpt-5","input":"hi","truncation":"auto"}`, wantField: "truncation"},
		{name: "top logprobs", raw: `{"model":"gpt-5","input":"hi","top_logprobs":2}`, wantField: "top_logprobs"},
		{name: "moderation extension", raw: `{"model":"gpt-5","input":"hi","moderation":{"model":"latest"}}`, wantField: "moderation"},
		{name: "reasoning summary", raw: `{"model":"gpt-5","input":"hi","reasoning":{"effort":"high","summary":"auto"}}`, wantField: "reasoning.summary"},
		{name: "reasoning context", raw: `{"model":"gpt-5","input":"hi","reasoning":{"effort":"high","context":"auto"}}`, wantField: "reasoning.context"},
		{name: "reasoning mode", raw: `{"model":"gpt-5","input":"hi","reasoning":{"effort":"high","mode":"standard"}}`, wantField: "reasoning.mode"},
		{name: "reasoning generated summary", raw: `{"model":"gpt-5","input":"hi","reasoning":{"effort":"high","generate_summary":"auto"}}`, wantField: "reasoning.generate_summary"},
		{name: "non-boolean function strictness", raw: `{"model":"gpt-5","input":"hi","tools":[{"type":"function","name":"lookup","parameters":{"type":"object"},"strict":"yes"}]}`, wantField: "tools[0].strict"},
		{name: "function allowed callers", raw: `{"model":"gpt-5","input":"hi","tools":[{"type":"function","name":"lookup","parameters":{"type":"object"},"allowed_callers":["direct"]}]}`, wantField: "tools[0].allowed_callers"},
		{name: "deferred function loading", raw: `{"model":"gpt-5","input":"hi","tools":[{"type":"function","name":"lookup","parameters":{"type":"object"},"defer_loading":true}]}`, wantField: "tools[0].defer_loading"},
		{name: "function output schema", raw: `{"model":"gpt-5","input":"hi","tools":[{"type":"function","name":"lookup","parameters":{"type":"object"},"output_schema":{"type":"object"}}]}`, wantField: "tools[0].output_schema"},
		{name: "hosted tool", raw: `{"model":"gpt-5","input":"hi","tools":[{"type":"web_search"}]}`, wantField: "tools[0].type"},
		{name: "reasoning input item", raw: `{"model":"gpt-5","input":[{"type":"reasoning","encrypted_content":"abc"}]}`, wantField: "input[0].type"},
		{name: "image file reference", raw: `{"model":"gpt-5","input":[{"type":"message","role":"user","content":[{"type":"input_image","file_id":"file_1"}]}]}`, wantField: "input[0].content[0].file_id"},
		{name: "empty lifecycle id", raw: `{"model":"gpt-5","input":[{"type":"message","id":"","role":"user","content":"hi"}]}`, wantField: "input[0].id"},
		{name: "incomplete lifecycle status", raw: `{"model":"gpt-5","input":[{"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{}","status":"in_progress"}]}`, wantField: "input[0].status"},
		{name: "unknown message phase", raw: `{"model":"gpt-5","input":[{"type":"message","role":"assistant","content":"hi","phase":"analysis"}]}`, wantField: "input[0].phase"},
		{name: "foreign nested input field", raw: `{"model":"gpt-5","input":[{"type":"message","role":"user","content":"hi","vendor_hint":true}]}`, wantField: "input[0].vendor_hint"},
		{name: "unknown message role", raw: `{"model":"gpt-5","input":[{"type":"message","role":"critic","content":"hi"}]}`, wantField: "input[0].role"},
		{name: "non-string text", raw: `{"model":"gpt-5","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":{"value":"hi"}}]}]}`, wantField: "input[0].content[0].text"},
		{name: "input cache breakpoint", raw: `{"model":"gpt-5","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi","prompt_cache_breakpoint":{"type":"ephemeral"}}]}]}`, wantField: "input[0].content[0].prompt_cache_breakpoint"},
		{name: "function call without name", raw: `{"model":"gpt-5","input":[{"type":"function_call","call_id":"call_1","arguments":"{}"}]}`, wantField: "input[0].name"},
		{name: "structured function output", raw: `{"model":"gpt-5","input":[{"type":"function_call_output","call_id":"call_1","output":[{"type":"input_text","text":"ok"}]}]}`, wantField: "input[0].output"},
		{name: "allowed tools choice", raw: `{"model":"gpt-5","input":"hi","tool_choice":{"type":"allowed_tools","mode":"auto","tools":[]}}`, wantField: "tool_choice.type"},
		{name: "future field", raw: `{"model":"gpt-5","input":"hi","future_responses_control":true}`, wantField: "future_responses_control"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := handler.validateChatCompatibility(
				routing.Target{Provider: "openai-chat", UpstreamRequestType: requestTypeChatCompletions},
				&models.UnifiedRequest{SourceRequestType: requestTypeResponses, RawJSON: json.RawMessage(tt.raw)},
			)
			var compatibilityErr *models.CompatibilityError
			if !errors.As(err, &compatibilityErr) {
				t.Fatalf("error = %v, want CompatibilityError", err)
			}
			if compatibilityErr.Field != tt.wantField || compatibilityErr.Provider != "openai-chat" {
				t.Fatalf("compatibility error = %#v, want field=%q provider=openai-chat", compatibilityErr, tt.wantField)
			}
		})
	}
}

func TestTranslatedResponsesAllowsFaithfullyMappedFields(t *testing.T) {
	handler := &Handler{registry: providers.NewRegistry(map[string]config.ProviderConfig{
		"openai-chat": {Type: "openai"},
	})}
	raw := json.RawMessage(`{
		"model":"gpt-5",
		"input":[
			{"type":"message","role":"user","content":[
				{"type":"input_text","text":"hi"},
				{"type":"input_image","image_url":"data:image/png;base64,aGVsbG8=","detail":"low"}
			],"id":"msg_1","status":"completed","phase":"final_answer"},
			{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":"{}","status":"completed"},
			{"type":"function_call_output","id":"fco_1","call_id":"call_1","output":"ok","status":"completed"}
		],
		"instructions":[{"type":"message","role":"developer","content":"be concise"}],
		"reasoning":{"effort":"high"},
		"text":{"format":{"type":"json_schema","name":"answer","schema":{"type":"object"},"strict":true}},
		"temperature":0.2,
		"top_p":0.9,
		"max_output_tokens":100,
		"tools":[{"type":"function","name":"lookup","description":"lookup","parameters":{"type":"object"},"strict":true}],
		"tool_choice":{"type":"function","name":"lookup"},
		"stream":false,
		"store":false,
		"user":"customer-1"
	}`)
	err := handler.validateChatCompatibility(
		routing.Target{Provider: "openai-chat", UpstreamRequestType: requestTypeChatCompletions},
		&models.UnifiedRequest{SourceRequestType: requestTypeResponses, RawJSON: raw},
	)
	if err != nil {
		t.Fatalf("mapped Responses controls were rejected: %v", err)
	}
}

func TestTranslatedResponsesAllowsNullFunctionStrictness(t *testing.T) {
	handler := &Handler{registry: providers.NewRegistry(map[string]config.ProviderConfig{
		"openai-chat": {Type: "openai"},
		"anthropic":   {Type: "anthropic"},
		"ollama":      {Type: "ollama"},
	})}
	raw := json.RawMessage(`{
		"model":"model",
		"input":"hi",
		"tools":[{"type":"function","name":"lookup","parameters":{"type":"object"},"strict":null}]
	}`)
	request := &models.UnifiedRequest{
		SourceRequestType: requestTypeResponses,
		RawJSON:           raw,
		Messages:          []models.Message{{Role: "user", Content: "hi"}},
		Tools: []models.Tool{{
			Type: "function",
			Function: models.ToolFunction{
				Name:       "lookup",
				Parameters: map[string]interface{}{"type": "object"},
			},
		}},
	}

	for _, providerID := range []string{"openai-chat", "anthropic", "ollama"} {
		t.Run(providerID, func(t *testing.T) {
			err := handler.validateChatCompatibility(
				routing.Target{Provider: providerID, UpstreamRequestType: requestTypeChatCompletions},
				request,
			)
			if err != nil {
				t.Fatalf("strict:null was rejected: %v", err)
			}
		})
	}

	got := handler.compatibleChatFallbacks([]routing.Target{
		{Provider: "openai-chat", UpstreamRequestType: requestTypeChatCompletions},
		{Provider: "anthropic", UpstreamRequestType: requestTypeChatCompletions},
		{Provider: "ollama", UpstreamRequestType: requestTypeChatCompletions},
	}, request)
	if len(got) != 3 {
		t.Fatalf("compatible fallbacks = %#v, want all translated targets", got)
	}
}

func TestTranslatedResponsesRejectsImageDetailWhenTargetCannotHonorIt(t *testing.T) {
	handler := &Handler{registry: providers.NewRegistry(map[string]config.ProviderConfig{
		"anthropic": {Type: "anthropic"},
	})}
	raw := json.RawMessage(`{"model":"claude","input":[{"type":"message","role":"user","content":[{"type":"input_image","image_url":"https://example.com/image.png","detail":"low"}]}]}`)

	err := handler.validateChatCompatibility(
		routing.Target{Provider: "anthropic", UpstreamRequestType: requestTypeChatCompletions},
		&models.UnifiedRequest{SourceRequestType: requestTypeResponses, RawJSON: raw},
	)
	var compatibilityErr *models.CompatibilityError
	if !errors.As(err, &compatibilityErr) {
		t.Fatalf("error = %v, want CompatibilityError", err)
	}
	if compatibilityErr.Field != "input[0].content[0].detail" || compatibilityErr.Provider != "anthropic" {
		t.Fatalf("compatibility error = %#v", compatibilityErr)
	}
}

func TestTranslatedResponsesFallbacksKeepOnlyCompatibleTargets(t *testing.T) {
	handler := &Handler{registry: providers.NewRegistry(map[string]config.ProviderConfig{
		"translated": {Type: "openai"},
		"native":     {Type: "openai"},
	})}
	req := &models.UnifiedRequest{
		SourceRequestType: requestTypeResponses,
		RawJSON:           json.RawMessage(`{"model":"gpt-5","input":"hi","metadata":{"trace":"x"}}`),
	}
	fallbacks := []routing.Target{
		{Provider: "translated", UpstreamRequestType: requestTypeChatCompletions},
		{Provider: "native", UpstreamRequestType: requestTypeResponses},
	}

	got := handler.compatibleChatFallbacks(fallbacks, req)
	if len(got) != 1 || got[0].Provider != "native" {
		t.Fatalf("compatible fallbacks = %#v, want only native target", got)
	}
}

func TestResponsesTranslatedTargetRejectsBeforeUpstream(t *testing.T) {
	upstreamCalls := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamCalls++
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer upstream.Close()
	handler, cache := newNativeContinuationTestHandler(t, upstream.URL+"/v1", requestTypeChatCompletions)
	defer cache.Stop()
	payload := []byte(`{"model":"gpt-5.4","input":"hi","metadata":{"trace":"x"}}`)
	recorder := httptest.NewRecorder()

	handler.Responses(recorder, httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(payload)))

	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400; body=%s", recorder.Code, recorder.Body.String())
	}
	if upstreamCalls != 0 {
		t.Fatalf("upstream calls = %d, want 0", upstreamCalls)
	}
	var response models.ErrorResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode error response: %v", err)
	}
	if response.Error.Param == nil || *response.Error.Param != "metadata" {
		t.Fatalf("error param = %#v, want metadata", response.Error.Param)
	}
}

func TestResponsesTranslatedTargetPreservesMappedImageAndTextFormat(t *testing.T) {
	var captured map[string]interface{}
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&captured); err != nil {
			t.Fatalf("decode upstream request: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl_1","object":"chat.completion","created":1,"model":"gpt-5.4","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	defer upstream.Close()
	handler, cache := newNativeContinuationTestHandler(t, upstream.URL+"/v1", requestTypeChatCompletions)
	defer cache.Stop()
	payload := []byte(`{
		"model":"gpt-5.4",
		"input":[{"type":"message","role":"user","content":[
			{"type":"input_text","text":"describe"},
			{"type":"input_image","image_url":"data:image/png;base64,aGVsbG8=","detail":"low"}
		]}],
		"text":{"format":{"type":"json_schema","name":"answer","schema":{"type":"object"},"strict":true}},
		"tools":[{"type":"function","name":"lookup","parameters":{"type":"object"},"strict":null}],
		"store":false
	}`)
	recorder := httptest.NewRecorder()

	handler.Responses(recorder, httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(payload)))

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", recorder.Code, recorder.Body.String())
	}
	messages, ok := captured["messages"].([]interface{})
	if !ok || len(messages) != 1 {
		t.Fatalf("messages = %#v", captured["messages"])
	}
	message := messages[0].(map[string]interface{})
	content, ok := message["content"].([]interface{})
	if !ok || len(content) != 2 {
		t.Fatalf("content = %#v", message["content"])
	}
	imagePart := content[1].(map[string]interface{})
	imageURL, ok := imagePart["image_url"].(map[string]interface{})
	if imagePart["type"] != "image_url" || !ok || imageURL["url"] != "data:image/png;base64,aGVsbG8=" || imageURL["detail"] != "low" {
		t.Fatalf("mapped image = %#v", imagePart)
	}
	responseFormat, ok := captured["response_format"].(map[string]interface{})
	if !ok || responseFormat["type"] != "json_schema" {
		t.Fatalf("response_format = %#v", captured["response_format"])
	}
	jsonSchema, ok := responseFormat["json_schema"].(map[string]interface{})
	if !ok || jsonSchema["name"] != "answer" || jsonSchema["strict"] != true {
		t.Fatalf("json_schema = %#v", responseFormat["json_schema"])
	}
	tools, ok := captured["tools"].([]interface{})
	if !ok || len(tools) != 1 {
		t.Fatalf("tools = %#v", captured["tools"])
	}
	tool, _ := tools[0].(map[string]interface{})
	function, _ := tool["function"].(map[string]interface{})
	if function["name"] != "lookup" {
		t.Fatalf("mapped tool = %#v", tool)
	}
	if _, exists := function["strict"]; exists {
		t.Fatalf("strict:null should use the translated target's non-strict default: %#v", function)
	}
}

func TestResponsesNativeTargetPassesCompleteEnvelopeThrough(t *testing.T) {
	var captured map[string]interface{}
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("read upstream request: %v", err)
		}
		if err := json.Unmarshal(body, &captured); err != nil {
			t.Fatalf("decode upstream request: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"resp_native","object":"response","created_at":1,"status":"completed","model":"gpt-5.4","output":[],"output_text":"ok"}`))
	}))
	defer upstream.Close()
	handler, cache := newNativeContinuationTestHandler(t, upstream.URL+"/v1", requestTypeResponses)
	defer cache.Stop()
	payload := []byte(`{
		"model":"gpt-5.4",
		"input":[
			{"type":"reasoning","encrypted_content":"abc","summary":[{"type":"summary_text","text":"prior"}]},
			{"type":"message","role":"user","content":[
				{"type":"input_text","text":"describe","prompt_cache_breakpoint":{"type":"ephemeral"}},
				{"type":"input_file","file_id":"file_1"}
			]}
		],
		"instructions":[{"type":"message","role":"developer","content":"be concise"}],
		"reasoning":{"effort":"high","summary":"auto","context":"auto","mode":"standard","generate_summary":"auto"},
		"metadata":{"trace":"x"},
		"include":["reasoning.encrypted_content"],
		"max_output_tokens":512,
		"moderation":{"model":"omni-moderation-latest","policy":"default"},
		"text":{"format":{"type":"text"},"verbosity":"low"},
		"context_management":[{"type":"compact","compact_threshold":1000}],
		"stream_options":{"include_obfuscation":false},
		"parallel_tool_calls":false,
		"tools":[
			{"type":"function","name":"lookup","parameters":{"type":"object"},"strict":true,"allowed_callers":["direct"],"defer_loading":true,"output_schema":{"type":"object"},"future_tool_control":"x"},
			{"type":"function","name":"nullable","parameters":{"type":"object"},"strict":null}
		],
		"tool_choice":{"type":"allowed_tools","mode":"auto","tools":[{"type":"function","name":"lookup"}]},
		"prompt":{"id":"pmpt_1"},
		"prompt_cache_key":"tenant",
		"prompt_cache_options":{"mode":"explicit","ttl":"30m"},
		"prompt_cache_retention":"24h",
		"safety_identifier":"user_hash",
		"service_tier":"priority",
		"temperature":0.2,
		"top_logprobs":5,
		"top_p":0.9,
		"truncation":"disabled",
		"user":"legacy-user",
		"background":false,
		"max_tool_calls":2,
		"future_responses_control":{"enabled":true},
		"store":false
	}`)
	recorder := httptest.NewRecorder()

	handler.Responses(recorder, httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(payload)))

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", recorder.Code, recorder.Body.String())
	}
	for _, field := range []string{
		"input", "instructions", "reasoning", "metadata", "include", "text", "parallel_tool_calls", "tools", "tool_choice", "prompt",
		"context_management", "stream_options", "max_output_tokens", "moderation", "prompt_cache_key", "prompt_cache_options", "prompt_cache_retention",
		"safety_identifier", "service_tier", "temperature", "top_logprobs", "top_p", "truncation", "user", "background", "max_tool_calls",
		"future_responses_control",
	} {
		if _, ok := captured[field]; !ok {
			t.Fatalf("native Responses field %q was lost: %#v", field, captured)
		}
	}
	input := captured["input"].([]interface{})
	if input[0].(map[string]interface{})["encrypted_content"] != "abc" || input[1].(map[string]interface{})["type"] != "message" {
		t.Fatalf("native extended input changed: %#v", input)
	}
	messageContent := input[1].(map[string]interface{})["content"].([]interface{})
	if messageContent[0].(map[string]interface{})["prompt_cache_breakpoint"] == nil {
		t.Fatalf("native input cache breakpoint changed: %#v", messageContent)
	}
	reasoning := captured["reasoning"].(map[string]interface{})
	if reasoning["summary"] != "auto" || reasoning["context"] != "auto" || reasoning["mode"] != "standard" || reasoning["generate_summary"] != "auto" {
		t.Fatalf("native reasoning changed: %#v", reasoning)
	}
	tools := captured["tools"].([]interface{})
	if len(tools) != 2 {
		t.Fatalf("native tools = %#v, want two", tools)
	}
	tool := tools[0].(map[string]interface{})
	if tool["strict"] != true || tool["defer_loading"] != true || tool["allowed_callers"] == nil || tool["output_schema"] == nil || tool["future_tool_control"] != "x" {
		t.Fatalf("native tools changed: %#v", tools)
	}
	nullableTool := tools[1].(map[string]interface{})
	if strict, exists := nullableTool["strict"]; !exists || strict != nil {
		t.Fatalf("native strict:null changed: %#v", nullableTool)
	}
}
