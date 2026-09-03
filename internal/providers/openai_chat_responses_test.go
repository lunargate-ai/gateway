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
