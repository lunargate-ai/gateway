package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/pkg/models"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

func decodeOllamaRequestBody(t *testing.T, reqBody io.Reader) map[string]interface{} {
	t.Helper()

	body, err := io.ReadAll(reqBody)
	if err != nil {
		t.Fatalf("failed to read request body: %v", err)
	}

	var payload map[string]interface{}
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("failed to unmarshal request payload: %v", err)
	}

	return payload
}

func decodeOllamaOptions(t *testing.T, payload map[string]interface{}) map[string]interface{} {
	t.Helper()
	raw, ok := payload["options"]
	if !ok {
		t.Fatalf("expected options in payload, got %#v", payload)
	}
	opts, ok := raw.(map[string]interface{})
	if !ok {
		t.Fatalf("expected options object, got %#v", raw)
	}
	return opts
}

func decodeOllamaMessages(t *testing.T, payload map[string]interface{}) []map[string]interface{} {
	t.Helper()
	raw, ok := payload["messages"].([]interface{})
	if !ok {
		t.Fatalf("expected messages array, got %#v", payload["messages"])
	}
	messages := make([]map[string]interface{}, len(raw))
	for i := range raw {
		message, ok := raw[i].(map[string]interface{})
		if !ok {
			t.Fatalf("expected messages[%d] object, got %#v", i, raw[i])
		}
		messages[i] = message
	}
	return messages
}

func TestOllamaTranslator_TranslateRequest_MapsReasoningEffortToThink(t *testing.T) {
	translator := NewOllamaTranslator(config.ProviderConfig{
		BaseURL:      "http://localhost:11434",
		DefaultModel: "gemma3",
	})

	req, err := translator.TranslateRequest(context.Background(), &models.UnifiedRequest{
		Model:           "gemma3",
		ReasoningEffort: "high",
		Messages:        []models.Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("TranslateRequest returned error: %v", err)
	}

	payload := decodeOllamaRequestBody(t, req.Body)

	think, ok := payload["think"].(string)
	if !ok || think != "high" {
		t.Fatalf("expected think=high, got %#v", payload["think"])
	}
}

func TestOllamaTranslator_TranslateRequest_MapsReasoningEffortNoneToThinkFalse(t *testing.T) {
	translator := NewOllamaTranslator(config.ProviderConfig{
		BaseURL:      "http://localhost:11434",
		DefaultModel: "gemma3",
	})

	req, err := translator.TranslateRequest(context.Background(), &models.UnifiedRequest{
		Model:           "gemma3",
		ReasoningEffort: "none",
		Messages:        []models.Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("TranslateRequest returned error: %v", err)
	}

	payload := decodeOllamaRequestBody(t, req.Body)

	think, ok := payload["think"].(bool)
	if !ok || think {
		t.Fatalf("expected think=false, got %#v", payload["think"])
	}
}

func TestOllamaTranslator_TranslateRequest_UsesProviderDefaultThink(t *testing.T) {
	translator := NewOllamaTranslator(config.ProviderConfig{
		BaseURL:      "http://localhost:11434",
		DefaultModel: "gemma3",
		Extra: map[string]string{
			"think": "true",
		},
	})

	req, err := translator.TranslateRequest(context.Background(), &models.UnifiedRequest{
		Model:    "gemma3",
		Messages: []models.Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("TranslateRequest returned error: %v", err)
	}

	payload := decodeOllamaRequestBody(t, req.Body)

	think, ok := payload["think"].(bool)
	if !ok || !think {
		t.Fatalf("expected think=true from provider extra, got %#v", payload["think"])
	}
}

func TestOllamaTranslator_TranslateRequest_MapsSamplingOptions(t *testing.T) {
	translator := NewOllamaTranslator(config.ProviderConfig{
		BaseURL:      "http://localhost:11434",
		DefaultModel: "gemma3",
	})

	temperature := 1.0
	topP := 0.95
	topK := 64
	req, err := translator.TranslateRequest(context.Background(), &models.UnifiedRequest{
		Model:       "gemma3",
		Messages:    []models.Message{{Role: "user", Content: "hi"}},
		Temperature: &temperature,
		TopP:        &topP,
		TopK:        &topK,
	})
	if err != nil {
		t.Fatalf("TranslateRequest returned error: %v", err)
	}

	payload := decodeOllamaRequestBody(t, req.Body)
	options := decodeOllamaOptions(t, payload)

	if got, ok := options["temperature"].(float64); !ok || got != 1.0 {
		t.Fatalf("expected options.temperature=1.0, got %#v", options["temperature"])
	}
	if got, ok := options["top_p"].(float64); !ok || got != 0.95 {
		t.Fatalf("expected options.top_p=0.95, got %#v", options["top_p"])
	}
	if got, ok := options["top_k"].(float64); !ok || got != 64 {
		t.Fatalf("expected options.top_k=64, got %#v", options["top_k"])
	}
}

func TestOllamaTranslator_TranslateRequest_MapsSupportedGenerationControls(t *testing.T) {
	translator := NewOllamaTranslator(config.ProviderConfig{
		BaseURL:      "http://localhost:11434",
		DefaultModel: "gemma3",
	})

	presencePenalty := 0.25
	frequencyPenalty := -0.5
	seed := 0
	maxTokens := 123
	req, err := translator.TranslateRequest(context.Background(), &models.UnifiedRequest{
		Model:            "gemma3",
		Messages:         []models.Message{{Role: "user", Content: "hi"}},
		Stop:             []interface{}{"END", "STOP"},
		MaxTokens:        &maxTokens,
		PresencePenalty:  &presencePenalty,
		FrequencyPenalty: &frequencyPenalty,
		Seed:             &seed,
		ReasoningEffort:  "max",
	})
	if err != nil {
		t.Fatalf("TranslateRequest returned error: %v", err)
	}

	payload := decodeOllamaRequestBody(t, req.Body)
	options := decodeOllamaOptions(t, payload)
	if got := options["num_predict"]; got != float64(maxTokens) {
		t.Fatalf("options.num_predict = %#v, want %d", got, maxTokens)
	}
	if got := options["presence_penalty"]; got != presencePenalty {
		t.Fatalf("options.presence_penalty = %#v, want %v", got, presencePenalty)
	}
	if got := options["frequency_penalty"]; got != frequencyPenalty {
		t.Fatalf("options.frequency_penalty = %#v, want %v", got, frequencyPenalty)
	}
	if got := options["seed"]; got != float64(seed) {
		t.Fatalf("options.seed = %#v, want %d", got, seed)
	}
	if got := payload["think"]; got != "max" {
		t.Fatalf("think = %#v, want max", got)
	}
	stop, ok := options["stop"].([]interface{})
	if !ok || len(stop) != 2 || stop[0] != "END" || stop[1] != "STOP" {
		t.Fatalf("options.stop = %#v, want [END STOP]", options["stop"])
	}
}

func TestOllamaTranslator_TranslateRequest_MapsResponseFormats(t *testing.T) {
	tests := []struct {
		name       string
		formatType string
		wantFormat interface{}
	}{
		{name: "text is native default", formatType: "text"},
		{name: "json object", formatType: "json_object", wantFormat: "json"},
		{
			name:       "json schema",
			formatType: "json_schema",
			wantFormat: map[string]interface{}{
				"type":     "object",
				"required": []interface{}{"answer"},
				"properties": map[string]interface{}{
					"answer": map[string]interface{}{"type": "string"},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			translator := NewOllamaTranslator(config.ProviderConfig{BaseURL: "http://localhost:11434"})
			responseFormat := &models.ResponseFormat{Type: tt.formatType}
			if tt.formatType == "json_schema" {
				responseFormat.JSONSchema = &models.JSONSchemaResponseFormat{Schema: tt.wantFormat}
			}
			req, err := translator.TranslateRequest(context.Background(), &models.UnifiedRequest{
				Model:          "gemma3",
				Messages:       []models.Message{{Role: "user", Content: "hi"}},
				ResponseFormat: responseFormat,
			})
			if err != nil {
				t.Fatalf("TranslateRequest returned error: %v", err)
			}
			payload := decodeOllamaRequestBody(t, req.Body)
			got, present := payload["format"]
			if tt.wantFormat == nil {
				if present {
					t.Fatalf("format = %#v, want omitted", got)
				}
				return
			}
			gotJSON, _ := json.Marshal(got)
			wantJSON, _ := json.Marshal(tt.wantFormat)
			if !bytes.Equal(gotJSON, wantJSON) {
				t.Fatalf("format = %s, want %s", gotJSON, wantJSON)
			}
		})
	}
}

func TestOllamaTranslator_TranslateRequest_RejectsUnsupportedExplicitFields(t *testing.T) {
	two := 2
	store := true
	tests := []struct {
		name      string
		configure func(*models.UnifiedRequest)
		wantField string
	}{
		{name: "multiple choices", configure: func(req *models.UnifiedRequest) { req.N = &two }, wantField: "n"},
		{name: "logit bias", configure: func(req *models.UnifiedRequest) { req.LogitBias = map[string]int{"42": 10} }, wantField: "logit_bias"},
		{name: "user", configure: func(req *models.UnifiedRequest) { req.User = "end-user" }, wantField: "user"},
		{name: "stored response", configure: func(req *models.UnifiedRequest) { req.Store = &store }, wantField: "store"},
		{name: "reasoning effort", configure: func(req *models.UnifiedRequest) { req.ReasoningEffort = "extreme" }, wantField: "reasoning_effort"},
		{name: "invalid stop", configure: func(req *models.UnifiedRequest) { req.Stop = float64(42) }, wantField: "stop"},
		{
			name: "unsupported response format",
			configure: func(req *models.UnifiedRequest) {
				req.ResponseFormat = &models.ResponseFormat{Type: "yaml"}
			},
			wantField: "response_format.type",
		},
		{
			name: "json schema without schema",
			configure: func(req *models.UnifiedRequest) {
				req.ResponseFormat = &models.ResponseFormat{Type: "json_schema"}
			},
			wantField: "response_format.json_schema.schema",
		},
		{
			name: "json schema is not an object",
			configure: func(req *models.UnifiedRequest) {
				req.ResponseFormat = &models.ResponseFormat{
					Type:       "json_schema",
					JSONSchema: &models.JSONSchemaResponseFormat{Schema: "not-an-object"},
				}
			},
			wantField: "response_format.json_schema.schema",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			request := &models.UnifiedRequest{
				Model:    "gemma3",
				Messages: []models.Message{{Role: "user", Content: "hi"}},
			}
			tt.configure(request)
			translator := NewOllamaTranslator(config.ProviderConfig{BaseURL: "http://localhost:11434"})
			_, err := translator.TranslateRequest(context.Background(), request)
			var compatibilityErr *models.CompatibilityError
			if !errors.As(err, &compatibilityErr) {
				t.Fatalf("error = %v, want CompatibilityError", err)
			}
			if compatibilityErr.Field != tt.wantField || compatibilityErr.Provider != "ollama" {
				t.Fatalf("compatibility error = %#v, want field=%q provider=ollama", compatibilityErr, tt.wantField)
			}
		})
	}
}

func TestOllamaTranslator_TranslateRequest_AllowsEquivalentExplicitDefaults(t *testing.T) {
	one := 1
	store := false
	translator := NewOllamaTranslator(config.ProviderConfig{BaseURL: "http://localhost:11434"})
	_, err := translator.TranslateRequest(context.Background(), &models.UnifiedRequest{
		Model:          "gemma3",
		Messages:       []models.Message{{Role: "user", Content: "hi"}},
		N:              &one,
		LogitBias:      map[string]int{},
		User:           "",
		Store:          &store,
		ResponseFormat: &models.ResponseFormat{Type: "text"},
	})
	if err != nil {
		t.Fatalf("equivalent defaults were rejected: %v", err)
	}
}

func TestOllamaTranslator_TranslateRequest_AllowsResponsesStoreHandledByGateway(t *testing.T) {
	store := true
	translator := NewOllamaTranslator(config.ProviderConfig{BaseURL: "http://localhost:11434"})
	_, err := translator.TranslateRequest(context.Background(), &models.UnifiedRequest{
		SourceRequestType: "responses",
		Model:             "gemma3",
		Messages:          []models.Message{{Role: "user", Content: "hi"}},
		Store:             &store,
	})
	if err != nil {
		t.Fatalf("Responses store handled by the gateway was rejected: %v", err)
	}
}

func TestOllamaTranslator_TranslateRequest_PreservesReasoningAndToolHistory(t *testing.T) {
	translator := NewOllamaTranslator(config.ProviderConfig{BaseURL: "http://localhost:11434"})
	index := 2
	req, err := translator.TranslateRequest(context.Background(), &models.UnifiedRequest{
		Model: "qwen3.5",
		Messages: []models.Message{
			{
				Role:             "assistant",
				Content:          "I will check.",
				ReasoningContent: "Need current weather.",
				ToolCalls: []models.ToolCall{{
					Index: &index,
					ID:    "call_weather_7",
					Type:  "function",
					Function: models.ToolCallFunction{
						Name:      "get_weather",
						Arguments: `{"city":"Warsaw","units":"celsius"}`,
					},
				}},
			},
			{
				Role:       "tool",
				Content:    `{"temperature":18}`,
				Name:       "get_weather",
				ToolCallID: "call_weather_7",
			},
		},
	})
	if err != nil {
		t.Fatalf("TranslateRequest returned error: %v", err)
	}

	messages := decodeOllamaMessages(t, decodeOllamaRequestBody(t, req.Body))
	if len(messages) != 2 {
		t.Fatalf("messages length = %d, want 2", len(messages))
	}
	assistant := messages[0]
	if assistant["role"] != "assistant" || assistant["content"] != "I will check." {
		t.Fatalf("assistant message = %#v", assistant)
	}
	if assistant["thinking"] != "Need current weather." {
		t.Fatalf("assistant thinking = %#v", assistant["thinking"])
	}
	if _, exists := assistant["reasoning"]; exists {
		t.Fatalf("non-native reasoning field must be omitted: %#v", assistant)
	}
	calls, ok := assistant["tool_calls"].([]interface{})
	if !ok || len(calls) != 1 {
		t.Fatalf("assistant tool_calls = %#v", assistant["tool_calls"])
	}
	call, _ := calls[0].(map[string]interface{})
	if call["id"] != "call_weather_7" {
		t.Fatalf("tool-call id = %#v", call["id"])
	}
	function, _ := call["function"].(map[string]interface{})
	if function["index"] != float64(index) || function["name"] != "get_weather" {
		t.Fatalf("tool-call function = %#v", function)
	}
	arguments, _ := function["arguments"].(map[string]interface{})
	if arguments["city"] != "Warsaw" || arguments["units"] != "celsius" {
		t.Fatalf("tool-call arguments = %#v", arguments)
	}

	tool := messages[1]
	if tool["role"] != "tool" || tool["tool_name"] != "get_weather" || tool["tool_call_id"] != "call_weather_7" {
		t.Fatalf("tool-result message = %#v", tool)
	}
	if tool["content"] != `{"temperature":18}` {
		t.Fatalf("tool-result content = %#v", tool["content"])
	}
}

func TestOllamaTranslator_TranslateRequest_MapsInlineImageParts(t *testing.T) {
	const image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
	translator := NewOllamaTranslator(config.ProviderConfig{BaseURL: "http://localhost:11434"})
	req, err := translator.TranslateRequest(context.Background(), &models.UnifiedRequest{
		Model: "gemma3",
		Messages: []models.Message{{
			Role: "user",
			Content: []interface{}{
				map[string]interface{}{"type": "text", "text": "look "},
				map[string]interface{}{
					"type": "image_url",
					"image_url": map[string]interface{}{
						"url":    "data:image/png;base64," + image,
						"detail": "auto",
					},
				},
				map[string]interface{}{"type": "input_text", "text": "here"},
				map[string]interface{}{"type": "input_image", "image_url": "data:image/png;base64," + image},
			},
		}},
	})
	if err != nil {
		t.Fatalf("TranslateRequest returned error: %v", err)
	}

	messages := decodeOllamaMessages(t, decodeOllamaRequestBody(t, req.Body))
	if got := messages[0]["content"]; got != "look here" {
		t.Fatalf("content = %#v, want look here", got)
	}
	images, ok := messages[0]["images"].([]interface{})
	if !ok || len(images) != 2 || images[0] != image || images[1] != image {
		t.Fatalf("images = %#v, want two native base64 images", messages[0]["images"])
	}
}

func TestOllamaTranslator_TranslateRequest_RejectsLossyMessageFields(t *testing.T) {
	tests := []struct {
		name      string
		message   models.Message
		wantField string
	}{
		{name: "unsupported role", message: models.Message{Role: "developer", Content: "hi"}, wantField: "messages[0].role"},
		{name: "participant name", message: models.Message{Role: "user", Content: "hi", Name: "alice"}, wantField: "messages[0].name"},
		{name: "tool id on user", message: models.Message{Role: "user", Content: "hi", ToolCallID: "call_1"}, wantField: "messages[0].tool_call_id"},
		{name: "reasoning on user", message: models.Message{Role: "user", Content: "hi", ReasoningContent: "secret"}, wantField: "messages[0].reasoning_content"},
		{
			name: "tool calls on user",
			message: models.Message{Role: "user", ToolCalls: []models.ToolCall{{
				Function: models.ToolCallFunction{Name: "lookup", Arguments: `{}`},
			}}},
			wantField: "messages[0].tool_calls",
		},
		{name: "legacy function call", message: models.Message{Role: "assistant", FunctionCall: &models.FunctionCall{Name: "lookup", Arguments: `{}`}}, wantField: "messages[0].function_call"},
		{name: "scalar content", message: models.Message{Role: "user", Content: float64(42)}, wantField: "messages[0].content"},
		{name: "non-object part", message: models.Message{Role: "user", Content: []interface{}{"hi"}}, wantField: "messages[0].content[0]"},
		{name: "missing part type", message: models.Message{Role: "user", Content: []interface{}{map[string]interface{}{"text": "hi"}}}, wantField: "messages[0].content[0].type"},
		{
			name: "unsupported text metadata",
			message: models.Message{Role: "user", Content: []interface{}{map[string]interface{}{
				"type": "text", "text": "hi", "annotations": []interface{}{},
			}}},
			wantField: "messages[0].content[0].annotations",
		},
		{
			name: "remote image URL",
			message: models.Message{Role: "user", Content: []interface{}{map[string]interface{}{
				"type": "image_url", "image_url": map[string]interface{}{"url": "https://example.com/image.png"},
			}}},
			wantField: "messages[0].content[0].image_url.url",
		},
		{
			name: "image detail",
			message: models.Message{Role: "user", Content: []interface{}{map[string]interface{}{
				"type": "image_url", "image_url": map[string]interface{}{"url": "data:image/png;base64,aQ==", "detail": "high"},
			}}},
			wantField: "messages[0].content[0].image_url.detail",
		},
		{
			name: "image file",
			message: models.Message{Role: "user", Content: []interface{}{map[string]interface{}{
				"type": "input_image", "file_id": "file_123",
			}}},
			wantField: "messages[0].content[0].file_id",
		},
		{
			name: "malformed image data",
			message: models.Message{Role: "user", Content: []interface{}{map[string]interface{}{
				"type": "input_image", "image_url": "data:image/png;base64,%%%",
			}}},
			wantField: "messages[0].content[0].image_url",
		},
		{
			name: "audio part",
			message: models.Message{Role: "user", Content: []interface{}{map[string]interface{}{
				"type": "input_audio", "input_audio": map[string]interface{}{"data": "..."},
			}}},
			wantField: "messages[0].content[0].type",
		},
		{
			name: "unsupported tool-call type",
			message: models.Message{Role: "assistant", ToolCalls: []models.ToolCall{{
				Type: "custom", Function: models.ToolCallFunction{Name: "lookup", Arguments: `{}`},
			}}},
			wantField: "messages[0].tool_calls[0].type",
		},
		{
			name: "missing tool name",
			message: models.Message{Role: "assistant", ToolCalls: []models.ToolCall{{
				Type: "function", Function: models.ToolCallFunction{Arguments: `{}`},
			}}},
			wantField: "messages[0].tool_calls[0].function.name",
		},
		{
			name: "non-object tool arguments",
			message: models.Message{Role: "assistant", ToolCalls: []models.ToolCall{{
				Type: "function", Function: models.ToolCallFunction{Name: "lookup", Arguments: `[]`},
			}}},
			wantField: "messages[0].tool_calls[0].function.arguments",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			translator := NewOllamaTranslator(config.ProviderConfig{BaseURL: "http://localhost:11434"})
			_, err := translator.TranslateRequest(context.Background(), &models.UnifiedRequest{
				Model:    "gemma3",
				Messages: []models.Message{tt.message},
			})
			var compatibilityErr *models.CompatibilityError
			if !errors.As(err, &compatibilityErr) {
				t.Fatalf("error = %v, want CompatibilityError", err)
			}
			if compatibilityErr.Field != tt.wantField || compatibilityErr.Provider != "ollama" {
				t.Fatalf("compatibility error = %#v, want field=%q provider=ollama", compatibilityErr, tt.wantField)
			}
		})
	}
}

func TestOllamaTranslator_ValidateRequestCompatibility_UsesTargetProviderForMessageError(t *testing.T) {
	translator := NewOllamaTranslator(config.ProviderConfig{BaseURL: "http://localhost:11434"})
	err := translator.ValidateRequestCompatibility("ollama-backup", &models.UnifiedRequest{
		Messages: []models.Message{{Role: "user", Content: []interface{}{map[string]interface{}{
			"type": "image_url", "image_url": map[string]interface{}{"url": "https://example.com/image.png"},
		}}}},
	})
	var compatibilityErr *models.CompatibilityError
	if !errors.As(err, &compatibilityErr) {
		t.Fatalf("error = %v, want CompatibilityError", err)
	}
	if compatibilityErr.Provider != "ollama-backup" || compatibilityErr.Field != "messages[0].content[0].image_url.url" {
		t.Fatalf("compatibility error = %#v", compatibilityErr)
	}
}

func TestOllamaTranslator_ToolCallResponseRoundTripPreservesNativeMessageFields(t *testing.T) {
	translator := NewOllamaTranslator(config.ProviderConfig{BaseURL: "http://localhost:11434"})
	parsed, err := translator.ParseResponse(&http.Response{
		StatusCode: http.StatusOK,
		Body: io.NopCloser(strings.NewReader(`{
			"model":"qwen3.5",
			"message":{
				"role":"assistant",
				"content":"",
				"thinking":"Need a lookup.",
				"tool_calls":[{"id":"call_native_9","function":{"index":4,"name":"lookup","arguments":{"q":"moon"}}}]
			},
			"done":true,
			"done_reason":"tool_calls"
		}`)),
	})
	if err != nil {
		t.Fatalf("ParseResponse returned error: %v", err)
	}
	assistant := parsed.Choices[0].Message
	if assistant == nil || assistant.ReasoningContent != "Need a lookup." || len(assistant.ToolCalls) != 1 {
		t.Fatalf("parsed assistant = %#v", assistant)
	}
	call := assistant.ToolCalls[0]
	if call.ID != "call_native_9" || call.Index == nil || *call.Index != 4 || call.Function.Name != "lookup" || call.Function.Arguments != `{"q":"moon"}` {
		t.Fatalf("parsed tool call = %#v", call)
	}

	request, err := translator.TranslateRequest(context.Background(), &models.UnifiedRequest{
		Model: "qwen3.5",
		Messages: []models.Message{
			*assistant,
			{Role: "tool", Content: `{"result":"full"}`, Name: "lookup", ToolCallID: call.ID},
		},
	})
	if err != nil {
		t.Fatalf("round-trip TranslateRequest returned error: %v", err)
	}
	messages := decodeOllamaMessages(t, decodeOllamaRequestBody(t, request.Body))
	if messages[0]["thinking"] != "Need a lookup." {
		t.Fatalf("round-trip thinking = %#v", messages[0]["thinking"])
	}
	calls := messages[0]["tool_calls"].([]interface{})
	roundTrippedCall := calls[0].(map[string]interface{})
	function := roundTrippedCall["function"].(map[string]interface{})
	if roundTrippedCall["id"] != "call_native_9" || function["index"] != float64(4) {
		t.Fatalf("round-trip tool call = %#v", roundTrippedCall)
	}
	if messages[1]["tool_call_id"] != "call_native_9" || messages[1]["tool_name"] != "lookup" {
		t.Fatalf("round-trip tool result = %#v", messages[1])
	}
}

func TestOllamaStreamTranslator_PreservesNativeToolCallIdentity(t *testing.T) {
	translator := NewOllamaStreamTranslator(NewOllamaTranslator(config.ProviderConfig{BaseURL: "http://localhost:11434"}))
	chunk, err := translator.ParseStreamChunk([]byte(`{
		"model":"qwen3.5",
		"message":{
			"role":"assistant",
			"thinking":"checking",
			"tool_calls":[{"id":"call_stream_3","function":{"index":3,"name":"lookup","arguments":{"q":"sun"}}}]
		},
		"done":false
	}`))
	if err != nil {
		t.Fatalf("ParseStreamChunk returned error: %v", err)
	}
	delta := chunk.Choices[0].Delta
	if delta == nil || delta.ReasoningContent != "checking" || len(delta.ToolCalls) != 1 {
		t.Fatalf("stream delta = %#v", delta)
	}
	call := delta.ToolCalls[0]
	if call.ID != "call_stream_3" || call.Index == nil || *call.Index != 3 || call.Function.Name != "lookup" || call.Function.Arguments != `{"q":"sun"}` {
		t.Fatalf("stream tool call = %#v", call)
	}
}

func TestOllamaTranslator_TranslateRequest_UsesProviderDefaultSamplingOptions(t *testing.T) {
	defaultTemperature := 1.0
	defaultTopP := 0.95
	defaultTopK := 64
	translator := NewOllamaTranslator(config.ProviderConfig{
		BaseURL:      "http://localhost:11434",
		DefaultModel: "gemma3",
		Temperature:  &defaultTemperature,
		TopP:         &defaultTopP,
		TopK:         &defaultTopK,
	})

	req, err := translator.TranslateRequest(context.Background(), &models.UnifiedRequest{
		Model:    "gemma3",
		Messages: []models.Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("TranslateRequest returned error: %v", err)
	}

	payload := decodeOllamaRequestBody(t, req.Body)
	options := decodeOllamaOptions(t, payload)

	if got, ok := options["temperature"].(float64); !ok || got != 1.0 {
		t.Fatalf("expected provider default options.temperature=1.0, got %#v", options["temperature"])
	}
	if got, ok := options["top_p"].(float64); !ok || got != 0.95 {
		t.Fatalf("expected provider default options.top_p=0.95, got %#v", options["top_p"])
	}
	if got, ok := options["top_k"].(float64); !ok || got != 64 {
		t.Fatalf("expected provider default options.top_k=64, got %#v", options["top_k"])
	}
}

func TestOllamaTranslator_TranslateRequest_KeepsUpstreamStreamingWhenToolsArePresent(t *testing.T) {
	translator := NewOllamaTranslator(config.ProviderConfig{
		BaseURL:      "http://localhost:11434",
		DefaultModel: "gemma3",
	})

	req, err := translator.TranslateRequest(context.Background(), &models.UnifiedRequest{
		Model:    "gemma3",
		Stream:   true,
		Messages: []models.Message{{Role: "user", Content: "what is the weather?"}},
		Tools: []models.Tool{
			{
				Type: "function",
				Function: models.ToolFunction{
					Name:        "get_weather",
					Description: "Get weather by city",
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("TranslateRequest returned error: %v", err)
	}

	payload := decodeOllamaRequestBody(t, req.Body)

	stream, ok := payload["stream"].(bool)
	if !ok {
		t.Fatalf("expected boolean stream flag, got %#v", payload["stream"])
	}
	if !stream {
		t.Fatalf("expected upstream stream=true when tools are present, got false")
	}
}

func TestOllamaTranslator_TranslateRequest_KeepsUpstreamStreamingWithoutTools(t *testing.T) {
	translator := NewOllamaTranslator(config.ProviderConfig{
		BaseURL:      "http://localhost:11434",
		DefaultModel: "gemma3",
	})

	req, err := translator.TranslateRequest(context.Background(), &models.UnifiedRequest{
		Model:    "gemma3",
		Stream:   true,
		Messages: []models.Message{{Role: "user", Content: "hello"}},
	})
	if err != nil {
		t.Fatalf("TranslateRequest returned error: %v", err)
	}

	payload := decodeOllamaRequestBody(t, req.Body)

	stream, ok := payload["stream"].(bool)
	if !ok {
		t.Fatalf("expected boolean stream flag, got %#v", payload["stream"])
	}
	if !stream {
		t.Fatalf("expected upstream stream=true when no tools are present, got false")
	}
}

func TestOllamaTranslator_TranslateRequest_ToolChoiceNoneOmitsTools(t *testing.T) {
	translator := NewOllamaTranslator(config.ProviderConfig{
		BaseURL:      "http://localhost:11434",
		DefaultModel: "gemma3",
	})

	req, err := translator.TranslateRequest(context.Background(), &models.UnifiedRequest{
		Model:      "gemma3",
		Messages:   []models.Message{{Role: "user", Content: "hello"}},
		ToolChoice: "none",
		Tools: []models.Tool{{
			Type: "function",
			Function: models.ToolFunction{
				Name: "get_weather",
			},
		}},
	})
	if err != nil {
		t.Fatalf("TranslateRequest returned error: %v", err)
	}

	payload := decodeOllamaRequestBody(t, req.Body)
	if _, ok := payload["tools"]; ok {
		t.Fatalf("expected tools to be omitted for tool_choice=none, got %#v", payload["tools"])
	}
}

func TestOllamaTranslator_TranslateRequest_ToolChoiceRequiredAddsInstruction(t *testing.T) {
	translator := NewOllamaTranslator(config.ProviderConfig{
		BaseURL:      "http://localhost:11434",
		DefaultModel: "gemma3",
	})

	req, err := translator.TranslateRequest(context.Background(), &models.UnifiedRequest{
		Model:      "gemma3",
		ToolChoice: "required",
		Messages:   []models.Message{{Role: "system", Content: "You are helpful."}, {Role: "user", Content: "hello"}},
		Tools: []models.Tool{{
			Type: "function",
			Function: models.ToolFunction{
				Name: "get_weather",
			},
		}},
	})
	if err != nil {
		t.Fatalf("TranslateRequest returned error: %v", err)
	}

	payload := decodeOllamaRequestBody(t, req.Body)
	messages, ok := payload["messages"].([]interface{})
	if !ok || len(messages) == 0 {
		t.Fatalf("expected messages array, got %#v", payload["messages"])
	}
	first, _ := messages[0].(map[string]interface{})
	content, _ := first["content"].(string)
	if !strings.Contains(content, "You must call one of the available tools") {
		t.Fatalf("expected required tool instruction in system prompt, got %q", content)
	}
}

func TestOllamaTranslator_TranslateRequest_ToolChoiceFunctionFiltersToolsAndAddsInstruction(t *testing.T) {
	translator := NewOllamaTranslator(config.ProviderConfig{
		BaseURL:      "http://localhost:11434",
		DefaultModel: "gemma3",
	})

	req, err := translator.TranslateRequest(context.Background(), &models.UnifiedRequest{
		Model: "gemma3",
		ToolChoice: map[string]interface{}{
			"type": "function",
			"function": map[string]interface{}{
				"name": "exec_command",
			},
		},
		Messages: []models.Message{{Role: "user", Content: "hello"}},
		Tools: []models.Tool{
			{
				Type: "function",
				Function: models.ToolFunction{
					Name: "exec_command",
				},
			},
			{
				Type: "function",
				Function: models.ToolFunction{
					Name: "write_stdin",
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("TranslateRequest returned error: %v", err)
	}

	payload := decodeOllamaRequestBody(t, req.Body)
	tools, ok := payload["tools"].([]interface{})
	if !ok || len(tools) != 1 {
		t.Fatalf("expected exactly one filtered tool, got %#v", payload["tools"])
	}
	tool, _ := tools[0].(map[string]interface{})
	function, _ := tool["function"].(map[string]interface{})
	if got, _ := function["name"].(string); got != "exec_command" {
		t.Fatalf("expected filtered tool exec_command, got %q", got)
	}
	messages, _ := payload["messages"].([]interface{})
	first, _ := messages[0].(map[string]interface{})
	content, _ := first["content"].(string)
	if !strings.Contains(content, `You must call the function "exec_command"`) {
		t.Fatalf("expected forced tool instruction in system prompt, got %q", content)
	}
}

func TestOllamaTranslator_TranslateRequest_ToolChoiceFunctionUnknownToolReturnsProviderError(t *testing.T) {
	translator := NewOllamaTranslator(config.ProviderConfig{
		BaseURL:      "http://localhost:11434",
		DefaultModel: "gemma3",
	})

	_, err := translator.TranslateRequest(context.Background(), &models.UnifiedRequest{
		Model: "gemma3",
		ToolChoice: map[string]interface{}{
			"type": "function",
			"function": map[string]interface{}{
				"name": "does_not_exist",
			},
		},
		Messages: []models.Message{{Role: "user", Content: "hello"}},
		Tools: []models.Tool{{
			Type: "function",
			Function: models.ToolFunction{
				Name: "exec_command",
			},
		}},
	})
	if err == nil {
		t.Fatal("expected error for unknown forced tool, got nil")
	}

	var providerErr *ProviderError
	if !errors.As(err, &providerErr) {
		t.Fatalf("expected ProviderError, got %T", err)
	}
	if providerErr.StatusCode != 400 {
		t.Fatalf("expected status 400, got %d", providerErr.StatusCode)
	}
	if providerErr.Type != "invalid_request_error" {
		t.Fatalf("expected invalid_request_error, got %q", providerErr.Type)
	}
}

func TestOllamaTranslator_TranslateRequest_DebugLogRedactsRequestContent(t *testing.T) {
	var output bytes.Buffer
	previousLogger := log.Logger
	log.Logger = zerolog.New(&output).Level(zerolog.DebugLevel)
	t.Cleanup(func() {
		log.Logger = previousLogger
	})

	translator := NewOllamaTranslator(config.ProviderConfig{
		BaseURL:      "http://url-user:url-secret@localhost:11434",
		DefaultModel: "gemma3",
	})

	_, err := translator.TranslateRequest(context.Background(), &models.UnifiedRequest{
		Model: "gemma3",
		Messages: []models.Message{
			{Role: "system", Content: "system-secret-instruction"},
			{Role: "user", Content: "prompt-secret-content"},
			{Role: "tool", Content: "tool-result-secret-content"},
		},
		Tools: []models.Tool{{
			Type: "function",
			Function: models.ToolFunction{
				Name:        "safe_tool_name",
				Description: "tool-description-secret",
				Parameters: map[string]interface{}{
					"secret_default": "tool-schema-secret",
				},
			},
		}},
	})
	if err != nil {
		t.Fatalf("TranslateRequest returned error: %v", err)
	}

	logged := output.String()
	for _, sensitive := range []string{
		"system-secret-instruction",
		"prompt-secret-content",
		"tool-result-secret-content",
		"tool-description-secret",
		"tool-schema-secret",
		"url-user",
		"url-secret",
	} {
		if strings.Contains(logged, sensitive) {
			t.Fatalf("debug log exposed sensitive value %q: %s", sensitive, logged)
		}
	}

	var event map[string]interface{}
	if err := json.Unmarshal(output.Bytes(), &event); err != nil {
		t.Fatalf("failed to decode debug log: %v", err)
	}
	if _, ok := event["upstream_payload"]; ok {
		t.Fatalf("debug log must not contain upstream_payload: %#v", event)
	}
	if _, ok := event["upstream_url"]; ok {
		t.Fatalf("debug log must not contain an upstream URL: %#v", event)
	}
	if got := event["provider"]; got != "ollama" {
		t.Fatalf("expected provider metadata, got %#v", got)
	}
	if got := event["model"]; got != "gemma3" {
		t.Fatalf("expected model metadata, got %#v", got)
	}
	if got := event["messages_count"]; got != float64(3) {
		t.Fatalf("expected messages_count=3, got %#v", got)
	}
	if got := event["tools_count"]; got != float64(1) {
		t.Fatalf("expected tools_count=1, got %#v", got)
	}
	toolNames, ok := event["tool_names"].([]interface{})
	if !ok || len(toolNames) != 1 || toolNames[0] != "safe_tool_name" {
		t.Fatalf("expected safe tool-name metadata, got %#v", event["tool_names"])
	}
}
