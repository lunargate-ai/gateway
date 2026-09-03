package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
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
