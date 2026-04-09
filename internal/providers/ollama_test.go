package providers

import (
	"context"
	"encoding/json"
	"io"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/pkg/models"
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

func TestOllamaTranslator_TranslateRequest_DisablesUpstreamStreamingWhenToolsArePresent(t *testing.T) {
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
	if stream {
		t.Fatalf("expected upstream stream=false when tools are present, got true")
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
