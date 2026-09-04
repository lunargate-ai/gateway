package providers

import (
	"context"
	"errors"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestAnthropicTranslatorPreservesFunctionStrictness(t *testing.T) {
	for _, strict := range []bool{false, true} {
		t.Run(map[bool]string{false: "false", true: "true"}[strict], func(t *testing.T) {
			translator := NewAnthropicTranslator(config.ProviderConfig{APIKey: "dummy"})
			payload := translateAnthropicRequest(t, translator, &models.UnifiedRequest{
				Model:    "claude",
				Messages: []models.Message{{Role: "user", Content: "hi"}},
				Tools: []models.Tool{{Type: "function", Function: models.ToolFunction{
					Name:       "lookup",
					Parameters: map[string]interface{}{"type": "object"},
					Strict:     &strict,
				}}},
			})

			if len(payload.Tools) != 1 || payload.Tools[0].Strict == nil || *payload.Tools[0].Strict != strict {
				t.Fatalf("Anthropic tool strict = %#v, want %v", payload.Tools, strict)
			}
		})
	}
}

func TestOllamaTranslatorHandlesFunctionStrictnessFailClosed(t *testing.T) {
	strict := true
	translator := NewOllamaTranslator(config.ProviderConfig{})
	_, err := translator.TranslateRequest(context.Background(), &models.UnifiedRequest{
		Model:    "qwen",
		Messages: []models.Message{{Role: "user", Content: "hi"}},
		Tools: []models.Tool{{Type: "function", Function: models.ToolFunction{
			Name: "lookup", Strict: &strict,
		}}},
	})
	var compatibilityErr *models.CompatibilityError
	if !errors.As(err, &compatibilityErr) || compatibilityErr.Field != "tools[0].function.strict" {
		t.Fatalf("error = %#v, want tools[0].function.strict CompatibilityError", err)
	}

	strict = false
	req, err := translator.TranslateRequest(context.Background(), &models.UnifiedRequest{
		Model:    "qwen",
		Messages: []models.Message{{Role: "user", Content: "hi"}},
		Tools: []models.Tool{{Type: "function", Function: models.ToolFunction{
			Name: "lookup", Strict: &strict,
		}}},
	})
	if err != nil {
		t.Fatalf("strict=false should match Ollama default: %v", err)
	}
	payload := decodeOllamaRequestBody(t, req.Body)
	tools := payload["tools"].([]interface{})
	function := tools[0].(map[string]interface{})["function"].(map[string]interface{})
	if _, exists := function["strict"]; exists {
		t.Fatalf("unsupported strict=false leaked to Ollama payload: %#v", function)
	}
}

func TestTranslatedChatTargetsRejectUnknownTypedToolChoiceMembers(t *testing.T) {
	choice := map[string]interface{}{
		"type": "function",
		"function": map[string]interface{}{
			"name":   "lookup",
			"future": "x",
		},
	}
	validators := []struct {
		name     string
		validate func(string, *models.UnifiedRequest) error
	}{
		{name: "anthropic", validate: NewAnthropicTranslator(config.ProviderConfig{}).ValidateRequestCompatibility},
		{name: "ollama", validate: NewOllamaTranslator(config.ProviderConfig{}).ValidateRequestCompatibility},
	}
	for _, validator := range validators {
		t.Run(validator.name, func(t *testing.T) {
			err := validator.validate(validator.name, &models.UnifiedRequest{ToolChoice: choice})
			var compatibilityErr *models.CompatibilityError
			if !errors.As(err, &compatibilityErr) || compatibilityErr.Field != "tool_choice.function.future" {
				t.Fatalf("error = %#v, want tool_choice.function.future CompatibilityError", err)
			}
		})
	}
}
