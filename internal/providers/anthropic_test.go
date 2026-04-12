package providers

import (
	"context"
	"encoding/json"
	"io"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestAnthropicTranslator_UsesProviderDefaultSamplingOptions(t *testing.T) {
	defaultTemperature := 1.0
	defaultTopP := 0.95
	translator := NewAnthropicTranslator(config.ProviderConfig{
		APIKey:      "dummy",
		BaseURL:     "https://api.anthropic.com",
		APIVersion:  "2023-06-01",
		Temperature: &defaultTemperature,
		TopP:        &defaultTopP,
	})

	req, err := translator.TranslateRequest(context.Background(), &models.UnifiedRequest{
		Model:    "claude-sonnet-4-5",
		Messages: []models.Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("TranslateRequest returned error: %v", err)
	}

	body, err := io.ReadAll(req.Body)
	if err != nil {
		t.Fatalf("failed to read request body: %v", err)
	}

	var payload anthropicRequest
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("failed to unmarshal request payload: %v", err)
	}

	if payload.Temperature == nil || *payload.Temperature != 1.0 {
		t.Fatalf("expected temperature=1.0 in upstream payload, got %#v", payload.Temperature)
	}
	if payload.TopP == nil || *payload.TopP != 0.95 {
		t.Fatalf("expected top_p=0.95 in upstream payload, got %#v", payload.TopP)
	}
}
