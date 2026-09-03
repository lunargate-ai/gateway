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
	defaultTopK := 64
	translator := NewAnthropicTranslator(config.ProviderConfig{
		APIKey:      "dummy",
		BaseURL:     "https://api.anthropic.com",
		APIVersion:  "2023-06-01",
		Temperature: &defaultTemperature,
		TopP:        &defaultTopP,
		TopK:        &defaultTopK,
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
	if payload.TopK == nil || *payload.TopK != 64 {
		t.Fatalf("expected top_k=64 in upstream payload, got %#v", payload.TopK)
	}
}

func TestAnthropicTranslator_PreservesDeveloperInstruction(t *testing.T) {
	translator := NewAnthropicTranslator(config.ProviderConfig{
		APIKey:  "dummy",
		BaseURL: "https://api.anthropic.com",
	})

	payload := translateAnthropicRequest(t, translator, &models.UnifiedRequest{
		Model: "claude-sonnet-4-5",
		Messages: []models.Message{
			{Role: "developer", Content: "Follow the repository rules."},
			{Role: "user", Content: "Fix the bug."},
		},
	})

	assertAnthropicSystemText(t, payload.System, []string{"Follow the repository rules."})
	if len(payload.Messages) != 1 || payload.Messages[0].Role != "user" {
		t.Fatalf("expected only the user message downstream, got %#v", payload.Messages)
	}
}

func TestAnthropicTranslator_PreservesSystemAndDeveloperSegmentOrder(t *testing.T) {
	translator := NewAnthropicTranslator(config.ProviderConfig{
		APIKey:  "dummy",
		BaseURL: "https://api.anthropic.com",
	})

	payload := translateAnthropicRequest(t, translator, &models.UnifiedRequest{
		Model: "claude-sonnet-4-5",
		Messages: []models.Message{
			{Role: "system", Content: "System first."},
			{Role: "developer", Content: []interface{}{
				map[string]interface{}{"type": "text", "text": "Developer segment one."},
				map[string]interface{}{"type": "input_text", "text": "Developer segment two."},
			}},
			{Role: "system", Content: "System last."},
			{Role: "user", Content: "Hello."},
		},
	})

	assertAnthropicSystemText(t, payload.System, []string{
		"System first.",
		"Developer segment one.",
		"Developer segment two.",
		"System last.",
	})
	if len(payload.Messages) != 1 || payload.Messages[0].Role != "user" {
		t.Fatalf("expected instruction roles to stay out of messages, got %#v", payload.Messages)
	}
}

func translateAnthropicRequest(
	t *testing.T,
	translator *AnthropicTranslator,
	unified *models.UnifiedRequest,
) anthropicRequest {
	t.Helper()

	req, err := translator.TranslateRequest(context.Background(), unified)
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
	return payload
}

func assertAnthropicSystemText(t *testing.T, blocks []anthropicContentBlock, want []string) {
	t.Helper()
	if len(blocks) != len(want) {
		t.Fatalf("expected %d system blocks, got %d: %#v", len(want), len(blocks), blocks)
	}
	for i, block := range blocks {
		if block.Type != "text" {
			t.Fatalf("expected system block %d to be text, got %q", i, block.Type)
		}
		if block.Text != want[i] {
			t.Fatalf("expected system block %d text %q, got %q", i, want[i], block.Text)
		}
	}
}
