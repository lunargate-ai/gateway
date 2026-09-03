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

func TestAnthropicTranslator_MapsSupportedClientControls(t *testing.T) {
	translator := NewAnthropicTranslator(config.ProviderConfig{
		APIKey:  "dummy",
		BaseURL: "https://api.anthropic.com",
		Capabilities: config.ProviderCapabilities{
			ReasoningEffort:   true,
			StructuredOutputs: true,
		},
	})
	one := 1
	strict := true
	payload := translateAnthropicRequest(t, translator, &models.UnifiedRequest{
		Model:           "claude-opus-5",
		Messages:        []models.Message{{Role: "user", Content: "return JSON"}},
		N:               &one,
		User:            "customer-123",
		ReasoningEffort: "xhigh",
		Stop:            []interface{}{"END", "STOP"},
		ToolChoice:      "required",
		ResponseFormat: &models.ResponseFormat{
			Type: "json_schema",
			JSONSchema: &models.JSONSchemaResponseFormat{
				Name:   "answer",
				Schema: map[string]interface{}{"type": "object"},
				Strict: &strict,
			},
		},
	})

	if payload.Metadata == nil || payload.Metadata.UserID != "customer-123" {
		t.Fatalf("metadata = %#v, want mapped user_id", payload.Metadata)
	}
	if payload.Thinking == nil || payload.Thinking.Type != "adaptive" {
		t.Fatalf("thinking = %#v, want adaptive", payload.Thinking)
	}
	if payload.OutputConfig == nil || payload.OutputConfig.Effort != "xhigh" ||
		payload.OutputConfig.Format == nil || payload.OutputConfig.Format.Type != "json_schema" {
		t.Fatalf("output_config = %#v", payload.OutputConfig)
	}
	schema, ok := payload.OutputConfig.Format.Schema.(map[string]interface{})
	if !ok || schema["type"] != "object" {
		t.Fatalf("structured output schema = %#v", payload.OutputConfig.Format.Schema)
	}
	if len(payload.StopSequences) != 2 || payload.StopSequences[1] != "STOP" {
		t.Fatalf("stop_sequences = %#v", payload.StopSequences)
	}
	choice, ok := payload.ToolChoice.(map[string]interface{})
	if !ok || choice["type"] != "any" {
		t.Fatalf("tool_choice = %#v", payload.ToolChoice)
	}
}

func TestAnthropicTranslator_RejectsUnsupportedClientControls(t *testing.T) {
	two := 2
	zero := 0.0
	seed := 42
	store := true
	tests := []struct {
		name      string
		request   models.UnifiedRequest
		wantField string
	}{
		{name: "multiple choices", request: models.UnifiedRequest{N: &two}, wantField: "n"},
		{name: "presence penalty", request: models.UnifiedRequest{PresencePenalty: &zero}, wantField: "presence_penalty"},
		{name: "frequency penalty", request: models.UnifiedRequest{FrequencyPenalty: &zero}, wantField: "frequency_penalty"},
		{name: "logit bias", request: models.UnifiedRequest{LogitBias: map[string]int{}}, wantField: "logit_bias"},
		{name: "seed", request: models.UnifiedRequest{Seed: &seed}, wantField: "seed"},
		{name: "chat storage", request: models.UnifiedRequest{Store: &store}, wantField: "store"},
		{name: "invalid stop", request: models.UnifiedRequest{Stop: []interface{}{"END", 1}}, wantField: "stop"},
		{name: "invalid tool choice", request: models.UnifiedRequest{ToolChoice: "sometimes"}, wantField: "tool_choice"},
		{name: "reasoning without capability", request: models.UnifiedRequest{ReasoningEffort: "high"}, wantField: "reasoning_effort"},
		{name: "structured output without capability", request: models.UnifiedRequest{ResponseFormat: &models.ResponseFormat{Type: "json_schema", JSONSchema: &models.JSONSchemaResponseFormat{Schema: map[string]interface{}{"type": "object"}}}}, wantField: "response_format"},
		{name: "json object", request: models.UnifiedRequest{ResponseFormat: &models.ResponseFormat{Type: "json_object"}}, wantField: "response_format"},
	}

	translator := NewAnthropicTranslator(config.ProviderConfig{APIKey: "dummy"})
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := translator.ValidateRequestCompatibility("anthropic-backup", &tt.request)
			var compatibilityErr *models.CompatibilityError
			if !errors.As(err, &compatibilityErr) {
				t.Fatalf("error = %v, want CompatibilityError", err)
			}
			if compatibilityErr.Field != tt.wantField || compatibilityErr.Provider != "anthropic-backup" {
				t.Fatalf("compatibility error = %#v", compatibilityErr)
			}
		})
	}
}

func TestAnthropicTranslator_RejectsLossyMessageInput(t *testing.T) {
	tests := []struct {
		name      string
		message   models.Message
		wantField string
	}{
		{
			name:      "unknown role",
			message:   models.Message{Role: "critic", Content: "Review this."},
			wantField: "messages[0].role",
		},
		{
			name: "unsupported content part",
			message: models.Message{Role: "user", Content: []interface{}{
				map[string]interface{}{
					"type":        "input_audio",
					"input_audio": map[string]interface{}{"data": "...", "format": "wav"},
				},
			}},
			wantField: "messages[0].content[0].type",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			translator := NewAnthropicTranslator(config.ProviderConfig{APIKey: "dummy"})
			_, err := translator.TranslateRequest(context.Background(), &models.UnifiedRequest{
				Model:    "claude-sonnet-4-5",
				Messages: []models.Message{tt.message},
			})
			var compatibilityErr *models.CompatibilityError
			if !errors.As(err, &compatibilityErr) {
				t.Fatalf("error = %v, want CompatibilityError", err)
			}
			if compatibilityErr.Field != tt.wantField || compatibilityErr.Provider != "anthropic" {
				t.Fatalf("compatibility error = %#v, want field=%q provider=anthropic", compatibilityErr, tt.wantField)
			}
		})
	}
}

func TestAnthropicTranslator_AllowsNonStoringRequests(t *testing.T) {
	store := false
	one := 1
	translator := NewAnthropicTranslator(config.ProviderConfig{APIKey: "dummy"})
	err := translator.ValidateRequestCompatibility("anthropic", &models.UnifiedRequest{
		Store:          &store,
		N:              &one,
		ResponseFormat: &models.ResponseFormat{Type: "text"},
	})
	if err != nil {
		t.Fatalf("locally represented controls were rejected: %v", err)
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
