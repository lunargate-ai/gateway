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

func TestAnthropicTranslatorUsesCurrentDefaultAndModelCatalog(t *testing.T) {
	translator := NewAnthropicTranslator(config.ProviderConfig{})
	if got := translator.DefaultModel(); got != "claude-sonnet-4-6" {
		t.Fatalf("default model = %q, want claude-sonnet-4-6", got)
	}

	want := map[string]bool{
		"claude-fable-5-1":          false,
		"claude-opus-5":             false,
		"claude-sonnet-5":           false,
		"claude-sonnet-4-6":         false,
		"claude-haiku-4-5-20251001": false,
	}
	retired := map[string]bool{
		"claude-3-opus-20240229":     true,
		"claude-3-sonnet-20240229":   true,
		"claude-3-haiku-20240307":    true,
		"claude-3-5-sonnet-20241022": true,
	}
	for _, model := range translator.Models() {
		if retired[model.ID] {
			t.Fatalf("retired model %q remains in catalog", model.ID)
		}
		if _, exists := want[model.ID]; exists {
			want[model.ID] = true
		}
	}
	for model, found := range want {
		if !found {
			t.Errorf("current model %q missing from catalog", model)
		}
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
			ReasoningEffort:       true,
			ReasoningEffortLevels: []string{"xhigh"},
			AdaptiveThinking:      true,
			StructuredOutputs:     true,
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

func TestAnthropicTranslator_MapsEffortWithoutFabricatingThinkingBudget(t *testing.T) {
	tests := []struct {
		name             string
		model            string
		adaptiveThinking bool
		wantThinking     bool
	}{
		{
			name:         "effort without thinking",
			model:        "claude-opus-4-5-20251101",
			wantThinking: false,
		},
		{
			name:             "adaptive thinking",
			model:            "claude-opus-4-8",
			adaptiveThinking: true,
			wantThinking:     true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			translator := NewAnthropicTranslator(config.ProviderConfig{
				APIKey: "dummy",
				Capabilities: config.ProviderCapabilities{
					ReasoningEffort:  true,
					AdaptiveThinking: tt.adaptiveThinking,
				},
			})
			maxTokens := 512
			req, err := translator.TranslateRequest(context.Background(), &models.UnifiedRequest{
				Model:           tt.model,
				MaxTokens:       &maxTokens,
				Messages:        []models.Message{{Role: "user", Content: "Solve this."}},
				ReasoningEffort: "medium",
			})
			if err != nil {
				t.Fatalf("TranslateRequest returned error: %v", err)
			}
			body, err := io.ReadAll(req.Body)
			if err != nil {
				t.Fatalf("read request body: %v", err)
			}
			if bytes.Contains(body, []byte("budget_tokens")) {
				t.Fatalf("request fabricated a manual thinking budget: %s", body)
			}
			var payload anthropicRequest
			if err := json.Unmarshal(body, &payload); err != nil {
				t.Fatalf("decode request body: %v", err)
			}
			if payload.OutputConfig == nil || payload.OutputConfig.Effort != "medium" {
				t.Fatalf("output_config = %#v, want effort=medium", payload.OutputConfig)
			}
			if tt.wantThinking {
				if payload.Thinking == nil || payload.Thinking.Type != "adaptive" {
					t.Fatalf("thinking = %#v, want adaptive", payload.Thinking)
				}
			} else if payload.Thinking != nil {
				t.Fatalf("thinking = %#v, want omitted", payload.Thinking)
			}
		})
	}
}

func TestAnthropicTranslator_RejectsEffortOutsideConfiguredContract(t *testing.T) {
	tests := []struct {
		name         string
		effort       string
		enabled      []string
		wantFragment string
	}{
		{name: "unknown level", effort: "minimal", wantFragment: "unsupported Anthropic effort level"},
		{name: "xhigh needs opt in", effort: "xhigh", wantFragment: "is not enabled for this provider"},
		{name: "max needs opt in", effort: "max", wantFragment: "is not enabled for this provider"},
		{name: "allowlist remains exact", effort: "max", enabled: []string{"xhigh"}, wantFragment: "is not enabled for this provider"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			translator := NewAnthropicTranslator(config.ProviderConfig{
				APIKey: "dummy",
				Capabilities: config.ProviderCapabilities{
					ReasoningEffort:       true,
					ReasoningEffortLevels: tt.enabled,
				},
			})
			err := translator.ValidateRequestCompatibility("anthropic-backup", &models.UnifiedRequest{ReasoningEffort: tt.effort})
			var compatibilityErr *models.CompatibilityError
			if !errors.As(err, &compatibilityErr) {
				t.Fatalf("error = %v, want CompatibilityError", err)
			}
			if compatibilityErr.Field != "reasoning_effort" || compatibilityErr.Provider != "anthropic-backup" {
				t.Fatalf("compatibility error = %#v", compatibilityErr)
			}
			if !strings.Contains(compatibilityErr.Reason, tt.wantFragment) {
				t.Fatalf("reason = %q, want substring %q", compatibilityErr.Reason, tt.wantFragment)
			}
		})
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

func TestAnthropicTranslatorSaturatesUsageTotal(t *testing.T) {
	maximum := int(^uint(0) >> 1)
	body, err := json.Marshal(map[string]interface{}{
		"id":          "msg_overflow",
		"type":        "message",
		"role":        "assistant",
		"content":     []interface{}{map[string]interface{}{"type": "text", "text": "ok"}},
		"model":       "claude-sonnet-4-6",
		"stop_reason": "end_turn",
		"usage":       map[string]int{"input_tokens": maximum, "output_tokens": maximum},
	})
	if err != nil {
		t.Fatalf("encode response: %v", err)
	}

	response, err := NewAnthropicTranslator(config.ProviderConfig{}).ParseResponse(&http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(bytes.NewReader(body)),
	})
	if err != nil {
		t.Fatalf("ParseResponse returned error: %v", err)
	}
	if response.Usage == nil || response.Usage.TotalTokens != maximum {
		t.Fatalf("usage = %#v, want total saturated to %d", response.Usage, maximum)
	}
}
