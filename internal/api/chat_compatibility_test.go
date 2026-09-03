package api

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/internal/routing"
	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestValidateChatCompatibilityRejectsExplicitTopKForOpenAI(t *testing.T) {
	handler := &Handler{registry: providers.NewRegistry(map[string]config.ProviderConfig{
		"openai-primary": {Type: "openai"},
	})}

	err := handler.validateChatCompatibility(routing.Target{Provider: "openai-primary"}, &models.UnifiedRequest{
		RawJSON: json.RawMessage(`{"model":"gpt-5.4","messages":[],"top_k":20}`),
	})
	var compatibilityErr *models.CompatibilityError
	if !errors.As(err, &compatibilityErr) {
		t.Fatalf("error = %v, want CompatibilityError", err)
	}
	if compatibilityErr.Field != "top_k" || compatibilityErr.Provider != "openai-primary" {
		t.Fatalf("compatibility error = %#v", compatibilityErr)
	}
}

func TestCompatibleChatFallbacksDropsTargetsThatWouldChangeSemantics(t *testing.T) {
	handler := &Handler{registry: providers.NewRegistry(map[string]config.ProviderConfig{
		"primary-ollama": {Type: "ollama"},
		"backup-openai":  {Type: "openai"},
		"backup-ollama":  {Type: "ollama"},
	})}
	req := &models.UnifiedRequest{
		RawJSON: json.RawMessage(`{"model":"model","messages":[],"top_k":20}`),
	}
	fallbacks := []routing.Target{
		{Provider: "backup-openai", Model: "gpt-5.4"},
		{Provider: "backup-ollama", Model: "qwen3.5"},
	}

	got := handler.compatibleChatFallbacks(fallbacks, req)
	if len(got) != 1 || got[0].Provider != "backup-ollama" {
		t.Fatalf("compatible fallbacks = %#v", got)
	}
	if len(fallbacks) != 2 {
		t.Fatalf("input fallbacks were mutated: %#v", fallbacks)
	}
}

func TestValidateChatCompatibilityRejectsUnsupportedOllamaFieldsWithTargetID(t *testing.T) {
	handler := &Handler{registry: providers.NewRegistry(map[string]config.ProviderConfig{
		"local-ollama": {Type: "ollama"},
	})}
	two := 2
	store := true
	tests := []struct {
		name      string
		request   models.UnifiedRequest
		wantField string
	}{
		{name: "multiple choices", request: models.UnifiedRequest{N: &two}, wantField: "n"},
		{name: "logit bias", request: models.UnifiedRequest{LogitBias: map[string]int{"7": -10}}, wantField: "logit_bias"},
		{name: "user", request: models.UnifiedRequest{User: "customer-123"}, wantField: "user"},
		{name: "store", request: models.UnifiedRequest{Store: &store}, wantField: "store"},
		{
			name: "response format",
			request: models.UnifiedRequest{
				ResponseFormat: &models.ResponseFormat{Type: "xml"},
			},
			wantField: "response_format.type",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := handler.validateChatCompatibility(routing.Target{Provider: "local-ollama"}, &tt.request)
			var compatibilityErr *models.CompatibilityError
			if !errors.As(err, &compatibilityErr) {
				t.Fatalf("error = %v, want CompatibilityError", err)
			}
			if compatibilityErr.Field != tt.wantField || compatibilityErr.Provider != "local-ollama" {
				t.Fatalf("compatibility error = %#v, want field=%q provider=local-ollama", compatibilityErr, tt.wantField)
			}
		})
	}
}

func TestCompatibleChatFallbacksDropsOllamaForUnsupportedRequestSemantics(t *testing.T) {
	handler := &Handler{registry: providers.NewRegistry(map[string]config.ProviderConfig{
		"ollama-backup": {Type: "ollama"},
		"openai-backup": {Type: "openai"},
	})}
	two := 2
	fallbacks := []routing.Target{
		{Provider: "ollama-backup", Model: "qwen3.5"},
		{Provider: "openai-backup", Model: "gpt-5.4"},
	}

	got := handler.compatibleChatFallbacks(fallbacks, &models.UnifiedRequest{N: &two})
	if len(got) != 1 || got[0].Provider != "openai-backup" {
		t.Fatalf("compatible fallbacks = %#v, want only openai-backup", got)
	}
}

func TestCompatibleChatFallbacksKeepsOllamaForMappedGenerationControls(t *testing.T) {
	handler := &Handler{registry: providers.NewRegistry(map[string]config.ProviderConfig{
		"ollama-backup": {Type: "ollama"},
	})}
	presencePenalty := 0.25
	frequencyPenalty := -0.5
	seed := 42
	store := false
	one := 1
	req := &models.UnifiedRequest{
		N:                &one,
		Stop:             []interface{}{"END"},
		PresencePenalty:  &presencePenalty,
		FrequencyPenalty: &frequencyPenalty,
		Seed:             &seed,
		Store:            &store,
		ResponseFormat: &models.ResponseFormat{
			Type:       "json_schema",
			JSONSchema: &models.JSONSchemaResponseFormat{Schema: map[string]interface{}{"type": "object"}},
		},
	}
	fallbacks := []routing.Target{{Provider: "ollama-backup", Model: "qwen3.5"}}

	got := handler.compatibleChatFallbacks(fallbacks, req)
	if len(got) != 1 || got[0].Provider != "ollama-backup" {
		t.Fatalf("compatible fallbacks = %#v, want ollama-backup", got)
	}
}

func TestCompatibleChatFallbacksDropsOllamaForLossyMessageTranslation(t *testing.T) {
	handler := &Handler{registry: providers.NewRegistry(map[string]config.ProviderConfig{
		"ollama-backup": {Type: "ollama"},
		"openai-backup": {Type: "openai"},
	})}
	req := &models.UnifiedRequest{Messages: []models.Message{{
		Role: "user",
		Content: []interface{}{map[string]interface{}{
			"type": "image_url",
			"image_url": map[string]interface{}{
				"url": "https://example.com/private-image.png",
			},
		}},
	}}}
	fallbacks := []routing.Target{
		{Provider: "ollama-backup", Model: "gemma3"},
		{Provider: "openai-backup", Model: "gpt-5.4"},
	}

	got := handler.compatibleChatFallbacks(fallbacks, req)
	if len(got) != 1 || got[0].Provider != "openai-backup" {
		t.Fatalf("compatible fallbacks = %#v, want only openai-backup", got)
	}
}

func TestCompatibleChatFallbacksKeepsOllamaForNativeMessageHistory(t *testing.T) {
	const image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
	handler := &Handler{registry: providers.NewRegistry(map[string]config.ProviderConfig{
		"ollama-backup": {Type: "ollama"},
	})}
	index := 0
	req := &models.UnifiedRequest{Messages: []models.Message{
		{
			Role: "user",
			Content: []interface{}{
				map[string]interface{}{"type": "text", "text": "inspect"},
				map[string]interface{}{"type": "input_image", "image_url": "data:image/png;base64," + image},
			},
		},
		{
			Role:             "assistant",
			ReasoningContent: "Need a tool.",
			ToolCalls: []models.ToolCall{{
				Index: &index,
				ID:    "call_1",
				Type:  "function",
				Function: models.ToolCallFunction{
					Name:      "inspect",
					Arguments: `{}`,
				},
			}},
		},
		{Role: "tool", Content: "done", Name: "inspect", ToolCallID: "call_1"},
	}}

	got := handler.compatibleChatFallbacks([]routing.Target{{Provider: "ollama-backup", Model: "gemma3"}}, req)
	if len(got) != 1 || got[0].Provider != "ollama-backup" {
		t.Fatalf("compatible fallbacks = %#v, want ollama-backup", got)
	}
}

func TestValidateChatCompatibilityAllowsResponsesStoreHandledLocallyForOllama(t *testing.T) {
	handler := &Handler{registry: providers.NewRegistry(map[string]config.ProviderConfig{
		"local-ollama": {Type: "ollama"},
	})}
	store := true
	req := &models.UnifiedRequest{SourceRequestType: "responses", Store: &store}

	if err := handler.validateChatCompatibility(routing.Target{Provider: "local-ollama"}, req); err != nil {
		t.Fatalf("Responses store handled locally was rejected: %v", err)
	}
}

func TestValidateChatCompatibilityAllowsTopKForTranslatedProviders(t *testing.T) {
	handler := &Handler{registry: providers.NewRegistry(map[string]config.ProviderConfig{
		"anthropic-primary": {Type: "anthropic"},
		"ollama-local":      {Type: "ollama"},
	})}
	req := &models.UnifiedRequest{
		RawJSON: json.RawMessage(`{"model":"model","messages":[],"top_k":20}`),
	}
	for _, providerID := range []string{"anthropic-primary", "ollama-local"} {
		if err := handler.validateChatCompatibility(routing.Target{Provider: providerID}, req); err != nil {
			t.Fatalf("provider %s rejected top_k: %v", providerID, err)
		}
	}
}

func TestValidateChatCompatibilityIgnoresConfiguredTopKDefault(t *testing.T) {
	handler := &Handler{registry: providers.NewRegistry(map[string]config.ProviderConfig{
		"openai-primary": {Type: "openai"},
	})}
	topK := 20
	if err := handler.validateChatCompatibility(routing.Target{Provider: "openai-primary"}, &models.UnifiedRequest{
		RawJSON: json.RawMessage(`{"model":"gpt-5.4","messages":[]}`),
		TopK:    &topK,
	}); err != nil {
		t.Fatalf("provider default should not be treated as a client field: %v", err)
	}
}

func TestValidateChatCompatibilityRequiresHostedToolCapabilityAndResponsesTarget(t *testing.T) {
	handler := &Handler{registry: providers.NewRegistry(map[string]config.ProviderConfig{
		"disabled": {Type: "openai"},
		"enabled": {
			Type: "openai",
			Capabilities: config.ProviderCapabilities{
				HostedTools: []string{"web_search_preview"},
			},
		},
	})}
	req := &models.UnifiedRequest{
		RawJSON:           json.RawMessage(`{"model":"gpt-5.4","input":"hello","tools":[{"type":"web_search_preview"}]}`),
		SourceRequestType: "responses",
	}

	tests := []struct {
		name   string
		target routing.Target
		ok     bool
	}{
		{name: "missing capability", target: routing.Target{Provider: "disabled", UpstreamRequestType: "responses"}},
		{name: "translated chat", target: routing.Target{Provider: "enabled", UpstreamRequestType: "chat_completions"}},
		{name: "native responses", target: routing.Target{Provider: "enabled", UpstreamRequestType: "responses"}, ok: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := handler.validateChatCompatibility(tc.target, req)
			if tc.ok {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				return
			}
			var compatibilityErr *models.CompatibilityError
			if !errors.As(err, &compatibilityErr) {
				t.Fatalf("error = %v, want CompatibilityError", err)
			}
			if compatibilityErr.Field != "tools[0].type" {
				t.Fatalf("field = %q, want tools[0].type", compatibilityErr.Field)
			}
		})
	}
}

func TestCompatibleChatFallbacksFiltersHostedToolsPerTarget(t *testing.T) {
	handler := &Handler{registry: providers.NewRegistry(map[string]config.ProviderConfig{
		"chat":               {Type: "openai", Capabilities: config.ProviderCapabilities{HostedTools: []string{"web_search"}}},
		"responses-disabled": {Type: "openai"},
		"responses-enabled": {
			Type:         "openai",
			Capabilities: config.ProviderCapabilities{HostedTools: []string{"web_search"}},
		},
	})}
	req := &models.UnifiedRequest{
		RawJSON:           json.RawMessage(`{"model":"gpt-5","input":"hello","tools":[{"type":"web_search"}]}`),
		SourceRequestType: "responses",
	}
	fallbacks := []routing.Target{
		{Provider: "chat", UpstreamRequestType: "chat_completions"},
		{Provider: "responses-disabled", UpstreamRequestType: "responses"},
		{Provider: "responses-enabled", UpstreamRequestType: "responses"},
	}

	got := handler.compatibleChatFallbacks(fallbacks, req)
	if len(got) != 1 || got[0].Provider != "responses-enabled" {
		t.Fatalf("compatible fallbacks = %#v, want only responses-enabled", got)
	}
}

func TestValidateChatCompatibilityRequiresBackgroundResponsesCapability(t *testing.T) {
	handler := &Handler{registry: providers.NewRegistry(map[string]config.ProviderConfig{
		"disabled": {Type: "openai"},
		"enabled": {
			Type: "openai",
			Capabilities: config.ProviderCapabilities{
				BackgroundResponses: true,
			},
		},
	})}
	target := func(provider string) routing.Target {
		return routing.Target{Provider: provider, UpstreamRequestType: requestTypeResponses}
	}
	request := func(background string) *models.UnifiedRequest {
		return &models.UnifiedRequest{
			RawJSON:           json.RawMessage(`{"model":"gpt-5.4","input":"hello","background":` + background + `}`),
			SourceRequestType: requestTypeResponses,
		}
	}

	err := handler.validateChatCompatibility(target("disabled"), request("true"))
	var compatibilityErr *models.CompatibilityError
	if !errors.As(err, &compatibilityErr) || compatibilityErr.Field != "background" || compatibilityErr.Provider != "disabled" {
		t.Fatalf("compatibility error = %#v, want disabled background", err)
	}
	if err := handler.validateChatCompatibility(target("enabled"), request("true")); err != nil {
		t.Fatalf("enabled background Responses rejected: %v", err)
	}
	if err := handler.validateChatCompatibility(target("disabled"), request("false")); err != nil {
		t.Fatalf("background:false should not require capability: %v", err)
	}

	fallbacks := handler.compatibleChatFallbacks(
		[]routing.Target{target("disabled"), target("enabled")},
		request("true"),
	)
	if len(fallbacks) != 1 || fallbacks[0].Provider != "enabled" {
		t.Fatalf("compatible fallbacks = %#v, want only enabled", fallbacks)
	}
}
