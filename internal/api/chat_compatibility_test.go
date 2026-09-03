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
