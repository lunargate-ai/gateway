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

	err := handler.validateChatCompatibility("openai-primary", &models.UnifiedRequest{
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
		if err := handler.validateChatCompatibility(providerID, req); err != nil {
			t.Fatalf("provider %s rejected top_k: %v", providerID, err)
		}
	}
}

func TestValidateChatCompatibilityIgnoresConfiguredTopKDefault(t *testing.T) {
	handler := &Handler{registry: providers.NewRegistry(map[string]config.ProviderConfig{
		"openai-primary": {Type: "openai"},
	})}
	topK := 20
	if err := handler.validateChatCompatibility("openai-primary", &models.UnifiedRequest{
		RawJSON: json.RawMessage(`{"model":"gpt-5.4","messages":[]}`),
		TopK:    &topK,
	}); err != nil {
		t.Fatalf("provider default should not be treated as a client field: %v", err)
	}
}
