package providers

import (
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
)

func TestRegistry_UpdateProvidersConfig_RebuildsTranslatorState(t *testing.T) {
	reg := NewRegistry(map[string]config.ProviderConfig{
		"openai": {
			Type:         "openai",
			BaseURL:      "https://old.example/v1",
			DefaultModel: "old-model",
		},
	})

	if ok := reg.UpdateProvidersConfig(map[string]config.ProviderConfig{
		"openai": {
			Type:         "openai",
			BaseURL:      "https://new.example/v1",
			DefaultModel: "new-model",
		},
	}); !ok {
		t.Fatalf("expected provider registry update to succeed")
	}

	translatorAny, ok := reg.Get("openai")
	if !ok {
		t.Fatalf("expected openai translator to remain registered")
	}
	translator, ok := translatorAny.(*OpenAITranslator)
	if !ok {
		t.Fatalf("expected OpenAI translator, got %T", translatorAny)
	}
	if got := translator.BaseURL(); got != "https://new.example/v1" {
		t.Fatalf("expected updated base URL, got %q", got)
	}
	if got := translator.DefaultModel(); got != "new-model" {
		t.Fatalf("expected updated default model, got %q", got)
	}
}

func TestRegistry_UpdateProvidersConfig_PreservesExistingRegistryOnInvalidReload(t *testing.T) {
	reg := NewRegistry(map[string]config.ProviderConfig{
		"openai": {
			Type:         "openai",
			BaseURL:      "https://stable.example/v1",
			DefaultModel: "stable-model",
		},
	})

	if ok := reg.UpdateProvidersConfig(map[string]config.ProviderConfig{
		"broken": {
			BaseURL: "https://broken.example/v1",
		},
	}); ok {
		t.Fatalf("expected invalid provider reload to be rejected")
	}

	translatorAny, ok := reg.Get("openai")
	if !ok {
		t.Fatalf("expected existing registry to be preserved")
	}
	translator := translatorAny.(*OpenAITranslator)
	if got := translator.BaseURL(); got != "https://stable.example/v1" {
		t.Fatalf("expected original base URL to remain, got %q", got)
	}
}

func TestRegistry_RegistersCustomOpenAICompatibleProvider(t *testing.T) {
	reg := NewRegistry(map[string]config.ProviderConfig{
		"deepseek": {
			Type:                 "openai",
			APIKey:               "dummy",
			BaseURL:              "https://api.deepseek.com/v1",
			CompatibilityProfile: "deepseek",
		},
	})

	providerType, ok := reg.Type("deepseek")
	if !ok {
		t.Fatalf("expected deepseek provider type to be registered")
	}
	if providerType != "openai" {
		t.Fatalf("expected provider type openai, got %q", providerType)
	}

	translatorAny, ok := reg.Get("deepseek")
	if !ok {
		t.Fatalf("expected deepseek translator to be registered")
	}
	translator, ok := translatorAny.(*OpenAITranslator)
	if !ok {
		t.Fatalf("expected custom provider to use OpenAI-compatible translator, got %T", translatorAny)
	}
	if got := translator.Name(); got != "openai" {
		t.Fatalf("expected translator name openai, got %q", got)
	}
	if got := translator.BaseURL(); got != "https://api.deepseek.com/v1" {
		t.Fatalf("expected deepseek default base URL, got %q", got)
	}
}
