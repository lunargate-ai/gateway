package providers

import (
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
)

func TestRegistry_UpdateProvidersConfig_RebuildsTranslatorState(t *testing.T) {
	initial := map[string]config.ProviderConfig{
		"openai": {
			Type:         "openai",
			BaseURL:      "https://old.example/v1",
			DefaultModel: "old-model",
		},
	}
	reg := NewRegistry(initial)
	before, ok := reg.Get("openai")
	if !ok {
		t.Fatal("expected initial openai translator")
	}

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
	if translatorAny == before {
		t.Fatal("real provider config change reused the previous translator")
	}
}

func TestRegistry_UpdateProvidersConfig_PreservesTranslatorOnIdenticalReload(t *testing.T) {
	temperature := 0.25
	configs := map[string]config.ProviderConfig{
		"openai": {
			Type:         "openai",
			APIKey:       "test-key",
			BaseURL:      "https://stable.example/v1",
			DefaultModel: "stable-model",
			Temperature:  &temperature,
			Extra:        map[string]string{"header": "value"},
			Models: config.ProviderModelsConfig{
				Mode:   "static",
				Static: []string{"stable-model", "other-model"},
			},
			Capabilities: config.ProviderCapabilities{
				ResponsesLifecycle:    true,
				ReasoningEffortLevels: []string{"low", "high"},
				HostedTools:           []string{"web_search"},
			},
		},
	}
	reg := NewRegistry(configs)
	before, ok := reg.Get("openai")
	if !ok {
		t.Fatal("expected initial openai translator")
	}

	reloadedTemperature := 0.25
	identical := map[string]config.ProviderConfig{
		"openai": {
			Type:         "openai",
			APIKey:       "test-key",
			BaseURL:      "https://stable.example/v1",
			DefaultModel: "stable-model",
			Temperature:  &reloadedTemperature,
			Extra:        map[string]string{"header": "value"},
			Models: config.ProviderModelsConfig{
				Mode:   "static",
				Static: []string{"stable-model", "other-model"},
			},
			Capabilities: config.ProviderCapabilities{
				ResponsesLifecycle:    true,
				ReasoningEffortLevels: []string{"low", "high"},
				HostedTools:           []string{"web_search"},
			},
		},
	}
	if changed := reg.UpdateProvidersConfig(identical); changed {
		t.Fatal("identical provider reload reported a change")
	}
	after, ok := reg.Get("openai")
	if !ok {
		t.Fatal("expected openai translator after identical reload")
	}
	if after != before {
		t.Fatal("identical provider reload rebuilt the translator")
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

func TestRegistry_CapabilitiesAreExplicitAndCopied(t *testing.T) {
	reg := NewRegistry(map[string]config.ProviderConfig{
		"openai": {
			Type: "openai",
			Capabilities: config.ProviderCapabilities{
				ResponsesLifecycle:    true,
				ReasoningEffortLevels: []string{"low", "xhigh"},
				HostedTools:           []string{"web_search"},
			},
		},
	})

	capabilities, ok := reg.Capabilities("openai")
	if !ok {
		t.Fatal("expected provider capabilities")
	}
	if !capabilities.ResponsesLifecycle {
		t.Fatal("responses lifecycle capability was not preserved")
	}
	capabilities.HostedTools[0] = "mutated"
	capabilities.ReasoningEffortLevels[0] = "mutated"

	again, ok := reg.Capabilities("openai")
	if !ok || len(again.HostedTools) != 1 || again.HostedTools[0] != "web_search" {
		t.Fatalf("registry capability slice was aliased: %#v", again.HostedTools)
	}
	if len(again.ReasoningEffortLevels) != 2 || again.ReasoningEffortLevels[0] != "low" {
		t.Fatalf("registry reasoning effort levels were aliased: %#v", again.ReasoningEffortLevels)
	}

	if missing, ok := reg.Capabilities("missing"); ok || missing.ResponsesLifecycle || len(missing.HostedTools) != 0 {
		t.Fatalf("missing provider capabilities = %#v, %v", missing, ok)
	}
}
