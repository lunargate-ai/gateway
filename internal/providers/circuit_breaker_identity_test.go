package providers

import (
	"strings"
	"testing"
	"time"

	"github.com/lunargate-ai/gateway/internal/config"
)

func TestRegistryCircuitBreakerIdentityTracksEffectiveAccount(t *testing.T) {
	base := config.ProviderConfig{
		Type:         "openai",
		APIKey:       "secret-old",
		BaseURL:      "https://api.example.com/v1/",
		DefaultModel: "model-old",
		Organization: "org-old",
	}
	registry := NewRegistry(map[string]config.ProviderConfig{"shared": base})
	initial := mustProviderCircuitBreakerKey(t, registry, "shared")
	for _, sensitive := range []string{"secret-old", "api.example.com", "org-old"} {
		if strings.Contains(initial, sensitive) {
			t.Fatalf("circuit-breaker key exposes %q", sensitive)
		}
	}

	identical := base
	if changed := registry.UpdateProvidersConfig(map[string]config.ProviderConfig{"shared": identical}); changed {
		t.Fatal("identical reload reported a provider change")
	}
	if got := mustProviderCircuitBreakerKey(t, registry, "shared"); got != initial {
		t.Fatal("no-op reload changed breaker identity")
	}

	unrelated := base
	unrelated.DefaultModel = "model-new"
	unrelated.Timeout = 45 * time.Second
	unrelated.Capabilities.StructuredOutputs = true
	if changed := registry.UpdateProvidersConfig(map[string]config.ProviderConfig{"shared": unrelated}); !changed {
		t.Fatal("real model/config reload was not applied")
	}
	if got := mustProviderCircuitBreakerKey(t, registry, "shared"); got != initial {
		t.Fatal("model-only/unrelated reload changed breaker identity")
	}

	changes := []struct {
		name   string
		mutate func(*config.ProviderConfig)
	}{
		{name: "endpoint", mutate: func(cfg *config.ProviderConfig) { cfg.BaseURL = "https://other.example.com/v1" }},
		{name: "organization", mutate: func(cfg *config.ProviderConfig) { cfg.Organization = "org-new" }},
		{name: "credential", mutate: func(cfg *config.ProviderConfig) { cfg.APIKey = "secret-new" }},
	}
	for _, testCase := range changes {
		t.Run(testCase.name, func(t *testing.T) {
			changed := base
			testCase.mutate(&changed)
			other := NewRegistry(map[string]config.ProviderConfig{"shared": changed})
			if got := mustProviderCircuitBreakerKey(t, other, "shared"); got == initial {
				t.Fatalf("%s change preserved breaker identity", testCase.name)
			}
		})
	}
}

func TestRegistryCircuitBreakerIdentityUsesEffectiveProviderContract(t *testing.T) {
	account := config.ProviderConfig{
		Type:    "anthropic",
		APIKey:  "secret",
		BaseURL: "https://api.anthropic.com/",
	}
	defaultVersion := NewRegistry(map[string]config.ProviderConfig{"primary": account})
	defaultKey := mustProviderCircuitBreakerKey(t, defaultVersion, "primary")

	explicit := account
	explicit.APIVersion = anthropicDefaultAPIVersion
	explicitVersion := NewRegistry(map[string]config.ProviderConfig{"primary": explicit})
	if got := mustProviderCircuitBreakerKey(t, explicitVersion, "primary"); got != defaultKey {
		t.Fatal("implicit and explicit default API versions differ")
	}

	newVersion := account
	newVersion.APIVersion = "2024-01-01"
	if got := mustProviderCircuitBreakerKey(t, NewRegistry(map[string]config.ProviderConfig{"primary": newVersion}), "primary"); got == defaultKey {
		t.Fatal("Anthropic API version change preserved breaker identity")
	}

	aliasRegistry := NewRegistry(map[string]config.ProviderConfig{
		"primary":  account,
		"fallback": account,
	})
	if mustProviderCircuitBreakerKey(t, aliasRegistry, "primary") == mustProviderCircuitBreakerKey(t, aliasRegistry, "fallback") {
		t.Fatal("different provider aliases share breaker identity")
	}

	openAI := config.ProviderConfig{Type: "openai", APIKey: "secret", APIVersion: "ignored-a"}
	openAIKey := mustProviderCircuitBreakerKey(t, NewRegistry(map[string]config.ProviderConfig{"openai": openAI}), "openai")
	openAI.APIVersion = "ignored-b"
	if got := mustProviderCircuitBreakerKey(t, NewRegistry(map[string]config.ProviderConfig{"openai": openAI}), "openai"); got != openAIKey {
		t.Fatal("unused OpenAI API version changed breaker identity")
	}

	ollama := config.ProviderConfig{Type: "ollama", APIKey: "unused-a", Organization: "unused-a"}
	ollamaKey := mustProviderCircuitBreakerKey(t, NewRegistry(map[string]config.ProviderConfig{"ollama": ollama}), "ollama")
	ollama.APIKey = "unused-b"
	ollama.Organization = "unused-b"
	if got := mustProviderCircuitBreakerKey(t, NewRegistry(map[string]config.ProviderConfig{"ollama": ollama}), "ollama"); got != ollamaKey {
		t.Fatal("unused Ollama credentials changed breaker identity")
	}
}

func mustProviderCircuitBreakerKey(t *testing.T, registry *Registry, provider string) string {
	t.Helper()
	snapshot, ok := registry.Snapshot(provider)
	if !ok {
		t.Fatalf("provider %q not found", provider)
	}
	key := snapshot.CircuitBreakerKey()
	if key == "" {
		t.Fatalf("provider %q has empty circuit-breaker key", provider)
	}
	return key
}
