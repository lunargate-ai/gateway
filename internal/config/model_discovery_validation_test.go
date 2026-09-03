package config

import (
	"strings"
	"testing"
	"time"
)

func TestValidateConfigCanonicalizesProviderModelDiscovery(t *testing.T) {
	cfg := validRuntimeConfig()
	provider := cfg.Providers["openai"]
	provider.Models = ProviderModelsConfig{
		Mode:  " FETCH ",
		Fetch: ModelsFetchConfig{},
	}
	cfg.Providers["openai"] = provider

	if err := validateConfig(cfg); err != nil {
		t.Fatalf("validateConfig returned error: %v", err)
	}
	modelsConfig := cfg.Providers["openai"].Models
	if got, want := modelsConfig.Mode, "fetch"; got != want {
		t.Fatalf("models.mode = %q, want %q", got, want)
	}
	if got, want := modelsConfig.Fetch.TTL, DefaultModelsFetchTTL; got != want {
		t.Fatalf("models.fetch.ttl = %s, want %s", got, want)
	}
}

func TestValidateConfigRejectsInvalidProviderModelDiscovery(t *testing.T) {
	tests := []struct {
		name     string
		provider string
		config   ProviderConfig
		wantErr  string
	}{
		{
			name:     "unknown mode",
			provider: "openai",
			config: ProviderConfig{
				Models: ProviderModelsConfig{Mode: "fecth"},
			},
			wantErr: `providers["openai"].models.mode`,
		},
		{
			name:     "negative fetch TTL",
			provider: "openai",
			config: ProviderConfig{
				Models: ProviderModelsConfig{Mode: "fetch", Fetch: ModelsFetchConfig{TTL: -time.Second}},
			},
			wantErr: `providers["openai"].models.fetch.ttl`,
		},
		{
			name:     "unsupported Anthropic fetch",
			provider: "anthropic",
			config: ProviderConfig{
				Type:   "anthropic",
				Models: ProviderModelsConfig{Mode: "fetch"},
			},
			wantErr: `providers["anthropic"].models.mode=fetch`,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cfg := validRuntimeConfig()
			cfg.Providers = map[string]ProviderConfig{test.provider: test.config}
			cfg.Routing.Routes[0].Targets[0].Provider = test.provider

			err := validateConfig(cfg)
			if err == nil {
				t.Fatal("validateConfig returned nil error")
			}
			if !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("validateConfig error = %q, want substring %q", err, test.wantErr)
			}
		})
	}
}

func TestValidateConfigAcceptsSupportedProviderModelDiscoveryModes(t *testing.T) {
	tests := []struct {
		name     string
		provider string
		config   ProviderConfig
	}{
		{name: "OpenAI fetch", provider: "openai", config: ProviderConfig{Models: ProviderModelsConfig{Mode: "fetch"}}},
		{name: "Ollama fetch", provider: "ollama", config: ProviderConfig{Models: ProviderModelsConfig{Mode: "fetch"}}},
		{name: "Anthropic translator", provider: "anthropic", config: ProviderConfig{Models: ProviderModelsConfig{Mode: "translator"}}},
		{name: "Anthropic static", provider: "anthropic", config: ProviderConfig{Models: ProviderModelsConfig{Mode: "static", Static: []string{"claude"}}}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cfg := validRuntimeConfig()
			cfg.Providers = map[string]ProviderConfig{test.provider: test.config}
			cfg.Routing.Routes[0].Targets[0].Provider = test.provider

			if err := validateConfig(cfg); err != nil {
				t.Fatalf("validateConfig returned error: %v", err)
			}
		})
	}
}
