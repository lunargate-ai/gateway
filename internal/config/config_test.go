package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestNewManagerExpandsEnvAcrossConfig(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "test-openai-key")
	t.Setenv("DEFAULT_MODEL", "gpt-5.2-mini")
	t.Setenv("LIGHT_MODEL", "gpt-5.2-nano")
	t.Setenv("HEAVY_MODEL", "gpt-5.2")
	t.Setenv("BACKEND_URL", "https://api.lunargate.ai/v1")
	t.Setenv("GATEWAY_API_KEY", "lgw_test")

	configPath := filepath.Join(t.TempDir(), "config.yaml")
	configBody := `providers:
  openai:
    api_key: "${OPENAI_API_KEY}"
    default_model: "${DEFAULT_MODEL}"
routing:
  routes:
    - name: "default"
      targets:
        - provider: openai
          model: "${LIGHT_MODEL}"
          weight: 100
      fallback:
        - provider: openai
          model: "${HEAVY_MODEL}"
          weight: 100
data_sharing:
  enabled: true
  backend_url: "${BACKEND_URL}/collector"
  api_key: "${GATEWAY_API_KEY}"
`
	if err := os.WriteFile(configPath, []byte(configBody), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	manager, err := NewManager(configPath)
	if err != nil {
		t.Fatalf("NewManager returned error: %v", err)
	}

	cfg := manager.Get()
	provider := cfg.Providers["openai"]
	if provider.APIKey != "test-openai-key" {
		t.Fatalf("provider api key = %q, want %q", provider.APIKey, "test-openai-key")
	}
	if provider.DefaultModel != "gpt-5.2-mini" {
		t.Fatalf("provider default model = %q, want %q", provider.DefaultModel, "gpt-5.2-mini")
	}

	route := cfg.Routing.Routes[0]
	if route.Targets[0].Model != "gpt-5.2-nano" {
		t.Fatalf("route target model = %q, want %q", route.Targets[0].Model, "gpt-5.2-nano")
	}
	if route.Fallback[0].Model != "gpt-5.2" {
		t.Fatalf("route fallback model = %q, want %q", route.Fallback[0].Model, "gpt-5.2")
	}

	if cfg.DataSharing.BackendURL != "https://api.lunargate.ai/v1" {
		t.Fatalf("data_sharing backend_url = %q, want %q", cfg.DataSharing.BackendURL, "https://api.lunargate.ai/v1")
	}
	if cfg.DataSharing.APIKey != "lgw_test" {
		t.Fatalf("data_sharing api_key = %q, want %q", cfg.DataSharing.APIKey, "lgw_test")
	}
}

func TestNewManager_ParsesProviderCompatibilityFields(t *testing.T) {
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	configBody := `providers:
  deepseek:
    type: "openai"
    api_key: "test-key"
    base_url: "https://api.deepseek.com/v1"
    compatibility_profile: "deepseek"
    normalize_developer_role: true
routing:
  routes:
    - name: "default"
      targets:
        - provider: deepseek
`
	if err := os.WriteFile(configPath, []byte(configBody), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	manager, err := NewManager(configPath)
	if err != nil {
		t.Fatalf("NewManager returned error: %v", err)
	}

	provider := manager.Get().Providers["deepseek"]
	if provider.CompatibilityProfile != "deepseek" {
		t.Fatalf("provider compatibility_profile = %q, want %q", provider.CompatibilityProfile, "deepseek")
	}
	if !provider.NormalizeDeveloperRole {
		t.Fatalf("provider normalize_developer_role = false, want true")
	}
}

func TestNewManager_NormalizesSecurityAPIKeyConfig(t *testing.T) {
	t.Setenv("CLIENT_API_KEY", "lg_client_test")

	configPath := filepath.Join(t.TempDir(), "config.yaml")
	configBody := `providers:
  openai:
    api_key: "test-key"
routing:
  routes:
    - name: "default"
      targets:
        - provider: openai
security:
  enabled: true
  provider: "api_key"
  api_key:
    header: "Authorization"
    prefix: "Bearer"
    allow_x_api_key: true
    keys:
      - name: "dashboard"
        value: "${CLIENT_API_KEY}"
`
	if err := os.WriteFile(configPath, []byte(configBody), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	manager, err := NewManager(configPath)
	if err != nil {
		t.Fatalf("NewManager returned error: %v", err)
	}

	cfg := manager.Get()
	if !cfg.Security.Enabled {
		t.Fatalf("security.enabled = false, want true")
	}
	if cfg.Security.Provider != "api_key" {
		t.Fatalf("security.provider = %q, want %q", cfg.Security.Provider, "api_key")
	}
	if cfg.Security.APIKey.Header != "Authorization" {
		t.Fatalf("security.api_key.header = %q, want %q", cfg.Security.APIKey.Header, "Authorization")
	}
	if len(cfg.Security.APIKey.Keys) != 1 {
		t.Fatalf("security.api_key.keys length = %d, want 1", len(cfg.Security.APIKey.Keys))
	}
	if cfg.Security.APIKey.Keys[0].Value != "lg_client_test" {
		t.Fatalf("security.api_key.keys[0].value = %q, want %q", cfg.Security.APIKey.Keys[0].Value, "lg_client_test")
	}
}

func TestNewManager_LegacySecurityAPIKeysRemainSupported(t *testing.T) {
	t.Setenv("LEGACY_GATEWAY_KEY", "lg_legacy_test")

	configPath := filepath.Join(t.TempDir(), "config.yaml")
	configBody := `providers:
  openai:
    api_key: "test-key"
routing:
  routes:
    - name: "default"
      targets:
        - provider: openai
security:
  api_keys_enabled: true
  api_keys:
    - "${LEGACY_GATEWAY_KEY}"
`
	if err := os.WriteFile(configPath, []byte(configBody), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	manager, err := NewManager(configPath)
	if err != nil {
		t.Fatalf("NewManager returned error: %v", err)
	}

	cfg := manager.Get()
	if !cfg.Security.Enabled {
		t.Fatalf("security.enabled = false, want true")
	}
	if cfg.Security.Provider != "api_key" {
		t.Fatalf("security.provider = %q, want %q", cfg.Security.Provider, "api_key")
	}
	if len(cfg.Security.APIKey.Keys) != 1 {
		t.Fatalf("security.api_key.keys length = %d, want 1", len(cfg.Security.APIKey.Keys))
	}
	if cfg.Security.APIKey.Keys[0].Value != "lg_legacy_test" {
		t.Fatalf("security.api_key.keys[0].value = %q, want %q", cfg.Security.APIKey.Keys[0].Value, "lg_legacy_test")
	}
}
