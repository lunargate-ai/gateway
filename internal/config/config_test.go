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
	t.Setenv("GATEWAY_ID", "gw_test")
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
  gateway_id: "${GATEWAY_ID}"
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
	if cfg.DataSharing.GatewayID != "gw_test" {
		t.Fatalf("data_sharing gateway_id = %q, want %q", cfg.DataSharing.GatewayID, "gw_test")
	}
	if cfg.DataSharing.APIKey != "lgw_test" {
		t.Fatalf("data_sharing api_key = %q, want %q", cfg.DataSharing.APIKey, "lgw_test")
	}
}
