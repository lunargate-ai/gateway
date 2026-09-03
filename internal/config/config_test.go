package config

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"
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
  api_key: "${GATEWAY_API_KEY}"
general:
  backend_url: "${BACKEND_URL}/collector"
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

	if cfg.General.BackendURL != "https://api.lunargate.ai/v1" {
		t.Fatalf("general.backend_url = %q, want %q", cfg.General.BackendURL, "https://api.lunargate.ai/v1")
	}
	if cfg.DataSharing.APIKey != "lgw_test" {
		t.Fatalf("data_sharing api_key = %q, want %q", cfg.DataSharing.APIKey, "lgw_test")
	}
	if cfg.General.APIKey != "lgw_test" {
		t.Fatalf("general.api_key = %q, want %q", cfg.General.APIKey, "lgw_test")
	}
}

func TestValidateConfigRejectsInvalidUpstreamRequestTypes(t *testing.T) {
	tests := []struct {
		name         string
		providerID   string
		providerType string
		requestType  string
		fallback     bool
		wantErr      bool
	}{
		{name: "openai responses", providerID: "custom", providerType: "openai", requestType: "responses"},
		{name: "built-in openai responses", providerID: "openai", requestType: "responses"},
		{name: "anthropic responses", providerID: "anthropic", providerType: "anthropic", requestType: "responses", wantErr: true},
		{name: "ollama fallback responses", providerID: "ollama", providerType: "ollama", requestType: "responses", fallback: true, wantErr: true},
		{name: "unknown protocol", providerID: "custom", providerType: "openai", requestType: "messages", wantErr: true},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			target := TargetConfig{Provider: test.providerID, UpstreamRequestType: test.requestType}
			route := RouteConfig{Name: "test"}
			if test.fallback {
				route.Fallback = []TargetConfig{target}
			} else {
				route.Targets = []TargetConfig{target}
			}
			cfg := &Config{
				Providers: map[string]ProviderConfig{
					test.providerID: {Type: test.providerType},
				},
				Routing: RoutingConfig{Routes: []RouteConfig{route}},
			}
			err := validateConfig(cfg)
			if (err != nil) != test.wantErr {
				t.Fatalf("validateConfig() error = %v, wantErr %v", err, test.wantErr)
			}
		})
	}
}

func TestNewManager_DefaultsUpdateChecksOn(t *testing.T) {
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	configBody := `providers:
  openai:
    api_key: "test-key"
routing:
  routes:
    - name: "default"
      targets:
        - provider: openai
`
	if err := os.WriteFile(configPath, []byte(configBody), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	manager, err := NewManager(configPath)
	if err != nil {
		t.Fatalf("NewManager returned error: %v", err)
	}

	updateCheck := manager.Get().UpdateCheck
	if !updateCheck.Enabled {
		t.Fatal("update_check.enabled = false, want true")
	}
	if updateCheck.Endpoint != defaultUpdateCheckURL {
		t.Fatalf("update_check.endpoint = %q, want %q", updateCheck.Endpoint, defaultUpdateCheckURL)
	}
	if updateCheck.Interval != defaultUpdateCheckPeriod {
		t.Fatalf("update_check.interval = %s, want %s", updateCheck.Interval, defaultUpdateCheckPeriod)
	}
	if updateCheck.Timeout != defaultUpdateCheckTimeout {
		t.Fatalf("update_check.timeout = %s, want %s", updateCheck.Timeout, defaultUpdateCheckTimeout)
	}
}

func TestNewManager_CanDisableAndOverrideUpdateCheck(t *testing.T) {
	t.Setenv("UPDATE_URL", "https://updates.example/latest")

	configPath := filepath.Join(t.TempDir(), "config.yaml")
	configBody := `providers:
  openai:
    api_key: "test-key"
routing:
  routes:
    - name: "default"
      targets:
        - provider: openai
update_check:
  enabled: false
  endpoint: "${UPDATE_URL}"
  interval: 12h
  timeout: 2s
`
	if err := os.WriteFile(configPath, []byte(configBody), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	manager, err := NewManager(configPath)
	if err != nil {
		t.Fatalf("NewManager returned error: %v", err)
	}

	updateCheck := manager.Get().UpdateCheck
	if updateCheck.Enabled {
		t.Fatal("update_check.enabled = true, want false")
	}
	if updateCheck.Endpoint != "https://updates.example/latest" {
		t.Fatalf("update_check.endpoint = %q", updateCheck.Endpoint)
	}
	if updateCheck.Interval != 12*time.Hour {
		t.Fatalf("update_check.interval = %s, want 12h", updateCheck.Interval)
	}
	if updateCheck.Timeout != 2*time.Second {
		t.Fatalf("update_check.timeout = %s, want 2s", updateCheck.Timeout)
	}
}

func TestNewManager_UsesGeneralAPIKeyWhenBothGeneralAndLegacyConfigured(t *testing.T) {
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	configBody := `providers:
  openai:
    api_key: "test-openai-key"
routing:
  routes:
    - name: "default"
      targets:
        - provider: openai
general:
  api_key: "lgw_from_general"
data_sharing:
  enabled: true
  api_key: "lgw_from_legacy"
`
	if err := os.WriteFile(configPath, []byte(configBody), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	manager, err := NewManager(configPath)
	if err != nil {
		t.Fatalf("NewManager returned error: %v", err)
	}

	cfg := manager.Get()
	if cfg.General.APIKey != "lgw_from_general" {
		t.Fatalf("general.api_key = %q, want %q", cfg.General.APIKey, "lgw_from_general")
	}
	if cfg.DataSharing.APIKey != "lgw_from_general" {
		t.Fatalf("data_sharing.api_key = %q, want %q", cfg.DataSharing.APIKey, "lgw_from_general")
	}
}

func TestNewManager_FallsBackToLegacyDataSharingConfig(t *testing.T) {
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	configBody := `providers:
  openai:
    api_key: "test-openai-key"
routing:
  routes:
    - name: "default"
      targets:
        - provider: openai
data_sharing:
  enabled: true
  api_key: "lgw_from_legacy"
  backend_url: "https://legacy.example/v1/collector"
`
	if err := os.WriteFile(configPath, []byte(configBody), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	manager, err := NewManager(configPath)
	if err != nil {
		t.Fatalf("NewManager returned error: %v", err)
	}

	cfg := manager.Get()
	if cfg.General.APIKey != "lgw_from_legacy" {
		t.Fatalf("general.api_key = %q, want %q", cfg.General.APIKey, "lgw_from_legacy")
	}
	if cfg.DataSharing.APIKey != "lgw_from_legacy" {
		t.Fatalf("data_sharing.api_key = %q, want %q", cfg.DataSharing.APIKey, "lgw_from_legacy")
	}
	if cfg.General.BackendURL != "https://legacy.example/v1" {
		t.Fatalf("general.backend_url = %q, want %q", cfg.General.BackendURL, "https://legacy.example/v1")
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

func TestNewManager_ParsesAndNormalizesProviderCapabilities(t *testing.T) {
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	configBody := `providers:
  openai:
    api_key: "test-key"
    capabilities:
      chat_completions_lifecycle: true
      responses_lifecycle: true
      conversations: true
      background_responses: true
      response_cancellation: true
      response_compaction: true
      response_input_tokens: true
      embeddings_base64: true
      structured_outputs: true
      reasoning_effort: true
      reasoning_effort_levels: [" LOW ", "xhigh", "low", ""]
      adaptive_thinking: true
      hosted_tools: [" Web_Search ", "file_search", "web_search", ""]
routing:
  routes:
    - name: "default"
      targets:
        - provider: openai
`
	if err := os.WriteFile(configPath, []byte(configBody), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	manager, err := NewManager(configPath)
	if err != nil {
		t.Fatalf("NewManager returned error: %v", err)
	}

	capabilities := manager.Get().Providers["openai"].Capabilities
	if !capabilities.ChatCompletionsLifecycle || !capabilities.ResponsesLifecycle || !capabilities.Conversations ||
		!capabilities.BackgroundResponses || !capabilities.ResponseCancellation ||
		!capabilities.ResponseCompaction || !capabilities.ResponseInputTokens ||
		!capabilities.EmbeddingsBase64 || !capabilities.StructuredOutputs ||
		!capabilities.ReasoningEffort || !capabilities.AdaptiveThinking {
		t.Fatalf("capability flags were not preserved: %#v", capabilities)
	}
	wantEffortLevels := []string{"low", "xhigh"}
	if !reflect.DeepEqual(capabilities.ReasoningEffortLevels, wantEffortLevels) {
		t.Fatalf("reasoning_effort_levels = %#v, want %#v", capabilities.ReasoningEffortLevels, wantEffortLevels)
	}
	wantTools := []string{"web_search", "file_search"}
	if !reflect.DeepEqual(capabilities.HostedTools, wantTools) {
		t.Fatalf("hosted_tools = %#v, want %#v", capabilities.HostedTools, wantTools)
	}
}

func TestNewManager_ProviderCapabilitiesDefaultDisabled(t *testing.T) {
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	configBody := `providers:
  openai:
    api_key: "test-key"
routing:
  routes:
    - name: "default"
      targets:
        - provider: openai
`
	if err := os.WriteFile(configPath, []byte(configBody), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	manager, err := NewManager(configPath)
	if err != nil {
		t.Fatalf("NewManager returned error: %v", err)
	}

	if got := manager.Get().Providers["openai"].Capabilities; !reflect.DeepEqual(got, ProviderCapabilities{}) {
		t.Fatalf("capabilities default = %#v, want all disabled", got)
	}
}

func TestNewManager_ParsesProviderSamplingDefaults(t *testing.T) {
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	configBody := `providers:
  ollama:
    type: "ollama"
    base_url: "http://localhost:11434"
    default_model: "gemma4:26b"
    temperature: 1.0
    top_p: 0.95
    top_k: 64
routing:
  routes:
    - name: "default"
      targets:
        - provider: ollama
`
	if err := os.WriteFile(configPath, []byte(configBody), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	manager, err := NewManager(configPath)
	if err != nil {
		t.Fatalf("NewManager returned error: %v", err)
	}

	provider := manager.Get().Providers["ollama"]
	if provider.Temperature == nil || *provider.Temperature != 1.0 {
		t.Fatalf("provider temperature = %#v, want 1.0", provider.Temperature)
	}
	if provider.TopP == nil || *provider.TopP != 0.95 {
		t.Fatalf("provider top_p = %#v, want 0.95", provider.TopP)
	}
	if provider.TopK == nil || *provider.TopK != 64 {
		t.Fatalf("provider top_k = %#v, want 64", provider.TopK)
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
