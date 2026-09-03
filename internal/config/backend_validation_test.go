package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestValidateConfigNormalizesBackendBeforeValidation(t *testing.T) {
	tests := []struct {
		name       string
		generalURL string
		legacyURL  string
		wantURL    string
	}{
		{
			name:    "default backend",
			wantURL: defaultBackendURL,
		},
		{
			name:       "general backend",
			generalURL: " HTTPS://user:password@example.test/v1/collector/?token=secret ",
			wantURL:    "https://user:password@example.test/v1?token=secret",
		},
		{
			name:      "deprecated backend alias",
			legacyURL: "https://legacy.example/v1/collector/",
			wantURL:   "https://legacy.example/v1",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cfg := validRuntimeConfig()
			cfg.General.BackendURL = test.generalURL
			cfg.DataSharing.BackendURL = test.legacyURL

			if err := validateConfig(cfg); err != nil {
				t.Fatalf("validateConfig returned error: %v", err)
			}
			if got := cfg.General.BackendURL; got != test.wantURL {
				t.Fatalf("general.backend_url = %q, want %q", got, test.wantURL)
			}
		})
	}
}

func TestValidateConfigRejectsInvalidBackendWithoutLeakingURL(t *testing.T) {
	const secret = "do-not-leak-backend-secret"
	tests := []string{
		"relative/path?token=" + secret,
		"ftp://example.test/v1?token=" + secret,
		"https:///%zz?token=" + secret,
	}

	for _, backendURL := range tests {
		cfg := validRuntimeConfig()
		cfg.General.BackendURL = backendURL

		err := validateConfig(cfg)
		if err == nil {
			t.Fatalf("validateConfig returned nil error for %q", backendURL)
		}
		if !strings.Contains(err.Error(), "general.backend_url") {
			t.Fatalf("validateConfig error = %q", err)
		}
		if strings.Contains(err.Error(), secret) || strings.Contains(err.Error(), backendURL) {
			t.Fatalf("validation error leaked backend URL: %q", err)
		}
	}
}

func TestValidateConfigEnforcesDataSharingPrerequisites(t *testing.T) {
	tests := []struct {
		name        string
		dataSharing DataSharingConfig
		apiKey      string
		wantErr     string
	}{
		{
			name:        "enabled without API key",
			dataSharing: DataSharingConfig{Enabled: true},
			wantErr:     "general.api_key",
		},
		{
			name:        "remote control without master switch",
			dataSharing: DataSharingConfig{RemoteControl: true},
			apiKey:      "gateway-key",
			wantErr:     "data_sharing.enabled",
		},
		{
			name:        "enabled with API key",
			dataSharing: DataSharingConfig{Enabled: true, RemoteControl: true},
			apiKey:      "gateway-key",
		},
		{
			name: "disabled master preserves sharing preferences",
			dataSharing: DataSharingConfig{
				SharePrompts:   true,
				ShareResponses: true,
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cfg := validRuntimeConfig()
			cfg.General.APIKey = test.apiKey
			cfg.DataSharing = test.dataSharing

			err := validateConfig(cfg)
			if test.wantErr == "" {
				if err != nil {
					t.Fatalf("validateConfig returned error: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("validateConfig error = %v, want substring %q", err, test.wantErr)
			}
		})
	}
}

func TestNewManagerRejectsMissingExpandedGatewayAPIKey(t *testing.T) {
	const keyEnvironment = "LUNARGATE_TEST_MISSING_GATEWAY_API_KEY"
	if err := os.Unsetenv(keyEnvironment); err != nil {
		t.Fatalf("unset environment: %v", err)
	}
	t.Cleanup(func() { _ = os.Unsetenv(keyEnvironment) })

	configPath := filepath.Join(t.TempDir(), "config.yaml")
	configBody := `providers:
  openai:
    api_key: test-key
routing:
  routes:
    - name: default
      targets:
        - provider: openai
          weight: 1
general:
  api_key: "${LUNARGATE_TEST_MISSING_GATEWAY_API_KEY}"
data_sharing:
  enabled: true
`
	if err := os.WriteFile(configPath, []byte(configBody), 0o600); err != nil {
		t.Fatalf("write config: %v", err)
	}

	_, err := NewManager(configPath)
	if err == nil || !strings.Contains(err.Error(), "general.api_key") {
		t.Fatalf("NewManager error = %v, want missing general.api_key", err)
	}
}
