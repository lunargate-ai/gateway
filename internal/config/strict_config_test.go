package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestNewManagerRejectsUnknownConfigurationFieldsWithoutLeakingValues(t *testing.T) {
	tests := []struct {
		name          string
		providerField string
		unknownYAML   string
		wantField     string
	}{
		{
			name: "unknown top-level field",
			unknownYAML: `unexpected_section:
  credential: "do-not-leak-secret"`,
			wantField: "unexpected_section",
		},
		{
			name: "unknown nested security field",
			unknownYAML: `security:
  enabeld: "do-not-leak-secret"`,
			wantField: "enabeld",
		},
		{
			name: "unknown provider field",
			providerField: `    api_keey: "do-not-leak-secret"
`,
			wantField: "api_keey",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			configPath := filepath.Join(t.TempDir(), "config.yaml")
			configBody := `providers:
  openai:
    api_key: "test-key"
` + test.providerField + `routing:
  routes:
    - name: default
      targets:
        - provider: openai
          weight: 1
` + test.unknownYAML + "\n"
			if err := os.WriteFile(configPath, []byte(configBody), 0o600); err != nil {
				t.Fatalf("write config: %v", err)
			}

			_, err := NewManager(configPath)
			if err == nil {
				t.Fatal("NewManager returned nil error for an unknown configuration field")
			}
			errorText := err.Error()
			if !strings.Contains(errorText, test.wantField) {
				t.Fatalf("error = %q, want field name %q", errorText, test.wantField)
			}
			if strings.Contains(errorText, "do-not-leak-secret") {
				t.Fatalf("error leaked the unknown field value: %q", errorText)
			}
		})
	}
}

func TestNewManagerAcceptsDeprecatedDataSharingBackendURL(t *testing.T) {
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	configBody := `providers:
  openai:
    api_key: "test-key"
routing:
  routes:
    - name: default
      targets:
        - provider: openai
          weight: 1
data_sharing:
  backend_url: "https://legacy.example/v1/collector"
`
	if err := os.WriteFile(configPath, []byte(configBody), 0o600); err != nil {
		t.Fatalf("write config: %v", err)
	}

	manager, err := NewManager(configPath)
	if err != nil {
		t.Fatalf("NewManager returned error: %v", err)
	}
	if got, want := manager.Get().General.BackendURL, "https://legacy.example/v1"; got != want {
		t.Fatalf("general.backend_url = %q, want %q", got, want)
	}
}
