package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestNewManagerCanonicalizesRoutingStrategyAndHeaderNames(t *testing.T) {
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	configBody := `providers:
  openai:
    api_key: test-key
routing:
  default_strategy: " ROUND-ROBIN "
  routes:
    - name: default
      match:
        headers:
          " X-Team ": engineering
      targets:
        - provider: openai
          weight: 1
`
	if err := os.WriteFile(configPath, []byte(configBody), 0o600); err != nil {
		t.Fatalf("write config: %v", err)
	}

	manager, err := NewManager(configPath)
	if err != nil {
		t.Fatalf("NewManager returned error: %v", err)
	}
	cfg := manager.Get()
	if got, want := cfg.Routing.DefaultStrategy, "round-robin"; got != want {
		t.Fatalf("routing.default_strategy = %q, want %q", got, want)
	}
	headers := cfg.Routing.Routes[0].Match.Headers
	if got, want := headers["x-team"], "engineering"; got != want {
		t.Fatalf("normalized x-team value = %q, want %q", got, want)
	}
	if len(headers) != 1 {
		t.Fatalf("normalized headers = %#v, want one entry", headers)
	}
}

func TestNewManagerRejectsDuplicateRoutingHeadersAfterNormalization(t *testing.T) {
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	configBody := `providers:
  openai:
    api_key: test-key
routing:
  routes:
    - name: default
      match:
        headers:
          X-Team: engineering
          x-team: platform
      targets:
        - provider: openai
          weight: 1
`
	if err := os.WriteFile(configPath, []byte(configBody), 0o600); err != nil {
		t.Fatalf("write config: %v", err)
	}

	_, err := NewManager(configPath)
	if err == nil {
		t.Fatal("NewManager returned nil error for duplicate normalized headers")
	}
	for _, want := range []string{"routing.routes[0].match.headers", "duplicate", "x-team"} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("error = %q, want substring %q", err, want)
		}
	}
}

func TestValidateConfigCanonicalizesEmptyRoutingStrategy(t *testing.T) {
	cfg := validRuntimeConfig()
	cfg.Routing.DefaultStrategy = ""

	if err := validateConfig(cfg); err != nil {
		t.Fatalf("validateConfig returned error: %v", err)
	}
	if got, want := cfg.Routing.DefaultStrategy, "round-robin"; got != want {
		t.Fatalf("routing.default_strategy = %q, want %q", got, want)
	}
}
