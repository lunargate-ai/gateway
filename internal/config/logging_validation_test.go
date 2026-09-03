package config

import (
	"strings"
	"testing"
)

func TestValidateConfigNormalizesLogging(t *testing.T) {
	cfg := validRuntimeConfig()
	cfg.Logging = LoggingConfig{Level: " WARN ", Format: " JSON "}

	if err := validateConfig(cfg); err != nil {
		t.Fatalf("validateConfig returned error: %v", err)
	}
	if got, want := cfg.Logging.Level, "warn"; got != want {
		t.Fatalf("logging.level = %q, want %q", got, want)
	}
	if got, want := cfg.Logging.Format, "json"; got != want {
		t.Fatalf("logging.format = %q, want %q", got, want)
	}
}

func TestValidateConfigAppliesLoggingDefaults(t *testing.T) {
	cfg := validRuntimeConfig()
	cfg.Logging = LoggingConfig{}

	if err := validateConfig(cfg); err != nil {
		t.Fatalf("validateConfig returned error: %v", err)
	}
	if got, want := cfg.Logging.Level, "info"; got != want {
		t.Fatalf("logging.level = %q, want %q", got, want)
	}
	if got, want := cfg.Logging.Format, "console"; got != want {
		t.Fatalf("logging.format = %q, want %q", got, want)
	}
}

func TestValidateConfigRejectsInvalidLogging(t *testing.T) {
	tests := []struct {
		name    string
		logging LoggingConfig
		wantErr string
	}{
		{
			name:    "unknown level",
			logging: LoggingConfig{Level: "verbose", Format: "console"},
			wantErr: "logging.level",
		},
		{
			name:    "unknown format",
			logging: LoggingConfig{Level: "info", Format: "pretty"},
			wantErr: "logging.format",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cfg := validRuntimeConfig()
			cfg.Logging = test.logging
			err := validateConfig(cfg)
			if err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("validateConfig error = %v, want substring %q", err, test.wantErr)
			}
		})
	}
}
