package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

func TestLocalLoopbackAddress(t *testing.T) {
	tests := []struct {
		name    string
		host    string
		want    string
		wantErr bool
	}{
		{name: "empty wildcard", want: "127.0.0.1:8080"},
		{name: "IPv4 wildcard", host: "0.0.0.0", want: "127.0.0.1:8080"},
		{name: "IPv6 wildcard", host: "::", want: "[::1]:8080"},
		{name: "bracketed IPv6 wildcard", host: "[::]", want: "[::1]:8080"},
		{name: "expanded IPv6 wildcard", host: "0:0:0:0:0:0:0:0", want: "[::1]:8080"},
		{name: "localhost", host: "localhost", want: "127.0.0.1:8080"},
		{name: "IPv4 loopback", host: "127.0.0.2", want: "127.0.0.2:8080"},
		{name: "IPv6 loopback", host: "::1", want: "[::1]:8080"},
		{name: "bracketed IPv6 loopback", host: "[::1]", want: "[::1]:8080"},
		{name: "specific LAN address", host: "10.2.0.153", wantErr: true},
		{name: "unverified hostname", host: "gateway.internal", wantErr: true},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := localLoopbackAddress(config.ServerConfig{Host: test.host, Port: 8080})
			if test.wantErr {
				if err == nil {
					t.Fatalf("localLoopbackAddress() = %q, want error", got)
				}
				if strings.Contains(err.Error(), test.host) {
					t.Fatalf("error unexpectedly echoed configured host: %v", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("localLoopbackAddress returned error: %v", err)
			}
			if got != test.want {
				t.Fatalf("localLoopbackAddress() = %q, want %q", got, test.want)
			}
		})
	}
}

func TestLogRemoteControlStatusRedactsBackendURL(t *testing.T) {
	var output bytes.Buffer
	previousLogger := log.Logger
	log.Logger = zerolog.New(&output)
	t.Cleanup(func() { log.Logger = previousLogger })

	cfg := &config.Config{
		General: config.GeneralConfig{
			BackendURL: "https://url-user:url-password@private.example.test/root/v1?token=query-secret#fragment-secret",
		},
		DataSharing: config.DataSharingConfig{Enabled: true, RemoteControl: true},
	}
	logRemoteControlStatus(cfg, nil)

	logged := output.String()
	for _, secret := range []string{"url-user", "url-password", "query-secret", "fragment-secret"} {
		if strings.Contains(logged, secret) {
			t.Fatalf("remote control status log leaked %q: %s", secret, logged)
		}
	}
	if !strings.Contains(logged, "https://private.example.test/root/v1") {
		t.Fatalf("remote control status log lost sanitized backend endpoint: %s", logged)
	}
}

func TestLogRemoteControlStatusDoesNotEchoInvalidBackendURL(t *testing.T) {
	const secret = "invalid-backend-secret"
	var output bytes.Buffer
	previousLogger := log.Logger
	log.Logger = zerolog.New(&output)
	t.Cleanup(func() { log.Logger = previousLogger })

	logRemoteControlStatus(&config.Config{
		General: config.GeneralConfig{BackendURL: "https://example.test/%zz?token=" + secret},
	}, nil)

	logged := output.String()
	if strings.Contains(logged, secret) {
		t.Fatalf("remote control status log echoed invalid backend URL: %s", logged)
	}
	if !strings.Contains(logged, "[invalid]") {
		t.Fatalf("remote control status log did not classify invalid backend URL: %s", logged)
	}
}

func TestResolveLoggingConfigPreservesCLIOverride(t *testing.T) {
	for _, fileLevel := range []string{"debug", "error"} {
		resolved, err := resolveLoggingConfig(config.LoggingConfig{
			Level:  fileLevel,
			Format: "console",
		}, "WARN")
		if err != nil {
			t.Fatalf("resolve logging config after file level %q: %v", fileLevel, err)
		}
		if resolved.Level != "warn" {
			t.Fatalf("resolved level = %q, want warn", resolved.Level)
		}
	}
}

func TestResolveLoggingConfigRejectsInvalidCLIOverride(t *testing.T) {
	_, err := resolveLoggingConfig(config.LoggingConfig{Level: "info", Format: "json"}, "verbose")
	if err == nil {
		t.Fatal("invalid CLI log-level override was accepted")
	}
}

func TestSetupLoggingCanSelectJSONAfterConsole(t *testing.T) {
	previousLogger := log.Logger
	previousLevel := zerolog.GlobalLevel()
	previousTimeFormat := zerolog.TimeFieldFormat
	t.Cleanup(func() {
		log.Logger = previousLogger
		zerolog.SetGlobalLevel(previousLevel)
		zerolog.TimeFieldFormat = previousTimeFormat
	})

	var consoleOutput bytes.Buffer
	setupLoggingOutput(config.LoggingConfig{Level: "info", Format: "console"}, &consoleOutput)
	log.Info().Msg("console-only")

	var jsonOutput bytes.Buffer
	setupLoggingOutput(config.LoggingConfig{Level: "debug", Format: "json"}, &jsonOutput)
	log.Debug().Msg("json-after-console")

	if strings.Contains(consoleOutput.String(), "json-after-console") {
		t.Fatalf("JSON log was still written through console logger: %s", consoleOutput.String())
	}
	if !strings.Contains(jsonOutput.String(), `"message":"json-after-console"`) {
		t.Fatalf("JSON logger was not restored after console setup: %s", jsonOutput.String())
	}
}

func TestRedactedHTTPURLRemovesCredentials(t *testing.T) {
	got := redactedHTTPURL("https://user:password@example.test/latest?token=query-secret#fragment-secret")
	if got != "https://example.test/latest" {
		t.Fatalf("redacted HTTP URL = %q", got)
	}
	for _, secret := range []string{"user", "password", "query-secret", "fragment-secret"} {
		if strings.Contains(got, secret) {
			t.Fatalf("redacted HTTP URL leaked %q: %s", secret, got)
		}
	}
}
