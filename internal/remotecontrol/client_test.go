package remotecontrol

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
)

func TestNewClientRequiresDataSharingMasterSwitch(t *testing.T) {
	client := NewClient(
		config.GeneralConfig{
			BackendURL: "https://api.lunargate.ai/v1",
			APIKey:     "lgw_test",
		},
		config.DataSharingConfig{
			RemoteControl: true,
		},
		config.SecurityConfig{},
		"test",
		"http://127.0.0.1:8080",
		nil,
		nil,
	)
	if client != nil {
		t.Fatal("expected remote control client to remain disabled by data_sharing.enabled")
	}
}

func TestNewClientRequiresOnlyAPIKeyWhenDataSharingEnabled(t *testing.T) {
	client := NewClient(
		config.GeneralConfig{
			BackendURL: "https://api.lunargate.ai/v1",
			APIKey:     "lgw_test",
		},
		config.DataSharingConfig{
			Enabled:       true,
			RemoteControl: true,
		},
		config.SecurityConfig{},
		"test",
		"http://127.0.0.1:8080",
		nil,
		nil,
	)
	if client == nil {
		t.Fatal("expected remote control client to initialize without gateway_id")
	}
}

func TestClientWebsocketURLDoesNotRequireGatewayIDQuery(t *testing.T) {
	client := NewClient(
		config.GeneralConfig{
			BackendURL: "https://api.lunargate.ai/v1",
			APIKey:     "lgw_test",
		},
		config.DataSharingConfig{
			Enabled:       true,
			RemoteControl: true,
		},
		config.SecurityConfig{},
		"test",
		"http://127.0.0.1:8080",
		nil,
		nil,
	)
	if client == nil {
		t.Fatal("expected remote control client")
	}

	wsURL, err := client.websocketURL()
	if err != nil {
		t.Fatalf("websocketURL returned error: %v", err)
	}
	if strings.Contains(wsURL, "gateway_id=") {
		t.Fatalf("expected websocket URL without gateway_id query, got %q", wsURL)
	}
}

func TestExecuteSandboxUsesConfiguredInboundAPIKey(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("X-Gateway-Key"); got != "Token local-test-key" {
			t.Errorf("X-Gateway-Key = %q, want configured credential", got)
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl_test"}`))
	}))
	defer server.Close()

	client := NewClient(
		config.GeneralConfig{
			BackendURL: "https://api.lunargate.ai/v1",
			APIKey:     "lgw_test",
		},
		config.DataSharingConfig{Enabled: true, RemoteControl: true},
		config.SecurityConfig{
			Enabled:  true,
			Provider: "api_key",
			APIKey: config.APIKeyAuthConfig{
				Header: "X-Gateway-Key",
				Prefix: "Token",
				Keys: []config.APIKeyCredential{
					{Name: "sandbox", Value: "local-test-key"},
				},
			},
		},
		"test",
		server.URL,
		nil,
		nil,
	)
	if client == nil {
		t.Fatal("expected remote control client")
	}

	status, _, _, err := client.executeSandbox(context.Background(), sandboxExecuteMessage{
		Target:  sandboxTarget{Mode: "model", Value: "openai/gpt-test"},
		Request: map[string]interface{}{"messages": []interface{}{}},
	})
	if err != nil {
		t.Fatalf("executeSandbox returned error: %v", err)
	}
	if status != http.StatusOK {
		t.Fatalf("status = %d, want %d", status, http.StatusOK)
	}
}

func TestClassifyDialErrorWrapsHandshakeStatus(t *testing.T) {
	err := classifyDialError(
		io.EOF,
		&http.Response{
			StatusCode: http.StatusUnauthorized,
			Body:       io.NopCloser(strings.NewReader(`{"detail":"Invalid gateway API key"}`)),
		},
	)

	statusErr, ok := err.(*dialStatusError)
	if !ok {
		t.Fatalf("expected dialStatusError, got %T", err)
	}
	if statusErr.statusCode != http.StatusUnauthorized {
		t.Fatalf("statusCode = %d, want %d", statusErr.statusCode, http.StatusUnauthorized)
	}
	if !strings.Contains(statusErr.Error(), "Invalid gateway API key") {
		t.Fatalf("expected error to include response detail, got %q", statusErr.Error())
	}
}
