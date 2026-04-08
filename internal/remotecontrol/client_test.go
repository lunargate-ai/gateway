package remotecontrol

import (
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
)

func TestNewClientRequiresOnlyAPIKeyForRemoteControl(t *testing.T) {
	client := NewClient(
		config.DataSharingConfig{
			RemoteControl: true,
			BackendURL:    "https://api.lunargate.ai/v1",
			APIKey:        "lgw_test",
		},
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
		config.DataSharingConfig{
			RemoteControl: true,
			BackendURL:    "https://api.lunargate.ai/v1",
			APIKey:        "lgw_test",
		},
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
