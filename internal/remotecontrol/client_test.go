package remotecontrol

import (
	"context"
	"encoding/hex"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/lunargate-ai/gateway/internal/config"
)

func TestLocalInstanceIDFormatAndUniqueness(t *testing.T) {
	const sampleSize = 1024

	seen := make(map[string]struct{}, sampleSize)
	for i := 0; i < sampleSize; i++ {
		id, err := localInstanceID()
		if err != nil {
			t.Fatalf("localInstanceID returned error: %v", err)
		}
		if len(id) != 32 {
			t.Fatalf("instance ID length = %d, want 32: %q", len(id), id)
		}
		if id != strings.ToLower(id) {
			t.Fatalf("instance ID is not lowercase hexadecimal: %q", id)
		}
		decoded, err := hex.DecodeString(id)
		if err != nil {
			t.Fatalf("instance ID is not hexadecimal: %q: %v", id, err)
		}
		if len(decoded) != 16 {
			t.Fatalf("decoded instance ID length = %d, want 16", len(decoded))
		}
		if _, exists := seen[id]; exists {
			t.Fatalf("duplicate instance ID generated: %q", id)
		}
		seen[id] = struct{}{}
	}
}

func TestInstanceIDFromReaderFailsClosed(t *testing.T) {
	id, err := instanceIDFromReader(strings.NewReader(strings.Repeat("x", 15)))
	if !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("error = %v, want wrapped io.ErrUnexpectedEOF", err)
	}
	if id != "" {
		t.Fatalf("instance ID = %q, want empty on entropy failure", id)
	}
}

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
	requestURL, err := url.Parse("wss://url-user:url-password@private.example.test/v1/remote-control/ws/gateway?token=query-secret#fragment-secret")
	if err != nil {
		t.Fatalf("parse request URL: %v", err)
	}
	err = classifyDialError(
		io.EOF,
		&http.Response{
			StatusCode: http.StatusUnauthorized,
			Body:       io.NopCloser(strings.NewReader(`{"detail":"backend-response-secret"}`)),
		},
		requestURL,
	)

	statusErr, ok := err.(*dialStatusError)
	if !ok {
		t.Fatalf("expected dialStatusError, got %T", err)
	}
	if statusErr.statusCode != http.StatusUnauthorized {
		t.Fatalf("statusCode = %d, want %d", statusErr.statusCode, http.StatusUnauthorized)
	}
	for _, secret := range []string{"url-user", "url-password", "query-secret", "fragment-secret", "backend-response-secret"} {
		if strings.Contains(statusErr.Error(), secret) {
			t.Fatalf("handshake error leaked %q: %s", secret, statusErr.Error())
		}
	}
}

func TestConnectSendsSnapshotHelloBeforeSlowModelRefresh(t *testing.T) {
	server, hellos, heartbeats := newHelloCaptureServer(t)
	defer server.Close()

	refreshStarted := make(chan struct{})
	releaseRefresh := make(chan struct{})
	client := NewClient(
		config.GeneralConfig{BackendURL: server.URL, APIKey: "lgw_test"},
		config.DataSharingConfig{Enabled: true, RemoteControl: true},
		config.SecurityConfig{},
		"0.4.0",
		"http://127.0.0.1:8080",
		func() []string { return []string{"default"} },
		func() []string { return []string{"openai/cached"} },
		func(ctx context.Context) []string {
			close(refreshStarted)
			select {
			case <-releaseRefresh:
				return []string{"openai/fresh"}
			case <-ctx.Done():
				return nil
			}
		},
	)
	if client == nil {
		t.Fatal("expected remote control client")
	}
	client.heartbeatInterval = 20 * time.Millisecond
	client.modelRefreshTimeout = 5 * time.Second

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- client.connectAndServe(ctx) }()
	defer func() {
		cancel()
		select {
		case <-done:
		case <-time.After(time.Second):
			t.Error("remote control client did not stop")
		}
	}()

	first := awaitHello(t, hellos)
	if got := strings.Join(first.Models, ","); got != "openai/cached" {
		t.Fatalf("initial hello models = %q, want cached snapshot", got)
	}
	if got := strings.Join(first.Routes, ","); got != "default" {
		t.Fatalf("initial hello routes = %q, want default", got)
	}
	select {
	case <-refreshStarted:
	case <-time.After(time.Second):
		t.Fatal("model refresh did not start after initial hello")
	}
	select {
	case <-heartbeats:
	case <-time.After(time.Second):
		t.Fatal("heartbeat was blocked by slow model refresh")
	}

	close(releaseRefresh)
	second := awaitHello(t, hellos)
	if got := strings.Join(second.Models, ","); got != "openai/fresh" {
		t.Fatalf("refreshed hello models = %q, want fresh models", got)
	}
}

func TestConnectCancelsSlowModelRefreshAtBoundedTimeout(t *testing.T) {
	server, hellos, _ := newHelloCaptureServer(t)
	defer server.Close()

	refreshStarted := make(chan struct{})
	refreshStopped := make(chan struct{})
	client := NewClient(
		config.GeneralConfig{BackendURL: server.URL, APIKey: "lgw_test"},
		config.DataSharingConfig{Enabled: true, RemoteControl: true},
		config.SecurityConfig{},
		"0.4.0",
		"http://127.0.0.1:8080",
		nil,
		func() []string { return []string{"openai/local"} },
		func(ctx context.Context) []string {
			close(refreshStarted)
			<-ctx.Done()
			close(refreshStopped)
			return nil
		},
	)
	if client == nil {
		t.Fatal("expected remote control client")
	}
	client.modelRefreshTimeout = 20 * time.Millisecond

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- client.connectAndServe(ctx) }()
	defer func() {
		cancel()
		select {
		case <-done:
		case <-time.After(time.Second):
			t.Error("remote control client leaked a model refresh goroutine")
		}
	}()

	first := awaitHello(t, hellos)
	if got := strings.Join(first.Models, ","); got != "openai/local" {
		t.Fatalf("initial hello models = %q, want local snapshot", got)
	}
	select {
	case <-refreshStarted:
	case <-time.After(time.Second):
		t.Fatal("slow model refresh did not start")
	}
	select {
	case <-refreshStopped:
	case <-time.After(time.Second):
		t.Fatal("slow model refresh was not canceled by its timeout")
	}
}

func newHelloCaptureServer(t *testing.T) (*httptest.Server, <-chan helloMessage, <-chan struct{}) {
	t.Helper()
	hellos := make(chan helloMessage, 4)
	heartbeats := make(chan struct{}, 4)
	upgrader := websocket.Upgrader{CheckOrigin: func(*http.Request) bool { return true }}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer lgw_test" {
			t.Errorf("Authorization = %q, want Bearer lgw_test", got)
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			t.Errorf("upgrade websocket: %v", err)
			return
		}
		defer conn.Close()
		for {
			var message helloMessage
			if err := conn.ReadJSON(&message); err != nil {
				return
			}
			if message.Type == "hello" {
				hellos <- message
			} else if message.Type == "heartbeat" {
				select {
				case heartbeats <- struct{}{}:
				default:
				}
			}
		}
	}))
	return server, hellos, heartbeats
}

func awaitHello(t *testing.T, hellos <-chan helloMessage) helloMessage {
	t.Helper()
	select {
	case hello := <-hellos:
		return hello
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for remote control hello")
		return helloMessage{}
	}
}
