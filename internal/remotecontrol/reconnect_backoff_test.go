package remotecontrol

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/lunargate-ai/gateway/internal/config"
)

func TestReconnectBackoffCapsAtFifteenSeconds(t *testing.T) {
	backoff := newReconnectBackoff()
	want := []time.Duration{
		time.Second,
		2 * time.Second,
		4 * time.Second,
		8 * time.Second,
		15 * time.Second,
		15 * time.Second,
	}
	for index, wantDelay := range want {
		if got := backoff.nextDelay(false); got != wantDelay {
			t.Fatalf("delay[%d] = %s, want %s", index, got, wantDelay)
		}
	}
}

func TestReconnectBackoffResetsAfterHealthyConnection(t *testing.T) {
	backoff := newReconnectBackoff()
	for range 4 {
		_ = backoff.nextDelay(false)
	}
	if got := backoff.nextDelay(true); got != time.Second {
		t.Fatalf("delay after healthy connection = %s, want 1s", got)
	}
	if got := backoff.nextDelay(false); got != 2*time.Second {
		t.Fatalf("next delay = %s, want 2s", got)
	}
}

func TestValidBackendFrameMarksConnectionHealthy(t *testing.T) {
	upgrader := websocket.Upgrader{CheckOrigin: func(*http.Request) bool { return true }}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		connection, err := upgrader.Upgrade(w, request, nil)
		if err != nil {
			t.Errorf("upgrade WebSocket: %v", err)
			return
		}
		defer connection.Close()
		if _, _, err := connection.ReadMessage(); err != nil {
			return
		}
		if err := connection.WriteJSON(map[string]string{"type": "heartbeat"}); err != nil {
			t.Errorf("write heartbeat: %v", err)
			return
		}
		for {
			if _, _, err := connection.ReadMessage(); err != nil {
				return
			}
		}
	}))
	defer server.Close()

	client := NewClient(
		config.GeneralConfig{BackendURL: server.URL, APIKey: "lgw_test"},
		config.DataSharingConfig{Enabled: true, RemoteControl: true},
		config.SecurityConfig{},
		"0.4.0",
		"http://127.0.0.1:8080",
		nil,
		nil,
		nil,
	)
	if client == nil {
		t.Fatal("expected remote control client")
	}
	client.heartbeatInterval = time.Hour

	ctx, cancel := context.WithCancel(context.Background())
	healthy := make(chan struct{}, 1)
	done := make(chan error, 1)
	go func() {
		done <- client.connectAndServeWithHealth(ctx, healthy)
	}()

	select {
	case <-healthy:
	case <-time.After(time.Second):
		cancel()
		t.Fatal("valid backend frame did not mark connection healthy")
	}
	cancel()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("remote control client did not stop")
	}
}
