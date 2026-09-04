package remotecontrol

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/lunargate-ai/gateway/internal/config"
)

func TestConnectAcceptsIncomingMessageAtReadLimit(t *testing.T) {
	const command = `{"type":"sandbox.execute","command_id":"cmd-boundary","target":{"mode":"model","value":"openai/test"},"request_type":"chat_completions","request":{"model":"test","messages":[]}}`
	const padding = "                                "
	message := command + padding
	responses := make(chan sandboxResponseMessage, 1)

	upgrader := websocket.Upgrader{CheckOrigin: func(*http.Request) bool { return true }}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			t.Errorf("upgrade websocket: %v", err)
			return
		}
		defer conn.Close()

		var hello helloMessage
		if err := conn.ReadJSON(&hello); err != nil {
			t.Errorf("read hello: %v", err)
			return
		}
		if err := conn.WriteMessage(websocket.TextMessage, []byte(message)); err != nil {
			t.Errorf("write boundary command: %v", err)
			return
		}
		var response sandboxResponseMessage
		if err := conn.ReadJSON(&response); err != nil {
			t.Errorf("read sandbox response: %v", err)
			return
		}
		responses <- response
	}))
	defer server.Close()

	local := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer local.Close()

	client := newWebSocketLimitTestClient(t, server.URL, local.URL)
	client.websocketReadLimit = int64(len(message))
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- client.connectAndServe(ctx) }()

	select {
	case response := <-responses:
		if !response.OK || response.CommandID != "cmd-boundary" || response.StatusCode != http.StatusOK {
			t.Fatalf("sandbox response = %#v", response)
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for exact-limit sandbox response")
	}

	cancel()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("remote control client did not stop")
	}
}

func TestConnectRejectsIncomingMessageAboveReadLimit(t *testing.T) {
	const command = `{"type":"sandbox.execute","command_id":"cmd-oversized","target":{"mode":"model","value":"openai/test"},"request":{"model":"test"}}`
	var localRequests atomic.Int32

	upgrader := websocket.Upgrader{CheckOrigin: func(*http.Request) bool { return true }}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			t.Errorf("upgrade websocket: %v", err)
			return
		}
		defer conn.Close()
		var hello helloMessage
		if err := conn.ReadJSON(&hello); err != nil {
			t.Errorf("read hello: %v", err)
			return
		}
		_ = conn.WriteMessage(websocket.TextMessage, []byte(command))
	}))
	defer server.Close()

	local := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		localRequests.Add(1)
		w.WriteHeader(http.StatusNoContent)
	}))
	defer local.Close()

	client := newWebSocketLimitTestClient(t, server.URL, local.URL)
	client.websocketReadLimit = int64(len(command) - 1)
	err := client.connectAndServe(context.Background())
	if !errors.Is(err, websocket.ErrReadLimit) {
		t.Fatalf("connectAndServe error = %v, want websocket.ErrReadLimit", err)
	}
	if got := localRequests.Load(); got != 0 {
		t.Fatalf("local sandbox requests = %d, want zero", got)
	}
}

func newWebSocketLimitTestClient(t *testing.T, backendURL, localBaseURL string) *Client {
	t.Helper()
	client := NewClient(
		config.GeneralConfig{BackendURL: backendURL, APIKey: "lgw_test"},
		config.DataSharingConfig{Enabled: true, RemoteControl: true},
		config.SecurityConfig{},
		"test",
		localBaseURL,
		nil,
		nil,
		nil,
	)
	if client == nil {
		t.Fatal("expected remote control client")
	}
	client.heartbeatInterval = time.Hour
	client.instanceID = strings.Repeat("a", 32)
	return client
}
