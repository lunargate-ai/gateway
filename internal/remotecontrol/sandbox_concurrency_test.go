package remotecontrol

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gorilla/websocket"
)

func TestSandboxConcurrencyRejectsThenReleasesSlot(t *testing.T) {
	firstStarted := make(chan struct{})
	releaseFirst := make(chan struct{})
	released := false
	defer func() {
		if !released {
			close(releaseFirst)
		}
	}()
	var localRequests atomic.Int32

	local := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		localRequests.Add(1)
		var request map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Errorf("decode local sandbox request: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		marker, _ := request["marker"].(string)
		if marker == "first" {
			close(firstStarted)
			<-releaseFirst
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{"marker": marker})
	}))
	defer local.Close()

	responses := make(chan sandboxResponseMessage, 3)
	serverErrors := make(chan error, 1)
	upgrader := websocket.Upgrader{CheckOrigin: func(*http.Request) bool { return true }}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			serverErrors <- err
			return
		}
		defer conn.Close()

		var hello helloMessage
		if err := conn.ReadJSON(&hello); err != nil {
			serverErrors <- err
			return
		}
		if err := conn.WriteJSON(sandboxCommandForConcurrencyTest("cmd-first", "first")); err != nil {
			serverErrors <- err
			return
		}
		select {
		case <-firstStarted:
		case <-time.After(time.Second):
			serverErrors <- context.DeadlineExceeded
			return
		}
		if err := conn.WriteJSON(sandboxCommandForConcurrencyTest("cmd-overload", "overload")); err != nil {
			serverErrors <- err
			return
		}

		for i := 0; i < 2; i++ {
			var response sandboxResponseMessage
			if err := conn.ReadJSON(&response); err != nil {
				serverErrors <- err
				return
			}
			responses <- response
		}
		if err := conn.WriteJSON(sandboxCommandForConcurrencyTest("cmd-after-release", "after")); err != nil {
			serverErrors <- err
			return
		}
		var response sandboxResponseMessage
		if err := conn.ReadJSON(&response); err != nil {
			serverErrors <- err
			return
		}
		responses <- response
	}))
	defer server.Close()

	client := newWebSocketLimitTestClient(t, server.URL, local.URL)
	client.sandboxCommandLimit = 1
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- client.connectAndServe(ctx) }()

	overload := awaitSandboxResponse(t, responses, serverErrors)
	if overload.CommandID != "cmd-overload" || overload.OK || overload.StatusCode != http.StatusTooManyRequests || overload.Error != "sandbox command concurrency limit reached" {
		t.Fatalf("overload response = %#v", overload)
	}
	close(releaseFirst)
	released = true

	first := awaitSandboxResponse(t, responses, serverErrors)
	if first.CommandID != "cmd-first" || !first.OK || first.StatusCode != http.StatusOK {
		t.Fatalf("first response = %#v", first)
	}
	after := awaitSandboxResponse(t, responses, serverErrors)
	if after.CommandID != "cmd-after-release" || !after.OK || after.StatusCode != http.StatusOK {
		t.Fatalf("post-release response = %#v", after)
	}
	if got := localRequests.Load(); got != 2 {
		t.Fatalf("local sandbox requests = %d, want 2", got)
	}

	cancel()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("remote control client did not stop")
	}
}

func sandboxCommandForConcurrencyTest(commandID, marker string) sandboxExecuteMessage {
	return sandboxExecuteMessage{
		Type:      "sandbox.execute",
		CommandID: commandID,
		Target:    sandboxTarget{Mode: "model", Value: "openai/test"},
		Request:   map[string]interface{}{"model": "test", "marker": marker},
	}
}

func awaitSandboxResponse(t *testing.T, responses <-chan sandboxResponseMessage, serverErrors <-chan error) sandboxResponseMessage {
	t.Helper()
	select {
	case response := <-responses:
		return response
	case err := <-serverErrors:
		t.Fatalf("websocket test server failed: %v", err)
		return sandboxResponseMessage{}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for sandbox response")
		return sandboxResponseMessage{}
	}
}
