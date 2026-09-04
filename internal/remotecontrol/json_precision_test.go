package remotecontrol

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/lunargate-ai/gateway/internal/config"
)

func TestSandboxCommandPreservesLargeJSONInteger(t *testing.T) {
	const largeInteger = "9007199254740993"
	receivedBody := make(chan string, 1)
	localServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		body, err := io.ReadAll(request.Body)
		if err != nil {
			receivedBody <- "read error: " + err.Error()
			http.Error(w, "read error", http.StatusInternalServerError)
			return
		}
		receivedBody <- string(body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"ok":true}`)
	}))
	defer localServer.Close()

	serverErrors := make(chan error, 1)
	upgrader := websocket.Upgrader{CheckOrigin: func(*http.Request) bool { return true }}
	backendServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		conn, err := upgrader.Upgrade(w, request, nil)
		if err != nil {
			serverErrors <- fmt.Errorf("upgrade websocket: %w", err)
			return
		}
		defer conn.Close()

		if _, _, err := conn.ReadMessage(); err != nil {
			serverErrors <- fmt.Errorf("read hello: %w", err)
			return
		}
		command := `{"type":"sandbox.execute","command_id":"large-int","target":{"mode":"model","value":"openai/test"},"request_type":"chat_completions","request":{"model":"test","metadata":{"sequence":` + largeInteger + `}}}`
		if err := conn.WriteMessage(websocket.TextMessage, []byte(command)); err != nil {
			serverErrors <- fmt.Errorf("write command: %w", err)
			return
		}
		var response sandboxResponseMessage
		if err := conn.ReadJSON(&response); err != nil {
			serverErrors <- fmt.Errorf("read sandbox response: %w", err)
			return
		}
		serverErrors <- nil
	}))
	defer backendServer.Close()

	client := NewClient(
		config.GeneralConfig{APIKey: "gateway-key", BackendURL: backendServer.URL},
		config.DataSharingConfig{Enabled: true, RemoteControl: true},
		config.SecurityConfig{},
		"test",
		localServer.URL,
		nil,
		nil,
		nil,
	)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	_ = client.connectAndServe(ctx)

	select {
	case body := <-receivedBody:
		if strings.HasPrefix(body, "read error:") {
			t.Fatal(body)
		}
		decoder := json.NewDecoder(strings.NewReader(body))
		decoder.UseNumber()
		var decoded struct {
			Metadata struct {
				Sequence json.Number `json:"sequence"`
			} `json:"metadata"`
		}
		if err := decoder.Decode(&decoded); err != nil {
			t.Fatalf("decode local request: %v", err)
		}
		if got := decoded.Metadata.Sequence.String(); got != largeInteger {
			t.Fatalf("local sandbox integer = %s, want %s; body=%s", got, largeInteger, body)
		}
	case <-ctx.Done():
		t.Fatal("timed out waiting for local sandbox request")
	}
	if err := <-serverErrors; err != nil {
		t.Fatal(err)
	}
}

func TestDecodeSingleJSONDocumentRejectsMultipleDocuments(t *testing.T) {
	var message sandboxExecuteMessage
	err := decodeSingleJSONDocument([]byte(`{"type":"heartbeat"} {"type":"sandbox.execute"}`), &message)
	if err == nil || !strings.Contains(err.Error(), "multiple JSON documents") {
		t.Fatalf("decode error = %v, want multiple-document rejection", err)
	}
}

func TestParseBodyPreservesLargeJSONInteger(t *testing.T) {
	const largeInteger = "9007199254740993"
	parsed := parseBody([]byte(`{"sequence":` + largeInteger + `}`))
	body, ok := parsed.(map[string]interface{})
	if !ok {
		t.Fatalf("parsed body type = %T, want map", parsed)
	}
	sequence, ok := body["sequence"].(json.Number)
	if !ok || sequence.String() != largeInteger {
		t.Fatalf("parsed sequence = %#v, want json.Number(%s)", body["sequence"], largeInteger)
	}
}
