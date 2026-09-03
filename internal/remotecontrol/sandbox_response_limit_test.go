package remotecontrol

import (
	"context"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
)

func TestExecuteSandboxAcceptsResponseAtLimit(t *testing.T) {
	const document = `{"ok":true}`
	const padding = "          "
	limit := int64(len(document) + len(padding))

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(document + padding))
	}))
	defer server.Close()

	client := &Client{
		localBaseURL:         server.URL,
		httpClient:           server.Client(),
		sandboxResponseLimit: limit,
	}
	status, _, body, err := client.executeSandbox(context.Background(), sandboxExecuteMessage{
		Request: map[string]interface{}{"model": "test"},
	})
	if err != nil {
		t.Fatalf("executeSandbox returned error: %v", err)
	}
	if status != http.StatusOK {
		t.Fatalf("status = %d, want %d", status, http.StatusOK)
	}
	if !reflect.DeepEqual(body, map[string]interface{}{"ok": true}) {
		t.Fatalf("body = %#v", body)
	}
}

func TestExecuteSandboxRejectsOversizedStreamingResponse(t *testing.T) {
	const limit int64 = 64
	const secret = "secret-sandbox-response"
	stream := "data: {\"delta\":\"" + secret + "\"}\n\n" +
		"data: {\"delta\":\"" + strings.Repeat("x", 64) + "\"}\n\n" +
		"data: [DONE]\n\n"

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("X-LunarGate-Request-ID", "req-oversized")
		_, _ = w.Write([]byte(stream))
	}))
	defer server.Close()

	client := &Client{
		localBaseURL:         server.URL,
		httpClient:           server.Client(),
		sandboxResponseLimit: limit,
	}
	status, headers, body, err := client.executeSandbox(context.Background(), sandboxExecuteMessage{
		RequestType: "chat_completions",
		Request: map[string]interface{}{
			"model":  "test",
			"stream": true,
		},
	})
	if status != http.StatusOK {
		t.Fatalf("status = %d, want %d", status, http.StatusOK)
	}
	if headers["X-LunarGate-Request-ID"] != "req-oversized" {
		t.Fatalf("headers = %#v", headers)
	}
	if body != nil {
		t.Fatalf("body = %#v, want nil", body)
	}
	if err == nil || !strings.Contains(err.Error(), "exceeds 64 byte limit") {
		t.Fatalf("error = %v, want total response limit error", err)
	}
	if strings.Contains(err.Error(), secret) {
		t.Fatalf("oversized response content leaked in error: %v", err)
	}
}
