package remotecontrol

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
)

func TestExecuteSandboxRoutesRequestTypes(t *testing.T) {
	tests := []struct {
		name        string
		requestType string
		wantPath    string
		payload     map[string]interface{}
	}{
		{
			name:        "default remains chat completions",
			requestType: "",
			wantPath:    "/v1/chat/completions",
			payload: map[string]interface{}{
				"model": "gpt-test",
				"messages": []interface{}{
					map[string]interface{}{"role": "user", "content": "hello"},
				},
			},
		},
		{
			name:        "legacy chat alias is normalized",
			requestType: " ChAt ",
			wantPath:    "/v1/chat/completions",
			payload: map[string]interface{}{
				"model":    "gpt-test",
				"messages": []interface{}{},
				"metadata": map[string]interface{}{"source": "sandbox"},
			},
		},
		{
			name:        "canonical chat completions",
			requestType: "chat_completions",
			wantPath:    "/v1/chat/completions",
			payload: map[string]interface{}{
				"model":    "gpt-test",
				"messages": []interface{}{},
				"stream":   true,
			},
		},
		{
			name:        "responses",
			requestType: " Responses ",
			wantPath:    "/v1/responses",
			payload: map[string]interface{}{
				"model": "gpt-test",
				"input": []interface{}{
					map[string]interface{}{"role": "user", "content": "hello"},
				},
				"store": false,
			},
		},
		{
			name:        "embeddings",
			requestType: "embeddings",
			wantPath:    "/v1/embeddings",
			payload: map[string]interface{}{
				"model":           "embedding-test",
				"input":           []interface{}{"hello", "world"},
				"encoding_format": "base64",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodPost {
					t.Errorf("method = %q, want %q", r.Method, http.MethodPost)
				}
				if r.URL.Path != tt.wantPath {
					t.Errorf("path = %q, want %q", r.URL.Path, tt.wantPath)
				}
				if got := r.Header.Get("Content-Type"); got != "application/json" {
					t.Errorf("Content-Type = %q, want application/json", got)
				}
				if got := r.Header.Get("X-LunarGate-No-Cache"); got != "true" {
					t.Errorf("X-LunarGate-No-Cache = %q, want true", got)
				}

				body, err := io.ReadAll(r.Body)
				if err != nil {
					t.Errorf("read request body: %v", err)
					w.WriteHeader(http.StatusBadRequest)
					return
				}
				var gotPayload map[string]interface{}
				if err := json.Unmarshal(body, &gotPayload); err != nil {
					t.Errorf("decode request body: %v", err)
					w.WriteHeader(http.StatusBadRequest)
					return
				}
				if !reflect.DeepEqual(gotPayload, tt.payload) {
					t.Errorf("payload = %#v, want %#v", gotPayload, tt.payload)
				}

				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"ok":true}`))
			}))
			defer server.Close()

			client := &Client{
				localBaseURL: server.URL,
				httpClient:   server.Client(),
			}
			status, _, body, err := client.executeSandbox(context.Background(), sandboxExecuteMessage{
				Target:      sandboxTarget{Mode: "model", Value: "openai/gpt-test"},
				RequestType: tt.requestType,
				Request:     tt.payload,
			})
			if err != nil {
				t.Fatalf("executeSandbox returned error: %v", err)
			}
			if status != http.StatusOK {
				t.Fatalf("status = %d, want %d", status, http.StatusOK)
			}
			if !reflect.DeepEqual(body, map[string]interface{}{"ok": true}) {
				t.Fatalf("body = %#v, want success body", body)
			}
		})
	}
}

func TestExecuteSandboxRejectsUnknownRequestTypeWithoutSending(t *testing.T) {
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		requests.Add(1)
		w.WriteHeader(http.StatusNoContent)
	}))
	defer server.Close()

	client := &Client{
		localBaseURL: server.URL,
		httpClient:   server.Client(),
	}
	status, headers, body, err := client.executeSandbox(context.Background(), sandboxExecuteMessage{
		Target:      sandboxTarget{Mode: "model", Value: "openai/gpt-test"},
		RequestType: "audio_generation",
		Request:     map[string]interface{}{"model": "gpt-test", "input": "hello"},
	})

	if status != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d", status, http.StatusBadRequest)
	}
	if len(headers) != 0 {
		t.Fatalf("headers = %#v, want empty", headers)
	}
	if body != nil {
		t.Fatalf("body = %#v, want nil", body)
	}
	if err == nil || !strings.Contains(err.Error(), `unsupported sandbox request_type "audio_generation"`) {
		t.Fatalf("error = %v, want explicit request_type error", err)
	}
	if got := requests.Load(); got != 0 {
		t.Fatalf("local gateway received %d requests, want none", got)
	}
}

func TestSandboxExecuteMessageDecodesRequestType(t *testing.T) {
	var msg sandboxExecuteMessage
	if err := json.Unmarshal([]byte(`{
		"type":"sandbox.execute",
		"command_id":"cmd-1",
		"request_type":"responses",
		"target":{"mode":"model","value":"openai/gpt-test"},
		"request":{"model":"gpt-test","input":"hello"}
	}`), &msg); err != nil {
		t.Fatalf("decode sandbox command: %v", err)
	}

	if msg.RequestType != "responses" {
		t.Fatalf("request_type = %q, want responses", msg.RequestType)
	}
	if !reflect.DeepEqual(msg.Request, map[string]interface{}{
		"model": "gpt-test",
		"input": "hello",
	}) {
		t.Fatalf("request = %#v, want decoded payload", msg.Request)
	}
}
