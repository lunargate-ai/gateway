package remotecontrol

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
)

func TestExecuteSandboxDoesNotFollowRedirectWithLocalCredential(t *testing.T) {
	var redirectTargetCalls atomic.Int32
	redirectTarget := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		redirectTargetCalls.Add(1)
		w.WriteHeader(http.StatusOK)
	}))
	defer redirectTarget.Close()

	var sourceCalls atomic.Int32
	source := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sourceCalls.Add(1)
		if got := r.Header.Get("X-Gateway-Key"); got != "Token local-test-key" {
			t.Errorf("source credential = %q, want configured local credential", got)
		}
		w.Header().Set("Location", redirectTarget.URL+"/capture")
		w.WriteHeader(http.StatusTemporaryRedirect)
		_, _ = w.Write([]byte(`{"redirected":true}`))
	}))
	defer source.Close()

	client := NewClient(
		config.GeneralConfig{BackendURL: "https://api.lunargate.ai/v1", APIKey: "lgw-test"},
		config.DataSharingConfig{Enabled: true, RemoteControl: true},
		config.SecurityConfig{
			Enabled:  true,
			Provider: "api_key",
			APIKey: config.APIKeyAuthConfig{
				Header: "X-Gateway-Key",
				Prefix: "Token",
				Keys: []config.APIKeyCredential{{
					Name:  "sandbox",
					Value: "local-test-key",
				}},
			},
		},
		"test",
		source.URL,
		nil,
		nil,
		nil,
	)
	if client == nil {
		t.Fatal("expected remote control client")
	}

	status, _, body, err := client.executeSandbox(context.Background(), sandboxExecuteMessage{
		Target:  sandboxTarget{Mode: "model", Value: "openai/gpt-test"},
		Request: map[string]interface{}{"messages": []interface{}{}},
	})
	if err != nil {
		t.Fatalf("executeSandbox returned error: %v", err)
	}
	if status != http.StatusTemporaryRedirect {
		t.Fatalf("status = %d, want %d", status, http.StatusTemporaryRedirect)
	}
	if response, ok := body.(map[string]interface{}); !ok || response["redirected"] != true {
		t.Fatalf("body = %#v, want original redirect response", body)
	}
	if got := sourceCalls.Load(); got != 1 {
		t.Fatalf("source calls = %d, want 1", got)
	}
	if got := redirectTargetCalls.Load(); got != 0 {
		t.Fatalf("redirect target calls = %d, want 0", got)
	}
}
