package remotecontrol

import (
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
)

func TestNewClientRequiresLoopbackSandboxEndpoint(t *testing.T) {
	tests := []struct {
		name       string
		baseURL    string
		wantClient bool
	}{
		{name: "IPv4 loopback", baseURL: "http://127.0.0.1:8080", wantClient: true},
		{name: "IPv6 loopback", baseURL: "http://[::1]:8080", wantClient: true},
		{name: "unverified localhost name", baseURL: "http://localhost:8080"},
		{name: "LAN address", baseURL: "http://10.2.0.153:8080"},
		{name: "unverified hostname", baseURL: "http://gateway.internal:8080"},
		{name: "opaque URL", baseURL: "http:127.0.0.1:8080"},
		{name: "unsupported scheme", baseURL: "file://127.0.0.1/tmp/socket"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client := NewClient(
				config.GeneralConfig{BackendURL: "https://api.lunargate.ai/v1", APIKey: "gateway-key"},
				config.DataSharingConfig{Enabled: true, RemoteControl: true},
				config.SecurityConfig{
					Enabled:  true,
					Provider: "api_key",
					APIKey:   config.APIKeyAuthConfig{Keys: []config.APIKeyCredential{{Value: "local-secret"}}},
				},
				"test",
				test.baseURL,
				nil,
				nil,
				nil,
			)
			if got := client != nil; got != test.wantClient {
				t.Fatalf("NewClient() non-nil = %v, want %v", got, test.wantClient)
			}
		})
	}
}
