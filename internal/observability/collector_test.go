package observability

import (
	"net/http"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
)

func TestCollectorClient_UpdateConfig_TogglesEnabledState(t *testing.T) {
	client := NewCollectorClient(config.DataSharingConfig{}, "test")
	defer client.Stop()

	if client.Enabled() {
		t.Fatalf("expected collector to start disabled with empty config")
	}
	if client.SharePrompts() {
		t.Fatalf("expected prompts sharing to be disabled")
	}

	client.UpdateConfig(config.DataSharingConfig{
		Enabled:        true,
		SharePrompts:   true,
		ShareResponses: true,
		BackendURL:     "https://example.com/v1",
		APIKey:         "secret",
		GatewayLat:     "10.0",
		GatewayLon:     "20.0",
	})

	if !client.Enabled() {
		t.Fatalf("expected collector to become enabled after config update")
	}
	if !client.SharePrompts() || !client.ShareResponses() {
		t.Fatalf("expected prompt/response sharing to follow updated config")
	}
	if got := client.GatewayLat(); got != "10.0" {
		t.Fatalf("expected updated gateway lat, got %q", got)
	}
	if got := client.GatewayLon(); got != "20.0" {
		t.Fatalf("expected updated gateway lon, got %q", got)
	}
}

func TestIsRetryableSendError(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want bool
	}{
		{
			name: "unauthorized is permanent",
			err:  &httpStatusError{statusCode: http.StatusUnauthorized, detail: "invalid gateway API key"},
			want: false,
		},
		{
			name: "forbidden is permanent",
			err:  &httpStatusError{statusCode: http.StatusForbidden},
			want: false,
		},
		{
			name: "too many requests is retryable",
			err:  &httpStatusError{statusCode: http.StatusTooManyRequests},
			want: true,
		},
		{
			name: "server error is retryable",
			err:  &httpStatusError{statusCode: http.StatusBadGateway},
			want: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isRetryableSendError(tt.err); got != tt.want {
				t.Fatalf("isRetryableSendError() = %v, want %v", got, tt.want)
			}
		})
	}
}
