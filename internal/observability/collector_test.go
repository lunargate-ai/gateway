package observability

import (
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
		GatewayID:      "gw-1",
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
	if got := client.GatewayID(); got != "gw-1" {
		t.Fatalf("expected updated gateway ID, got %q", got)
	}
	if got := client.GatewayLat(); got != "10.0" {
		t.Fatalf("expected updated gateway lat, got %q", got)
	}
	if got := client.GatewayLon(); got != "20.0" {
		t.Fatalf("expected updated gateway lon, got %q", got)
	}
}
