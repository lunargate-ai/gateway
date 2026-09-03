package observability

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/lunargate-ai/gateway/internal/config"
)

func TestCollectorClient_DropsQueuedPayloadAfterIdentityChange(t *testing.T) {
	var firstRequests int
	first := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		firstRequests++
		w.WriteHeader(http.StatusAccepted)
	}))
	defer first.Close()

	var secondAuthorizations []string
	second := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		secondAuthorizations = append(secondAuthorizations, r.Header.Get("Authorization"))
		w.WriteHeader(http.StatusAccepted)
	}))
	defer second.Close()

	client := &CollectorClient{
		gatewayVersion: "test",
		httpClient:     &http.Client{Timeout: time.Second},
		queue:          make(chan collectorItem, 2),
		ctx:            context.Background(),
		cfg: normalizeCollectorConfig(config.GeneralConfig{
			APIKey:     "shared-secret",
			BackendURL: first.URL,
		}, config.DataSharingConfig{Enabled: true}),
	}

	client.Enqueue(context.Background(), "queued-before-reload", []Event{{Type: "metric"}})
	queuedBeforeReload := <-client.queue

	client.UpdateConfig(config.GeneralConfig{
		APIKey:     "shared-secret",
		BackendURL: second.URL,
	}, config.DataSharingConfig{Enabled: true})

	if err := client.send(context.Background(), queuedBeforeReload); err != nil {
		t.Fatalf("send queued payload after reload: %v", err)
	}
	if firstRequests != 0 || len(secondAuthorizations) != 0 {
		t.Fatalf("queued payload was sent after backend change: first=%d second=%d", firstRequests, len(secondAuthorizations))
	}

	client.Enqueue(context.Background(), "queued-after-reload", []Event{{Type: "metric"}})
	queuedAfterReload := <-client.queue
	if err := client.send(context.Background(), queuedAfterReload); err != nil {
		t.Fatalf("send payload queued after reload: %v", err)
	}
	if len(secondAuthorizations) != 1 || secondAuthorizations[0] != "Bearer shared-secret" {
		t.Fatalf("second collector authorizations = %q, want shared credential", secondAuthorizations)
	}

	client.Enqueue(context.Background(), "queued-before-key-reload", []Event{{Type: "metric"}})
	queuedBeforeKeyReload := <-client.queue
	client.UpdateConfig(config.GeneralConfig{
		APIKey:     "replacement-secret",
		BackendURL: second.URL,
	}, config.DataSharingConfig{Enabled: true})

	if err := client.send(context.Background(), queuedBeforeKeyReload); err != nil {
		t.Fatalf("send queued payload after credential reload: %v", err)
	}
	if len(secondAuthorizations) != 1 {
		t.Fatalf("queued payload was sent after credential change: second=%d", len(secondAuthorizations))
	}

	client.Enqueue(context.Background(), "queued-after-key-reload", []Event{{Type: "metric"}})
	queuedAfterKeyReload := <-client.queue
	if err := client.send(context.Background(), queuedAfterKeyReload); err != nil {
		t.Fatalf("send payload queued after credential reload: %v", err)
	}
	if len(secondAuthorizations) != 2 || secondAuthorizations[1] != "Bearer replacement-secret" {
		t.Fatalf("second collector authorizations = %q, want replacement credential", secondAuthorizations)
	}
}

func TestCollectorClient_UpdateConfig_TogglesEnabledState(t *testing.T) {
	client := NewCollectorClient(config.GeneralConfig{}, config.DataSharingConfig{}, "test")
	defer client.Stop()

	if client.Enabled() {
		t.Fatalf("expected collector to start disabled with empty config")
	}
	if client.SharePrompts() {
		t.Fatalf("expected prompts sharing to be disabled")
	}

	client.UpdateConfig(config.GeneralConfig{
		APIKey:     "secret",
		BackendURL: "https://example.com/v1",
	}, config.DataSharingConfig{
		Enabled:        true,
		SharePrompts:   true,
		ShareResponses: true,
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
