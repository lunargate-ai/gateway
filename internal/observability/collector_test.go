package observability

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
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
	queuedBytesBeforeReload := client.queuedPayloadBytes()
	if queuedBytesBeforeReload == 0 {
		t.Fatal("expected queued payload bytes before config reload")
	}

	client.UpdateConfig(config.GeneralConfig{
		APIKey:     "shared-secret",
		BackendURL: second.URL,
	}, config.DataSharingConfig{Enabled: true})
	if got := client.queuedPayloadBytes(); got != queuedBytesBeforeReload {
		t.Fatalf("queued payload bytes after config reload = %d, want %d", got, queuedBytesBeforeReload)
	}
	queuedBeforeReload := dequeueCollectorItem(t, client)

	if err := client.send(context.Background(), queuedBeforeReload); err != nil {
		t.Fatalf("send queued payload after reload: %v", err)
	}
	if firstRequests != 0 || len(secondAuthorizations) != 0 {
		t.Fatalf("queued payload was sent after backend change: first=%d second=%d", firstRequests, len(secondAuthorizations))
	}

	client.Enqueue(context.Background(), "queued-after-reload", []Event{{Type: "metric"}})
	queuedAfterReload := dequeueCollectorItem(t, client)
	if err := client.send(context.Background(), queuedAfterReload); err != nil {
		t.Fatalf("send payload queued after reload: %v", err)
	}
	if len(secondAuthorizations) != 1 || secondAuthorizations[0] != "Bearer shared-secret" {
		t.Fatalf("second collector authorizations = %q, want shared credential", secondAuthorizations)
	}

	client.Enqueue(context.Background(), "queued-before-key-reload", []Event{{Type: "metric"}})
	queuedBytesBeforeKeyReload := client.queuedPayloadBytes()
	client.UpdateConfig(config.GeneralConfig{
		APIKey:     "replacement-secret",
		BackendURL: second.URL,
	}, config.DataSharingConfig{Enabled: true})
	if got := client.queuedPayloadBytes(); got != queuedBytesBeforeKeyReload {
		t.Fatalf("queued payload bytes after credential reload = %d, want %d", got, queuedBytesBeforeKeyReload)
	}
	queuedBeforeKeyReload := dequeueCollectorItem(t, client)

	if err := client.send(context.Background(), queuedBeforeKeyReload); err != nil {
		t.Fatalf("send queued payload after credential reload: %v", err)
	}
	if len(secondAuthorizations) != 1 {
		t.Fatalf("queued payload was sent after credential change: second=%d", len(secondAuthorizations))
	}

	client.Enqueue(context.Background(), "queued-after-key-reload", []Event{{Type: "metric"}})
	queuedAfterKeyReload := dequeueCollectorItem(t, client)
	if err := client.send(context.Background(), queuedAfterKeyReload); err != nil {
		t.Fatalf("send payload queued after credential reload: %v", err)
	}
	if len(secondAuthorizations) != 2 || secondAuthorizations[1] != "Bearer replacement-secret" {
		t.Fatalf("second collector authorizations = %q, want replacement credential", secondAuthorizations)
	}
}

func TestCollectorClient_QueueByteLimits(t *testing.T) {
	client := newBoundedTestCollector(3, 8, 12)

	if status, queued := client.tryEnqueue(collectorItem{payload: make([]byte, 8)}); status != collectorEnqueueAccepted || queued != 8 {
		t.Fatalf("exact payload limit: status=%v queued=%d, want accepted and 8", status, queued)
	}
	if status, queued := client.tryEnqueue(collectorItem{payload: make([]byte, 9)}); status != collectorEnqueuePayloadTooLarge || queued != 8 {
		t.Fatalf("oversized payload: status=%v queued=%d, want payload-too-large and 8", status, queued)
	}
	if status, queued := client.tryEnqueue(collectorItem{payload: make([]byte, 5)}); status != collectorEnqueueBudgetExceeded || queued != 8 {
		t.Fatalf("over queue budget: status=%v queued=%d, want budget-exceeded and 8", status, queued)
	}
	if status, queued := client.tryEnqueue(collectorItem{payload: make([]byte, 4)}); status != collectorEnqueueAccepted || queued != 12 {
		t.Fatalf("exact queue limit: status=%v queued=%d, want accepted and 12", status, queued)
	}

	dequeueCollectorItem(t, client)
	if got := client.queuedPayloadBytes(); got != 4 {
		t.Fatalf("queued bytes after dequeue = %d, want 4", got)
	}
	if status, queued := client.tryEnqueue(collectorItem{payload: make([]byte, 8)}); status != collectorEnqueueAccepted || queued != 12 {
		t.Fatalf("reused queue budget: status=%v queued=%d, want accepted and 12", status, queued)
	}

	client.drainQueue()
	if got := client.queuedPayloadBytes(); got != 0 {
		t.Fatalf("queued bytes after drain = %d, want 0", got)
	}
}

func TestCollectorClient_QueueCapacityDropDoesNotConsumeBytes(t *testing.T) {
	client := newBoundedTestCollector(1, 8, 16)

	if status, queued := client.tryEnqueue(collectorItem{payload: make([]byte, 4)}); status != collectorEnqueueAccepted || queued != 4 {
		t.Fatalf("first enqueue: status=%v queued=%d, want accepted and 4", status, queued)
	}
	if status, queued := client.tryEnqueue(collectorItem{payload: make([]byte, 4)}); status != collectorEnqueueQueueFull || queued != 4 {
		t.Fatalf("full queue: status=%v queued=%d, want queue-full and 4", status, queued)
	}

	client.drainQueue()
	if got := client.queuedPayloadBytes(); got != 0 {
		t.Fatalf("queued bytes after drain = %d, want 0", got)
	}
}

func TestCollectorClient_OversizedPayloadIsDroppedAtomically(t *testing.T) {
	client := newBoundedTestCollector(1, 64, 64)
	client.cfg = normalizeCollectorConfig(config.GeneralConfig{
		APIKey:     "secret",
		BackendURL: "https://example.com/v1",
	}, config.DataSharingConfig{Enabled: true})

	client.Enqueue(context.Background(), "oversized", []Event{{
		Type: "request_log",
		Data: RequestLogEventData{Request: map[string]string{"input": "payload"}},
	}})

	if got := len(client.queue); got != 0 {
		t.Fatalf("queued items = %d, want oversized batch dropped", got)
	}
	if got := client.queuedPayloadBytes(); got != 0 {
		t.Fatalf("queued bytes = %d, want 0", got)
	}
}

func TestCollectorClient_ConcurrentQueueByteBudget(t *testing.T) {
	const (
		goroutines     = 32
		attemptsEach   = 100
		payloadBytes   = 17
		maxQueuedItems = 1000
	)
	client := newBoundedTestCollector(goroutines*attemptsEach, payloadBytes, payloadBytes*maxQueuedItems)

	var accepted atomic.Int64
	var budgetDrops atomic.Int64
	var wg sync.WaitGroup
	for range goroutines {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range attemptsEach {
				status, _ := client.tryEnqueue(collectorItem{payload: make([]byte, payloadBytes)})
				switch status {
				case collectorEnqueueAccepted:
					accepted.Add(1)
				case collectorEnqueueBudgetExceeded:
					budgetDrops.Add(1)
				default:
					t.Errorf("unexpected concurrent enqueue status: %v", status)
				}
			}
		}()
	}
	wg.Wait()

	if got := accepted.Load(); got != maxQueuedItems {
		t.Fatalf("accepted items = %d, want %d", got, maxQueuedItems)
	}
	if got := budgetDrops.Load(); got != goroutines*attemptsEach-maxQueuedItems {
		t.Fatalf("budget drops = %d, want %d", got, goroutines*attemptsEach-maxQueuedItems)
	}
	if got := client.queuedPayloadBytes(); got != payloadBytes*maxQueuedItems {
		t.Fatalf("queued bytes = %d, want %d", got, payloadBytes*maxQueuedItems)
	}

	client.drainQueue()
	if got := client.queuedPayloadBytes(); got != 0 {
		t.Fatalf("queued bytes after concurrent drain = %d, want 0", got)
	}
}

func TestCollectorClient_StopDrainsQueuedByteAccounting(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	client := newBoundedTestCollector(2, 8, 16)
	client.ctx = ctx
	client.cancel = cancel

	if status, _ := client.tryEnqueue(collectorItem{payload: make([]byte, 8)}); status != collectorEnqueueAccepted {
		t.Fatalf("enqueue status = %v, want accepted", status)
	}
	client.Stop()

	if got := len(client.queue); got != 0 {
		t.Fatalf("queued items after stop = %d, want 0", got)
	}
	if got := client.queuedPayloadBytes(); got != 0 {
		t.Fatalf("queued bytes after stop = %d, want 0", got)
	}
	if status, queued := client.tryEnqueue(collectorItem{payload: make([]byte, 1)}); status != collectorEnqueueStopped || queued != 0 {
		t.Fatalf("enqueue after stop: status=%v queued=%d, want stopped and 0", status, queued)
	}
}

func TestCollectorClient_ConcurrentStopLeavesNoQueuedBytes(t *testing.T) {
	const producers = 256
	ctx, cancel := context.WithCancel(context.Background())
	client := newBoundedTestCollector(producers, 8, producers*8)
	client.ctx = ctx
	client.cancel = cancel

	start := make(chan struct{})
	var wg sync.WaitGroup
	for range producers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			client.tryEnqueue(collectorItem{payload: make([]byte, 8)})
		}()
	}
	close(start)
	client.Stop()
	wg.Wait()

	if got := len(client.queue); got != 0 {
		t.Fatalf("queued items after concurrent stop = %d, want 0", got)
	}
	if got := client.queuedPayloadBytes(); got != 0 {
		t.Fatalf("queued bytes after concurrent stop = %d, want 0", got)
	}
}

func newBoundedTestCollector(capacity int, maxPayloadBytes, maxQueueBytes int64) *CollectorClient {
	return &CollectorClient{
		queue:           make(chan collectorItem, capacity),
		ctx:             context.Background(),
		maxPayloadBytes: maxPayloadBytes,
		maxQueueBytes:   maxQueueBytes,
	}
}

func dequeueCollectorItem(t *testing.T, client *CollectorClient) collectorItem {
	t.Helper()
	select {
	case item := <-client.queue:
		client.releaseQueuedPayload(item)
		return item
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for collector item")
		return collectorItem{}
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
