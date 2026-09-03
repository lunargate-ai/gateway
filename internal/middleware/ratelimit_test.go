package middleware

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
)

func TestRateLimiterConfigSnapshotOwnsBuckets(t *testing.T) {
	rl := NewRateLimiter(config.RateLimitConfig{
		Enabled:           true,
		RequestsPerMinute: 60,
		BurstSize:         1,
	})

	oldSnapshot := rl.current.Load()
	oldBucket := rl.getBucket("client", oldSnapshot)

	rl.UpdateConfig(config.RateLimitConfig{
		Enabled:           true,
		RequestsPerMinute: 600,
		BurstSize:         5,
	})

	newSnapshot := rl.current.Load()
	newBucket := rl.getBucket("client", newSnapshot)

	if oldSnapshot == newSnapshot {
		t.Fatal("config update reused the previous snapshot")
	}
	if oldBucket == newBucket {
		t.Fatal("config update reused a bucket created with the previous config")
	}
	if got := rl.getBucket("client", oldSnapshot); got != oldBucket {
		t.Fatal("in-flight request did not retain its original snapshot bucket")
	}
	if oldBucket.maxTokens != 1 || oldBucket.refillRate != 1 {
		t.Fatalf("old bucket used mixed config: max=%v refill=%v", oldBucket.maxTokens, oldBucket.refillRate)
	}
	if newBucket.maxTokens != 5 || newBucket.refillRate != 10 {
		t.Fatalf("new bucket used mixed config: max=%v refill=%v", newBucket.maxTokens, newBucket.refillRate)
	}
}

func TestRateLimiterConcurrentConfigUpdatesAndRequests(t *testing.T) {
	configs := []config.RateLimitConfig{
		{Enabled: true, RequestsPerMinute: 60, BurstSize: 1},
		{Enabled: true, RequestsPerMinute: 600, BurstSize: 5},
		{Enabled: false, RequestsPerMinute: 120, BurstSize: 2},
	}
	rl := NewRateLimiter(configs[0])

	var handled atomic.Int64
	handler := rl.Middleware(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		handled.Add(1)
		w.WriteHeader(http.StatusNoContent)
	}))

	start := make(chan struct{})
	errs := make(chan error, 1)
	report := func(err error) {
		select {
		case errs <- err:
		default:
		}
	}

	var wg sync.WaitGroup
	for updater := 0; updater < 4; updater++ {
		wg.Add(1)
		go func(offset int) {
			defer wg.Done()
			<-start
			for i := 0; i < 64; i++ {
				rl.UpdateConfig(configs[(offset+i)%len(configs)])
			}
		}(updater)
	}

	for worker := 0; worker < 12; worker++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			<-start
			for i := 0; i < 128; i++ {
				req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
				req.RemoteAddr = fmt.Sprintf("192.0.2.%d:1234", worker+1)
				recorder := httptest.NewRecorder()
				handler.ServeHTTP(recorder, req)

				switch recorder.Code {
				case http.StatusNoContent:
				case http.StatusTooManyRequests:
				default:
					report(fmt.Errorf("unexpected status %d", recorder.Code))
					return
				}

				if limit := recorder.Header().Get("X-RateLimit-Limit"); limit != "" && limit != "60" && limit != "600" {
					report(fmt.Errorf("unexpected rate limit snapshot %q", limit))
					return
				}
			}
		}(worker)
	}

	close(start)
	wg.Wait()
	close(errs)

	for err := range errs {
		t.Fatal(err)
	}
	if handled.Load() == 0 {
		t.Fatal("all concurrent requests were rejected")
	}
}
