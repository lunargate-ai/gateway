package middleware

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"

	chimw "github.com/go-chi/chi/v5/middleware"
	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/security"
)

func TestRateLimitKeyIgnoresUnverifiedCredentialHeaders(t *testing.T) {
	first := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	first.RemoteAddr = "192.0.2.10:1234"
	first.Header.Set("Authorization", "Bearer attacker-one")
	first.Header.Set("X-API-Key", "attacker-one")
	second := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	second.RemoteAddr = "192.0.2.10:5678"
	second.Header.Set("Authorization", "Bearer attacker-two")
	second.Header.Set("X-API-Key", "attacker-two")

	if firstKey, secondKey := extractRateLimitKey(first), extractRateLimitKey(second); firstKey != secondKey {
		t.Fatalf("unverified headers split one peer into %q and %q", firstKey, secondKey)
	}
}

func TestRateLimitKeyUsesVerifiedSubject(t *testing.T) {
	first := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	first.RemoteAddr = "192.0.2.10:1234"
	first = first.WithContext(security.ContextWithAuthInfo(first.Context(), security.AuthInfo{Subject: "verified-client"}))
	second := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	second.RemoteAddr = "198.51.100.20:5678"
	second = second.WithContext(security.ContextWithAuthInfo(second.Context(), security.AuthInfo{Subject: "verified-client"}))

	if firstKey, secondKey := extractRateLimitKey(first), extractRateLimitKey(second); firstKey != secondKey || firstKey == "ip:192.0.2.10" {
		t.Fatalf("verified subject keys = %q and %q", firstKey, secondKey)
	}
}

func TestRateLimiterUsesSocketPeerBeforeRealIPRewrite(t *testing.T) {
	rl := NewRateLimiter(config.RateLimitConfig{
		Enabled:           true,
		RequestsPerMinute: 1,
		BurstSize:         1,
	})
	handler := CapturePeerAddress(chimw.RealIP(rl.Middleware(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))))

	for index, forwardedFor := range []string{"203.0.113.1", "203.0.113.2"} {
		request := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
		request.RemoteAddr = "192.0.2.10:1234"
		request.Header.Set("X-Forwarded-For", forwardedFor)
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		want := http.StatusNoContent
		if index == 1 {
			want = http.StatusTooManyRequests
		}
		if response.Code != want {
			t.Fatalf("request %d status = %d, want %d", index+1, response.Code, want)
		}
	}
}

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
