package middleware

import (
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
)

func TestRateLimiterAutoBurstHasMinimumCapacity(t *testing.T) {
	tests := []struct {
		name      string
		rpm       int
		burstSize int
		wantBurst float64
	}{
		{name: "one request per minute", rpm: 1, wantBurst: 1},
		{name: "five requests per minute", rpm: 5, wantBurst: 1},
		{name: "six requests per minute", rpm: 6, wantBurst: 1},
		{name: "automatic ten second capacity", rpm: 60, wantBurst: 10},
		{name: "explicit burst is unchanged", rpm: 1, burstSize: 3, wantBurst: 3},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			rl := NewRateLimiter(config.RateLimitConfig{
				Enabled:           true,
				RequestsPerMinute: test.rpm,
				BurstSize:         test.burstSize,
			})
			snapshot := rl.current.Load()
			bucket := rl.getBucket("client", snapshot)
			if bucket.maxTokens != test.wantBurst {
				t.Fatalf("maxTokens = %v, want %v", bucket.maxTokens, test.wantBurst)
			}
			allowed, _, _ := bucket.allow()
			if !allowed {
				t.Fatal("new rate-limit bucket rejected its first request")
			}
		})
	}
}
