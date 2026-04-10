package middleware

import (
	"crypto/sha256"
	"encoding/hex"
	"net"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/security"
	"github.com/rs/zerolog/log"
)

// TokenBucket implements a simple in-memory token bucket rate limiter.
type TokenBucket struct {
	mu         sync.Mutex
	tokens     float64
	maxTokens  float64
	refillRate float64 // tokens per second
	lastRefill time.Time
}

func newTokenBucket(maxTokens float64, refillRate float64) *TokenBucket {
	return &TokenBucket{
		tokens:     maxTokens,
		maxTokens:  maxTokens,
		refillRate: refillRate,
		lastRefill: time.Now(),
	}
}

func (tb *TokenBucket) allow() (bool, float64) {
	tb.mu.Lock()
	defer tb.mu.Unlock()

	now := time.Now()
	elapsed := now.Sub(tb.lastRefill).Seconds()
	tb.tokens += elapsed * tb.refillRate
	if tb.tokens > tb.maxTokens {
		tb.tokens = tb.maxTokens
	}
	tb.lastRefill = now

	if tb.tokens >= 1 {
		tb.tokens--
		return true, tb.tokens
	}

	return false, 0
}

// RateLimiter is a middleware that limits request rates using token bucket algorithm.
type RateLimiter struct {
	mu         sync.RWMutex
	buckets    map[string]*bucketEntry
	cfg        config.RateLimitConfig
	maxBuckets int
	bucketTTL  time.Duration
}

type bucketEntry struct {
	bucket   *TokenBucket
	lastSeen time.Time
}

// NewRateLimiter creates a new rate limiter middleware.
func NewRateLimiter(cfg config.RateLimitConfig) *RateLimiter {
	return &RateLimiter{
		buckets:    make(map[string]*bucketEntry),
		cfg:        cfg,
		maxBuckets: 10000,
		bucketTTL:  15 * time.Minute,
	}
}

// UpdateConfig hot-reloads rate limit config.
func (rl *RateLimiter) UpdateConfig(cfg config.RateLimitConfig) {
	rl.mu.Lock()
	defer rl.mu.Unlock()
	rl.cfg = cfg
	// Reset buckets on config change
	rl.buckets = make(map[string]*bucketEntry)
	log.Info().Msg("rate limiter config updated")
}

// Middleware returns the HTTP middleware handler.
func (rl *RateLimiter) Middleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !rl.cfg.Enabled {
			next.ServeHTTP(w, r)
			return
		}

		key := extractRateLimitKey(r)
		bucket := rl.getBucket(key)

		allowed, remaining := bucket.allow()

		limit := rl.cfg.RequestsPerMinute
		w.Header().Set("X-RateLimit-Limit", strconv.Itoa(limit))
		w.Header().Set("X-RateLimit-Remaining", strconv.FormatFloat(remaining, 'f', 0, 64))

		if !allowed {
			w.Header().Set("Retry-After", "60")
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte(`{"error":{"message":"Rate limit exceeded","type":"rate_limit_error","code":"rate_limit_exceeded"}}`))
			return
		}

		next.ServeHTTP(w, r)
	})
}

func (rl *RateLimiter) getBucket(key string) *TokenBucket {
	now := time.Now()
	rl.mu.RLock()
	entry, ok := rl.buckets[key]
	rl.mu.RUnlock()
	if ok {
		rl.mu.Lock()
		if entry2, ok2 := rl.buckets[key]; ok2 {
			entry2.lastSeen = now
			b := entry2.bucket
			rl.mu.Unlock()
			return b
		}
		rl.mu.Unlock()
	}

	rl.mu.Lock()
	defer rl.mu.Unlock()

	if entry, ok = rl.buckets[key]; ok {
		entry.lastSeen = now
		return entry.bucket
	}

	rl.evictLocked(now)

	rpm := float64(rl.cfg.RequestsPerMinute)
	burst := float64(rl.cfg.BurstSize)
	if burst <= 0 {
		burst = rpm / 6 // default burst = 10s worth
	}
	b := newTokenBucket(burst, rpm/60.0)
	rl.buckets[key] = &bucketEntry{bucket: b, lastSeen: now}
	return b
}

func (rl *RateLimiter) evictLocked(now time.Time) {
	if rl.bucketTTL > 0 {
		for k, e := range rl.buckets {
			if e == nil {
				delete(rl.buckets, k)
				continue
			}
			if now.Sub(e.lastSeen) > rl.bucketTTL {
				delete(rl.buckets, k)
			}
		}
	}

	if rl.maxBuckets <= 0 {
		return
	}
	for len(rl.buckets) >= rl.maxBuckets {
		var oldestKey string
		oldestTime := now
		for k, e := range rl.buckets {
			if e == nil {
				oldestKey = k
				break
			}
			if e.lastSeen.Before(oldestTime) {
				oldestTime = e.lastSeen
				oldestKey = k
			}
		}
		if oldestKey == "" {
			return
		}
		delete(rl.buckets, oldestKey)
	}
}

func hashKey(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:])
}

func extractRateLimitKey(r *http.Request) string {
	if info, ok := security.AuthInfoFromContext(r.Context()); ok && strings.TrimSpace(info.Subject) != "" {
		return "subject:" + hashKey(info.Subject)
	}

	// Prefer API key header, then IP
	if key := strings.TrimSpace(r.Header.Get("X-API-Key")); key != "" {
		return "key:" + hashKey(key)
	}
	if key := strings.TrimSpace(r.Header.Get("Authorization")); key != "" {
		return "auth:" + hashKey(key)
	}
	addr := strings.TrimSpace(r.RemoteAddr)
	if host, _, err := net.SplitHostPort(addr); err == nil && host != "" {
		return "ip:" + host
	}
	return "ip:" + addr
}
