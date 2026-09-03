package middleware

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"net"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/security"
	"github.com/rs/zerolog/log"
)

type peerAddressContextKey struct{}

// CapturePeerAddress retains the socket peer before proxy-derived middleware
// rewrites RemoteAddr. Until trusted proxies are configurable, unverified
// forwarding headers must not create independent rate-limit identities.
func CapturePeerAddress(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		peer := remoteHost(r.RemoteAddr)
		ctx := context.WithValue(r.Context(), peerAddressContextKey{}, peer)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

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
	current    atomic.Pointer[rateLimitSnapshot]
	maxBuckets int
	bucketTTL  time.Duration
}

type rateLimitSnapshot struct {
	cfg     config.RateLimitConfig
	mu      sync.Mutex
	buckets map[string]*bucketEntry
}

type bucketEntry struct {
	bucket   *TokenBucket
	lastSeen time.Time
}

// NewRateLimiter creates a new rate limiter middleware.
func NewRateLimiter(cfg config.RateLimitConfig) *RateLimiter {
	rl := &RateLimiter{
		maxBuckets: 10000,
		bucketTTL:  15 * time.Minute,
	}
	rl.current.Store(newRateLimitSnapshot(cfg))
	return rl
}

func newRateLimitSnapshot(cfg config.RateLimitConfig) *rateLimitSnapshot {
	return &rateLimitSnapshot{
		cfg:     cfg,
		buckets: make(map[string]*bucketEntry),
	}
}

// UpdateConfig hot-reloads rate limit config.
func (rl *RateLimiter) UpdateConfig(cfg config.RateLimitConfig) {
	if rl == nil {
		return
	}
	rl.current.Store(newRateLimitSnapshot(cfg))
	log.Info().Msg("rate limiter config updated")
}

// Middleware returns the HTTP middleware handler.
func (rl *RateLimiter) Middleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		snapshot := rl.current.Load()
		if snapshot == nil || !snapshot.cfg.Enabled {
			next.ServeHTTP(w, r)
			return
		}

		key := extractRateLimitKey(r)
		bucket := rl.getBucket(key, snapshot)

		allowed, remaining := bucket.allow()

		limit := snapshot.cfg.RequestsPerMinute
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

func (rl *RateLimiter) getBucket(key string, snapshot *rateLimitSnapshot) *TokenBucket {
	now := time.Now()
	snapshot.mu.Lock()
	defer snapshot.mu.Unlock()

	if entry, ok := snapshot.buckets[key]; ok {
		entry.lastSeen = now
		return entry.bucket
	}

	rl.evictLocked(snapshot, now)

	rpm := float64(snapshot.cfg.RequestsPerMinute)
	burst := float64(snapshot.cfg.BurstSize)
	if burst <= 0 {
		burst = rpm / 6 // default burst = 10s worth
	}
	b := newTokenBucket(burst, rpm/60.0)
	snapshot.buckets[key] = &bucketEntry{bucket: b, lastSeen: now}
	return b
}

func (rl *RateLimiter) evictLocked(snapshot *rateLimitSnapshot, now time.Time) {
	if rl.bucketTTL > 0 {
		for k, e := range snapshot.buckets {
			if e == nil {
				delete(snapshot.buckets, k)
				continue
			}
			if now.Sub(e.lastSeen) > rl.bucketTTL {
				delete(snapshot.buckets, k)
			}
		}
	}

	if rl.maxBuckets <= 0 {
		return
	}
	for len(snapshot.buckets) >= rl.maxBuckets {
		var oldestKey string
		oldestTime := now
		for k, e := range snapshot.buckets {
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
		delete(snapshot.buckets, oldestKey)
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

	if peer, ok := r.Context().Value(peerAddressContextKey{}).(string); ok {
		return "ip:" + strings.TrimSpace(peer)
	}
	return "ip:" + remoteHost(r.RemoteAddr)
}

func remoteHost(remoteAddr string) string {
	addr := strings.TrimSpace(remoteAddr)
	if host, _, err := net.SplitHostPort(addr); err == nil && host != "" {
		return host
	}
	return addr
}
