package middleware

import (
	"testing"
	"time"

	"github.com/lunargate-ai/gateway/internal/config"
)

func TestCache_StopIsIdempotent(t *testing.T) {
	cache := NewCache(config.CacheConfig{
		Enabled: true,
		TTL:     time.Minute,
		MaxSize: 8,
	})

	cache.Set("k", "v")
	if got := cache.Get("k"); got != "v" {
		t.Fatalf("expected cached value %q, got %#v", "v", got)
	}

	cache.Stop()
	cache.Stop()

	if got := cache.Get("k"); got != "v" {
		t.Fatalf("expected cached value to remain readable after Stop, got %#v", got)
	}
}

func TestCache_UpdateConfig_ResetsEntriesAndDisablesCache(t *testing.T) {
	cache := NewCache(config.CacheConfig{
		Enabled: true,
		TTL:     time.Minute,
		MaxSize: 8,
	})

	cache.Set("k", "v")
	cache.UpdateConfig(config.CacheConfig{
		Enabled: false,
		TTL:     time.Minute,
		MaxSize: 8,
	})

	if cache.Enabled() {
		t.Fatalf("expected cache to be disabled after config update")
	}
	if got := cache.Get("k"); got != nil {
		t.Fatalf("expected cache entries to be cleared after update, got %#v", got)
	}
}
