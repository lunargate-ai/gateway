package middleware

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/pkg/models"
	"github.com/rs/zerolog/log"
)

// CacheEntry stores a cached response with its expiration time.
type CacheEntry struct {
	Response  interface{}
	CreatedAt time.Time
	ExpiresAt time.Time
}

// Cache provides in-memory exact-match caching for LLM responses.
type Cache struct {
	mu       sync.RWMutex
	entries  map[string]*CacheEntry
	cfg      config.CacheConfig
	stopCh   chan struct{}
	stopOnce sync.Once
}

// NewCache creates a new in-memory cache.
func NewCache(cfg config.CacheConfig) *Cache {
	c := &Cache{
		entries: make(map[string]*CacheEntry),
		cfg:     cfg,
		stopCh:  make(chan struct{}),
	}

	// Start cleanup goroutine
	go c.cleanup()

	return c
}

// GenerateKey creates a deterministic cache key from a request.
func GenerateKey(req *models.UnifiedRequest) string {
	// Create a normalized version of the request for hashing
	normalized := struct {
		Model      string           `json:"model"`
		Messages   []models.Message `json:"messages"`
		Temp       *float64         `json:"temperature,omitempty"`
		MaxTok     *int             `json:"max_tokens,omitempty"`
		Tools      []models.Tool    `json:"tools,omitempty"`
		ToolChoice interface{}      `json:"tool_choice,omitempty"`
	}{
		Model:      req.Model,
		Messages:   req.Messages,
		Temp:       req.Temperature,
		MaxTok:     req.MaxTokens,
		Tools:      req.Tools,
		ToolChoice: req.ToolChoice,
	}

	data, err := json.Marshal(normalized)
	if err != nil {
		return ""
	}

	hash := sha256.Sum256(data)
	return fmt.Sprintf("%x", hash[:16])
}

func GenerateEmbeddingsKey(req *models.EmbeddingsRequest) string {
	normalized := struct {
		Model          string      `json:"model"`
		Input          interface{} `json:"input"`
		EncodingFormat string      `json:"encoding_format,omitempty"`
		Dimensions     *int        `json:"dimensions,omitempty"`
		User           string      `json:"user,omitempty"`
	}{
		Model:          req.Model,
		Input:          req.Input,
		EncodingFormat: req.EncodingFormat,
		Dimensions:     req.Dimensions,
		User:           req.User,
	}

	data, err := json.Marshal(normalized)
	if err != nil {
		return ""
	}

	hash := sha256.Sum256(data)
	return fmt.Sprintf("%x", hash[:16])
}

// Get looks up a cached response. Returns nil if not found or expired.
func (c *Cache) Get(key string) interface{} {
	c.mu.RLock()
	if !c.cfg.Enabled {
		c.mu.RUnlock()
		return nil
	}
	entry, ok := c.entries[key]
	c.mu.RUnlock()

	if !ok {
		return nil
	}

	if time.Now().After(entry.ExpiresAt) {
		c.mu.Lock()
		delete(c.entries, key)
		c.mu.Unlock()
		return nil
	}

	log.Debug().Str("cache_key", key).Msg("cache hit")
	return entry.Response
}

// Set stores a response in the cache.
func (c *Cache) Set(key string, resp interface{}) {
	if key == "" {
		return
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.cfg.Enabled {
		return
	}

	// Evict if at capacity
	if len(c.entries) >= c.cfg.MaxSize {
		c.evictOldest()
	}

	now := time.Now()
	c.entries[key] = &CacheEntry{
		Response:  resp,
		CreatedAt: now,
		ExpiresAt: now.Add(c.cfg.TTL),
	}

	log.Debug().Str("cache_key", key).Dur("ttl", c.cfg.TTL).Msg("cached response")
}

// Enabled returns whether caching is enabled.
func (c *Cache) Enabled() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.cfg.Enabled
}

// UpdateConfig hot-reloads cache settings and resets entries so TTL/max-size
// semantics are immediately consistent with the new config.
func (c *Cache) UpdateConfig(cfg config.CacheConfig) {
	if c == nil {
		return
	}
	c.mu.Lock()
	c.cfg = cfg
	c.entries = make(map[string]*CacheEntry)
	c.mu.Unlock()
	log.Info().Msg("cache config updated")
}

// Stop shuts down the background cleanup loop.
func (c *Cache) Stop() {
	if c == nil {
		return
	}
	c.stopOnce.Do(func() {
		close(c.stopCh)
	})
}

func (c *Cache) evictOldest() {
	var oldestKey string
	var oldestTime time.Time

	for key, entry := range c.entries {
		if oldestKey == "" || entry.CreatedAt.Before(oldestTime) {
			oldestKey = key
			oldestTime = entry.CreatedAt
		}
	}

	if oldestKey != "" {
		delete(c.entries, oldestKey)
	}
}

func (c *Cache) cleanup() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			c.mu.Lock()
			now := time.Now()
			for key, entry := range c.entries {
				if now.After(entry.ExpiresAt) {
					delete(c.entries, key)
				}
			}
			c.mu.Unlock()
		case <-c.stopCh:
			return
		}
	}
}
