package middleware

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"strings"
	"sync"
	"time"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/pkg/models"
	"github.com/rs/zerolog/log"
)

// CacheEntry stores a cached response with its expiration time.
type CacheEntry struct {
	response  cachedResponse
	CreatedAt time.Time
	ExpiresAt time.Time
	sizeBytes int
}

// cachedResponse retains only owned byte slices. The dynamic type is metadata
// used to reconstruct the public value on every Get; callers never share the
// stored representation with the cache or with one another.
type cachedResponse struct {
	responseType reflect.Type
	payload      []byte
	rawJSON      json.RawMessage
}

// Cache provides in-memory exact-match caching for LLM responses.
type Cache struct {
	mu         sync.RWMutex
	entries    map[string]*CacheEntry
	cfg        config.CacheConfig
	totalBytes int
	stopCh     chan struct{}
	stopOnce   sync.Once
}

// NewCache creates a new in-memory cache.
func NewCache(cfg config.CacheConfig) *Cache {
	c := &Cache{
		entries: make(map[string]*CacheEntry),
		cfg:     normalizeCacheConfig(cfg),
		stopCh:  make(chan struct{}),
	}

	// Start cleanup goroutine
	go c.cleanup()

	return c
}

// GenerateKey creates a deterministic cache key from a request.
func GenerateKey(req *models.UnifiedRequest) string {
	return GenerateKeyForTarget(req, "", "")
}

// GenerateKeyForTarget includes the complete client document and the resolved
// upstream contract. Unknown additive API fields therefore cannot collide in
// cache with a request that the native provider observes differently.
func GenerateKeyForTarget(req *models.UnifiedRequest, provider string, upstreamRequestType string) string {
	return GenerateKeyForTargetWithHeaders(req, provider, upstreamRequestType, nil)
}

// GenerateKeyForTargetWithHeaders includes forwarded provider-control headers
// whose values may change the upstream API contract or idempotency semantics.
func GenerateKeyForTargetWithHeaders(req *models.UnifiedRequest, provider string, upstreamRequestType string, headers http.Header) string {
	return generateKeyForResolvedTargetWithHeaders(req, provider, "", upstreamRequestType, headers)
}

// GenerateKeyForResolvedTarget includes the effective model selected for the
// upstream target in addition to its provider and protocol.
func GenerateKeyForResolvedTarget(req *models.UnifiedRequest, provider string, model string, upstreamRequestType string) string {
	return GenerateKeyForResolvedTargetWithHeaders(req, provider, model, upstreamRequestType, nil)
}

// GenerateKeyForResolvedTargetWithHeaders includes both the effective target
// model and forwarded provider-control headers in the cache contract.
func GenerateKeyForResolvedTargetWithHeaders(req *models.UnifiedRequest, provider string, model string, upstreamRequestType string, headers http.Header) string {
	return generateKeyForResolvedTargetWithHeaders(req, provider, model, upstreamRequestType, headers)
}

func generateKeyForResolvedTargetWithHeaders(req *models.UnifiedRequest, provider string, model string, upstreamRequestType string, headers http.Header) string {
	if req == nil {
		return ""
	}

	data, err := json.Marshal(cacheKeyMaterial{
		Raw:                 canonicalJSONDocument(req.RawJSON),
		Normalized:          req,
		Provider:            strings.TrimSpace(provider),
		ResolvedModel:       strings.TrimSpace(model),
		UpstreamRequestType: strings.ToLower(strings.TrimSpace(upstreamRequestType)),
		ProviderHeaders:     canonicalProviderControlHeaders(headers),
	})
	if err != nil {
		return ""
	}

	hash := sha256.Sum256(data)
	return fmt.Sprintf("%x", hash[:16])
}

func GenerateEmbeddingsKey(req *models.EmbeddingsRequest) string {
	return GenerateEmbeddingsKeyForTarget(req, "", "")
}

func GenerateEmbeddingsKeyForTarget(req *models.EmbeddingsRequest, provider string, upstreamRequestType string) string {
	return GenerateEmbeddingsKeyForTargetWithHeaders(req, provider, upstreamRequestType, nil)
}

// GenerateEmbeddingsKeyForTargetWithHeaders is the embeddings equivalent of
// GenerateKeyForTargetWithHeaders.
func GenerateEmbeddingsKeyForTargetWithHeaders(req *models.EmbeddingsRequest, provider string, upstreamRequestType string, headers http.Header) string {
	return generateEmbeddingsKeyForResolvedTargetWithHeaders(req, provider, "", upstreamRequestType, headers)
}

// GenerateEmbeddingsKeyForResolvedTarget is the embeddings equivalent of
// GenerateKeyForResolvedTarget.
func GenerateEmbeddingsKeyForResolvedTarget(req *models.EmbeddingsRequest, provider string, model string, upstreamRequestType string) string {
	return GenerateEmbeddingsKeyForResolvedTargetWithHeaders(req, provider, model, upstreamRequestType, nil)
}

// GenerateEmbeddingsKeyForResolvedTargetWithHeaders is the embeddings
// equivalent of GenerateKeyForResolvedTargetWithHeaders.
func GenerateEmbeddingsKeyForResolvedTargetWithHeaders(req *models.EmbeddingsRequest, provider string, model string, upstreamRequestType string, headers http.Header) string {
	return generateEmbeddingsKeyForResolvedTargetWithHeaders(req, provider, model, upstreamRequestType, headers)
}

func generateEmbeddingsKeyForResolvedTargetWithHeaders(req *models.EmbeddingsRequest, provider string, model string, upstreamRequestType string, headers http.Header) string {
	if req == nil {
		return ""
	}

	data, err := json.Marshal(cacheKeyMaterial{
		Raw:                 canonicalJSONDocument(req.RawJSON),
		Normalized:          req,
		Provider:            strings.TrimSpace(provider),
		ResolvedModel:       strings.TrimSpace(model),
		UpstreamRequestType: strings.ToLower(strings.TrimSpace(upstreamRequestType)),
		ProviderHeaders:     canonicalProviderControlHeaders(headers),
	})
	if err != nil {
		return ""
	}

	hash := sha256.Sum256(data)
	return fmt.Sprintf("%x", hash[:16])
}

type cacheKeyMaterial struct {
	Raw                 interface{} `json:"raw,omitempty"`
	Normalized          interface{} `json:"normalized"`
	Provider            string      `json:"provider,omitempty"`
	ResolvedModel       string      `json:"resolved_model,omitempty"`
	UpstreamRequestType string      `json:"upstream_request_type,omitempty"`
	ProviderHeaders     http.Header `json:"provider_headers,omitempty"`
}

var cacheProviderControlHeaders = []string{
	"Anthropic-Beta",
	"Idempotency-Key",
	"OpenAI-Beta",
}

func canonicalProviderControlHeaders(headers http.Header) http.Header {
	if len(headers) == 0 {
		return nil
	}
	canonical := make(http.Header)
	for rawName, values := range headers {
		for _, allowedName := range cacheProviderControlHeaders {
			if !strings.EqualFold(strings.TrimSpace(rawName), allowedName) {
				continue
			}
			for _, value := range values {
				if value = strings.TrimSpace(value); value != "" {
					canonical.Add(allowedName, value)
				}
			}
			break
		}
	}
	if len(canonical) == 0 {
		return nil
	}
	return canonical
}

func canonicalJSONDocument(raw json.RawMessage) interface{} {
	if len(bytes.TrimSpace(raw)) == 0 {
		return nil
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	document, duplicateKey, err := decodeCanonicalJSONValue(decoder)
	if err != nil {
		return string(raw)
	}
	if _, err := decoder.Token(); err != io.EOF {
		return string(raw)
	}
	// Duplicate object names have parser-dependent semantics. Preserve the
	// complete document in the key rather than treating it as equivalent to a
	// single-key object and risking a cross-request cache hit.
	if duplicateKey {
		return string(raw)
	}
	return document
}

func decodeCanonicalJSONValue(decoder *json.Decoder) (interface{}, bool, error) {
	token, err := decoder.Token()
	if err != nil {
		return nil, false, err
	}
	delimiter, composite := token.(json.Delim)
	if !composite {
		return token, false, nil
	}

	switch delimiter {
	case '{':
		object := make(map[string]interface{})
		seen := make(map[string]struct{})
		duplicateKey := false
		for decoder.More() {
			keyToken, err := decoder.Token()
			if err != nil {
				return nil, false, err
			}
			key, ok := keyToken.(string)
			if !ok {
				return nil, false, fmt.Errorf("JSON object key is not a string")
			}
			if _, exists := seen[key]; exists {
				duplicateKey = true
			}
			seen[key] = struct{}{}
			value, nestedDuplicate, err := decodeCanonicalJSONValue(decoder)
			if err != nil {
				return nil, false, err
			}
			duplicateKey = duplicateKey || nestedDuplicate
			object[key] = value
		}
		closing, err := decoder.Token()
		if err != nil || closing != json.Delim('}') {
			return nil, false, fmt.Errorf("invalid JSON object terminator")
		}
		return object, duplicateKey, nil
	case '[':
		array := make([]interface{}, 0)
		duplicateKey := false
		for decoder.More() {
			value, nestedDuplicate, err := decodeCanonicalJSONValue(decoder)
			if err != nil {
				return nil, false, err
			}
			duplicateKey = duplicateKey || nestedDuplicate
			array = append(array, value)
		}
		closing, err := decoder.Token()
		if err != nil || closing != json.Delim(']') {
			return nil, false, fmt.Errorf("invalid JSON array terminator")
		}
		return array, duplicateKey, nil
	default:
		return nil, false, fmt.Errorf("unexpected JSON delimiter %q", delimiter)
	}
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
	if entry == nil {
		c.deleteObservedEntry(key, nil)
		return nil
	}

	if !time.Now().Before(entry.ExpiresAt) {
		c.deleteObservedEntry(key, entry)
		return nil
	}

	response, err := entry.response.clone()
	if err != nil {
		c.deleteObservedEntry(key, entry)
		log.Error().Err(err).Str("cache_key", key).Msg("failed to clone cached response")
		return nil
	}

	log.Debug().Str("cache_key", key).Msg("cache hit")
	return response
}

// deleteObservedEntry removes only the exact entry seen by Get. A concurrent
// Set may already have refreshed the same key; deleting by key alone would
// incorrectly discard that newer value.
func (c *Cache) deleteObservedEntry(key string, observed *CacheEntry) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if current, ok := c.entries[key]; ok && current == observed {
		c.removeLocked(key, current)
	}
}

// Set stores a response in the cache.
func (c *Cache) Set(key string, resp interface{}) {
	if key == "" {
		return
	}
	c.mu.RLock()
	enabled := c.cfg.Enabled
	c.mu.RUnlock()
	if !enabled {
		return
	}

	stored, err := newCachedResponse(resp)
	if err != nil {
		log.Warn().Err(err).Str("cache_key", key).Msg("response cannot be cached")
		return
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.cfg.Enabled {
		return
	}
	if c.cfg.MaxSize <= 0 || c.cfg.MaxEntryBytes <= 0 || c.cfg.MaxBytes <= 0 {
		return
	}

	now := time.Now()
	c.cleanupExpiredLocked(now)
	size, withinLimit := stored.sizeWithinLimit(key, c.cfg.MaxEntryBytes)
	if !withinLimit || size > c.cfg.MaxBytes {
		log.Warn().
			Str("cache_key", key).
			Int("max_entry_bytes", c.cfg.MaxEntryBytes).
			Msg("response exceeds cache entry limit")
		return
	}

	if existing, ok := c.entries[key]; ok {
		c.removeLocked(key, existing)
	}
	for len(c.entries) >= c.cfg.MaxSize || size > c.cfg.MaxBytes-c.totalBytes {
		if !c.evictOldestLocked() {
			return
		}
	}
	c.entries[key] = &CacheEntry{
		response:  stored,
		CreatedAt: now,
		ExpiresAt: now.Add(c.cfg.TTL),
		sizeBytes: size,
	}
	c.totalBytes += size

	log.Debug().Str("cache_key", key).Dur("ttl", c.cfg.TTL).Int("entry_bytes", size).Msg("cached response")
}

// Enabled returns whether caching is enabled.
func (c *Cache) Enabled() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.cfg.Enabled
}

// UpdateConfig hot-reloads cache settings and resets entries when their
// semantics change. An identical reload keeps valid cached responses intact.
func (c *Cache) UpdateConfig(cfg config.CacheConfig) {
	if c == nil {
		return
	}
	cfg = normalizeCacheConfig(cfg)
	c.mu.Lock()
	if c.cfg == cfg {
		c.mu.Unlock()
		return
	}
	c.cfg = cfg
	c.entries = make(map[string]*CacheEntry)
	c.totalBytes = 0
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

func (c *Cache) evictOldestLocked() bool {
	var oldestKey string
	var oldestTime time.Time

	for key, entry := range c.entries {
		if entry == nil {
			c.removeLocked(key, nil)
			return true
		}
		if oldestKey == "" || entry.CreatedAt.Before(oldestTime) {
			oldestKey = key
			oldestTime = entry.CreatedAt
		}
	}

	if oldestKey == "" {
		return false
	}
	c.removeLocked(oldestKey, c.entries[oldestKey])
	return true
}

func (c *Cache) cleanupExpiredLocked(now time.Time) {
	for key, entry := range c.entries {
		if entry == nil || !now.Before(entry.ExpiresAt) {
			c.removeLocked(key, entry)
		}
	}
}

func (c *Cache) cleanupExpired(now time.Time) {
	c.mu.Lock()
	c.cleanupExpiredLocked(now)
	c.mu.Unlock()
}

func (c *Cache) removeLocked(key string, entry *CacheEntry) {
	delete(c.entries, key)
	if entry != nil {
		c.totalBytes -= entry.sizeBytes
		if c.totalBytes < 0 {
			c.totalBytes = 0
		}
	}
}

func (c *Cache) cleanup() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			c.cleanupExpired(time.Now())
		case <-c.stopCh:
			return
		}
	}
}

func normalizeCacheConfig(cfg config.CacheConfig) config.CacheConfig {
	// Config files receive these defaults from Viper. Applying them here as
	// well keeps direct constructors and no-op reloads backward-compatible.
	if cfg.MaxEntryBytes == 0 {
		cfg.MaxEntryBytes = config.DefaultCacheMaxEntryBytes
	}
	if cfg.MaxBytes == 0 {
		cfg.MaxBytes = config.DefaultCacheMaxBytes
	}
	return cfg
}

func newCachedResponse(response interface{}) (cachedResponse, error) {
	if response == nil {
		return cachedResponse{}, fmt.Errorf("response is nil")
	}
	value := reflect.ValueOf(response)
	switch value.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		if value.IsNil() {
			return cachedResponse{}, fmt.Errorf("response is nil")
		}
	}

	payload, err := json.Marshal(response)
	if err != nil {
		return cachedResponse{}, fmt.Errorf("failed to encode response: %w", err)
	}
	stored := cachedResponse{
		responseType: reflect.TypeOf(response),
		payload:      append([]byte(nil), payload...),
		rawJSON:      cachedRawJSON(response),
	}
	if _, err := stored.clone(); err != nil {
		return cachedResponse{}, fmt.Errorf("failed to clone response: %w", err)
	}
	return stored, nil
}

func cachedRawJSON(response interface{}) json.RawMessage {
	var raw json.RawMessage
	switch typed := response.(type) {
	case *models.UnifiedResponse:
		if typed != nil {
			raw = typed.RawJSON
		}
	case models.UnifiedResponse:
		raw = typed.RawJSON
	case *models.EmbeddingsResponse:
		if typed != nil {
			raw = typed.RawJSON
		}
	case models.EmbeddingsResponse:
		raw = typed.RawJSON
	}
	return append(json.RawMessage(nil), raw...)
}

func (r cachedResponse) sizeWithinLimit(key string, limit int) (int, bool) {
	if limit <= 0 || len(key) > limit {
		return 0, false
	}
	size := len(key)
	if len(r.payload) > limit-size {
		return 0, false
	}
	size += len(r.payload)
	if len(r.rawJSON) > limit-size {
		return 0, false
	}
	return size + len(r.rawJSON), true
}

func (r cachedResponse) clone() (interface{}, error) {
	if r.responseType == nil || len(r.payload) == 0 {
		return nil, fmt.Errorf("cached response is empty")
	}

	switch r.responseType {
	case reflect.TypeOf((*models.UnifiedResponse)(nil)):
		var response models.UnifiedResponse
		if err := json.Unmarshal(r.payload, &response); err != nil {
			return nil, err
		}
		response.RawJSON = append(json.RawMessage(nil), r.rawJSON...)
		return &response, nil
	case reflect.TypeOf(models.UnifiedResponse{}):
		var response models.UnifiedResponse
		if err := json.Unmarshal(r.payload, &response); err != nil {
			return nil, err
		}
		response.RawJSON = append(json.RawMessage(nil), r.rawJSON...)
		return response, nil
	case reflect.TypeOf((*models.EmbeddingsResponse)(nil)):
		var response models.EmbeddingsResponse
		if err := json.Unmarshal(r.payload, &response); err != nil {
			return nil, err
		}
		response.RawJSON = append(json.RawMessage(nil), r.rawJSON...)
		return &response, nil
	case reflect.TypeOf(models.EmbeddingsResponse{}):
		var response models.EmbeddingsResponse
		if err := json.Unmarshal(r.payload, &response); err != nil {
			return nil, err
		}
		response.RawJSON = append(json.RawMessage(nil), r.rawJSON...)
		return response, nil
	}

	if r.responseType.Kind() == reflect.Pointer {
		response := reflect.New(r.responseType.Elem())
		if err := json.Unmarshal(r.payload, response.Interface()); err != nil {
			return nil, err
		}
		return response.Interface(), nil
	}
	response := reflect.New(r.responseType)
	if err := json.Unmarshal(r.payload, response.Interface()); err != nil {
		return nil, err
	}
	return response.Elem().Interface(), nil
}
