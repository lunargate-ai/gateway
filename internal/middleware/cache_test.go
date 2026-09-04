package middleware

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestGenerateKeyIncludesCompleteRequest(t *testing.T) {
	newRequest := func() *models.UnifiedRequest {
		return &models.UnifiedRequest{
			Model: "openai/gpt-4o-mini",
			Messages: []models.Message{{
				Role:    "user",
				Content: "hello",
			}},
		}
	}

	floatPtr := func(value float64) *float64 { return &value }
	intPtr := func(value int) *int { return &value }

	tests := []struct {
		name   string
		mutate func(*models.UnifiedRequest)
	}{
		{name: "model", mutate: func(req *models.UnifiedRequest) { req.Model = "anthropic/claude-sonnet-4-5" }},
		{name: "messages", mutate: func(req *models.UnifiedRequest) { req.Messages[0].Content = "different" }},
		{name: "temperature", mutate: func(req *models.UnifiedRequest) { req.Temperature = floatPtr(0.2) }},
		{name: "top_p", mutate: func(req *models.UnifiedRequest) { req.TopP = floatPtr(0.8) }},
		{name: "top_k", mutate: func(req *models.UnifiedRequest) { req.TopK = intPtr(20) }},
		{name: "n", mutate: func(req *models.UnifiedRequest) { req.N = intPtr(2) }},
		{name: "stream", mutate: func(req *models.UnifiedRequest) { req.Stream = true }},
		{name: "stream_options", mutate: func(req *models.UnifiedRequest) {
			req.StreamOptions = &models.StreamOptions{IncludeUsage: true}
		}},
		{name: "stop", mutate: func(req *models.UnifiedRequest) { req.Stop = []interface{}{"END"} }},
		{name: "max_tokens", mutate: func(req *models.UnifiedRequest) { req.MaxTokens = intPtr(128) }},
		{name: "presence_penalty", mutate: func(req *models.UnifiedRequest) { req.PresencePenalty = floatPtr(0.4) }},
		{name: "frequency_penalty", mutate: func(req *models.UnifiedRequest) { req.FrequencyPenalty = floatPtr(0.6) }},
		{name: "logit_bias", mutate: func(req *models.UnifiedRequest) { req.LogitBias = map[string]int{"42": 10} }},
		{name: "user", mutate: func(req *models.UnifiedRequest) { req.User = "user-123" }},
		{name: "tools", mutate: func(req *models.UnifiedRequest) {
			req.Tools = []models.Tool{{
				Type: "function",
				Function: models.ToolFunction{
					Name:       "lookup",
					Parameters: map[string]interface{}{"type": "object"},
				},
			}}
		}},
		{name: "tool_choice", mutate: func(req *models.UnifiedRequest) { req.ToolChoice = "required" }},
		{name: "functions", mutate: func(req *models.UnifiedRequest) {
			req.Functions = []models.ToolFunction{{Name: "legacy_lookup"}}
		}},
		{name: "function_call", mutate: func(req *models.UnifiedRequest) {
			req.FunctionCall = map[string]interface{}{"name": "legacy_lookup"}
		}},
		{name: "response_format", mutate: func(req *models.UnifiedRequest) {
			req.ResponseFormat = &models.ResponseFormat{Type: "json_object"}
		}},
		{name: "reasoning_effort", mutate: func(req *models.UnifiedRequest) { req.ReasoningEffort = "high" }},
		{name: "reasoning", mutate: func(req *models.UnifiedRequest) {
			req.Reasoning = &models.Reasoning{Effort: "medium"}
		}},
		{name: "seed", mutate: func(req *models.UnifiedRequest) { req.Seed = intPtr(1234) }},
		{name: "previous_response_id", mutate: func(req *models.UnifiedRequest) {
			req.PreviousResponseID = "resp_previous"
		}},
	}

	baseKey := GenerateKey(newRequest())
	if baseKey == "" {
		t.Fatal("expected non-empty cache key")
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := newRequest()
			tt.mutate(req)
			if got := GenerateKey(req); got == baseKey {
				t.Fatalf("expected %s to change cache key", tt.name)
			}
		})
	}
}

func TestGenerateKeyIsDeterministic(t *testing.T) {
	first := &models.UnifiedRequest{
		Model:     "openai/gpt-4o-mini",
		Messages:  []models.Message{{Role: "user", Content: "hello"}},
		LogitBias: map[string]int{"10": 1, "20": -1},
		ToolChoice: map[string]interface{}{
			"type": "function",
			"function": map[string]interface{}{
				"name": "lookup",
			},
		},
	}
	second := &models.UnifiedRequest{
		Model:     "openai/gpt-4o-mini",
		Messages:  []models.Message{{Role: "user", Content: "hello"}},
		LogitBias: map[string]int{"20": -1, "10": 1},
		ToolChoice: map[string]interface{}{
			"function": map[string]interface{}{
				"name": "lookup",
			},
			"type": "function",
		},
	}

	if firstKey, secondKey := GenerateKey(first), GenerateKey(second); firstKey != secondKey {
		t.Fatalf("expected equivalent requests to have the same key: %s != %s", firstKey, secondKey)
	}
}

func TestGenerateKeyMatchesEquivalentNormalizedRequests(t *testing.T) {
	tests := []struct {
		name      string
		legacy    *models.UnifiedRequest
		canonical *models.UnifiedRequest
	}{
		{
			name: "legacy functions",
			legacy: &models.UnifiedRequest{
				Model:        "openai/gpt-4o-mini",
				Messages:     []models.Message{{Role: "user", Content: "hello"}},
				Functions:    []models.ToolFunction{{Name: "lookup"}},
				FunctionCall: map[string]interface{}{"name": "lookup"},
			},
			canonical: &models.UnifiedRequest{
				Model:    "openai/gpt-4o-mini",
				Messages: []models.Message{{Role: "user", Content: "hello"}},
				Tools: []models.Tool{{
					Type:     "function",
					Function: models.ToolFunction{Name: "lookup"},
				}},
				ToolChoice: map[string]interface{}{
					"type": "function",
					"function": map[string]interface{}{
						"name": "lookup",
					},
				},
			},
		},
		{
			name: "reasoning object",
			legacy: &models.UnifiedRequest{
				Model:     "openai/gpt-4o-mini",
				Messages:  []models.Message{{Role: "user", Content: "hello"}},
				Reasoning: &models.Reasoning{Effort: "high"},
			},
			canonical: &models.UnifiedRequest{
				Model:           "openai/gpt-4o-mini",
				Messages:        []models.Message{{Role: "user", Content: "hello"}},
				ReasoningEffort: "high",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := models.NormalizeUnifiedRequest(tt.legacy); err != nil {
				t.Fatalf("normalize legacy request: %v", err)
			}
			if err := models.NormalizeUnifiedRequest(tt.canonical); err != nil {
				t.Fatalf("normalize canonical request: %v", err)
			}

			legacyKey := GenerateKey(tt.legacy)
			canonicalKey := GenerateKey(tt.canonical)
			if legacyKey != canonicalKey {
				t.Fatalf("expected equivalent normalized requests to match: %s != %s", legacyKey, canonicalKey)
			}
		})
	}
}

func TestGenerateKeyReturnsEmptyForNilRequest(t *testing.T) {
	if got := GenerateKey(nil); got != "" {
		t.Fatalf("expected empty key for nil request, got %q", got)
	}
}

func TestGenerateKeyIncludesUnknownRawFieldsAndTargetContract(t *testing.T) {
	base := &models.UnifiedRequest{
		RawJSON: json.RawMessage(`{"model":"gpt-5.4","messages":[],"service_tier":"default"}`),
		Model:   "openai/gpt-5.4",
	}
	changedRaw := *base
	changedRaw.RawJSON = json.RawMessage(`{"messages":[],"service_tier":"priority","model":"gpt-5.4"}`)
	if GenerateKeyForTarget(base, "openai", "chat_completions") == GenerateKeyForTarget(&changedRaw, "openai", "chat_completions") {
		t.Fatal("unknown native field did not change the cache key")
	}

	reordered := *base
	reordered.RawJSON = json.RawMessage(`{"service_tier":"default","messages":[],"model":"gpt-5.4"}`)
	if GenerateKeyForTarget(base, "openai", "chat_completions") != GenerateKeyForTarget(&reordered, "openai", "chat_completions") {
		t.Fatal("equivalent raw object ordering changed the cache key")
	}

	baseKey := GenerateKeyForTarget(base, "openai", "chat_completions")
	if baseKey == GenerateKeyForTarget(base, "openai-secondary", "chat_completions") {
		t.Fatal("provider did not change the cache key")
	}
	if baseKey == GenerateKeyForTarget(base, "openai", "responses") {
		t.Fatal("upstream request type did not change the cache key")
	}
}

func TestGenerateKeyForResolvedTargetIncludesEffectiveModel(t *testing.T) {
	request := &models.UnifiedRequest{
		RawJSON: json.RawMessage(`{"model":"client-model","messages":[]}`),
		Model:   "shared/client-model",
	}
	primary := GenerateKeyForResolvedTarget(request, "shared", "model-one", "chat_completions")
	fallback := GenerateKeyForResolvedTarget(request, "shared", "model-two", "chat_completions")
	if primary == fallback {
		t.Fatal("effective target model did not change the cache key")
	}
	if padded := GenerateKeyForResolvedTarget(request, "shared", " model-one ", "chat_completions"); primary != padded {
		t.Fatal("effective target model whitespace changed the cache key")
	}
}

func TestGenerateKeyIncludesProviderControlHeaders(t *testing.T) {
	request := &models.UnifiedRequest{
		RawJSON: json.RawMessage(`{"model":"gpt-5.4","messages":[]}`),
		Model:   "openai/gpt-5.4",
	}
	base := GenerateKeyForTargetWithHeaders(request, "openai", "chat_completions", nil)
	first := GenerateKeyForTargetWithHeaders(request, "openai", "chat_completions", http.Header{
		"openai-beta":     {"responses=v1"},
		"Idempotency-Key": {"request-one"},
	})
	second := GenerateKeyForTargetWithHeaders(request, "openai", "chat_completions", http.Header{
		"OpenAI-Beta":     {"responses=v2"},
		"Idempotency-Key": {"request-one"},
	})
	if base == first {
		t.Fatal("provider-control headers did not change the cache key")
	}
	if first == second {
		t.Fatal("OpenAI-Beta value did not change the cache key")
	}
	equivalent := GenerateKeyForTargetWithHeaders(request, "openai", "chat_completions", http.Header{
		"OPENAI-BETA":     {" responses=v1 "},
		"idempotency-key": {"request-one"},
	})
	if first != equivalent {
		t.Fatal("header casing or surrounding whitespace changed the cache key")
	}
}

func TestGenerateEmbeddingsKeyIncludesUnknownRawFieldsAndTarget(t *testing.T) {
	base := &models.EmbeddingsRequest{
		RawJSON: json.RawMessage(`{"model":"embed","input":"hello","future":"one"}`),
		Model:   "openai/embed",
		Input:   "hello",
	}
	changed := *base
	changed.RawJSON = json.RawMessage(`{"model":"embed","input":"hello","future":"two"}`)
	if GenerateEmbeddingsKeyForTarget(base, "openai", "embeddings") == GenerateEmbeddingsKeyForTarget(&changed, "openai", "embeddings") {
		t.Fatal("unknown embeddings field did not change the cache key")
	}
	if GenerateEmbeddingsKeyForTarget(base, "openai", "embeddings") == GenerateEmbeddingsKeyForTarget(base, "ollama", "embeddings") {
		t.Fatal("embeddings provider did not change the cache key")
	}
}

func TestGenerateEmbeddingsKeyForResolvedTargetIncludesEffectiveModel(t *testing.T) {
	request := &models.EmbeddingsRequest{
		RawJSON: json.RawMessage(`{"model":"client-model","input":"hello"}`),
		Model:   "shared/client-model",
		Input:   "hello",
	}
	primary := GenerateEmbeddingsKeyForResolvedTarget(request, "shared", "model-one", "embeddings")
	fallback := GenerateEmbeddingsKeyForResolvedTarget(request, "shared", "model-two", "embeddings")
	if primary == fallback {
		t.Fatal("effective embeddings target model did not change the cache key")
	}
	if padded := GenerateEmbeddingsKeyForResolvedTarget(request, "shared", " model-one ", "embeddings"); primary != padded {
		t.Fatal("effective embeddings target model whitespace changed the cache key")
	}
}

func TestGenerateEmbeddingsKeyIncludesProviderControlHeaders(t *testing.T) {
	request := &models.EmbeddingsRequest{
		RawJSON: json.RawMessage(`{"model":"embed","input":"hello"}`),
		Model:   "openai/embed",
		Input:   "hello",
	}
	first := GenerateEmbeddingsKeyForTargetWithHeaders(request, "openai", "embeddings", http.Header{
		"OpenAI-Beta": {"embeddings=v1"},
	})
	second := GenerateEmbeddingsKeyForTargetWithHeaders(request, "openai", "embeddings", http.Header{
		"OpenAI-Beta": {"embeddings=v2"},
	})
	if first == second {
		t.Fatal("embeddings provider-control header did not change the cache key")
	}
}

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

func TestCache_UpdateConfig_PreservesEntriesOnIdenticalReload(t *testing.T) {
	cfg := config.CacheConfig{Enabled: true, TTL: time.Minute, MaxSize: 8}
	cache := NewCache(cfg)
	t.Cleanup(cache.Stop)

	cache.Set("k", "v")
	cache.mu.RLock()
	originalEntry := cache.entries["k"]
	originalBytes := cache.totalBytes
	cache.mu.RUnlock()
	cache.UpdateConfig(cfg)

	if got := cache.Get("k"); got != "v" {
		t.Fatalf("identical reload discarded cached value: got %#v", got)
	}
	cache.mu.RLock()
	retainedEntry := cache.entries["k"]
	retainedBytes := cache.totalBytes
	cache.mu.RUnlock()
	if retainedEntry != originalEntry || retainedBytes != originalBytes {
		t.Fatalf("identical reload changed cache state: entry_retained=%t bytes=%d, want %d", retainedEntry == originalEntry, retainedBytes, originalBytes)
	}

	changed := cfg
	changed.TTL = 2 * time.Minute
	cache.UpdateConfig(changed)
	if got := cache.Get("k"); got != nil {
		t.Fatalf("real config change retained stale cached value: got %#v", got)
	}
	cache.mu.RLock()
	changedBytes := cache.totalBytes
	cache.mu.RUnlock()
	if changedBytes != 0 {
		t.Fatalf("real config change retained byte accounting: got %d", changedBytes)
	}
}

func TestCacheExpiredObservationDoesNotDeleteConcurrentRefresh(t *testing.T) {
	cache := NewCache(config.CacheConfig{Enabled: true, TTL: time.Minute, MaxSize: 10})
	t.Cleanup(cache.Stop)

	expiredResponse, err := newCachedResponse("expired")
	if err != nil {
		t.Fatalf("prepare expired response: %v", err)
	}
	expiredSize, ok := expiredResponse.sizeWithinLimit("same-key", cache.cfg.MaxEntryBytes)
	if !ok {
		t.Fatal("expired response unexpectedly exceeds cache limit")
	}
	expired := &CacheEntry{
		response:  expiredResponse,
		CreatedAt: time.Now().Add(-2 * time.Minute),
		ExpiresAt: time.Now().Add(-time.Minute),
		sizeBytes: expiredSize,
	}
	refreshedResponse, err := newCachedResponse("fresh")
	if err != nil {
		t.Fatalf("prepare refreshed response: %v", err)
	}
	refreshedSize, ok := refreshedResponse.sizeWithinLimit("same-key", cache.cfg.MaxEntryBytes)
	if !ok {
		t.Fatal("refreshed response unexpectedly exceeds cache limit")
	}
	refreshed := &CacheEntry{
		response:  refreshedResponse,
		CreatedAt: time.Now(),
		ExpiresAt: time.Now().Add(time.Minute),
		sizeBytes: refreshedSize,
	}

	cache.mu.Lock()
	cache.entries["same-key"] = refreshed
	cache.totalBytes = refreshedSize
	cache.mu.Unlock()

	// Simulate Get observing an expired entry immediately before Set replaces
	// it. Cleanup must compare entry identity, not only the shared key.
	cache.deleteObservedEntry("same-key", expired)

	cache.mu.RLock()
	got := cache.entries["same-key"]
	cache.mu.RUnlock()
	if got != refreshed {
		t.Fatalf("concurrent refresh was removed: got %#v, want %#v", got, refreshed)
	}
	assertCacheAccounting(t, cache)
}

func TestCacheRejectsOversizedEntryWithoutReplacingExisting(t *testing.T) {
	const key = "same"
	entryLimit := testCacheResponseSize(t, key, "small")
	cache := NewCache(config.CacheConfig{
		Enabled:       true,
		TTL:           time.Minute,
		MaxSize:       8,
		MaxEntryBytes: entryLimit,
		MaxBytes:      entryLimit * 4,
	})
	t.Cleanup(cache.Stop)

	cache.Set(key, "small")
	cache.mu.RLock()
	originalEntry := cache.entries[key]
	originalBytes := cache.totalBytes
	cache.mu.RUnlock()

	cache.Set(key, strings.Repeat("x", entryLimit+1))
	if got := cache.Get(key); got != "small" {
		t.Fatalf("oversized overwrite changed existing value: got %#v", got)
	}
	cache.Set("new", strings.Repeat("y", entryLimit+1))
	if got := cache.Get("new"); got != nil {
		t.Fatalf("oversized new entry was retained: got %#v", got)
	}

	cache.mu.RLock()
	retainedEntry := cache.entries[key]
	retainedBytes := cache.totalBytes
	cache.mu.RUnlock()
	if retainedEntry != originalEntry || retainedBytes != originalBytes {
		t.Fatalf("oversized insert changed cache state: entry_retained=%t bytes=%d, want %d", retainedEntry == originalEntry, retainedBytes, originalBytes)
	}
	assertCacheAccounting(t, cache)
}

func TestCacheEvictsUntilByteAndCountLimitsHold(t *testing.T) {
	const small = "12345678"
	const large = "1234567890123456"
	smallSize := testCacheResponseSize(t, "aa", small)
	largeSize := testCacheResponseSize(t, "dd", large)
	maxBytes := smallSize * 3
	if largeSize+smallSize > maxBytes || largeSize+2*smallSize <= maxBytes {
		t.Fatalf("invalid test sizes: small=%d large=%d max=%d", smallSize, largeSize, maxBytes)
	}
	cache := NewCache(config.CacheConfig{
		Enabled:       true,
		TTL:           time.Minute,
		MaxSize:       10,
		MaxEntryBytes: largeSize,
		MaxBytes:      maxBytes,
	})
	t.Cleanup(cache.Stop)

	for index, key := range []string{"aa", "bb", "cc"} {
		cache.Set(key, small)
		cache.mu.Lock()
		cache.entries[key].CreatedAt = time.Unix(int64(index+1), 0)
		cache.mu.Unlock()
	}
	cache.Set("dd", large)

	if got := cache.Get("aa"); got != nil {
		t.Fatalf("oldest entry survived byte eviction: %#v", got)
	}
	if got := cache.Get("bb"); got != nil {
		t.Fatalf("second-oldest entry survived required repeated eviction: %#v", got)
	}
	if got := cache.Get("cc"); got != small {
		t.Fatalf("newest small entry was evicted: %#v", got)
	}
	if got := cache.Get("dd"); got != large {
		t.Fatalf("new entry was not retained: %#v", got)
	}
	assertCacheAccounting(t, cache)

	countCache := NewCache(config.CacheConfig{
		Enabled:       true,
		TTL:           time.Minute,
		MaxSize:       2,
		MaxEntryBytes: 1024,
		MaxBytes:      4096,
	})
	t.Cleanup(countCache.Stop)
	for index, key := range []string{"aa", "bb", "cc"} {
		countCache.Set(key, small)
		countCache.mu.Lock()
		if entry := countCache.entries[key]; entry != nil {
			entry.CreatedAt = time.Unix(int64(index+1), 0)
		}
		countCache.mu.Unlock()
	}
	if got := countCache.Get("aa"); got != nil {
		t.Fatalf("oldest entry survived count eviction: %#v", got)
	}
	assertCacheAccounting(t, countCache)
}

func TestCacheOverwriteMaintainsExactAccounting(t *testing.T) {
	cache := NewCache(config.CacheConfig{
		Enabled:       true,
		TTL:           time.Minute,
		MaxSize:       2,
		MaxEntryBytes: 1024,
		MaxBytes:      4096,
	})
	t.Cleanup(cache.Stop)

	cache.Set("aa", "first")
	cache.Set("bb", "second")
	cache.mu.Lock()
	cache.entries["aa"].CreatedAt = time.Unix(2, 0)
	cache.entries["bb"].CreatedAt = time.Unix(1, 0)
	cache.mu.Unlock()

	cache.Set("aa", strings.Repeat("x", 64))
	if got := cache.Get("bb"); got != "second" {
		t.Fatalf("overwrite evicted an unrelated entry: got %#v", got)
	}
	wantBytes := testCacheResponseSize(t, "aa", strings.Repeat("x", 64)) + testCacheResponseSize(t, "bb", "second")
	cache.mu.RLock()
	gotBytes := cache.totalBytes
	cache.mu.RUnlock()
	if gotBytes != wantBytes {
		t.Fatalf("bytes after growing overwrite = %d, want %d", gotBytes, wantBytes)
	}

	cache.Set("aa", "x")
	wantBytes = testCacheResponseSize(t, "aa", "x") + testCacheResponseSize(t, "bb", "second")
	cache.mu.RLock()
	gotBytes = cache.totalBytes
	cache.mu.RUnlock()
	if gotBytes != wantBytes {
		t.Fatalf("bytes after shrinking overwrite = %d, want %d", gotBytes, wantBytes)
	}
	assertCacheAccounting(t, cache)
}

func TestCacheExpiryAndCleanupReleaseBytes(t *testing.T) {
	cache := NewCache(config.CacheConfig{
		Enabled:       true,
		TTL:           time.Minute,
		MaxSize:       2,
		MaxEntryBytes: 1024,
		MaxBytes:      4096,
	})
	t.Cleanup(cache.Stop)

	cache.Set("aa", "expired-on-get")
	cache.Set("bb", "expired-on-cleanup")
	cache.mu.Lock()
	cache.entries["aa"].ExpiresAt = time.Now().Add(-time.Second)
	cache.entries["bb"].ExpiresAt = time.Now().Add(-time.Second)
	cache.mu.Unlock()

	if got := cache.Get("aa"); got != nil {
		t.Fatalf("expired entry returned from Get: %#v", got)
	}
	cache.mu.RLock()
	bytesAfterGet := cache.totalBytes
	cache.mu.RUnlock()
	if want := testCacheResponseSize(t, "bb", "expired-on-cleanup"); bytesAfterGet != want {
		t.Fatalf("bytes after Get expiry = %d, want %d", bytesAfterGet, want)
	}

	cache.cleanupExpired(time.Now())
	cache.mu.RLock()
	remaining := len(cache.entries)
	remainingBytes := cache.totalBytes
	cache.mu.RUnlock()
	if remaining != 0 || remainingBytes != 0 {
		t.Fatalf("cleanup retained expired state: entries=%d bytes=%d", remaining, remainingBytes)
	}

	cache.Set("cc", "stale")
	cache.mu.Lock()
	cache.entries["cc"].ExpiresAt = time.Now().Add(-time.Second)
	cache.mu.Unlock()
	cache.Set("dd", "fresh")
	if got := cache.Get("cc"); got != nil {
		t.Fatalf("Set did not remove expired entry before insertion: %#v", got)
	}
	if got := cache.Get("dd"); got != "fresh" {
		t.Fatalf("fresh entry missing after expired cleanup: %#v", got)
	}
	assertCacheAccounting(t, cache)
}

func TestCacheReturnsIsolatedResponseClones(t *testing.T) {
	original := &models.UnifiedResponse{
		RawJSON: json.RawMessage(`{"id":"raw-id","future":{"value":1}}`),
		ID:      "typed-id",
		Choices: []models.Choice{{
			Message: &models.Message{
				Role: "assistant",
				Content: []interface{}{
					map[string]interface{}{"type": "text", "text": "original"},
				},
			},
		}},
	}
	cache := NewCache(config.CacheConfig{
		Enabled:       true,
		TTL:           time.Minute,
		MaxSize:       2,
		MaxEntryBytes: 4096,
		MaxBytes:      8192,
	})
	t.Cleanup(cache.Stop)
	cache.Set("response", original)

	original.ID = "mutated-source"
	original.RawJSON[0] = '['
	original.Choices[0].Message.Content.([]interface{})[0].(map[string]interface{})["text"] = "mutated-source"

	first, ok := cache.Get("response").(*models.UnifiedResponse)
	if !ok || first == nil {
		t.Fatalf("cached response type = %T", first)
	}
	if first.ID != "typed-id" || first.Choices[0].Message.Content.([]interface{})[0].(map[string]interface{})["text"] != "original" {
		t.Fatalf("source mutation reached cache: %#v", first)
	}
	if string(first.RawJSON) != `{"id":"raw-id","future":{"value":1}}` {
		t.Fatalf("raw response changed in cache: %s", first.RawJSON)
	}

	first.ID = "mutated-read"
	first.RawJSON[0] = '['
	first.Choices[0].Message.Content.([]interface{})[0].(map[string]interface{})["text"] = "mutated-read"
	second := cache.Get("response").(*models.UnifiedResponse)
	if second.ID != "typed-id" || second.Choices[0].Message.Content.([]interface{})[0].(map[string]interface{})["text"] != "original" {
		t.Fatalf("returned response shared cache state: %#v", second)
	}
	if string(second.RawJSON) != `{"id":"raw-id","future":{"value":1}}` {
		t.Fatalf("returned raw response shared cache state: %s", second.RawJSON)
	}
	assertCacheAccounting(t, cache)
}

func TestCacheReturnsIsolatedEmbeddingClones(t *testing.T) {
	original := &models.EmbeddingsResponse{
		RawJSON: json.RawMessage(`{"object":"list","future":true}`),
		Object:  "list",
		Data: []models.EmbeddingData{{
			Object:    "embedding",
			Embedding: models.NewFloatEmbeddingValue([]float64{0.25, 0.5}),
			Index:     0,
		}},
	}
	cache := NewCache(config.CacheConfig{
		Enabled:       true,
		TTL:           time.Minute,
		MaxSize:       2,
		MaxEntryBytes: 4096,
		MaxBytes:      8192,
	})
	t.Cleanup(cache.Stop)
	cache.Set("embedding", original)

	original.RawJSON[0] = '['
	original.Data[0].Embedding[0] = '9'
	first, ok := cache.Get("embedding").(*models.EmbeddingsResponse)
	if !ok || first == nil {
		t.Fatalf("cached embeddings type = %T", first)
	}
	if string(first.RawJSON) != `{"object":"list","future":true}` || string(first.Data[0].Embedding) != `[0.25,0.5]` {
		t.Fatalf("source mutation reached embeddings cache: raw=%s embedding=%s", first.RawJSON, first.Data[0].Embedding)
	}

	first.RawJSON[0] = '['
	first.Data[0].Embedding[0] = '8'
	second := cache.Get("embedding").(*models.EmbeddingsResponse)
	if string(second.RawJSON) != `{"object":"list","future":true}` || string(second.Data[0].Embedding) != `[0.25,0.5]` {
		t.Fatalf("returned embeddings shared cache state: raw=%s embedding=%s", second.RawJSON, second.Data[0].Embedding)
	}
	assertCacheAccounting(t, cache)
}

func TestCacheConcurrentAccountingRemainsBounded(t *testing.T) {
	cfg := config.CacheConfig{
		Enabled:       true,
		TTL:           250 * time.Microsecond,
		MaxSize:       16,
		MaxEntryBytes: 512,
		MaxBytes:      2048,
	}
	cache := NewCache(cfg)
	t.Cleanup(cache.Stop)

	var workers sync.WaitGroup
	for worker := 0; worker < 8; worker++ {
		worker := worker
		workers.Add(1)
		go func() {
			defer workers.Done()
			for index := 0; index < 400; index++ {
				key := fmt.Sprintf("key-%02d", (worker+index)%32)
				cache.Set(key, map[string]interface{}{
					"worker": worker,
					"value":  strings.Repeat("x", 16+(index%96)),
				})
				_ = cache.Get(key)
				if index%11 == 0 {
					cache.cleanupExpired(time.Now())
				}
			}
		}()
	}
	workers.Add(1)
	go func() {
		defer workers.Done()
		for index := 0; index < 400; index++ {
			cache.UpdateConfig(cfg)
		}
	}()
	workers.Wait()
	cache.cleanupExpired(time.Now())
	assertCacheAccounting(t, cache)
}

func testCacheResponseSize(t *testing.T, key string, response interface{}) int {
	t.Helper()
	stored, err := newCachedResponse(response)
	if err != nil {
		t.Fatalf("newCachedResponse: %v", err)
	}
	size, ok := stored.sizeWithinLimit(key, 1<<30)
	if !ok {
		t.Fatal("test response unexpectedly exceeded size helper limit")
	}
	return size
}

func assertCacheAccounting(t *testing.T, cache *Cache) {
	t.Helper()
	cache.mu.RLock()
	defer cache.mu.RUnlock()

	calculated := 0
	for key, entry := range cache.entries {
		if entry == nil {
			t.Fatalf("cache retained nil entry for %q", key)
		}
		size, ok := entry.response.sizeWithinLimit(key, cache.cfg.MaxEntryBytes)
		if !ok {
			t.Fatalf("entry %q exceeds per-entry limit", key)
		}
		if entry.sizeBytes != size {
			t.Fatalf("entry %q size = %d, want %d", key, entry.sizeBytes, size)
		}
		calculated += size
	}
	if cache.totalBytes != calculated {
		t.Fatalf("totalBytes = %d, calculated %d", cache.totalBytes, calculated)
	}
	if cache.totalBytes > cache.cfg.MaxBytes {
		t.Fatalf("totalBytes = %d exceeds max %d", cache.totalBytes, cache.cfg.MaxBytes)
	}
	if len(cache.entries) > cache.cfg.MaxSize {
		t.Fatalf("entries = %d exceeds max %d", len(cache.entries), cache.cfg.MaxSize)
	}
}
