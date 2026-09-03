package middleware

import (
	"encoding/json"
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
