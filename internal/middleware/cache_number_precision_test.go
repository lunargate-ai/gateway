package middleware

import (
	"encoding/json"
	"testing"

	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestGenerateKeyForTargetDistinguishesLargeRawIntegers(t *testing.T) {
	first := &models.UnifiedRequest{
		RawJSON: json.RawMessage(`{"model":"test","messages":[],"opaque_id":9007199254740992}`),
		Model:   "test",
	}
	second := *first
	second.RawJSON = json.RawMessage(`{"model":"test","messages":[],"opaque_id":9007199254740993}`)

	firstKey := GenerateKeyForTarget(first, "openai", "chat_completions")
	secondKey := GenerateKeyForTarget(&second, "openai", "chat_completions")
	if firstKey == "" || secondKey == "" {
		t.Fatal("expected non-empty cache keys")
	}
	if firstKey == secondKey {
		t.Fatal("distinct integers above float64 precision produced the same chat cache key")
	}
}

func TestGenerateEmbeddingsKeyForTargetDistinguishesLargeRawIntegers(t *testing.T) {
	first := &models.EmbeddingsRequest{
		RawJSON: json.RawMessage(`{"model":"test","input":"hello","opaque_id":9007199254740992}`),
		Model:   "test",
		Input:   "hello",
	}
	second := *first
	second.RawJSON = json.RawMessage(`{"model":"test","input":"hello","opaque_id":9007199254740993}`)

	firstKey := GenerateEmbeddingsKeyForTarget(first, "openai", "embeddings")
	secondKey := GenerateEmbeddingsKeyForTarget(&second, "openai", "embeddings")
	if firstKey == "" || secondKey == "" {
		t.Fatal("expected non-empty cache keys")
	}
	if firstKey == secondKey {
		t.Fatal("distinct integers above float64 precision produced the same embeddings cache key")
	}
}
