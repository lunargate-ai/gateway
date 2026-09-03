package middleware

import (
	"encoding/json"
	"testing"

	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestGenerateKeyForTargetDistinguishesDuplicateObjectNames(t *testing.T) {
	single := &models.UnifiedRequest{
		RawJSON: json.RawMessage(`{"model":"test","messages":[],"future":{"mode":2}}`),
		Model:   "test",
	}
	duplicate := *single
	duplicate.RawJSON = json.RawMessage(`{"model":"test","messages":[],"future":{"mode":1,"mode":2}}`)

	singleKey := GenerateKeyForTarget(single, "openai", "chat_completions")
	duplicateKey := GenerateKeyForTarget(&duplicate, "openai", "chat_completions")
	if singleKey == "" || duplicateKey == "" {
		t.Fatal("expected non-empty cache keys")
	}
	if singleKey == duplicateKey {
		t.Fatal("ambiguous duplicate object name collided with the effective single-key request")
	}
}

func TestGenerateEmbeddingsKeyForTargetDistinguishesDuplicateObjectNames(t *testing.T) {
	single := &models.EmbeddingsRequest{
		RawJSON: json.RawMessage(`{"model":"test","input":"hello","future":2}`),
		Model:   "test",
		Input:   "hello",
	}
	duplicate := *single
	duplicate.RawJSON = json.RawMessage(`{"model":"test","input":"hello","future":1,"future":2}`)

	singleKey := GenerateEmbeddingsKeyForTarget(single, "openai", "embeddings")
	duplicateKey := GenerateEmbeddingsKeyForTarget(&duplicate, "openai", "embeddings")
	if singleKey == "" || duplicateKey == "" {
		t.Fatal("expected non-empty cache keys")
	}
	if singleKey == duplicateKey {
		t.Fatal("ambiguous duplicate object name collided with the effective single-key embeddings request")
	}
}
