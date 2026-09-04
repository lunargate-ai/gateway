package streaming

import (
	"encoding/json"
	"testing"

	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestChatStreamEnvelopeNormalizesUntrustedUsage(t *testing.T) {
	chunk := &models.StreamChunk{
		RawJSON: json.RawMessage(`{"id":"chatcmpl-negative","object":"chat.completion.chunk","choices":[],"usage":{"prompt_tokens":-7,"completion_tokens":-3,"total_tokens":-10,"prompt_tokens_details":{"cached_tokens":-5,"kept":true}}}`),
		ID:      "chatcmpl-negative",
		Object:  "chat.completion.chunk",
		Choices: []models.Choice{},
		Usage: &models.Usage{
			PromptTokens:     -7,
			CompletionTokens: -3,
			TotalTokens:      -10,
		},
	}

	normalized := newChatStreamEnvelopeNormalizer("gpt-4o").normalize(chunk)
	if normalized.Usage == nil || normalized.Usage.PromptTokens != 0 || normalized.Usage.CompletionTokens != 0 || normalized.Usage.TotalTokens != 0 {
		t.Fatalf("typed usage = %#v, want all counters clamped to zero", normalized.Usage)
	}

	payload, err := marshalStreamChunk(normalized)
	if err != nil {
		t.Fatalf("marshal normalized chunk: %v", err)
	}
	var envelope map[string]interface{}
	if err := json.Unmarshal(payload, &envelope); err != nil {
		t.Fatalf("decode normalized chunk: %v", err)
	}
	usage := envelope["usage"].(map[string]interface{})
	for _, field := range []string{"prompt_tokens", "completion_tokens", "total_tokens"} {
		if got := usage[field]; got != float64(0) {
			t.Errorf("%s = %#v, want 0", field, got)
		}
	}
	details := usage["prompt_tokens_details"].(map[string]interface{})
	if got := details["cached_tokens"]; got != float64(0) {
		t.Errorf("cached_tokens = %#v, want 0", got)
	}
	if got := details["kept"]; got != true {
		t.Errorf("additive detail = %#v, want true", got)
	}
}

func TestStreamUsageAccumulatorSaturatesComponentTotal(t *testing.T) {
	maximum := int(^uint(0) >> 1)
	var usage streamUsageAccumulator
	usage.add(&models.StreamChunk{Usage: &models.Usage{
		PromptTokens:     maximum,
		CompletionTokens: maximum,
	}})

	if usage.totalTokens != maximum {
		t.Fatalf("totalTokens = %d, want %d", usage.totalTokens, maximum)
	}
}

func TestChatStreamEnvelopeRaisesTypedAndRawTotalToComponentSum(t *testing.T) {
	chunk := &models.StreamChunk{
		RawJSON: json.RawMessage(`{"id":"chatcmpl-inconsistent","object":"chat.completion.chunk","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":7,"total_tokens":1}}`),
		ID:      "chatcmpl-inconsistent",
		Object:  "chat.completion.chunk",
		Choices: []models.Choice{},
		Usage: &models.Usage{
			PromptTokens:     5,
			CompletionTokens: 7,
			TotalTokens:      1,
		},
	}

	normalized := newChatStreamEnvelopeNormalizer("gpt-4o").normalize(chunk)
	if normalized.Usage == nil || normalized.Usage.TotalTokens != 12 {
		t.Fatalf("typed usage = %#v, want total_tokens=12", normalized.Usage)
	}

	payload, err := marshalStreamChunk(normalized)
	if err != nil {
		t.Fatalf("marshal normalized chunk: %v", err)
	}
	var envelope struct {
		Usage models.Usage `json:"usage"`
	}
	if err := json.Unmarshal(payload, &envelope); err != nil {
		t.Fatalf("decode normalized chunk: %v", err)
	}
	if envelope.Usage.TotalTokens != 12 {
		t.Fatalf("raw total_tokens = %d, want 12", envelope.Usage.TotalTokens)
	}
}
