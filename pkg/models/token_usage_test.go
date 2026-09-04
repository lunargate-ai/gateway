package models

import (
	"encoding/json"
	"strconv"
	"testing"
)

func TestSaturatingTokenSum(t *testing.T) {
	maximum := int(^uint(0) >> 1)
	tests := []struct {
		name   string
		values []int
		want   int
	}{
		{name: "ordinary", values: []int{3, 4, 5}, want: 12},
		{name: "negative is zero", values: []int{-10, 4}, want: 4},
		{name: "exact maximum", values: []int{maximum - 1, 1}, want: maximum},
		{name: "overflow saturates", values: []int{maximum, maximum}, want: maximum},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := SaturatingTokenSum(test.values...); got != test.want {
				t.Fatalf("SaturatingTokenSum() = %d, want %d", got, test.want)
			}
		})
	}
}

func TestNormalizeRawUsageCountersClampsNestedTokenFields(t *testing.T) {
	raw := json.RawMessage(`{"id":"chatcmpl-1","usage":{"prompt_tokens":-3,"completion_tokens":2,"prompt_tokens_details":{"cached_tokens":-4,"kept":true}},"other":{"negative_tokens":-9}}`)
	normalized := NormalizeRawUsageCounters(raw)

	var payload map[string]interface{}
	if err := json.Unmarshal(normalized, &payload); err != nil {
		t.Fatalf("decode normalized payload: %v", err)
	}
	usage := payload["usage"].(map[string]interface{})
	if got := usage["prompt_tokens"]; got != float64(0) {
		t.Fatalf("prompt_tokens = %#v, want 0", got)
	}
	if got := usage["completion_tokens"]; got != float64(2) {
		t.Fatalf("completion_tokens = %#v, want 2", got)
	}
	details := usage["prompt_tokens_details"].(map[string]interface{})
	if got := details["cached_tokens"]; got != float64(0) {
		t.Fatalf("cached_tokens = %#v, want 0", got)
	}
	if got := details["kept"]; got != true {
		t.Fatalf("kept = %#v, want true", got)
	}
	other := payload["other"].(map[string]interface{})
	if got := other["negative_tokens"]; got != float64(-9) {
		t.Fatalf("non-usage field changed to %#v", got)
	}
}

func TestNormalizeUsageBoundsCacheDetailsAndTotal(t *testing.T) {
	usage := &Usage{
		PromptTokens:     10,
		CompletionTokens: 7,
		TotalTokens:      1,
		PromptTokensDetails: &InputTokensDetails{
			CachedTokens:       100,
			CacheWriteTokens:   100,
			CacheWriteTokens5m: 80,
			CacheWriteTokens1h: 80,
		},
		CompletionTokensDetails: &CompletionTokensDetails{
			AcceptedPredictionTokens: -2,
			AudioTokens:              -3,
			ReasoningTokens:          100,
			RejectedPredictionTokens: -4,
		},
	}

	NormalizeUsage(usage)
	if usage.TotalTokens != 17 {
		t.Fatalf("total_tokens = %d, want 17", usage.TotalTokens)
	}
	if got := usage.PromptTokensDetails.CacheWriteTokens; got != 10 {
		t.Fatalf("cache_write_tokens = %d, want 10", got)
	}
	if got := usage.PromptTokensDetails.CachedTokens; got != 0 {
		t.Fatalf("cached_tokens = %d, want 0 after cache writes consume input total", got)
	}
	if got := usage.PromptTokensDetails.CacheWriteTokens5m; got != 10 {
		t.Fatalf("cache_write_tokens_5m = %d, want 10", got)
	}
	if got := usage.PromptTokensDetails.CacheWriteTokens1h; got != 0 {
		t.Fatalf("cache_write_tokens_1h = %d, want 0", got)
	}
	if got := usage.CompletionTokensDetails.ReasoningTokens; got != 7 {
		t.Fatalf("reasoning_tokens = %d, want bounded output total 7", got)
	}
	if usage.CompletionTokensDetails.AcceptedPredictionTokens != 0 ||
		usage.CompletionTokensDetails.AudioTokens != 0 ||
		usage.CompletionTokensDetails.RejectedPredictionTokens != 0 {
		t.Fatalf("negative completion details were not clamped: %#v", usage.CompletionTokensDetails)
	}
}

func TestCloneCompletionTokensDetailsDoesNotAlias(t *testing.T) {
	original := &CompletionTokensDetails{ReasoningTokens: 3, AcceptedPredictionTokens: 2}
	cloned := CloneCompletionTokensDetails(original)
	if cloned == nil || *cloned != *original {
		t.Fatalf("clone = %#v, want %#v", cloned, original)
	}
	cloned.ReasoningTokens = 1
	if original.ReasoningTokens != 3 {
		t.Fatalf("clone aliases original: %#v", original)
	}
}

func TestNormalizeRawUsageCountersRepairsTotalsAndCacheDetails(t *testing.T) {
	maximum := int(^uint(0) >> 1)
	raw := json.RawMessage(`{"id":"resp-1","usage":{"input_tokens":` +
		jsonNumber(maximum) + `,"output_tokens":` + jsonNumber(maximum) +
		`,"total_tokens":1,"input_tokens_details":{"cached_tokens":` + jsonNumber(maximum) +
		`,"cache_write_tokens":` + jsonNumber(maximum) + `,"kept":true}},"future":{"kept":true}}`)
	normalized := NormalizeRawUsageCounters(raw)

	var payload map[string]interface{}
	if err := json.Unmarshal(normalized, &payload); err != nil {
		t.Fatalf("decode normalized payload: %v", err)
	}
	usage := payload["usage"].(map[string]interface{})
	if got := usage["total_tokens"]; got != float64(maximum) {
		t.Fatalf("total_tokens = %#v, want %d", got, maximum)
	}
	details := usage["input_tokens_details"].(map[string]interface{})
	if got := details["cache_write_tokens"]; got != float64(maximum) {
		t.Fatalf("cache_write_tokens = %#v, want %d", got, maximum)
	}
	if got := details["cached_tokens"]; got != float64(0) {
		t.Fatalf("cached_tokens = %#v, want 0", got)
	}
	if got := details["kept"]; got != true {
		t.Fatalf("additive detail = %#v, want true", got)
	}
	if got := payload["future"].(map[string]interface{})["kept"]; got != true {
		t.Fatalf("additive top-level field = %#v, want true", got)
	}
}

func jsonNumber(value int) string {
	return strconv.Itoa(value)
}
