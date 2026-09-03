package models

import (
	"bytes"
	"encoding/json"
	"strconv"
	"strings"
)

// NonNegativeTokenCount clamps an untrusted token counter before it is used
// by metrics, billing estimates, or response normalization.
func NonNegativeTokenCount(value int) int {
	if value < 0 {
		return 0
	}
	return value
}

// SaturatingTokenSum adds untrusted token counters without allowing negative
// inputs or integer overflow to produce a negative observable total.
func SaturatingTokenSum(values ...int) int {
	maximum := int(^uint(0) >> 1)
	total := 0
	for _, value := range values {
		value = NonNegativeTokenCount(value)
		if value > maximum-total {
			return maximum
		}
		total += value
	}
	return total
}

// NormalizeUsage clamps the typed token counters supplied by an upstream.
func NormalizeUsage(usage *Usage) {
	if usage == nil {
		return
	}
	usage.PromptTokens = NonNegativeTokenCount(usage.PromptTokens)
	usage.CompletionTokens = NonNegativeTokenCount(usage.CompletionTokens)
	usage.TotalTokens = NonNegativeTokenCount(usage.TotalTokens)
	if componentTotal := SaturatingTokenSum(usage.PromptTokens, usage.CompletionTokens); componentTotal > usage.TotalTokens {
		usage.TotalTokens = componentTotal
	}
	if usage.PromptTokensDetails != nil {
		normalized := TokenUsageFromUsage(usage)
		usage.PromptTokensDetails.CachedTokens = normalized.CachedInputTokens
		usage.PromptTokensDetails.CacheWriteTokens = normalized.CacheWriteInputTokens
		usage.PromptTokensDetails.CacheWriteTokens5m = normalized.CacheWriteInputTokens5m
		usage.PromptTokensDetails.CacheWriteTokens1h = normalized.CacheWriteInputTokens1h
	}
}

// TokenUsage is an inclusive token total with mutually exclusive cache
// categories. Cache-write TTL fields decompose CacheWriteInputTokens.
type TokenUsage struct {
	InputTokens             int
	OutputTokens            int
	CachedInputTokens       int
	CacheWriteInputTokens   int
	CacheWriteInputTokens5m int
	CacheWriteInputTokens1h int
}

// TokenUsageFromUsage produces a bounded, non-overlapping accounting view.
func TokenUsageFromUsage(usage *Usage) TokenUsage {
	if usage == nil {
		return TokenUsage{}
	}
	result := TokenUsage{
		InputTokens:  usage.PromptTokens,
		OutputTokens: usage.CompletionTokens,
	}
	if details := usage.PromptTokensDetails; details != nil {
		result.CachedInputTokens = details.CachedTokens
		result.CacheWriteInputTokens = details.CacheWriteTokens
		result.CacheWriteInputTokens5m = details.CacheWriteTokens5m
		result.CacheWriteInputTokens1h = details.CacheWriteTokens1h
	}
	return result.Normalized()
}

// Normalized clamps every counter and ensures cache categories never exceed
// the inclusive input total or overlap one another.
func (usage TokenUsage) Normalized() TokenUsage {
	usage.InputTokens = NonNegativeTokenCount(usage.InputTokens)
	usage.OutputTokens = NonNegativeTokenCount(usage.OutputTokens)
	usage.CachedInputTokens = NonNegativeTokenCount(usage.CachedInputTokens)
	usage.CacheWriteInputTokens = NonNegativeTokenCount(usage.CacheWriteInputTokens)
	usage.CacheWriteInputTokens5m = NonNegativeTokenCount(usage.CacheWriteInputTokens5m)
	usage.CacheWriteInputTokens1h = NonNegativeTokenCount(usage.CacheWriteInputTokens1h)

	classifiedWrites := SaturatingTokenSum(usage.CacheWriteInputTokens5m, usage.CacheWriteInputTokens1h)
	if classifiedWrites > usage.CacheWriteInputTokens {
		usage.CacheWriteInputTokens = classifiedWrites
	}
	if usage.CacheWriteInputTokens > usage.InputTokens {
		usage.CacheWriteInputTokens = usage.InputTokens
	}
	remaining := usage.InputTokens - usage.CacheWriteInputTokens
	if usage.CachedInputTokens > remaining {
		usage.CachedInputTokens = remaining
	}

	if usage.CacheWriteInputTokens5m > usage.CacheWriteInputTokens {
		usage.CacheWriteInputTokens5m = usage.CacheWriteInputTokens
	}
	remainingWrites := usage.CacheWriteInputTokens - usage.CacheWriteInputTokens5m
	if usage.CacheWriteInputTokens1h > remainingWrites {
		usage.CacheWriteInputTokens1h = remainingWrites
	}
	return usage
}

func (usage TokenUsage) UncachedInputTokens() int {
	usage = usage.Normalized()
	return usage.InputTokens - usage.CachedInputTokens - usage.CacheWriteInputTokens
}

func (usage TokenUsage) UnclassifiedCacheWriteInputTokens() int {
	usage = usage.Normalized()
	return usage.CacheWriteInputTokens - usage.CacheWriteInputTokens5m - usage.CacheWriteInputTokens1h
}

func CloneInputTokensDetails(details *InputTokensDetails) *InputTokensDetails {
	if details == nil {
		return nil
	}
	copyDetails := *details
	return &copyDetails
}

// NormalizeRawUsageCounters clamps every numeric usage field whose name ends
// in _tokens, including nested provider-specific detail objects. Unrelated raw
// response fields are preserved.
func NormalizeRawUsageCounters(raw json.RawMessage) json.RawMessage {
	if len(bytes.TrimSpace(raw)) == 0 || !json.Valid(raw) {
		return raw
	}

	var envelope map[string]json.RawMessage
	if err := json.Unmarshal(raw, &envelope); err != nil || envelope == nil {
		return raw
	}
	usage, ok := envelope["usage"]
	if !ok {
		return raw
	}

	normalizedUsage, changed := clampNegativeRawTokenCounters(usage)
	normalizedUsage, accountingChanged := normalizeRawUsageAccounting(normalizedUsage)
	changed = changed || accountingChanged
	if !changed {
		return raw
	}
	envelope["usage"] = normalizedUsage
	normalized, err := json.Marshal(envelope)
	if err != nil {
		return raw
	}
	return normalized
}

func normalizeRawUsageAccounting(raw json.RawMessage) (json.RawMessage, bool) {
	var usage map[string]json.RawMessage
	if err := json.Unmarshal(raw, &usage); err != nil || usage == nil {
		return raw, false
	}

	input, hasInput := firstRawTokenCount(usage, "prompt_tokens", "input_tokens")
	output, hasOutput := firstRawTokenCount(usage, "completion_tokens", "output_tokens")
	changed := false
	if hasInput || hasOutput {
		minimumTotal := SaturatingTokenSum(input, output)
		rawTotal, totalPresent := usage["total_tokens"]
		total, totalValid := rawTokenCount(rawTotal)
		if !totalPresent || (totalValid && total < minimumTotal) {
			usage["total_tokens"] = rawTokenCountJSON(minimumTotal)
			changed = true
		}
	}

	if hasInput {
		for _, field := range []string{"prompt_tokens_details", "input_tokens_details"} {
			details, ok := usage[field]
			if !ok {
				continue
			}
			normalized, detailsChanged := normalizeRawInputTokenDetails(details, input)
			if detailsChanged {
				usage[field] = normalized
				changed = true
			}
		}
	}

	if !changed {
		return raw, false
	}
	normalized, err := json.Marshal(usage)
	if err != nil {
		return raw, false
	}
	return normalized, true
}

func normalizeRawInputTokenDetails(raw json.RawMessage, inputTokens int) (json.RawMessage, bool) {
	var details map[string]json.RawMessage
	if err := json.Unmarshal(raw, &details); err != nil || details == nil {
		return raw, false
	}

	cached, cachedOK := rawTokenCount(details["cached_tokens"])
	cacheWrite, cacheWriteOK := rawTokenCount(details["cache_write_tokens"])
	cacheWrite5m, cacheWrite5mOK := rawTokenCount(details["cache_write_tokens_5m"])
	cacheWrite1h, cacheWrite1hOK := rawTokenCount(details["cache_write_tokens_1h"])
	accounting := (TokenUsage{
		InputTokens:             inputTokens,
		CachedInputTokens:       cached,
		CacheWriteInputTokens:   cacheWrite,
		CacheWriteInputTokens5m: cacheWrite5m,
		CacheWriteInputTokens1h: cacheWrite1h,
	}).Normalized()

	changed := setNormalizedRawTokenCount(details, "cached_tokens", cachedOK, accounting.CachedInputTokens)
	changed = setNormalizedRawTokenCount(details, "cache_write_tokens", cacheWriteOK, accounting.CacheWriteInputTokens) || changed
	changed = setNormalizedRawTokenCount(details, "cache_write_tokens_5m", cacheWrite5mOK, accounting.CacheWriteInputTokens5m) || changed
	changed = setNormalizedRawTokenCount(details, "cache_write_tokens_1h", cacheWrite1hOK, accounting.CacheWriteInputTokens1h) || changed
	if !changed {
		return raw, false
	}
	normalized, err := json.Marshal(details)
	if err != nil {
		return raw, false
	}
	return normalized, true
}

func firstRawTokenCount(object map[string]json.RawMessage, fields ...string) (int, bool) {
	for _, field := range fields {
		if value, ok := rawTokenCount(object[field]); ok {
			return value, true
		}
	}
	return 0, false
}

func rawTokenCount(raw json.RawMessage) (int, bool) {
	if len(bytes.TrimSpace(raw)) == 0 {
		return 0, false
	}
	var value int
	if err := json.Unmarshal(raw, &value); err != nil {
		return 0, false
	}
	return NonNegativeTokenCount(value), true
}

func setNormalizedRawTokenCount(object map[string]json.RawMessage, field string, present bool, value int) bool {
	if !present {
		return false
	}
	current, ok := rawTokenCount(object[field])
	if ok && current == value {
		return false
	}
	object[field] = rawTokenCountJSON(value)
	return true
}

func rawTokenCountJSON(value int) json.RawMessage {
	return json.RawMessage(strconv.Itoa(NonNegativeTokenCount(value)))
}

func clampNegativeRawTokenCounters(raw json.RawMessage) (json.RawMessage, bool) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 {
		return raw, false
	}

	switch trimmed[0] {
	case '{':
		var object map[string]json.RawMessage
		if err := json.Unmarshal(trimmed, &object); err != nil {
			return raw, false
		}
		changed := false
		for key, value := range object {
			if strings.HasSuffix(strings.ToLower(strings.TrimSpace(key)), "_tokens") && isNegativeJSONNumber(value) {
				object[key] = json.RawMessage("0")
				changed = true
				continue
			}
			normalized, childChanged := clampNegativeRawTokenCounters(value)
			if childChanged {
				object[key] = normalized
				changed = true
			}
		}
		if !changed {
			return raw, false
		}
		normalized, err := json.Marshal(object)
		if err != nil {
			return raw, false
		}
		return normalized, true

	case '[':
		var array []json.RawMessage
		if err := json.Unmarshal(trimmed, &array); err != nil {
			return raw, false
		}
		changed := false
		for index, value := range array {
			normalized, childChanged := clampNegativeRawTokenCounters(value)
			if childChanged {
				array[index] = normalized
				changed = true
			}
		}
		if !changed {
			return raw, false
		}
		normalized, err := json.Marshal(array)
		if err != nil {
			return raw, false
		}
		return normalized, true
	}

	return raw, false
}

func isNegativeJSONNumber(raw json.RawMessage) bool {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || trimmed[0] != '-' {
		return false
	}
	var number json.Number
	return json.Unmarshal(trimmed, &number) == nil
}
