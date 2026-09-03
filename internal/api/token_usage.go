package api

import "github.com/lunargate-ai/gateway/pkg/models"

func mergeObservedTokenUsage(current *models.TokenUsage, update *models.Usage) {
	if current == nil || update == nil {
		return
	}
	next := models.TokenUsageFromUsage(update)
	if next.InputTokens > current.InputTokens {
		current.InputTokens = next.InputTokens
	}
	if next.OutputTokens > current.OutputTokens {
		current.OutputTokens = next.OutputTokens
	}
	if next.CachedInputTokens > current.CachedInputTokens {
		current.CachedInputTokens = next.CachedInputTokens
	}
	if next.CacheWriteInputTokens > current.CacheWriteInputTokens {
		current.CacheWriteInputTokens = next.CacheWriteInputTokens
	}
	if next.CacheWriteInputTokens5m > current.CacheWriteInputTokens5m {
		current.CacheWriteInputTokens5m = next.CacheWriteInputTokens5m
	}
	if next.CacheWriteInputTokens1h > current.CacheWriteInputTokens1h {
		current.CacheWriteInputTokens1h = next.CacheWriteInputTokens1h
	}
	*current = current.Normalized()
}

func inputTokenDetailsFromTokenUsage(usage models.TokenUsage) *models.InputTokensDetails {
	usage = usage.Normalized()
	if usage.CachedInputTokens == 0 && usage.CacheWriteInputTokens == 0 {
		return nil
	}
	return &models.InputTokensDetails{
		CachedTokens:       usage.CachedInputTokens,
		CacheWriteTokens:   usage.CacheWriteInputTokens,
		CacheWriteTokens5m: usage.CacheWriteInputTokens5m,
		CacheWriteTokens1h: usage.CacheWriteInputTokens1h,
	}
}
