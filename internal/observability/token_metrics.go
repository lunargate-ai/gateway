package observability

import "github.com/lunargate-ai/gateway/pkg/models"

// ObserveTokenUsage records the inclusive input/output totals once and emits
// cache categories through a separate counter, avoiding double counting in the
// established tokens_total metric.
func (m *Metrics) ObserveTokenUsage(provider string, model string, usage models.TokenUsage) {
	if m == nil {
		return
	}
	usage = usage.Normalized()
	if usage.InputTokens > 0 {
		m.TokensTotal.WithLabelValues(provider, model, "input").Add(float64(usage.InputTokens))
	}
	if usage.OutputTokens > 0 {
		m.TokensTotal.WithLabelValues(provider, model, "output").Add(float64(usage.OutputTokens))
	}
	if usage.CachedInputTokens > 0 {
		m.CacheTokensTotal.WithLabelValues(provider, model, "read").Add(float64(usage.CachedInputTokens))
	}
	if count := usage.UnclassifiedCacheWriteInputTokens(); count > 0 {
		m.CacheTokensTotal.WithLabelValues(provider, model, "write").Add(float64(count))
	}
	if usage.CacheWriteInputTokens5m > 0 {
		m.CacheTokensTotal.WithLabelValues(provider, model, "write_5m").Add(float64(usage.CacheWriteInputTokens5m))
	}
	if usage.CacheWriteInputTokens1h > 0 {
		m.CacheTokensTotal.WithLabelValues(provider, model, "write_1h").Add(float64(usage.CacheWriteInputTokens1h))
	}
}
