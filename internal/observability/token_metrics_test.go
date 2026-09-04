package observability

import (
	"testing"

	"github.com/lunargate-ai/gateway/pkg/models"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
)

func TestObserveTokenUsageKeepsCacheCategoriesOutOfInclusiveTotal(t *testing.T) {
	metrics := NewMetricsWithRegisterer(prometheus.NewRegistry())
	metrics.ObserveTokenUsage("anthropic", "claude-sonnet-4-6", models.TokenUsage{
		InputTokens:             100,
		OutputTokens:            20,
		CachedInputTokens:       40,
		CacheWriteInputTokens:   30,
		CacheWriteInputTokens5m: 10,
		CacheWriteInputTokens1h: 5,
	})

	assertCounterValue(t, metrics.TokensTotal.WithLabelValues("anthropic", "claude-sonnet-4-6", "input"), 100)
	assertCounterValue(t, metrics.TokensTotal.WithLabelValues("anthropic", "claude-sonnet-4-6", "output"), 20)
	assertCounterValue(t, metrics.CacheTokensTotal.WithLabelValues("anthropic", "claude-sonnet-4-6", "read"), 40)
	assertCounterValue(t, metrics.CacheTokensTotal.WithLabelValues("anthropic", "claude-sonnet-4-6", "write"), 15)
	assertCounterValue(t, metrics.CacheTokensTotal.WithLabelValues("anthropic", "claude-sonnet-4-6", "write_5m"), 10)
	assertCounterValue(t, metrics.CacheTokensTotal.WithLabelValues("anthropic", "claude-sonnet-4-6", "write_1h"), 5)
}

func assertCounterValue(t *testing.T, counter prometheus.Counter, want float64) {
	t.Helper()
	if got := testutil.ToFloat64(counter); got != want {
		t.Fatalf("counter = %v, want %v", got, want)
	}
}
