package observability

import (
	"strings"

	"github.com/lunargate-ai/gateway/pkg/models"
)

type modelPricing struct {
	// Rates are in USD per one million tokens. Zero cache-specific rates
	// conservatively fall back to the uncached input rate.
	InputPerMTokensUSD        float64
	CachedInputPerMTokensUSD  float64
	CacheWritePerMTokensUSD   float64
	CacheWrite5mPerMTokensUSD float64
	CacheWrite1hPerMTokensUSD float64
	OutputPerMTokensUSD       float64
}

// EstimateCostUSD returns a catalog estimate only for a built-in provider
// identity paired with its native type. Compatible custom endpoints fail
// closed to zero because their billing cannot be inferred from API shape.
func EstimateCostUSD(providerID string, providerType string, model string, tokensIn int, tokensOut int) float64 {
	return EstimateTokenUsageCostUSD(providerID, providerType, model, models.TokenUsage{
		InputTokens:  tokensIn,
		OutputTokens: tokensOut,
	})
}

// EstimateTokenUsageCostUSD applies cache-category rates to an inclusive input
// total without charging any token twice.
func EstimateTokenUsageCostUSD(providerID string, providerType string, model string, usage models.TokenUsage) float64 {
	usage = usage.Normalized()
	if usage.InputTokens == 0 && usage.OutputTokens == 0 {
		return 0
	}

	p, ok := lookupPricing(providerID, providerType, model)
	if !ok {
		return 0
	}

	cacheReadRate := cacheRateOrInput(p.CachedInputPerMTokensUSD, p.InputPerMTokensUSD)
	cacheWriteRate := cacheRateOrInput(p.CacheWritePerMTokensUSD, p.InputPerMTokensUSD)
	cacheWrite5mRate := cacheRateOrInput(p.CacheWrite5mPerMTokensUSD, cacheWriteRate)
	cacheWrite1hRate := cacheRateOrInput(p.CacheWrite1hPerMTokensUSD, cacheWriteRate)

	costPerMillion := float64(usage.UncachedInputTokens())*p.InputPerMTokensUSD +
		float64(usage.CachedInputTokens)*cacheReadRate +
		float64(usage.UnclassifiedCacheWriteInputTokens())*cacheWriteRate +
		float64(usage.CacheWriteInputTokens5m)*cacheWrite5mRate +
		float64(usage.CacheWriteInputTokens1h)*cacheWrite1hRate +
		float64(usage.OutputTokens)*p.OutputPerMTokensUSD
	return costPerMillion / 1_000_000.0
}

func cacheRateOrInput(rate float64, inputRate float64) float64 {
	if rate > 0 {
		return rate
	}
	return inputRate
}

func lookupPricing(providerID string, providerType string, model string) (modelPricing, bool) {
	provider := strings.TrimSpace(providerID)
	p := strings.ToLower(strings.TrimSpace(providerType))
	m := strings.ToLower(strings.TrimSpace(model))
	if provider != p {
		return modelPricing{}, false
	}

	switch p {
	case "openai":
		switch m {
		case "gpt-4o":
			return modelPricing{InputPerMTokensUSD: 2.5, CachedInputPerMTokensUSD: 1.25, OutputPerMTokensUSD: 10}, true
		case "gpt-4o-mini":
			return modelPricing{InputPerMTokensUSD: 0.15, CachedInputPerMTokensUSD: 0.075, OutputPerMTokensUSD: 0.6}, true
		case "gpt-4-turbo":
			return modelPricing{InputPerMTokensUSD: 10, OutputPerMTokensUSD: 30}, true
		case "gpt-4":
			return modelPricing{InputPerMTokensUSD: 30, OutputPerMTokensUSD: 60}, true
		case "gpt-3.5-turbo":
			return modelPricing{InputPerMTokensUSD: 0.5, OutputPerMTokensUSD: 1.5}, true
		case "text-embedding-3-small":
			return modelPricing{InputPerMTokensUSD: 0.02, OutputPerMTokensUSD: 0}, true
		case "text-embedding-3-large":
			return modelPricing{InputPerMTokensUSD: 0.13, OutputPerMTokensUSD: 0}, true
		case "text-embedding-ada-002":
			return modelPricing{InputPerMTokensUSD: 0.1, OutputPerMTokensUSD: 0}, true
		default:
			return modelPricing{}, false
		}

	case "anthropic":
		switch m {
		case "claude-sonnet-5", "claude-sonnet-4-6":
			return modelPricing{
				InputPerMTokensUSD:        3,
				CachedInputPerMTokensUSD:  0.3,
				CacheWritePerMTokensUSD:   3.75,
				CacheWrite5mPerMTokensUSD: 3.75,
				CacheWrite1hPerMTokensUSD: 6,
				OutputPerMTokensUSD:       15,
			}, true
		case "claude-3-haiku-20240307":
			return modelPricing{InputPerMTokensUSD: 0.25, OutputPerMTokensUSD: 1.25}, true
		case "claude-3-sonnet-20240229":
			return modelPricing{InputPerMTokensUSD: 3, OutputPerMTokensUSD: 15}, true
		case "claude-3-opus-20240229":
			return modelPricing{InputPerMTokensUSD: 15, OutputPerMTokensUSD: 75}, true
		case "claude-3-5-sonnet-20241022":
			return modelPricing{InputPerMTokensUSD: 3, OutputPerMTokensUSD: 15}, true
		default:
			return modelPricing{}, false
		}
	default:
		return modelPricing{}, false
	}
}
