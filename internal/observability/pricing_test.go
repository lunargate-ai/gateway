package observability

import (
	"testing"

	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestEstimateCostUSDRequiresBuiltInProviderIdentity(t *testing.T) {
	tests := []struct {
		name         string
		providerID   string
		providerType string
		model        string
		want         float64
	}{
		{
			name:         "official OpenAI provider",
			providerID:   "openai",
			providerType: "openai",
			model:        "gpt-4o",
			want:         12.5,
		},
		{
			name:         "OpenAI-compatible custom provider",
			providerID:   "abacus",
			providerType: "openai",
			model:        "gpt-4o",
			want:         0,
		},
		{
			name:         "custom fallback provider",
			providerID:   "fallback",
			providerType: "openai",
			model:        "gpt-4o",
			want:         0,
		},
		{
			name:         "provider type mismatch",
			providerID:   "openai",
			providerType: "anthropic",
			model:        "gpt-4o",
			want:         0,
		},
		{
			name:         "unknown model",
			providerID:   "openai",
			providerType: "openai",
			model:        "unknown-model",
			want:         0,
		},
		{
			name:         "model without confirmed catalog rate",
			providerID:   "openai",
			providerType: "openai",
			model:        "gpt-5.6-terra",
			want:         0,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := EstimateCostUSD(test.providerID, test.providerType, test.model, 1_000_000, 1_000_000)
			if got != test.want {
				t.Fatalf("EstimateCostUSD() = %v, want %v", got, test.want)
			}
		})
	}
}

func TestEstimateCostUSDUsesCurrentPerMillionTokenRates(t *testing.T) {
	tests := []struct {
		name         string
		providerID   string
		providerType string
		model        string
		want         float64
	}{
		{name: "GPT-4o", providerID: "openai", providerType: "openai", model: "gpt-4o", want: 12.5},
		{name: "Claude Sonnet 4.6", providerID: "anthropic", providerType: "anthropic", model: "claude-sonnet-4-6", want: 18},
		{name: "Claude Sonnet 5", providerID: "anthropic", providerType: "anthropic", model: "claude-sonnet-5", want: 18},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := EstimateCostUSD(test.providerID, test.providerType, test.model, 1_000_000, 1_000_000)
			if got != test.want {
				t.Fatalf("cost for 1M input + 1M output tokens = %v USD, want %v USD", got, test.want)
			}
		})
	}
}

func TestEstimateCostUSDClampsNegativeTokenCounts(t *testing.T) {
	tests := []struct {
		name      string
		tokensIn  int
		tokensOut int
		want      float64
	}{
		{name: "negative input", tokensIn: -1_000_000, tokensOut: 1_000_000, want: 10},
		{name: "negative output", tokensIn: 1_000_000, tokensOut: -1_000_000, want: 2.5},
		{name: "both negative", tokensIn: -1_000_000, tokensOut: -1_000_000, want: 0},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := EstimateCostUSD("openai", "openai", "gpt-4o", test.tokensIn, test.tokensOut)
			if got != test.want {
				t.Fatalf("EstimateCostUSD() = %v, want %v", got, test.want)
			}
		})
	}
}

func TestEstimateTokenUsageCostUSDAppliesCacheRatesWithoutDoubleCounting(t *testing.T) {
	tests := []struct {
		name         string
		providerID   string
		providerType string
		model        string
		usage        models.TokenUsage
		want         float64
	}{
		{
			name:         "OpenAI cached read and unclassified write",
			providerID:   "openai",
			providerType: "openai",
			model:        "gpt-4o",
			usage: models.TokenUsage{
				InputTokens:           1_000_000,
				OutputTokens:          1_000_000,
				CachedInputTokens:     400_000,
				CacheWriteInputTokens: 100_000,
			},
			want: 12,
		},
		{
			name:         "Anthropic cache TTL breakdown",
			providerID:   "anthropic",
			providerType: "anthropic",
			model:        "claude-sonnet-4-6",
			usage: models.TokenUsage{
				InputTokens:             1_000_000,
				OutputTokens:            1_000_000,
				CachedInputTokens:       400_000,
				CacheWriteInputTokens:   300_000,
				CacheWriteInputTokens5m: 100_000,
				CacheWriteInputTokens1h: 100_000,
			},
			want: 17.37,
		},
		{
			name:         "custom compatible provider remains unpriced",
			providerID:   "abacus",
			providerType: "openai",
			model:        "gpt-4o",
			usage: models.TokenUsage{
				InputTokens:       1_000_000,
				CachedInputTokens: 1_000_000,
			},
			want: 0,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := EstimateTokenUsageCostUSD(test.providerID, test.providerType, test.model, test.usage)
			if got != test.want {
				t.Fatalf("EstimateTokenUsageCostUSD() = %v, want %v", got, test.want)
			}
		})
	}
}
