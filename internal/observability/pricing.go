package observability

import "strings"

type modelPricing struct {
	InputPerMTokensUSD  float64
	OutputPerMTokensUSD float64
}

func EstimateCostUSD(provider string, model string, tokensIn int, tokensOut int) float64 {
	if tokensIn <= 0 && tokensOut <= 0 {
		return 0
	}

	p, ok := lookupPricing(provider, model)
	if !ok {
		return 0
	}

	inCost := (float64(tokensIn) / 1_000_000.0) * p.InputPerMTokensUSD
	outCost := (float64(tokensOut) / 1_000_000.0) * p.OutputPerMTokensUSD
	return inCost + outCost
}

func lookupPricing(provider string, model string) (modelPricing, bool) {
	p := strings.ToLower(strings.TrimSpace(provider))
	m := strings.ToLower(strings.TrimSpace(model))

	switch p {
	case "openai":
		switch m {
		case "gpt-4o":
			return modelPricing{InputPerMTokensUSD: 5, OutputPerMTokensUSD: 15}, true
		case "gpt-4o-mini":
			return modelPricing{InputPerMTokensUSD: 0.15, OutputPerMTokensUSD: 0.6}, true
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
