package providers

import (
	"net/http"
	"strings"

	"github.com/lunargate-ai/gateway/pkg/models"
)

func (t *AnthropicTranslator) requestSpeed(providerID string, req *models.UnifiedRequest) (string, error) {
	tier := models.RequestServiceTier(req)
	if tier == nil {
		return "", nil
	}
	unsupported := func(reason string) (string, error) {
		return "", &models.CompatibilityError{Field: "service_tier", Provider: providerID, Reason: reason}
	}
	switch *tier {
	case "auto":
		return "", nil
	case "default":
		return "standard", nil
	case "fast", "priority":
		if !t.cfg.Capabilities.FastMode {
			return unsupported("enable provider capability fast_mode for a compatible Anthropic account/model")
		}
		return "fast", nil
	default:
		return unsupported("requested service tier has no Anthropic speed equivalent")
	}
}

func anthropicServiceTier(speed string) *string {
	var value string
	switch speed {
	case "standard":
		value = "default"
	case "fast":
		value = "fast"
	default:
		return nil
	}
	return &value
}

func anthropicSpeedNeedsBeta(model string) bool {
	for _, name := range []string{"claude-opus-5", "claude-opus-4-8", "claude-opus-4-6"} {
		if model == name || strings.HasPrefix(model, name+"-20") {
			return true
		}
	}
	return false
}

func addAnthropicBeta(headers http.Header, beta string) {
	var values []string
	seen := make(map[string]bool)
	for _, line := range append(headers.Values("Anthropic-Beta"), beta) {
		for _, value := range strings.Split(line, ",") {
			value = strings.TrimSpace(value)
			if value != "" && !seen[value] {
				values = append(values, value)
				seen[value] = true
			}
		}
	}
	headers.Set("Anthropic-Beta", strings.Join(values, ","))
}
