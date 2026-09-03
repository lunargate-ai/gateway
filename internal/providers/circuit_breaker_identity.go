package providers

import (
	"strings"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/provideridentity"
	"github.com/lunargate-ai/gateway/pkg/models"
)

const anthropicDefaultAPIVersion = "2023-06-01"

func providerCircuitBreakerKey(
	providerID string,
	providerType string,
	cfg config.ProviderConfig,
	translator models.ProviderTranslator,
) string {
	providerType = strings.ToLower(strings.TrimSpace(providerType))
	baseURL := ""
	if translator != nil {
		baseURL = translator.BaseURL()
	}

	organization := cfg.Organization
	apiKey := cfg.APIKey
	apiVersion := ""
	switch providerType {
	case "anthropic":
		organization = ""
		apiVersion = strings.TrimSpace(cfg.APIVersion)
		if apiVersion == "" {
			apiVersion = anthropicDefaultAPIVersion
		}
	case "ollama":
		organization = ""
		apiKey = ""
	}

	accountFingerprint := provideridentity.AccountFingerprint(
		providerType,
		baseURL,
		organization,
		apiKey,
	)
	return provideridentity.CircuitBreakerKey(providerID, accountFingerprint, apiVersion)
}
