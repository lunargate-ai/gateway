package api

import (
	"encoding/json"

	"github.com/lunargate-ai/gateway/pkg/models"
)

func normalizeUnifiedResponseUsage(response *models.UnifiedResponse) {
	if response == nil {
		return
	}
	models.NormalizeUsage(response.Usage)
	response.RawJSON = normalizeRawUsageCounters(response.RawJSON)
}

func normalizeEmbeddingsResponseUsage(response *models.EmbeddingsResponse) {
	if response == nil {
		return
	}
	if response.Usage != nil {
		response.Usage.PromptTokens = nonNegativeTokenCount(response.Usage.PromptTokens)
		response.Usage.TotalTokens = nonNegativeTokenCount(response.Usage.TotalTokens)
		if response.Usage.PromptTokens > response.Usage.TotalTokens {
			response.Usage.TotalTokens = response.Usage.PromptTokens
		}
	}
	response.RawJSON = normalizeRawUsageCounters(response.RawJSON)
}

func nonNegativeTokenCount(value int) int {
	return models.NonNegativeTokenCount(value)
}

func normalizeRawUsageCounters(raw json.RawMessage) json.RawMessage {
	return models.NormalizeRawUsageCounters(raw)
}
