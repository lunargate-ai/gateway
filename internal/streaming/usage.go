package streaming

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/lunargate-ai/gateway/pkg/models"
)

type streamUsageAccumulator struct {
	id               string
	object           string
	created          int64
	model            string
	promptTokens     int
	completionTokens int
	totalTokens      int
	inputDetails     models.InputTokensDetails
	hasInputDetails  bool
}

func (a *streamUsageAccumulator) add(chunk *models.StreamChunk) {
	if chunk == nil {
		return
	}
	if chunk.ID != "" {
		a.id = chunk.ID
	}
	if chunk.Object != "" {
		a.object = chunk.Object
	}
	if chunk.Created != 0 {
		a.created = chunk.Created
	}
	if chunk.Model != "" {
		a.model = chunk.Model
	}
	usage := chunk.Usage
	if usage == nil {
		return
	}
	if usage.PromptTokens > a.promptTokens {
		a.promptTokens = usage.PromptTokens
	}
	if usage.CompletionTokens > a.completionTokens {
		a.completionTokens = usage.CompletionTokens
	}
	if usage.TotalTokens > a.totalTokens {
		a.totalTokens = usage.TotalTokens
	}
	if details := usage.PromptTokensDetails; details != nil {
		a.hasInputDetails = true
		if details.CachedTokens > a.inputDetails.CachedTokens {
			a.inputDetails.CachedTokens = details.CachedTokens
		}
		if details.CacheWriteTokens > a.inputDetails.CacheWriteTokens {
			a.inputDetails.CacheWriteTokens = details.CacheWriteTokens
		}
		if details.CacheWriteTokens5m > a.inputDetails.CacheWriteTokens5m {
			a.inputDetails.CacheWriteTokens5m = details.CacheWriteTokens5m
		}
		if details.CacheWriteTokens1h > a.inputDetails.CacheWriteTokens1h {
			a.inputDetails.CacheWriteTokens1h = details.CacheWriteTokens1h
		}
	}
	if componentTotal := models.SaturatingTokenSum(a.promptTokens, a.completionTokens); componentTotal > a.totalTokens {
		a.totalTokens = componentTotal
	}
}

func withoutStreamUsage(chunk *models.StreamChunk) *models.StreamChunk {
	if chunk == nil {
		return chunk
	}
	filteredRaw, rawUsageRemoved := removeRawStreamField(chunk.RawJSON, "usage")
	if chunk.Usage == nil && !rawUsageRemoved {
		return chunk
	}
	copyChunk := *chunk
	copyChunk.Usage = nil
	if rawUsageRemoved {
		copyChunk.RawJSON = filteredRaw
	}
	return &copyChunk
}

func writeCanonicalUsageTrailer(
	w http.ResponseWriter,
	controller *http.ResponseController,
	envelope *chatStreamEnvelopeNormalizer,
	usage streamUsageAccumulator,
) error {
	var inputDetails *models.InputTokensDetails
	if usage.hasInputDetails {
		inputDetails = models.CloneInputTokensDetails(&usage.inputDetails)
	}
	chunk := envelope.normalize(&models.StreamChunk{
		ID:      usage.id,
		Object:  usage.object,
		Created: usage.created,
		Model:   usage.model,
		Choices: []models.Choice{},
		Usage: &models.Usage{
			PromptTokens:        usage.promptTokens,
			CompletionTokens:    usage.completionTokens,
			TotalTokens:         usage.totalTokens,
			PromptTokensDetails: inputDetails,
		},
	})
	payload, err := json.Marshal(chunk)
	if err != nil {
		return fmt.Errorf("failed to marshal usage trailer: %w", err)
	}
	return writeSSEFrame(w, controller, payload, "usage trailer")
}
