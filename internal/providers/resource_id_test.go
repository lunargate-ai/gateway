package providers

import (
	"testing"

	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestUnifiedToResponsesPayloadNeverCanonicalizesPreviousResponseID(t *testing.T) {
	for _, previousResponseID := range []string{
		"resp_exact",
		"resp_internal space",
		" resp_invalid_boundary_whitespace ",
	} {
		payload := unifiedToResponsesPayload(&models.UnifiedRequest{
			Model:              "mock-gpt",
			PreviousResponseID: previousResponseID,
		})
		if payload.PreviousResponseID != previousResponseID {
			t.Fatalf(
				"previous_response_id=%q translated as %q",
				previousResponseID,
				payload.PreviousResponseID,
			)
		}
	}
}
