package api

import (
	"strings"

	"github.com/google/uuid"
)

func translatedResponseID(upstreamID string) string {
	candidate := strings.TrimSpace(upstreamID)
	if strings.HasPrefix(candidate, "resp_") && len(candidate) > len("resp_") {
		return candidate
	}
	return "resp_" + strings.ReplaceAll(uuid.NewString(), "-", "")
}
