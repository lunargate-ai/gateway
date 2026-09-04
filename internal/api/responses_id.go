package api

import (
	"strings"

	"github.com/google/uuid"
)

func translatedResponseID(upstreamID string) string {
	if validOpaqueResourceID(upstreamID) && strings.HasPrefix(upstreamID, "resp_") && len(upstreamID) > len("resp_") {
		return upstreamID
	}
	return "resp_" + strings.ReplaceAll(uuid.NewString(), "-", "")
}
