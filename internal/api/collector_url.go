package api

import "github.com/lunargate-ai/gateway/internal/safeurl"

// sanitizeCollectorUpstreamBaseURL keeps routing context useful to the
// collector without exporting credentials commonly embedded in provider URLs.
func sanitizeCollectorUpstreamBaseURL(raw string) (string, bool) {
	return safeurl.RedactedHTTPURL(raw)
}
