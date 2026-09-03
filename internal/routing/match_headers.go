package routing

import (
	"sort"
	"strings"

	"github.com/lunargate-ai/gateway/internal/config"
)

// MatchHeaderNames returns the normalized inbound header names referenced by
// the current routing generation. Callers can capture only those additional
// request headers for matching without adding arbitrary headers to logs or
// observability payloads.
func (e *Engine) MatchHeaderNames() []string {
	if e == nil {
		return nil
	}
	cfg, _ := e.config.Load().(*config.RoutingConfig)
	if cfg == nil {
		return nil
	}

	seen := make(map[string]struct{})
	for _, route := range cfg.Routes {
		for rawName := range route.Match.Headers {
			name := strings.ToLower(strings.TrimSpace(rawName))
			if name != "" {
				seen[name] = struct{}{}
			}
		}
	}
	result := make([]string, 0, len(seen))
	for name := range seen {
		result = append(result, name)
	}
	sort.Strings(result)
	return result
}
