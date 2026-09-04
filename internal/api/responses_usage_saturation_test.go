package api

import (
	"testing"

	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestResponsesStreamUsageSaturatesComponentTotal(t *testing.T) {
	maximum := int(^uint(0) >> 1)
	proxy := newResponsesStreamProxy(nil)
	proxy.mergeUsage(&models.Usage{
		PromptTokens:     maximum,
		CompletionTokens: maximum,
	})

	if proxy.usage == nil || proxy.usage.TotalTokens != maximum {
		t.Fatalf("usage = %#v, want total saturated to %d", proxy.usage, maximum)
	}
}
