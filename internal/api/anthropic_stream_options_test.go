package api

import (
	"testing"

	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestIncludeClientStreamUsageFollowsClientContract(t *testing.T) {
	include := &models.UnifiedRequest{StreamOptions: &models.StreamOptions{IncludeUsage: true}}
	exclude := &models.UnifiedRequest{StreamOptions: &models.StreamOptions{IncludeUsage: false}}
	if !includeClientStreamUsage(requestTypeChatCompletions, include) {
		t.Fatal("explicit Chat Completions include_usage=true was ignored")
	}
	if includeClientStreamUsage(requestTypeChatCompletions, exclude) {
		t.Fatal("Chat Completions include_usage=false was ignored")
	}
	if !includeClientStreamUsage(requestTypeResponses, nil) {
		t.Fatal("Responses terminal usage must always be retained")
	}
}
