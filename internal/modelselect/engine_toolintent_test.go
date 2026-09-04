package modelselect

import (
	"testing"

	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestRequestContainsToolIntent(t *testing.T) {
	tests := []struct {
		name       string
		toolChoice interface{}
		want       bool
	}{
		{name: "omitted", want: true},
		{name: "auto", toolChoice: "auto", want: true},
		{name: "required", toolChoice: "required", want: true},
		{name: "named function", toolChoice: map[string]interface{}{"type": "function"}, want: true},
		{name: "none", toolChoice: "none", want: false},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			req := &models.UnifiedRequest{
				Tools:      []models.Tool{{Type: "function", Function: models.ToolFunction{Name: "terminal"}}},
				ToolChoice: test.toolChoice,
				Messages:   []models.Message{{Role: "user", Content: "odpowiedz zwyczajnie"}},
			}
			if got := requestContainsToolIntent(req); got != test.want {
				t.Fatalf("requestContainsToolIntent() = %v, want %v", got, test.want)
			}
		})
	}
}

func TestRequestContainsToolIntent_ContinuationOverridesNone(t *testing.T) {
	req := &models.UnifiedRequest{
		Tools:      []models.Tool{{Type: "function", Function: models.ToolFunction{Name: "terminal"}}},
		ToolChoice: "none",
		Messages: []models.Message{{
			Role:       "tool",
			ToolCallID: "call_123",
			Content:    "done",
		}},
	}
	if !requestContainsToolIntent(req) {
		t.Fatal("expected tool continuation to require tool-capable routing")
	}
}
