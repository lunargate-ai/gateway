package models

import "testing"

func TestNormalizeUnifiedRequest_MapsReasoningObjectToReasoningEffort(t *testing.T) {
	req := &UnifiedRequest{
		Model: "gpt-5.2",
		Messages: []Message{
			{Role: "user", Content: "hi"},
		},
		Reasoning: &Reasoning{Effort: "low"},
	}

	if err := NormalizeUnifiedRequest(req); err != nil {
		t.Fatalf("NormalizeUnifiedRequest returned error: %v", err)
	}
	if req.ReasoningEffort != "low" {
		t.Fatalf("expected reasoning_effort=low, got %q", req.ReasoningEffort)
	}
	if req.Reasoning != nil {
		t.Fatalf("expected canonicalized reasoning object to be nil")
	}
}

func TestNormalizeUnifiedRequest_MapsMaxCompletionTokens(t *testing.T) {
	maxCompletionTokens := 321
	req := &UnifiedRequest{MaxCompletionTokens: &maxCompletionTokens}

	if err := NormalizeUnifiedRequest(req); err != nil {
		t.Fatalf("NormalizeUnifiedRequest returned error: %v", err)
	}
	if req.MaxTokens == nil || *req.MaxTokens != maxCompletionTokens {
		t.Fatalf("max_tokens = %#v, want %d", req.MaxTokens, maxCompletionTokens)
	}
	if req.MaxCompletionTokens != nil {
		t.Fatalf("max_completion_tokens was not canonicalized: %#v", req.MaxCompletionTokens)
	}
}
