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
