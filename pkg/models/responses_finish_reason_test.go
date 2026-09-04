package models

import "testing"

func TestUnifiedResponseToResponses_MapsTerminalFinishReasons(t *testing.T) {
	tests := []struct {
		name             string
		finishReason     string
		wantStatus       string
		wantIncomplete   string
		wantOutputStatus string
	}{
		{
			name:             "stop remains completed",
			finishReason:     "stop",
			wantStatus:       "completed",
			wantOutputStatus: "completed",
		},
		{
			name:             "tool calls remain completed",
			finishReason:     "tool_calls",
			wantStatus:       "completed",
			wantOutputStatus: "completed",
		},
		{
			name:             "length is max output tokens",
			finishReason:     "length",
			wantStatus:       "incomplete",
			wantIncomplete:   ResponsesIncompleteReasonMaxOutputTokens,
			wantOutputStatus: "incomplete",
		},
		{
			name:             "content filter is incomplete",
			finishReason:     "content_filter",
			wantStatus:       "incomplete",
			wantIncomplete:   ResponsesIncompleteReasonContentFilter,
			wantOutputStatus: "incomplete",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			finishReason := tc.finishReason
			resp := &UnifiedResponse{
				ID:      "resp_finish",
				Created: 123,
				Model:   "mock-gpt",
				Choices: []Choice{{
					Index:        0,
					FinishReason: &finishReason,
					Message: &Message{
						Role:             "assistant",
						Content:          "partial answer",
						ReasoningContent: "partial reasoning",
						ToolCalls: []ToolCall{{
							ID:   "call_finish",
							Type: "function",
							Function: ToolCallFunction{
								Name:      "lookup",
								Arguments: `{"query":"partial"}`,
							},
						}},
					},
				}},
			}

			got := UnifiedResponseToResponses(resp)
			if got.Status != tc.wantStatus {
				t.Fatalf("status = %q, want %q", got.Status, tc.wantStatus)
			}
			if tc.wantIncomplete == "" {
				if got.IncompleteDetails != nil {
					t.Fatalf("incomplete_details = %#v, want nil", got.IncompleteDetails)
				}
			} else {
				if got.IncompleteDetails == nil {
					t.Fatal("incomplete_details is nil")
				}
				if got.IncompleteDetails.Reason != tc.wantIncomplete {
					t.Fatalf("incomplete_details.reason = %q, want %q", got.IncompleteDetails.Reason, tc.wantIncomplete)
				}
			}
			if len(got.Output) != 3 {
				t.Fatalf("output items = %d, want 3", len(got.Output))
			}
			for i, item := range got.Output {
				if item.Status != tc.wantOutputStatus {
					t.Errorf("output[%d].status = %q, want %q", i, item.Status, tc.wantOutputStatus)
				}
			}
		})
	}
}

func TestUnifiedResponseToResponses_ContentFilterTakesPrecedence(t *testing.T) {
	length := "length"
	contentFilter := "content_filter"
	resp := &UnifiedResponse{
		ID:    "resp_multi",
		Model: "mock-gpt",
		Choices: []Choice{
			{Index: 0, FinishReason: &length},
			{Index: 1, FinishReason: &contentFilter},
		},
	}

	got := UnifiedResponseToResponses(resp)
	if got.Status != "incomplete" {
		t.Fatalf("status = %q, want incomplete", got.Status)
	}
	if got.IncompleteDetails == nil || got.IncompleteDetails.Reason != ResponsesIncompleteReasonContentFilter {
		t.Fatalf("incomplete_details = %#v, want content_filter", got.IncompleteDetails)
	}
}
