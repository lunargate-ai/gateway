package models

import "testing"

func TestResponsesToUnifiedRequestNeverCanonicalizesPreviousResponseID(t *testing.T) {
	for _, previousResponseID := range []string{
		"resp_exact",
		"resp_internal space",
		" resp_invalid_boundary_whitespace ",
	} {
		req := &ResponsesRequest{
			Model:              "mock-gpt",
			Input:              "continue",
			PreviousResponseID: previousResponseID,
		}
		unified, err := ResponsesToUnifiedRequest(req)
		if err != nil {
			t.Fatalf("previous_response_id=%q translation failed: %v", previousResponseID, err)
		}
		if unified.PreviousResponseID != previousResponseID {
			t.Fatalf(
				"previous_response_id=%q translated as %q",
				previousResponseID,
				unified.PreviousResponseID,
			)
		}
	}
}

func TestResponsesToUnifiedRequestNeverCanonicalizesToolCallIDs(t *testing.T) {
	for _, callID := range []string{"call_exact", "call internal space", " call_boundary_whitespace "} {
		req := &ResponsesRequest{
			Model: "mock-gpt",
			Input: []interface{}{
				map[string]interface{}{
					"type":    "function_call_output",
					"call_id": callID,
					"output":  "done",
				},
			},
		}
		unified, err := ResponsesToUnifiedRequest(req)
		if err != nil {
			t.Fatalf("call_id=%q translation failed: %v", callID, err)
		}
		if len(unified.Messages) != 1 || unified.Messages[0].ToolCallID != callID {
			t.Fatalf("call_id=%q translated as %#v", callID, unified.Messages)
		}
	}
}

func TestUnifiedResponseToResponsesNeverCanonicalizesDerivedItemIDs(t *testing.T) {
	response := UnifiedResponseToResponses(&UnifiedResponse{
		ID: " chat response id ",
		Choices: []Choice{{
			Index:   0,
			Message: &Message{Role: "assistant", Content: "done"},
		}},
	})
	if response == nil || len(response.Output) != 1 {
		t.Fatalf("response output = %#v", response)
	}
	if response.Output[0].ID != "msg_ chat response id _0" {
		t.Fatalf("derived item id = %q, want source identity preserved exactly", response.Output[0].ID)
	}
}
