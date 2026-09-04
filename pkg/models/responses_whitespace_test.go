package models

import "testing"

func TestResponsesToUnifiedRequest_PreservesSignificantTextWhitespace(t *testing.T) {
	instructions := "  keep the layout exactly\n"
	inputText := "\n  if ready {\n    run()\n  }\n"

	unified, err := ResponsesToUnifiedRequest(&ResponsesRequest{
		Model:        "gpt-5.4",
		Instructions: instructions,
		Input: []interface{}{
			map[string]interface{}{
				"type": "message",
				"role": "user",
				"content": []interface{}{
					map[string]interface{}{
						"type": "input_text",
						"text": inputText,
					},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("ResponsesToUnifiedRequest returned error: %v", err)
	}
	if len(unified.Messages) != 2 {
		t.Fatalf("messages = %#v, want developer and user messages", unified.Messages)
	}
	if got, _ := unified.Messages[0].Content.(string); got != instructions {
		t.Fatalf("instructions = %q, want %q", got, instructions)
	}
	if got, _ := unified.Messages[1].Content.(string); got != inputText {
		t.Fatalf("input text = %q, want %q", got, inputText)
	}
}

func TestResponsesToUnifiedRequest_PreservesFallbackTextWhitespace(t *testing.T) {
	want := "\n  fallback text  \n"
	unified, err := ResponsesToUnifiedRequest(&ResponsesRequest{
		Model: "gpt-5.4",
		Input: []interface{}{
			map[string]interface{}{
				"type": "message",
				"role": "user",
				"text": want,
			},
		},
	})
	if err != nil {
		t.Fatalf("ResponsesToUnifiedRequest returned error: %v", err)
	}
	if len(unified.Messages) != 1 {
		t.Fatalf("messages = %#v, want one user message", unified.Messages)
	}
	if got, _ := unified.Messages[0].Content.(string); got != want {
		t.Fatalf("fallback text = %q, want %q", got, want)
	}
}

func TestUnifiedResponseToResponses_PreservesSignificantTextWhitespace(t *testing.T) {
	wantText := "\n  result := value\n    + offset  \n"
	wantReasoning := "  preserve indentation\n    before returning  "

	out := UnifiedResponseToResponses(&UnifiedResponse{
		ID:      "chatcmpl_whitespace",
		Created: 123,
		Model:   "openai/gpt-5.4",
		Choices: []Choice{{
			Index: 0,
			Message: &Message{
				Role:             "assistant",
				Content:          wantText,
				ReasoningContent: wantReasoning,
			},
		}},
	})
	if out == nil || len(out.Output) != 2 {
		t.Fatalf("response = %#v, want message and reasoning output", out)
	}
	if len(out.Output[0].Content) != 1 || out.Output[0].Content[0].Text != wantText {
		t.Fatalf("message content = %#v, want exact text %q", out.Output[0].Content, wantText)
	}
	if out.OutputText != wantText {
		t.Fatalf("output_text = %q, want %q", out.OutputText, wantText)
	}
	if len(out.Output[1].Summary) != 1 || out.Output[1].Summary[0].Text != wantReasoning {
		t.Fatalf("reasoning summary = %#v, want exact text %q", out.Output[1].Summary, wantReasoning)
	}
}
