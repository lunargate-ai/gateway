package models

import "testing"

func TestResponsesToUnifiedRequestMapsInstructionsAndTextFormat(t *testing.T) {
	strict := true
	req := &ResponsesRequest{
		Model: "gpt-5",
		Input: "hello",
		Instructions: []interface{}{
			map[string]interface{}{
				"type":    "message",
				"role":    "developer",
				"content": "be concise",
			},
		},
		Text: &ResponsesText{Format: &ResponsesTextFormat{
			Type:        "json_schema",
			Name:        "answer",
			Description: "structured answer",
			Schema:      map[string]interface{}{"type": "object"},
			Strict:      &strict,
		}},
	}

	unified, err := ResponsesToUnifiedRequest(req)
	if err != nil {
		t.Fatalf("ResponsesToUnifiedRequest returned error: %v", err)
	}
	if len(unified.Messages) != 2 || unified.Messages[0].Role != "developer" || unified.Messages[0].Content != "be concise" {
		t.Fatalf("instructions were not preserved ahead of input: %#v", unified.Messages)
	}
	if unified.ResponseFormat == nil || unified.ResponseFormat.Type != "json_schema" || unified.ResponseFormat.JSONSchema == nil {
		t.Fatalf("response format = %#v", unified.ResponseFormat)
	}
	if unified.ResponseFormat.JSONSchema.Name != "answer" || unified.ResponseFormat.JSONSchema.Strict == nil || !*unified.ResponseFormat.JSONSchema.Strict {
		t.Fatalf("json schema options = %#v", unified.ResponseFormat.JSONSchema)
	}
}

func TestResponsesToUnifiedRequestMapsStringInstructionsAsDeveloper(t *testing.T) {
	unified, err := ResponsesToUnifiedRequest(&ResponsesRequest{
		Model:        "gpt-5",
		Input:        "hello",
		Instructions: "follow policy",
	})
	if err != nil {
		t.Fatalf("ResponsesToUnifiedRequest returned error: %v", err)
	}
	if len(unified.Messages) != 2 || unified.Messages[0].Role != "developer" || unified.Messages[0].Content != "follow policy" {
		t.Fatalf("string instructions = %#v", unified.Messages)
	}
}

func TestResponsesToUnifiedRequestCarriesExtendedInputUntilTargetValidation(t *testing.T) {
	unified, err := ResponsesToUnifiedRequest(&ResponsesRequest{
		Model: "gpt-5",
		Input: []interface{}{
			map[string]interface{}{
				"type":              "reasoning",
				"encrypted_content": "abc",
			},
		},
	})
	if err != nil {
		t.Fatalf("native Responses input was rejected before target resolution: %v", err)
	}
	if len(unified.Messages) != 1 {
		t.Fatalf("placeholder messages = %#v, want one", unified.Messages)
	}
}

func TestResponsesToUnifiedRequestMapsImageContentToChatShape(t *testing.T) {
	unified, err := ResponsesToUnifiedRequest(&ResponsesRequest{
		Model: "gpt-5",
		Input: []interface{}{
			map[string]interface{}{
				"type": "message",
				"role": "user",
				"content": []interface{}{
					map[string]interface{}{
						"type":      "input_image",
						"image_url": "data:image/png;base64,aGVsbG8=",
						"detail":    "low",
					},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("ResponsesToUnifiedRequest returned error: %v", err)
	}
	parts, ok := unified.Messages[0].Content.([]interface{})
	if !ok || len(parts) != 1 {
		t.Fatalf("image content = %#v", unified.Messages[0].Content)
	}
	part, ok := parts[0].(map[string]interface{})
	if !ok || part["type"] != "image_url" {
		t.Fatalf("image part = %#v", parts[0])
	}
	image, ok := part["image_url"].(map[string]interface{})
	if !ok || image["url"] != "data:image/png;base64,aGVsbG8=" || image["detail"] != "low" {
		t.Fatalf("image_url = %#v", part["image_url"])
	}
}
