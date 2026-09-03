package providers

import (
	"context"
	"errors"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestAnthropicTranslatorRejectsLossyTypedMessageHistory(t *testing.T) {
	tests := []struct {
		name      string
		message   models.Message
		wantField string
	}{
		{name: "participant name", message: models.Message{Role: "user", Content: "hi", Name: "alice"}, wantField: "messages[0].name"},
		{name: "assistant refusal", message: models.Message{Role: "assistant", Refusal: "no"}, wantField: "messages[0].refusal"},
		{name: "assistant reasoning", message: models.Message{Role: "assistant", ReasoningContent: "private"}, wantField: "messages[0].reasoning_content"},
		{
			name: "image detail",
			message: models.Message{Role: "user", Content: []interface{}{map[string]interface{}{
				"type": "image_url", "image_url": map[string]interface{}{"url": "data:image/png;base64,aQ==", "detail": "high"},
			}}},
			wantField: "messages[0].content[0].image_url.detail",
		},
		{
			name: "custom tool call",
			message: models.Message{Role: "assistant", ToolCalls: []models.ToolCall{{
				ID: "call_1", Type: "custom", Function: models.ToolCallFunction{Name: "lookup", Arguments: `{}`},
			}}},
			wantField: "messages[0].tool_calls[0].type",
		},
		{
			name: "non-object tool arguments",
			message: models.Message{Role: "assistant", ToolCalls: []models.ToolCall{{
				ID: "call_1", Type: "function", Function: models.ToolCallFunction{Name: "lookup", Arguments: `[]`},
			}}},
			wantField: "messages[0].tool_calls[0].function.arguments",
		},
		{
			name: "image tool result",
			message: models.Message{Role: "tool", ToolCallID: "call_1", Content: []interface{}{map[string]interface{}{
				"type": "image_url", "image_url": map[string]interface{}{"url": "data:image/png;base64,aQ=="},
			}}},
			wantField: "messages[0].content[0].type",
		},
	}

	translator := NewAnthropicTranslator(config.ProviderConfig{APIKey: "dummy"})
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := translator.TranslateRequest(context.Background(), &models.UnifiedRequest{
				Model:    "claude",
				Messages: []models.Message{tt.message},
			})
			var compatibilityErr *models.CompatibilityError
			if !errors.As(err, &compatibilityErr) {
				t.Fatalf("error = %v, want CompatibilityError", err)
			}
			if compatibilityErr.Field != tt.wantField {
				t.Fatalf("field = %q, want %q", compatibilityErr.Field, tt.wantField)
			}
		})
	}
}

func TestAnthropicTranslatorRejectsEmptyConversationMessages(t *testing.T) {
	tests := []struct {
		name    string
		message models.Message
	}{
		{name: "null user", message: models.Message{Role: "user"}},
		{name: "empty user string", message: models.Message{Role: "user", Content: ""}},
		{name: "empty user array", message: models.Message{Role: "user", Content: []interface{}{}}},
		{name: "empty user text part", message: models.Message{Role: "user", Content: []interface{}{map[string]interface{}{"type": "text", "text": ""}}}},
		{name: "empty assistant without tool calls", message: models.Message{Role: "assistant", Content: ""}},
	}

	translator := NewAnthropicTranslator(config.ProviderConfig{APIKey: "dummy"})
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := translator.TranslateRequest(context.Background(), &models.UnifiedRequest{
				Model: "claude", Messages: []models.Message{tt.message},
			})
			var compatibilityErr *models.CompatibilityError
			if !errors.As(err, &compatibilityErr) || compatibilityErr.Field != "messages[0].content" {
				t.Fatalf("error = %#v, want messages[0].content CompatibilityError", err)
			}
		})
	}
}

func TestOllamaTranslatorRejectsRefusalHistory(t *testing.T) {
	translator := NewOllamaTranslator(config.ProviderConfig{})
	_, err := translator.TranslateRequest(context.Background(), &models.UnifiedRequest{
		Model:    "qwen",
		Messages: []models.Message{{Role: "assistant", Refusal: "no"}},
	})
	var compatibilityErr *models.CompatibilityError
	if !errors.As(err, &compatibilityErr) || compatibilityErr.Field != "messages[0].refusal" {
		t.Fatalf("error = %#v, want messages[0].refusal CompatibilityError", err)
	}
}

func TestAnthropicTranslatorPreservesToolResultTextParts(t *testing.T) {
	translator := NewAnthropicTranslator(config.ProviderConfig{APIKey: "dummy"})
	payload := translateAnthropicRequest(t, translator, &models.UnifiedRequest{
		Model: "claude",
		Messages: []models.Message{{
			Role:       "tool",
			ToolCallID: "call_1",
			Content: []interface{}{
				map[string]interface{}{"type": "text", "text": "first"},
				map[string]interface{}{"type": "input_text", "text": "second"},
			},
		}},
	})

	if len(payload.Messages) != 1 {
		t.Fatalf("messages = %#v, want one tool-result message", payload.Messages)
	}
	outer, ok := payload.Messages[0].Content.([]interface{})
	if !ok || len(outer) != 1 {
		t.Fatalf("tool result blocks = %#v", payload.Messages[0].Content)
	}
	toolResult, ok := outer[0].(map[string]interface{})
	if !ok {
		t.Fatalf("tool result = %#v", outer[0])
	}
	content, ok := toolResult["content"].([]interface{})
	if !ok || len(content) != 2 {
		t.Fatalf("tool result content = %#v, want two text blocks", toolResult["content"])
	}
	if content[0].(map[string]interface{})["text"] != "first" || content[1].(map[string]interface{})["text"] != "second" {
		t.Fatalf("tool result content = %#v", content)
	}
}

func TestAnthropicTranslatorOmitsEmptyToolResultContent(t *testing.T) {
	tests := []struct {
		name    string
		content interface{}
	}{
		{name: "null", content: nil},
		{name: "empty string", content: ""},
		{name: "empty array", content: []interface{}{}},
		{name: "empty text part", content: []interface{}{map[string]interface{}{"type": "text", "text": ""}}},
	}

	translator := NewAnthropicTranslator(config.ProviderConfig{APIKey: "dummy"})
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			payload := translateAnthropicRequest(t, translator, &models.UnifiedRequest{
				Model: "claude",
				Messages: []models.Message{{
					Role: "tool", ToolCallID: "call_1", Content: tt.content,
				}},
			})

			outer := payload.Messages[0].Content.([]interface{})
			toolResult := outer[0].(map[string]interface{})
			if _, exists := toolResult["content"]; exists {
				t.Fatalf("tool result = %#v, want optional content omitted", toolResult)
			}
		})
	}
}

func TestAnthropicTranslatorSendsEmptyToolArgumentsAsObject(t *testing.T) {
	translator := NewAnthropicTranslator(config.ProviderConfig{APIKey: "dummy"})
	payload := translateAnthropicRequest(t, translator, &models.UnifiedRequest{
		Model: "claude",
		Messages: []models.Message{{Role: "assistant", ToolCalls: []models.ToolCall{{
			ID: "call_1", Type: "function", Function: models.ToolCallFunction{Name: "lookup"},
		}}}},
	})

	blocks, ok := payload.Messages[0].Content.([]interface{})
	if !ok || len(blocks) != 1 {
		t.Fatalf("assistant blocks = %#v", payload.Messages[0].Content)
	}
	input, ok := blocks[0].(map[string]interface{})["input"].(map[string]interface{})
	if !ok || len(input) != 0 {
		t.Fatalf("tool input = %#v, want empty object", blocks[0])
	}
}

func TestAnthropicTranslatorRejectsAmbiguousOrInvalidImageReferences(t *testing.T) {
	tests := []struct {
		name      string
		part      map[string]interface{}
		wantField string
	}{
		{
			name:      "missing reference",
			part:      map[string]interface{}{"type": "input_image"},
			wantField: "messages[0].content[0].image_url",
		},
		{
			name:      "wrong reference type",
			part:      map[string]interface{}{"type": "image_url", "image_url": true},
			wantField: "messages[0].content[0].image_url",
		},
		{
			name: "missing nested URL",
			part: map[string]interface{}{
				"type": "image_url", "image_url": map[string]interface{}{"detail": "auto"},
			},
			wantField: "messages[0].content[0].image_url.url",
		},
		{
			name: "conflicting references",
			part: map[string]interface{}{
				"type": "input_image", "url": "https://example.com/a.png", "image_url": "https://example.com/b.png",
			},
			wantField: "messages[0].content[0].image_url",
		},
		{
			name: "unsupported URL",
			part: map[string]interface{}{
				"type": "image_url", "image_url": map[string]interface{}{"url": "file:///tmp/image.png"},
			},
			wantField: "messages[0].content[0].image_url.url",
		},
		{
			name: "unsupported data URL media type",
			part: map[string]interface{}{
				"type": "image_url", "image_url": map[string]interface{}{"url": "data:text/plain;base64,aQ=="},
			},
			wantField: "messages[0].content[0].image_url.url",
		},
		{
			name: "malformed base64 image",
			part: map[string]interface{}{
				"type": "image_url", "image_url": map[string]interface{}{"url": "data:image/png;base64,not-base64"},
			},
			wantField: "messages[0].content[0].image_url.url",
		},
	}

	translator := NewAnthropicTranslator(config.ProviderConfig{APIKey: "dummy"})
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := translator.TranslateRequest(context.Background(), &models.UnifiedRequest{
				Model: "claude",
				Messages: []models.Message{{
					Role: "user", Content: []interface{}{tt.part},
				}},
			})
			var compatibilityErr *models.CompatibilityError
			if !errors.As(err, &compatibilityErr) || compatibilityErr.Field != tt.wantField {
				t.Fatalf("error = %#v, want field %q", err, tt.wantField)
			}
		})
	}
}

func TestAnthropicTranslatorNormalizesCaseInsensitiveImageDataURL(t *testing.T) {
	translator := NewAnthropicTranslator(config.ProviderConfig{APIKey: "dummy"})
	payload := translateAnthropicRequest(t, translator, &models.UnifiedRequest{
		Model: "claude",
		Messages: []models.Message{{
			Role: "user",
			Content: []interface{}{map[string]interface{}{
				"type": "image_url", "image_url": "DATA:IMAGE/PNG;BASE64,aQ==",
			}},
		}},
	})

	blocks := payload.Messages[0].Content.([]interface{})
	source := blocks[0].(map[string]interface{})["source"].(map[string]interface{})
	if source["type"] != "base64" || source["media_type"] != "image/png" || source["data"] != "aQ==" {
		t.Fatalf("image source = %#v", source)
	}
}

func TestAnthropicTranslatorAcceptsIdenticalImageReferenceAliases(t *testing.T) {
	const imageURL = "https://example.com/image.png"
	translator := NewAnthropicTranslator(config.ProviderConfig{APIKey: "dummy"})
	payload := translateAnthropicRequest(t, translator, &models.UnifiedRequest{
		Model: "claude",
		Messages: []models.Message{{
			Role: "user",
			Content: []interface{}{map[string]interface{}{
				"type": "input_image", "url": imageURL, "image_url": map[string]interface{}{"url": imageURL},
			}},
		}},
	})

	blocks := payload.Messages[0].Content.([]interface{})
	source := blocks[0].(map[string]interface{})["source"].(map[string]interface{})
	if source["type"] != "url" || source["url"] != imageURL {
		t.Fatalf("image source = %#v", source)
	}
}
