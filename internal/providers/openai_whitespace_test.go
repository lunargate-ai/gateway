package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestOpenAITranslator_ResponsesToChatRequestPreservesSignificantWhitespace(t *testing.T) {
	want := "\n  if ready {\n    run()\n  }\n"
	unified, err := models.ResponsesToUnifiedRequest(&models.ResponsesRequest{
		Model: "gpt-5.4",
		Input: []interface{}{
			map[string]interface{}{
				"type": "message",
				"role": "user",
				"content": []interface{}{
					map[string]interface{}{"type": "input_text", "text": want},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("ResponsesToUnifiedRequest: %v", err)
	}

	translator := NewOpenAITranslator(config.ProviderConfig{APIKey: "dummy"})
	request, err := translator.TranslateRequest(context.Background(), unified)
	if err != nil {
		t.Fatalf("TranslateRequest: %v", err)
	}
	body, err := io.ReadAll(request.Body)
	if err != nil {
		t.Fatalf("read request body: %v", err)
	}
	var payload struct {
		Messages []models.Message `json:"messages"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("decode request body: %v", err)
	}
	if len(payload.Messages) != 1 {
		t.Fatalf("messages = %#v, want one message", payload.Messages)
	}
	if got, _ := payload.Messages[0].Content.(string); got != want {
		t.Fatalf("upstream Chat content = %q, want %q", got, want)
	}
}

func TestOpenAITranslator_ChatToResponsesRequestPreservesSignificantWhitespace(t *testing.T) {
	want := "  first line\n    indented line\n"
	translator := NewOpenAITranslator(config.ProviderConfig{APIKey: "dummy"})
	request, err := translator.TranslateRequest(
		WithUpstreamRequestType(context.Background(), "responses"),
		&models.UnifiedRequest{
			RawJSON:           mustMarshalOpenAIWhitespaceTest(t, map[string]interface{}{"model": "gpt-5.4", "messages": []interface{}{map[string]interface{}{"role": "user", "content": want}}}),
			SourceRequestType: "chat_completions",
			Model:             "gpt-5.4",
			Messages:          []models.Message{{Role: "user", Content: want}},
		},
	)
	if err != nil {
		t.Fatalf("TranslateRequest: %v", err)
	}
	body, err := io.ReadAll(request.Body)
	if err != nil {
		t.Fatalf("read request body: %v", err)
	}
	var payload map[string]interface{}
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("decode request body: %v", err)
	}
	input, _ := payload["input"].([]interface{})
	if len(input) != 1 {
		t.Fatalf("input = %#v, want one message", payload["input"])
	}
	content, _ := input[0].(map[string]interface{})["content"].([]interface{})
	if len(content) != 1 {
		t.Fatalf("content = %#v, want one text part", input[0])
	}
	if got := content[0].(map[string]interface{})["text"]; got != want {
		t.Fatalf("upstream Responses content = %#v, want %q", got, want)
	}
}

func TestOpenAITranslator_ResponsesHTTPAndSSEPreserveIdenticalWhitespace(t *testing.T) {
	want := "\n  result := value\n    + offset  \n"
	translator := NewOpenAITranslator(config.ProviderConfig{APIKey: "dummy"})
	responseBody := mustMarshalOpenAIWhitespaceTest(t, map[string]interface{}{
		"id":          "resp_whitespace",
		"object":      "response",
		"created_at":  123,
		"status":      "completed",
		"model":       "gpt-5.4",
		"output":      []interface{}{},
		"output_text": want,
	})
	request := (&http.Request{}).WithContext(
		WithSourceRequestType(WithUpstreamRequestType(context.Background(), "responses"), "chat_completions"),
	)
	unified, err := translator.ParseResponse(&http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(bytes.NewReader(responseBody)),
		Request:    request,
	})
	if err != nil {
		t.Fatalf("ParseResponse: %v", err)
	}
	if len(unified.Choices) != 1 || unified.Choices[0].Message == nil {
		t.Fatalf("HTTP response choices = %#v", unified.Choices)
	}
	httpText, _ := unified.Choices[0].Message.Content.(string)
	if httpText != want {
		t.Fatalf("HTTP content = %q, want %q", httpText, want)
	}

	events := []struct {
		name  string
		event map[string]interface{}
	}{
		{
			name: "delta",
			event: map[string]interface{}{
				"type": "response.output_text.delta", "response_id": "resp_whitespace",
				"output_index": 0, "content_index": 0, "delta": want,
			},
		},
		{
			name: "text done snapshot",
			event: map[string]interface{}{
				"type": "response.output_text.done", "response_id": "resp_whitespace",
				"output_index": 0, "content_index": 0, "text": want,
			},
		},
		{
			name: "content part snapshot",
			event: map[string]interface{}{
				"type": "response.content_part.done", "response_id": "resp_whitespace",
				"output_index": 0, "content_index": 0,
				"part": map[string]interface{}{"type": "output_text", "text": want},
			},
		},
		{
			name: "output item snapshot",
			event: map[string]interface{}{
				"type": "response.output_item.done", "response_id": "resp_whitespace", "output_index": 0,
				"item": map[string]interface{}{
					"id": "msg_whitespace", "type": "message",
					"content": []interface{}{map[string]interface{}{"type": "output_text", "text": want}},
				},
			},
		},
	}
	for _, test := range events {
		t.Run(test.name, func(t *testing.T) {
			streamTranslator := NewOpenAIStreamTranslator(translator)
			chunk, parseErr := streamTranslator.ParseStreamChunk(mustMarshalOpenAIWhitespaceTest(t, test.event))
			if parseErr != nil {
				t.Fatalf("ParseStreamChunk: %v", parseErr)
			}
			if chunk == nil || len(chunk.Choices) != 1 || chunk.Choices[0].Delta == nil {
				t.Fatalf("SSE chunk = %#v", chunk)
			}
			sseText, _ := chunk.Choices[0].Delta.Content.(string)
			if sseText != httpText {
				t.Fatalf("SSE content = %q, HTTP content = %q", sseText, httpText)
			}
		})
	}
}

func TestOpenAITranslator_ResponsesHTTPFallbackPartPreservesWhitespace(t *testing.T) {
	want := "  fallback\n    content  "
	body := mustMarshalOpenAIWhitespaceTest(t, map[string]interface{}{
		"id": "resp_fallback", "object": "response", "created_at": 123,
		"status": "completed", "model": "gpt-5.4", "output_text": "",
		"output": []interface{}{map[string]interface{}{
			"type": "message", "role": "assistant", "status": "completed",
			"content": []interface{}{map[string]interface{}{"type": "output_text", "text": want}},
		}},
	})
	unified, err := NewOpenAITranslator(config.ProviderConfig{APIKey: "dummy"}).ParseResponse(&http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(bytes.NewReader(body)),
	})
	if err != nil {
		t.Fatalf("ParseResponse: %v", err)
	}
	got, _ := unified.Choices[0].Message.Content.(string)
	if got != want {
		t.Fatalf("fallback content = %q, want %q", got, want)
	}
}

func mustMarshalOpenAIWhitespaceTest(t *testing.T, value interface{}) json.RawMessage {
	t.Helper()
	b, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("marshal JSON: %v", err)
	}
	return b
}
