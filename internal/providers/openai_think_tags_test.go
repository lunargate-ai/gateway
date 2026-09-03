package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestOpenAITranslator_ReasoningTagExtractionIsDisabledByDefault(t *testing.T) {
	body := []byte(`{"id":"chatcmpl_literal","object":"chat.completion","created":1,"model":"deepseek-chat","choices":[{"index":0,"message":{"role":"assistant","content":"Use <think>literal markup</think> here","x_message":{"kept":true}},"finish_reason":"stop","x_choice":9007199254740993}],"x_vendor":{"kept":true}}`)
	translator := NewOpenAITranslator(config.ProviderConfig{
		CompatibilityProfile: "deepseek",
	})

	response, err := translator.ParseResponse(&http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(bytes.NewReader(body)),
	})
	if err != nil {
		t.Fatalf("ParseResponse returned error: %v", err)
	}
	if got := response.Choices[0].Message.ContentString(); got != "Use <think>literal markup</think> here" {
		t.Fatalf("content = %q, want literal tags", got)
	}
	if got := response.Choices[0].Message.ReasoningContent; got != "" {
		t.Fatalf("reasoning_content = %q, want empty", got)
	}
	if !bytes.Equal(response.RawJSON, body) {
		t.Fatalf("raw response changed:\n got: %s\nwant: %s", response.RawJSON, body)
	}
}

func TestOpenAITranslator_ReasoningTagExtractionPreservesHTTPEnvelope(t *testing.T) {
	body := `{"id":"chatcmpl_extract","object":"chat.completion","created":1,"model":"custom","choices":[{"index":0,"message":{"role":"assistant","content":"  prefix <think> plan \n</think> suffix  ","reasoning_content":"existing ","refusal":"kept","x_message":{"kept":true}},"finish_reason":"stop","x_choice":9007199254740993}],"service_tier":"priority","x_vendor":{"nested":{"kept":true}}}`
	translator := NewOpenAITranslator(config.ProviderConfig{ExtractReasoningTags: true})

	response, err := translator.ParseResponse(&http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(body)),
	})
	if err != nil {
		t.Fatalf("ParseResponse returned error: %v", err)
	}
	message := response.Choices[0].Message
	if got := message.ContentString(); got != "  prefix  suffix  " {
		t.Fatalf("content = %q, want surrounding whitespace preserved", got)
	}
	if got := message.ReasoningContent; got != "existing \n plan \n" {
		t.Fatalf("reasoning_content = %q, want exact combined content", got)
	}

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(response.RawJSON, &raw); err != nil {
		t.Fatalf("decode preserved response: %v", err)
	}
	if _, ok := raw["x_vendor"]; !ok {
		t.Fatalf("top-level extension was lost: %s", response.RawJSON)
	}
	var choices []map[string]json.RawMessage
	if err := json.Unmarshal(raw["choices"], &choices); err != nil || len(choices) != 1 {
		t.Fatalf("decode choices: %v (%s)", err, raw["choices"])
	}
	if got := string(choices[0]["x_choice"]); got != "9007199254740993" {
		t.Fatalf("large additive number = %s", got)
	}
	var rawMessage map[string]json.RawMessage
	if err := json.Unmarshal(choices[0]["message"], &rawMessage); err != nil {
		t.Fatalf("decode raw message: %v", err)
	}
	for _, field := range []string{"refusal", "x_message"} {
		if _, ok := rawMessage[field]; !ok {
			t.Fatalf("message field %q was lost: %s", field, choices[0]["message"])
		}
	}
	var rawContent string
	if err := json.Unmarshal(rawMessage["content"], &rawContent); err != nil || rawContent != "  prefix  suffix  " {
		t.Fatalf("raw content = %q, err=%v", rawContent, err)
	}
	var rawReasoning string
	if err := json.Unmarshal(rawMessage["reasoning_content"], &rawReasoning); err != nil || rawReasoning != "existing \n plan \n" {
		t.Fatalf("raw reasoning_content = %q, err=%v", rawReasoning, err)
	}
}

func TestOpenAITranslator_ReasoningTagOptionDoesNotRewriteRequestContent(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		t.Run(map[bool]string{false: "disabled", true: "enabled"}[enabled], func(t *testing.T) {
			raw := json.RawMessage(`{"model":"custom","messages":[{"role":"user","content":"  <think>literal request</think>  ","x_message":true}],"x_request":{"kept":true}}`)
			translator := NewOpenAITranslator(config.ProviderConfig{
				APIKey:               "dummy",
				ExtractReasoningTags: enabled,
			})
			request, err := translator.TranslateRequest(context.Background(), &models.UnifiedRequest{
				RawJSON: raw,
				Model:   "custom",
				Messages: []models.Message{{
					Role:    "user",
					Content: "  <think>literal request</think>  ",
				}},
			})
			if err != nil {
				t.Fatalf("TranslateRequest returned error: %v", err)
			}
			body, err := io.ReadAll(request.Body)
			if err != nil {
				t.Fatalf("read request: %v", err)
			}
			var payload map[string]json.RawMessage
			if err := json.Unmarshal(body, &payload); err != nil {
				t.Fatalf("decode request: %v", err)
			}
			if _, ok := payload["x_request"]; !ok {
				t.Fatalf("request extension was lost: %s", body)
			}
			var messages []map[string]json.RawMessage
			if err := json.Unmarshal(payload["messages"], &messages); err != nil || len(messages) != 1 {
				t.Fatalf("decode messages: %v", err)
			}
			var content string
			if err := json.Unmarshal(messages[0]["content"], &content); err != nil {
				t.Fatalf("decode content: %v", err)
			}
			if content != "  <think>literal request</think>  " {
				t.Fatalf("request content = %q, want exact literal", content)
			}
			if _, ok := messages[0]["x_message"]; !ok {
				t.Fatalf("message extension was lost: %s", payload["messages"])
			}
		})
	}
}

func TestOpenAITranslator_ReasoningTagExtractionInStreamChunkIsExplicit(t *testing.T) {
	data := []byte(`{"id":"chatcmpl_stream","object":"chat.completion.chunk","created":1,"model":"custom","choices":[{"index":0,"delta":{"content":" before <think>stream plan</think> after ","x_delta":true},"finish_reason":null,"x_choice":true}],"x_vendor":true}`)
	for _, enabled := range []bool{false, true} {
		t.Run(map[bool]string{false: "disabled", true: "enabled"}[enabled], func(t *testing.T) {
			translator := NewOpenAITranslator(config.ProviderConfig{ExtractReasoningTags: enabled})
			chunk, err := translator.ParseStreamChunk(data)
			if err != nil {
				t.Fatalf("ParseStreamChunk returned error: %v", err)
			}
			if !bytes.Equal(chunk.RawJSON, data) {
				t.Fatalf("raw stream chunk changed:\n got: %s\nwant: %s", chunk.RawJSON, data)
			}
			message := chunk.Choices[0].Delta
			if enabled {
				if got := message.ContentString(); got != " before  after " {
					t.Fatalf("content = %q, want extracted content", got)
				}
				if got := message.ReasoningContent; got != "stream plan" {
					t.Fatalf("reasoning_content = %q", got)
				}
				return
			}
			if got := message.ContentString(); got != " before <think>stream plan</think> after " {
				t.Fatalf("content = %q, want literal tags", got)
			}
			if message.ReasoningContent != "" {
				t.Fatalf("reasoning_content = %q, want empty", message.ReasoningContent)
			}
		})
	}
}
