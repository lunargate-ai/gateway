package providers

import (
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
)

func TestAnthropicStopReasonMapping(t *testing.T) {
	tests := []struct {
		name    string
		reason  string
		details *anthropicStopDetails
		want    string
	}{
		{name: "natural stop", reason: "end_turn", want: "stop"},
		{name: "token limit", reason: "max_tokens", want: "length"},
		{name: "context limit", reason: "model_context_window_exceeded", want: "length"},
		{name: "paused server tool loop", reason: "pause_turn", want: "length"},
		{name: "custom stop", reason: "stop_sequence", want: "stop"},
		{name: "client tool", reason: "tool_use", want: "tool_calls"},
		{name: "refusal reason", reason: "refusal", want: "content_filter"},
		{name: "refusal details", reason: "end_turn", details: &anthropicStopDetails{Type: "refusal"}, want: "content_filter"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := mapAnthropicStopReason(&tt.reason, tt.details)
			if got == nil || *got != tt.want {
				t.Fatalf("finish reason = %#v, want %q", got, tt.want)
			}
		})
	}
}

func TestAnthropicParseResponsePreservesRefusalExplanation(t *testing.T) {
	translator := NewAnthropicTranslator(config.ProviderConfig{})
	response, err := translator.ParseResponse(&http.Response{
		StatusCode: http.StatusOK,
		Body: io.NopCloser(strings.NewReader(`{
			"id":"msg_refusal","type":"message","role":"assistant","model":"claude-opus-5",
			"content":[
				{"type":"thinking","thinking":"private reasoning","signature":"signed"},
				{"type":"text","text":"partial visible answer"},
				{"type":"tool_use","id":"toolu_partial","name":"unsafe_partial_tool","input":{"value":1}}
			],
			"stop_reason":"refusal",
			"stop_details":{"type":"refusal","category":"cyber","explanation":"This request was declined."},
			"usage":{"input_tokens":3,"output_tokens":4}
		}`)),
	})
	if err != nil {
		t.Fatalf("ParseResponse returned error: %v", err)
	}
	choice := response.Choices[0]
	if choice.FinishReason == nil || *choice.FinishReason != "content_filter" {
		t.Fatalf("finish reason = %#v, want content_filter", choice.FinishReason)
	}
	if choice.Message.Refusal != "This request was declined." {
		t.Fatalf("refusal = %q", choice.Message.Refusal)
	}
	if choice.Message.ContentString() != "" {
		t.Fatalf("refusal retained partial content = %#v", choice.Message.Content)
	}
	if len(choice.Message.ToolCalls) != 0 {
		t.Fatalf("refusal retained partial tool calls = %#v", choice.Message.ToolCalls)
	}
	if strings.Contains(choice.Message.ContentString(), "private reasoning") {
		t.Fatal("thinking content was exposed")
	}
}

func TestAnthropicStreamPreservesRefusalExplanation(t *testing.T) {
	translator := NewAnthropicStreamTranslator(NewAnthropicTranslator(config.ProviderConfig{}))
	partial, err := translator.ParseStreamChunk([]byte(`{
		"type":"content_block_delta",
		"index":0,
		"delta":{"type":"text_delta","text":"partial visible answer"}
	}`))
	if err != nil {
		t.Fatalf("parse partial stream chunk: %v", err)
	}
	if partial == nil || partial.Choices[0].Delta.Content != "partial visible answer" {
		t.Fatalf("partial stream chunk = %#v", partial)
	}

	chunk, err := translator.ParseStreamChunk([]byte(`{
		"type":"message_delta",
		"delta":{
			"stop_reason":"refusal",
			"stop_details":{"type":"refusal","category":"cyber","explanation":"This request was declined."}
		},
		"usage":{"output_tokens":4}
	}`))
	if err != nil {
		t.Fatalf("ParseStreamChunk returned error: %v", err)
	}
	choice := chunk.Choices[0]
	if choice.FinishReason == nil || *choice.FinishReason != "content_filter" {
		t.Fatalf("finish reason = %#v, want content_filter", choice.FinishReason)
	}
	if choice.Delta.Refusal != "This request was declined." {
		t.Fatalf("refusal delta = %q", choice.Delta.Refusal)
	}
	// Streaming cannot retract the partial delta already delivered above; the
	// final content_filter + refusal signal tells clients to discard it.
}
