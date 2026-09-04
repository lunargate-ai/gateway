package providers

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestOpenAITranslator_ResponsesRefusalPreservedInChatResponse(t *testing.T) {
	const body = `{"id":"resp_refusal","object":"response","created_at":123,"status":"completed","model":"gpt-5.4","output":[{"type":"message","id":"msg_refusal","status":"completed","role":"assistant","content":[{"type":"refusal","refusal":"I can't help with that."}]}],"output_text":"","usage":{"input_tokens":3,"output_tokens":4,"total_tokens":7}}`
	request, err := http.NewRequestWithContext(
		WithSourceRequestType(WithUpstreamRequestType(context.Background(), "responses"), "chat_completions"),
		http.MethodPost,
		"https://api.openai.com/v1/responses",
		nil,
	)
	if err != nil {
		t.Fatalf("create request: %v", err)
	}

	translator := NewOpenAITranslator(config.ProviderConfig{APIKey: "dummy"})
	response, err := translator.ParseResponse(&http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(body)),
		Request:    request,
	})
	if err != nil {
		t.Fatalf("ParseResponse returned error: %v", err)
	}
	if response == nil || len(response.Choices) != 1 || response.Choices[0].Message == nil {
		t.Fatalf("expected one Chat Completions message, got %#v", response)
	}
	choice := response.Choices[0]
	if choice.Message.Refusal != "I can't help with that." {
		t.Fatalf("refusal = %q", choice.Message.Refusal)
	}
	if choice.Message.Content != nil {
		t.Fatalf("refusal was disguised as content: %#v", choice.Message.Content)
	}
	if choice.FinishReason == nil || *choice.FinishReason != "stop" {
		t.Fatalf("finish_reason = %#v, want stop", choice.FinishReason)
	}
	if len(response.RawJSON) != 0 {
		t.Fatalf("translated Responses envelope leaked as RawJSON: %s", response.RawJSON)
	}
}

func TestOpenAITranslator_NativeResponsesRefusalKeepsRawEnvelope(t *testing.T) {
	const body = `{"id":"resp_native_refusal","object":"response","created_at":123,"status":"completed","model":"gpt-5.4","output":[{"type":"message","id":"msg_refusal","status":"completed","role":"assistant","content":[{"type":"refusal","refusal":"declined","future_part_field":{"kept":true}}]}],"output_text":"","future_top_level":{"large_integer":9007199254740993}}`
	request, err := http.NewRequestWithContext(
		WithSourceRequestType(WithUpstreamRequestType(context.Background(), "responses"), "responses"),
		http.MethodPost,
		"https://api.openai.com/v1/responses",
		nil,
	)
	if err != nil {
		t.Fatalf("create request: %v", err)
	}

	translator := NewOpenAITranslator(config.ProviderConfig{APIKey: "dummy"})
	response, err := translator.ParseResponse(&http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(body)),
		Request:    request,
	})
	if err != nil {
		t.Fatalf("ParseResponse returned error: %v", err)
	}
	if response == nil || string(response.RawJSON) != body {
		t.Fatalf("native RawJSON changed:\n got: %s\nwant: %s", response.RawJSON, body)
	}
	if got := response.Choices[0].Message.Refusal; got != "declined" {
		t.Fatalf("typed refusal = %q, want declined", got)
	}
}

func TestOpenAITranslator_ResponsesRefusalEventsMapToChatDeltas(t *testing.T) {
	tests := []struct {
		name  string
		event string
		want  string
	}{
		{
			name:  "delta",
			event: `{"type":"response.refusal.delta","response_id":"resp_1","item_id":"msg_1","output_index":0,"content_index":0,"delta":"decl"}`,
			want:  "decl",
		},
		{
			name:  "done",
			event: `{"type":"response.refusal.done","response_id":"resp_1","item_id":"msg_1","output_index":0,"content_index":0,"refusal":"declined"}`,
			want:  "declined",
		},
		{
			name:  "content part snapshot",
			event: `{"type":"response.content_part.done","response_id":"resp_1","item_id":"msg_1","output_index":0,"content_index":0,"part":{"type":"refusal","refusal":"declined"}}`,
			want:  "declined",
		},
		{
			name:  "message item snapshot",
			event: `{"type":"response.output_item.done","response_id":"resp_1","output_index":0,"item":{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"refusal","refusal":"declined"}]}}`,
			want:  "declined",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			translator := NewOpenAITranslator(config.ProviderConfig{APIKey: "dummy"})
			chunk, err := translator.ParseStreamChunk([]byte(tt.event))
			if err != nil {
				t.Fatalf("ParseStreamChunk returned error: %v", err)
			}
			if got := streamChunkRefusal(t, chunk); got != tt.want {
				t.Fatalf("refusal delta = %q, want %q", got, tt.want)
			}
			if chunk.ID != "resp_1" || chunk.Object != "chat.completion.chunk" {
				t.Fatalf("unexpected Chat SSE envelope: %#v", chunk)
			}
		})
	}
}

func TestOpenAIStreamTranslator_RefusalSnapshotsDoNotDuplicateDeltas(t *testing.T) {
	translator := newOpenAIStreamTranslatorForTest()

	created, err := translator.ParseStreamChunk([]byte(`{"type":"response.created","response":{"id":"resp_refusal","object":"response","created_at":123,"status":"in_progress","model":"gpt-5.4","output":[]}}`))
	if err != nil || created == nil {
		t.Fatalf("response.created: chunk=%#v err=%v", created, err)
	}

	for i, delta := range []string{"No", "No"} {
		chunk, parseErr := translator.ParseStreamChunk([]byte(`{"type":"response.refusal.delta","item_id":"msg_refusal","output_index":0,"content_index":0,"delta":"` + delta + `"}`))
		if parseErr != nil {
			t.Fatalf("refusal delta %d: %v", i+1, parseErr)
		}
		if got := streamChunkRefusal(t, chunk); got != delta {
			t.Fatalf("refusal delta %d = %q, want %q", i+1, got, delta)
		}
		if chunk.ID != "resp_refusal" || chunk.Model != "gpt-5.4" || chunk.Created != 123 {
			t.Fatalf("refusal delta metadata = %#v", chunk)
		}
	}

	done, err := translator.ParseStreamChunk([]byte(`{"type":"response.refusal.done","item_id":"msg_refusal","output_index":0,"content_index":0,"refusal":"NoNo!"}`))
	if err != nil {
		t.Fatalf("refusal.done: %v", err)
	}
	if got := streamChunkRefusal(t, done); got != "!" {
		t.Fatalf("refusal.done tail = %q, want !", got)
	}

	duplicateSnapshots := [][]byte{
		[]byte(`{"type":"response.content_part.done","item_id":"msg_refusal","output_index":0,"content_index":0,"part":{"type":"refusal","refusal":"NoNo!"}}`),
		[]byte(`{"type":"response.output_item.done","output_index":0,"item":{"id":"msg_refusal","type":"message","role":"assistant","status":"completed","content":[{"type":"refusal","refusal":"NoNo!"}]}}`),
	}
	for i, event := range duplicateSnapshots {
		chunk, parseErr := translator.ParseStreamChunk(event)
		if parseErr != nil {
			t.Fatalf("duplicate refusal snapshot %d: %v", i+1, parseErr)
		}
		if chunk != nil {
			t.Fatalf("duplicate refusal snapshot %d emitted content: %#v", i+1, chunk)
		}
	}

	terminal, terminalErr := translator.ParseStreamChunk([]byte(`{"type":"response.completed","response":{"id":"resp_refusal","object":"response","created_at":123,"status":"completed","model":"gpt-5.4","output":[{"type":"message","id":"msg_refusal","status":"completed","role":"assistant","content":[{"type":"refusal","refusal":"NoNo!"}]}],"output_text":"","usage":{"input_tokens":3,"output_tokens":4,"total_tokens":7}}}`))
	if !errors.Is(terminalErr, ErrStreamDone) {
		t.Fatalf("terminal error = %v, want ErrStreamDone", terminalErr)
	}
	if terminal == nil || len(terminal.Choices) != 1 || terminal.Choices[0].FinishReason == nil || *terminal.Choices[0].FinishReason != "stop" {
		t.Fatalf("terminal chunk = %#v, want one stop choice", terminal)
	}
	if got := terminal.Choices[0].Delta.Refusal; got != "" {
		t.Fatalf("terminal duplicated refusal = %q", got)
	}
}

func TestOpenAIStreamTranslator_DoneOnlyRefusalEmitsFullSnapshot(t *testing.T) {
	tests := []struct {
		name  string
		event string
	}{
		{
			name:  "refusal done",
			event: `{"type":"response.refusal.done","response_id":"resp_1","item_id":"msg_1","output_index":0,"content_index":0,"refusal":"done only"}`,
		},
		{
			name:  "content part done",
			event: `{"type":"response.content_part.done","response_id":"resp_1","item_id":"msg_1","output_index":0,"content_index":0,"part":{"type":"refusal","refusal":"done only"}}`,
		},
		{
			name:  "output item done",
			event: `{"type":"response.output_item.done","response_id":"resp_1","output_index":0,"item":{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"refusal","refusal":"done only"}]}}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			translator := newOpenAIStreamTranslatorForTest()
			chunk, err := translator.ParseStreamChunk([]byte(tt.event))
			if err != nil {
				t.Fatalf("parse done-only refusal: %v", err)
			}
			if got := streamChunkRefusal(t, chunk); got != "done only" {
				t.Fatalf("done-only refusal = %q, want done only", got)
			}
		})
	}
}

func TestOpenAIStreamTranslator_RefusalUsesSharedStateLimits(t *testing.T) {
	t.Run("state bytes", func(t *testing.T) {
		translator := newBoundedOpenAIStreamTranslatorForTest()
		translator.stateBytes = openAIStreamStateMaxBytes - 1
		chunk, err := translator.ParseStreamChunk([]byte(`{"type":"response.refusal.delta","output_index":0,"content_index":0,"delta":"xx"}`))
		if chunk != nil || !errors.Is(err, errOpenAIStreamStateTooLarge) {
			t.Fatalf("chunk=%#v err=%v, want state limit error", chunk, err)
		}
	})

	t.Run("content parts", func(t *testing.T) {
		translator := newBoundedOpenAIStreamTranslatorForTest()
		translator.partCount = openAIStreamMaxParts
		chunk, err := translator.ParseStreamChunk([]byte(`{"type":"response.refusal.delta","output_index":0,"content_index":0,"delta":"x"}`))
		if chunk != nil || !errors.Is(err, errOpenAIStreamTooManyParts) {
			t.Fatalf("chunk=%#v err=%v, want part limit error", chunk, err)
		}
	})
}

func streamChunkRefusal(t *testing.T, chunk *models.StreamChunk) string {
	t.Helper()
	if chunk == nil || len(chunk.Choices) != 1 || chunk.Choices[0].Delta == nil {
		t.Fatalf("expected one stream delta, got %#v", chunk)
	}
	return chunk.Choices[0].Delta.Refusal
}
