package providers

import (
	"encoding/json"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestOpenAIStreamTranslator_PreservesRepeatedTextDeltas(t *testing.T) {
	translator := newOpenAIStreamTranslatorForTest()
	event := []byte(`{"type":"response.output_text.delta","response_id":"resp_1","item_id":"msg_1","output_index":0,"content_index":0,"delta":"ha"}`)

	for i := 0; i < 2; i++ {
		chunk, err := translator.ParseStreamChunk(event)
		if err != nil {
			t.Fatalf("delta %d: %v", i+1, err)
		}
		if got := streamChunkText(t, chunk); got != "ha" {
			t.Fatalf("delta %d = %q, want %q", i+1, got, "ha")
		}
	}

	chunk, err := translator.ParseStreamChunk([]byte(`{"type":"response.output_text.done","response_id":"resp_1","item_id":"msg_1","output_index":0,"content_index":0,"text":"haha"}`))
	if err != nil {
		t.Fatalf("done snapshot: %v", err)
	}
	if chunk != nil {
		t.Fatalf("matching done snapshot must not repeat content: %#v", chunk)
	}
}

func TestOpenAIStreamTranslator_TextSnapshotsEmitOnlyMissingTail(t *testing.T) {
	translator := newOpenAIStreamTranslatorForTest()

	chunk, err := translator.ParseStreamChunk([]byte(`{"type":"response.output_text.delta","response_id":"resp_1","item_id":"msg_1","output_index":0,"content_index":0,"delta":"Hello"}`))
	if err != nil || streamChunkText(t, chunk) != "Hello" {
		t.Fatalf("initial delta: chunk=%#v err=%v", chunk, err)
	}

	chunk, err = translator.ParseStreamChunk([]byte(`{"type":"response.output_text.done","response_id":"resp_1","item_id":"msg_1","output_index":0,"content_index":0,"text":"Hello world"}`))
	if err != nil {
		t.Fatalf("output_text.done: %v", err)
	}
	if got := streamChunkText(t, chunk); got != " world" {
		t.Fatalf("missing tail = %q, want %q", got, " world")
	}

	duplicateSnapshots := [][]byte{
		[]byte(`{"type":"response.content_part.done","response_id":"resp_1","item_id":"msg_1","output_index":0,"content_index":0,"part":{"type":"output_text","text":"Hello world"}}`),
		[]byte(`{"type":"response.output_item.done","response_id":"resp_1","output_index":0,"item":{"id":"msg_1","type":"message","content":[{"type":"output_text","text":"Hello world"}]}}`),
	}
	for i, event := range duplicateSnapshots {
		chunk, err = translator.ParseStreamChunk(event)
		if err != nil {
			t.Fatalf("duplicate snapshot %d: %v", i+1, err)
		}
		if chunk != nil {
			t.Fatalf("duplicate snapshot %d emitted content: %#v", i+1, chunk)
		}
	}
}

func TestOpenAIStreamTranslator_DoneOnlyTextEmitsFullSnapshot(t *testing.T) {
	tests := []struct {
		name  string
		event string
	}{
		{
			name:  "output text done",
			event: `{"type":"response.output_text.done","response_id":"resp_1","item_id":"msg_1","output_index":0,"content_index":0,"text":"done only"}`,
		},
		{
			name:  "content part done",
			event: `{"type":"response.content_part.done","response_id":"resp_1","item_id":"msg_1","output_index":0,"content_index":0,"part":{"type":"output_text","text":"done only"}}`,
		},
		{
			name:  "output item done",
			event: `{"type":"response.output_item.done","response_id":"resp_1","output_index":0,"item":{"id":"msg_1","type":"message","content":[{"type":"output_text","text":"done only"}]}}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			translator := newOpenAIStreamTranslatorForTest()
			chunk, err := translator.ParseStreamChunk([]byte(tt.event))
			if err != nil {
				t.Fatalf("parse done-only event: %v", err)
			}
			if got := streamChunkText(t, chunk); got != "done only" {
				t.Fatalf("done-only content = %q, want %q", got, "done only")
			}
		})
	}
}

func TestOpenAIStreamTranslator_ReasoningSnapshotsDoNotDuplicateDeltas(t *testing.T) {
	translator := newOpenAIStreamTranslatorForTest()
	delta := []byte(`{"type":"response.reasoning_summary_text.delta","response_id":"resp_1","item_id":"rs_1","output_index":1,"summary_index":0,"delta":"ha"}`)

	for i := 0; i < 2; i++ {
		chunk, err := translator.ParseStreamChunk(delta)
		if err != nil {
			t.Fatalf("reasoning delta %d: %v", i+1, err)
		}
		if got := streamChunkReasoning(t, chunk); got != "ha" {
			t.Fatalf("reasoning delta %d = %q, want %q", i+1, got, "ha")
		}
	}

	chunk, err := translator.ParseStreamChunk([]byte(`{"type":"response.reasoning_summary_text.done","response_id":"resp_1","item_id":"rs_1","output_index":1,"summary_index":0,"text":"haha!"}`))
	if err != nil {
		t.Fatalf("reasoning done: %v", err)
	}
	if got := streamChunkReasoning(t, chunk); got != "!" {
		t.Fatalf("reasoning missing tail = %q, want %q", got, "!")
	}

	duplicateSnapshots := [][]byte{
		[]byte(`{"type":"response.reasoning_summary_part.done","response_id":"resp_1","item_id":"rs_1","output_index":1,"summary_index":0,"part":{"type":"summary_text","text":"haha!"}}`),
		[]byte(`{"type":"response.output_item.done","response_id":"resp_1","output_index":1,"item":{"id":"rs_1","type":"reasoning","summary":[{"type":"summary_text","text":"haha!"}]}}`),
	}
	for i, event := range duplicateSnapshots {
		chunk, err = translator.ParseStreamChunk(event)
		if err != nil {
			t.Fatalf("reasoning duplicate snapshot %d: %v", i+1, err)
		}
		if chunk != nil {
			t.Fatalf("reasoning duplicate snapshot %d emitted content: %#v", i+1, chunk)
		}
	}
}

func TestOpenAIStreamTranslator_FunctionArgumentsStayStableAndLossless(t *testing.T) {
	translator := newOpenAIStreamTranslatorForTest()

	added, err := translator.ParseStreamChunk([]byte(`{"type":"response.output_item.added","response_id":"resp_1","output_index":2,"item":{"id":"fc_abc","type":"function_call","call_id":"call_abc","name":"lookup","arguments":""}}`))
	if err != nil {
		t.Fatalf("function item added: %v", err)
	}
	assertStreamToolCall(t, added, 0, "call_abc", "lookup", "")

	deltas := []string{`{"x":"`, "ha", "ha", `"`}
	for i, delta := range deltas {
		event := []byte(`{"type":"response.function_call_arguments.delta","response_id":"resp_1","item_id":"fc_abc","output_index":2,"delta":` + mustJSONQuote(t, delta) + `}`)
		chunk, parseErr := translator.ParseStreamChunk(event)
		if parseErr != nil {
			t.Fatalf("arguments delta %d: %v", i+1, parseErr)
		}
		assertStreamToolCall(t, chunk, 0, "call_abc", "lookup", delta)
	}

	doneArgs := `{"x":"haha"}`
	done, err := translator.ParseStreamChunk([]byte(`{"type":"response.function_call_arguments.done","response_id":"resp_1","item_id":"fc_abc","output_index":2,"name":"lookup","arguments":` + mustJSONQuote(t, doneArgs) + `}`))
	if err != nil {
		t.Fatalf("arguments done: %v", err)
	}
	assertStreamToolCall(t, done, 0, "call_abc", "lookup", "}")

	itemDone, err := translator.ParseStreamChunk([]byte(`{"type":"response.output_item.done","response_id":"resp_1","output_index":2,"item":{"id":"fc_abc","type":"function_call","call_id":"call_abc","name":"lookup","arguments":` + mustJSONQuote(t, doneArgs) + `}}`))
	if err != nil {
		t.Fatalf("function item done: %v", err)
	}
	if itemDone != nil {
		t.Fatalf("output item snapshot must not repeat arguments: %#v", itemDone)
	}

	doneOnly, err := translator.ParseStreamChunk([]byte(`{"type":"response.output_item.done","response_id":"resp_1","output_index":4,"item":{"id":"fc_def","type":"function_call","call_id":"call_def","name":"finish","arguments":"{}"}}`))
	if err != nil {
		t.Fatalf("done-only function item: %v", err)
	}
	assertStreamToolCall(t, doneOnly, 1, "call_def", "finish", "{}")
}

func newOpenAIStreamTranslatorForTest() models.ProviderTranslator {
	return NewOpenAIStreamTranslator(NewOpenAITranslator(config.ProviderConfig{
		APIKey:  "dummy",
		BaseURL: "https://api.openai.com/v1",
	}))
}

func streamChunkText(t *testing.T, chunk *models.StreamChunk) string {
	t.Helper()
	if chunk == nil || len(chunk.Choices) != 1 || chunk.Choices[0].Delta == nil {
		t.Fatalf("expected one stream delta, got %#v", chunk)
	}
	text, _ := chunk.Choices[0].Delta.Content.(string)
	return text
}

func streamChunkReasoning(t *testing.T, chunk *models.StreamChunk) string {
	t.Helper()
	if chunk == nil || len(chunk.Choices) != 1 || chunk.Choices[0].Delta == nil {
		t.Fatalf("expected one stream delta, got %#v", chunk)
	}
	return chunk.Choices[0].Delta.ReasoningContent
}

func assertStreamToolCall(t *testing.T, chunk *models.StreamChunk, index int, id, name, arguments string) {
	t.Helper()
	if chunk == nil || len(chunk.Choices) != 1 || chunk.Choices[0].Delta == nil || len(chunk.Choices[0].Delta.ToolCalls) != 1 {
		t.Fatalf("expected one tool-call delta, got %#v", chunk)
	}
	call := chunk.Choices[0].Delta.ToolCalls[0]
	if call.Index == nil || *call.Index != index {
		t.Fatalf("tool index = %#v, want %d", call.Index, index)
	}
	if call.ID != id || call.Function.Name != name || call.Function.Arguments != arguments {
		t.Fatalf("tool call = %#v, want id=%q name=%q arguments=%q", call, id, name, arguments)
	}
}

func mustJSONQuote(t *testing.T, value string) string {
	t.Helper()
	b, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("marshal JSON string: %v", err)
	}
	return string(b)
}
