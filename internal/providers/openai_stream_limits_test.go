package providers

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
)

func TestOpenAIStreamTranslatorStateBudgetHandlesManySmallDeltas(t *testing.T) {
	translator := newBoundedOpenAIStreamTranslatorForTest()
	delta := strings.Repeat("x", 4<<10)
	event := []byte(`{"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":` + mustJSONQuote(t, delta) + `}`)
	for i := 0; i < openAIStreamStateMaxBytes/len(delta); i++ {
		chunk, err := translator.ParseStreamChunk(event)
		if err != nil {
			t.Fatalf("delta %d at boundary: %v", i, err)
		}
		if chunk == nil || len(chunk.Choices) != 1 || chunk.Choices[0].Delta == nil || chunk.Choices[0].Delta.Content != delta {
			t.Fatalf("delta %d was not emitted intact", i)
		}
	}
	if translator.stateBytes != openAIStreamStateMaxBytes {
		t.Fatalf("state bytes = %d, want %d", translator.stateBytes, openAIStreamStateMaxBytes)
	}
	key := openAIStreamPartKey{outputIndex: 0, contentIndex: 0, kind: "output_text"}
	if got := translator.textParts[key].content.Len(); got != openAIStreamStateMaxBytes {
		t.Fatalf("stored text bytes = %d, want %d", got, openAIStreamStateMaxBytes)
	}

	chunk, err := translator.ParseStreamChunk([]byte(`{"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"!"}`))
	if !errors.Is(err, errOpenAIStreamStateTooLarge) || chunk != nil {
		t.Fatalf("overflow result = chunk:%#v error:%v", chunk, err)
	}
	if got := translator.textParts[key].content.Len(); got != openAIStreamStateMaxBytes {
		t.Fatalf("overflow mutated stored text to %d bytes", got)
	}
}

func TestOpenAIStreamTranslatorPartLimit(t *testing.T) {
	translator := newBoundedOpenAIStreamTranslatorForTest()
	for index := 0; index < openAIStreamMaxParts; index++ {
		event := []byte(fmt.Sprintf(`{"type":"response.output_text.delta","output_index":%d,"content_index":0,"delta":"x"}`, index))
		if chunk, err := translator.ParseStreamChunk(event); err != nil || chunk == nil {
			t.Fatalf("part %d = chunk:%#v error:%v", index, chunk, err)
		}
	}
	chunk, err := translator.ParseStreamChunk([]byte(fmt.Sprintf(`{"type":"response.output_text.delta","output_index":%d,"content_index":0,"delta":"x"}`, openAIStreamMaxParts)))
	if !errors.Is(err, errOpenAIStreamTooManyParts) || chunk != nil {
		t.Fatalf("part overflow = chunk:%#v error:%v", chunk, err)
	}
	if translator.partCount != openAIStreamMaxParts {
		t.Fatalf("part count = %d, want %d", translator.partCount, openAIStreamMaxParts)
	}
}

func TestOpenAIStreamTranslatorToolAndAliasLimits(t *testing.T) {
	t.Run("tools", func(t *testing.T) {
		translator := newBoundedOpenAIStreamTranslatorForTest()
		for index := 0; index < openAIStreamMaxTools; index++ {
			event := []byte(fmt.Sprintf(`{"type":"response.output_item.added","output_index":%d,"item":{"id":"fc_%d","type":"function_call","call_id":"call_%d","name":"lookup","arguments":""}}`, index, index, index))
			if chunk, err := translator.ParseStreamChunk(event); err != nil || chunk == nil {
				t.Fatalf("tool %d = chunk:%#v error:%v", index, chunk, err)
			}
		}
		event := []byte(fmt.Sprintf(`{"type":"response.output_item.added","output_index":%d,"item":{"id":"fc_over","type":"function_call","call_id":"call_over","name":"lookup","arguments":""}}`, openAIStreamMaxTools))
		chunk, err := translator.ParseStreamChunk(event)
		if !errors.Is(err, errOpenAIStreamTooManyTools) || chunk != nil {
			t.Fatalf("tool overflow = chunk:%#v error:%v", chunk, err)
		}
		if translator.toolCount != openAIStreamMaxTools {
			t.Fatalf("tool count = %d, want %d", translator.toolCount, openAIStreamMaxTools)
		}
	})

	t.Run("aliases per tool", func(t *testing.T) {
		translator := newBoundedOpenAIStreamTranslatorForTest()
		added := []byte(`{"type":"response.output_item.added","output_index":0,"item":{"id":"fc_0","type":"function_call","call_id":"call_0","name":"lookup","arguments":""}}`)
		if chunk, err := translator.ParseStreamChunk(added); err != nil || chunk == nil {
			t.Fatalf("initial tool = chunk:%#v error:%v", chunk, err)
		}
		for aliasIndex := 1; aliasIndex <= openAIStreamMaxAliasesPerTool-2; aliasIndex++ {
			event := []byte(fmt.Sprintf(`{"type":"response.function_call_arguments.delta","output_index":0,"item_id":"fc_alias_%d","delta":""}`, aliasIndex))
			if _, err := translator.ParseStreamChunk(event); err != nil {
				t.Fatalf("alias %d: %v", aliasIndex, err)
			}
		}
		overflow := []byte(`{"type":"response.function_call_arguments.delta","output_index":0,"item_id":"fc_alias_over","delta":""}`)
		chunk, err := translator.ParseStreamChunk(overflow)
		if !errors.Is(err, errOpenAIStreamTooManyAliases) || chunk != nil {
			t.Fatalf("alias overflow = chunk:%#v error:%v", chunk, err)
		}
		if got := len(translator.toolAliases); got != openAIStreamMaxAliasesPerTool {
			t.Fatalf("alias count = %d, want %d", got, openAIStreamMaxAliasesPerTool)
		}
	})
}

func newBoundedOpenAIStreamTranslatorForTest() *openAIStreamTranslator {
	return NewOpenAIStreamTranslator(NewOpenAITranslator(config.ProviderConfig{
		APIKey:  "dummy",
		BaseURL: "https://api.openai.com/v1",
	})).(*openAIStreamTranslator)
}
