package providers

import (
	"bytes"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
)

func TestOpenAITranslator_ParseStreamChunkPreservesRawChatJSON(t *testing.T) {
	translator := NewOpenAITranslator(config.ProviderConfig{
		APIKey:  "dummy",
		BaseURL: "https://api.openai.com/v1",
	})
	data := []byte(`{"id":"chatcmpl_raw","object":"chat.completion.chunk","created":1,"model":"gpt-5.4","service_tier":"priority","choices":[{"index":0,"delta":{"content":"hello","audio":{"id":"audio_1"},"x_delta":{"kept":true}},"finish_reason":null,"logprobs":{"refusal":[],"x_logprobs":"kept"},"x_choice":9007199254740993}],"x_vendor":{"kept":true}}`)
	want := append([]byte(nil), data...)

	chunk, err := translator.ParseStreamChunk(data)
	if err != nil {
		t.Fatalf("ParseStreamChunk returned error: %v", err)
	}
	if chunk == nil {
		t.Fatal("expected parsed chunk")
	}
	if !bytes.Equal(chunk.RawJSON, want) {
		t.Fatalf("raw chunk changed:\n got: %s\nwant: %s", chunk.RawJSON, want)
	}

	data[0] = '['
	if !bytes.Equal(chunk.RawJSON, want) {
		t.Fatal("raw chunk aliases the caller's input buffer")
	}
}
