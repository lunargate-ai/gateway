package providers

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"strconv"
	"strings"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
)

func TestOpenAIParseResponseCompletesAdditiveChatEnvelope(t *testing.T) {
	body := []byte(`{
  "created": 1788382926,
  "model": "claude-haiku-4-5-20251001",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop", "native_finish_reason": "end_turn"}],
  "usage": {"input_tokens": 12, "output_tokens": 4, "raw_input_tokens": 12},
  "provider_extension": {"kept": true}
}`)
	translator := NewOpenAITranslator(config.ProviderConfig{APIKey: "dummy", DefaultModel: "fallback-model"})

	response, err := translator.ParseResponse(&http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(bytes.NewReader(body)),
	})
	if err != nil {
		t.Fatalf("ParseResponse returned error: %v", err)
	}
	if !strings.HasPrefix(response.ID, "chatcmpl-") {
		t.Fatalf("id = %q, want generated chatcmpl id", response.ID)
	}
	if response.Object != "chat.completion" {
		t.Fatalf("object = %q, want chat.completion", response.Object)
	}
	if response.Model != "claude-haiku-4-5-20251001" {
		t.Fatalf("model = %q", response.Model)
	}
	if response.Usage == nil || response.Usage.PromptTokens != 12 || response.Usage.CompletionTokens != 4 || response.Usage.TotalTokens != 16 {
		t.Fatalf("usage = %#v", response.Usage)
	}

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(response.RawJSON, &raw); err != nil {
		t.Fatalf("raw response is not JSON: %v", err)
	}
	if _, ok := raw["provider_extension"]; !ok {
		t.Fatal("provider extension was dropped")
	}
	var usage map[string]json.RawMessage
	if err := json.Unmarshal(raw["usage"], &usage); err != nil {
		t.Fatalf("raw usage is not JSON: %v", err)
	}
	for _, key := range []string{"input_tokens", "output_tokens", "raw_input_tokens", "prompt_tokens", "completion_tokens", "total_tokens"} {
		if _, ok := usage[key]; !ok {
			t.Fatalf("raw usage missing %q: %s", key, raw["usage"])
		}
	}
}

func TestOpenAIParseResponseKeepsCompliantEnvelopeByteForByte(t *testing.T) {
	body := []byte(`{"id":"chatcmpl-original","object":"chat.completion","created":1,"model":"gpt-test","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3},"future":"kept"}`)
	translator := NewOpenAITranslator(config.ProviderConfig{APIKey: "dummy"})

	response, err := translator.ParseResponse(&http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(bytes.NewReader(body)),
	})
	if err != nil {
		t.Fatalf("ParseResponse returned error: %v", err)
	}
	if !bytes.Equal(response.RawJSON, body) {
		t.Fatalf("raw envelope changed:\n got: %s\nwant: %s", response.RawJSON, body)
	}
}

func TestOpenAIParseResponseSaturatesAliasedUsageTotal(t *testing.T) {
	maximum := int(^uint(0) >> 1)
	body := []byte(`{"created":1,"model":"gpt-4o","choices":[],"usage":{"input_tokens":` +
		strconv.Itoa(maximum) + `,"output_tokens":` + strconv.Itoa(maximum) + `}}`)
	translator := NewOpenAITranslator(config.ProviderConfig{APIKey: "dummy"})

	response, err := translator.ParseResponse(&http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(bytes.NewReader(body)),
	})
	if err != nil {
		t.Fatalf("ParseResponse returned error: %v", err)
	}
	if response.Usage == nil || response.Usage.PromptTokens != maximum || response.Usage.CompletionTokens != maximum || response.Usage.TotalTokens != maximum {
		t.Fatalf("usage = %#v, want component and total saturation at %d", response.Usage, maximum)
	}

	var raw struct {
		Usage struct {
			TotalTokens int `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(response.RawJSON, &raw); err != nil {
		t.Fatalf("decode raw response: %v", err)
	}
	if raw.Usage.TotalTokens != maximum {
		t.Fatalf("raw total_tokens = %d, want %d", raw.Usage.TotalTokens, maximum)
	}
}
