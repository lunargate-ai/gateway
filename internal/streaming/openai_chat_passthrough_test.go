package streaming

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestStreamResponsePreservesAdditiveOpenAIChatFields(t *testing.T) {
	contentChunk := `{"id":"chatcmpl_passthrough","choices":[{"index":0,"delta":{"role":"assistant","content":"<think>private plan</think>visible answer","refusal":"policy refusal","audio":{"id":"audio_1","transcript":"spoken"},"annotations":[{"type":"url_citation","url":"https://example.com"}],"x_delta":{"kept":true}},"finish_reason":null,"logprobs":{"content":[{"token":"visible","logprob":-0.1,"bytes":[118],"top_logprobs":[{"token":"v","logprob":-0.2}],"x_token":"kept"}],"refusal":[{"token":"policy","logprob":-0.3}],"x_logprobs":"kept"},"x_choice":9007199254740993}],"system_fingerprint":"fp_passthrough","service_tier":"priority","obfuscation":"pad","usage":null,"x_vendor":{"nested":{"kept":true}}}`
	usageChunk := `{"id":"chatcmpl_passthrough","choices":[],"system_fingerprint":"fp_passthrough","service_tier":"priority","usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5,"prompt_tokens_details":{"cached_tokens":1},"x_usage":"kept"},"x_usage_chunk":{"kept":true}}`

	for _, includeUsage := range []bool{false, true} {
		name := map[bool]string{false: "usage excluded", true: "usage included"}[includeUsage]
		t.Run(name, func(t *testing.T) {
			translator := providers.NewOpenAITranslator(config.ProviderConfig{
				APIKey:       "dummy",
				DefaultModel: "fallback-model",
			})
			providerResponse := &http.Response{
				StatusCode: http.StatusOK,
				Body: io.NopCloser(strings.NewReader(
					"data: " + contentChunk + "\n\n" +
						"data: " + usageChunk + "\n\n" +
						"data: [DONE]\n\n",
				)),
			}
			recorder := httptest.NewRecorder()
			observed := make([]*models.StreamChunk, 0, 2)

			err := NewHandler().StreamResponseWithObserverAndUsage(
				context.Background(),
				recorder,
				providerResponse,
				translator,
				func(chunk *models.StreamChunk) { observed = append(observed, chunk) },
				includeUsage,
			)
			if err != nil {
				t.Fatalf("stream response: %v", err)
			}
			if len(observed) != 2 {
				t.Fatalf("observer received %d chunks, want 2", len(observed))
			}
			if observed[0].ID != "chatcmpl_passthrough" || observed[0].Object != "chat.completion.chunk" || observed[0].Created == 0 || observed[0].Model != "fallback-model" {
				t.Fatalf("observer saw unnormalized envelope: %#v", observed[0])
			}
			if observed[1].ID != observed[0].ID || observed[1].Created != observed[0].Created || observed[1].Model != observed[0].Model {
				t.Fatalf("observer saw inconsistent envelopes: first=%#v second=%#v", observed[0], observed[1])
			}
			if observed[1].Usage == nil || observed[1].Usage.TotalTokens != 5 {
				t.Fatalf("observer lost usage: %#v", observed[1])
			}
			if !strings.Contains(string(observed[0].RawJSON), `"usage":null`) {
				t.Fatalf("client usage filtering mutated observer raw data: %s", observed[0].RawJSON)
			}
			if len(observed[0].Choices) != 1 || observed[0].Choices[0].Delta == nil {
				t.Fatalf("observer lost content choice: %#v", observed[0])
			}
			if got := observed[0].Choices[0].Delta.ContentString(); got != "visible answer" {
				t.Fatalf("observer content = %q, want normalized visible answer", got)
			}
			if got := observed[0].Choices[0].Delta.ReasoningContent; got != "private plan" {
				t.Fatalf("observer reasoning = %q, want private plan", got)
			}

			frames := strings.Split(strings.TrimSpace(recorder.Body.String()), "\n\n")
			if len(frames) != 3 || frames[2] != "data: [DONE]" {
				t.Fatalf("unexpected frames: %q", recorder.Body.String())
			}
			content := decodeStreamFrameObject(t, frames[0])
			usage := decodeStreamFrameObject(t, frames[1])

			assertOpenAIChatAdditiveFields(t, content)
			if got := jsonStringField(t, content, "object"); got != "chat.completion.chunk" {
				t.Fatalf("object = %q, want chat.completion.chunk", got)
			}
			if got := jsonStringField(t, content, "model"); got != "fallback-model" {
				t.Fatalf("model = %q, want fallback-model", got)
			}
			if _, ok := content["created"]; !ok {
				t.Fatal("normalized created field is missing")
			}
			if _, ok := usage["x_usage_chunk"]; !ok {
				t.Fatalf("usage chunk vendor extension missing: %s", frames[1])
			}

			contentUsage, contentHasUsage := content["usage"]
			usageValue, usageHasUsage := usage["usage"]
			if !includeUsage {
				if contentHasUsage || usageHasUsage {
					t.Fatalf("usage leaked without opt-in: content=%s usage=%s", contentUsage, usageValue)
				}
				return
			}
			if !contentHasUsage || string(contentUsage) != "null" {
				t.Fatalf("ordinary chunk usage = %s, want preserved null", contentUsage)
			}
			if !usageHasUsage {
				t.Fatal("opted-in usage chunk lost usage")
			}
			usageObject := decodeJSONObject(t, usageValue)
			if _, ok := usageObject["prompt_tokens_details"]; !ok {
				t.Fatalf("usage details missing: %s", usageValue)
			}
			if got := jsonStringField(t, usageObject, "x_usage"); got != "kept" {
				t.Fatalf("usage extension = %q, want kept", got)
			}
		})
	}
}

func assertOpenAIChatAdditiveFields(t *testing.T, content map[string]json.RawMessage) {
	t.Helper()
	for key, want := range map[string]string{
		"service_tier":       "priority",
		"system_fingerprint": "fp_passthrough",
		"obfuscation":        "pad",
	} {
		if got := jsonStringField(t, content, key); got != want {
			t.Fatalf("top-level field %q = %q, want %q", key, got, want)
		}
	}
	if _, ok := content["x_vendor"]; !ok {
		t.Fatalf("top-level vendor field missing: %#v", content)
	}

	choices := decodeJSONArray(t, content["choices"])
	if len(choices) != 1 {
		t.Fatalf("choices = %s, want one", content["choices"])
	}
	choice := decodeJSONObject(t, choices[0])
	if _, ok := choice["x_choice"]; !ok {
		t.Fatal("choice extension missing")
	}
	if got := string(choice["x_choice"]); got != "9007199254740993" {
		t.Fatalf("large vendor integer = %s, want exact preservation", choice["x_choice"])
	}

	delta := decodeJSONObject(t, choice["delta"])
	for _, key := range []string{"audio", "annotations", "x_delta"} {
		if _, ok := delta[key]; !ok {
			t.Fatalf("delta field %q missing: %s", key, choice["delta"])
		}
	}
	if got := jsonStringField(t, delta, "content"); got != "visible answer" {
		t.Fatalf("normalized content = %q, want visible answer", got)
	}
	if got := jsonStringField(t, delta, "reasoning_content"); got != "private plan" {
		t.Fatalf("normalized reasoning = %q, want private plan", got)
	}
	if got := jsonStringField(t, delta, "refusal"); got != "policy refusal" {
		t.Fatalf("refusal = %q, want policy refusal", got)
	}

	logprobs := decodeJSONObject(t, choice["logprobs"])
	for _, key := range []string{"refusal", "x_logprobs"} {
		if _, ok := logprobs[key]; !ok {
			t.Fatalf("logprobs field %q missing: %s", key, choice["logprobs"])
		}
	}
	contentLogprobs := decodeJSONArray(t, logprobs["content"])
	if len(contentLogprobs) != 1 {
		t.Fatalf("content logprobs = %s, want one token", logprobs["content"])
	}
	token := decodeJSONObject(t, contentLogprobs[0])
	for _, key := range []string{"top_logprobs", "x_token"} {
		if _, ok := token[key]; !ok {
			t.Fatalf("token logprobs field %q missing: %s", key, contentLogprobs[0])
		}
	}
}

func decodeStreamFrameObject(t *testing.T, frame string) map[string]json.RawMessage {
	t.Helper()
	return decodeJSONObject(t, json.RawMessage(strings.TrimPrefix(frame, "data: ")))
}

func decodeJSONObject(t *testing.T, raw json.RawMessage) map[string]json.RawMessage {
	t.Helper()
	var object map[string]json.RawMessage
	if err := json.Unmarshal(raw, &object); err != nil {
		t.Fatalf("decode JSON object %s: %v", raw, err)
	}
	return object
}

func decodeJSONArray(t *testing.T, raw json.RawMessage) []json.RawMessage {
	t.Helper()
	var array []json.RawMessage
	if err := json.Unmarshal(raw, &array); err != nil {
		t.Fatalf("decode JSON array %s: %v", raw, err)
	}
	return array
}

func jsonStringField(t *testing.T, object map[string]json.RawMessage, key string) string {
	t.Helper()
	var value string
	if err := json.Unmarshal(object[key], &value); err != nil {
		t.Fatalf("decode string field %q from %s: %v", key, object[key], err)
	}
	return value
}
