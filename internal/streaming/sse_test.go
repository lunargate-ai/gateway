package streaming

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestStreamResponseEmitsTerminalUsageBeforeDone(t *testing.T) {
	translator := providers.NewOpenAITranslator(config.ProviderConfig{
		APIKey:  "dummy",
		BaseURL: "https://api.openai.com/v1",
	})
	providerResp := &http.Response{
		StatusCode: http.StatusOK,
		Body: io.NopCloser(strings.NewReader(
			"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_usage\",\"created_at\":123,\"model\":\"gpt-5.3-codex\",\"usage\":{\"input_tokens\":17,\"output_tokens\":9,\"total_tokens\":26}}}\n\n",
		)),
	}
	recorder := httptest.NewRecorder()
	var observed *models.StreamChunk

	err := NewHandler().StreamResponseWithObserver(
		context.Background(),
		recorder,
		providerResp,
		translator,
		func(chunk *models.StreamChunk) { observed = chunk },
	)
	if err != nil {
		t.Fatalf("StreamResponseWithObserver returned error: %v", err)
	}
	if observed == nil || observed.Usage == nil {
		t.Fatalf("expected observer to receive terminal usage, got %#v", observed)
	}

	frames := strings.Split(strings.TrimSpace(recorder.Body.String()), "\n\n")
	if len(frames) != 2 {
		t.Fatalf("expected usage frame and done frame, got %d: %q", len(frames), recorder.Body.String())
	}
	if frames[1] != "data: [DONE]" {
		t.Fatalf("last frame = %q, want data: [DONE]", frames[1])
	}
	var chunk models.StreamChunk
	if err := json.Unmarshal([]byte(strings.TrimPrefix(frames[0], "data: ")), &chunk); err != nil {
		t.Fatalf("decode terminal chunk: %v", err)
	}
	if chunk.Usage == nil || chunk.Usage.PromptTokens != 17 || chunk.Usage.CompletionTokens != 9 || chunk.Usage.TotalTokens != 26 {
		t.Fatalf("unexpected terminal usage: %#v", chunk.Usage)
	}
}

func TestStreamResponseRejectsEOFWithoutTerminalEvent(t *testing.T) {
	translator := providers.NewOpenAITranslator(config.ProviderConfig{APIKey: "dummy", BaseURL: "https://api.openai.com/v1"})
	providerResp := &http.Response{
		StatusCode: http.StatusOK,
		Body: io.NopCloser(strings.NewReader(
			"data: {\"id\":\"chatcmpl_partial\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"}}]}\n\n",
		)),
	}

	err := NewHandler().StreamResponse(context.Background(), httptest.NewRecorder(), providerResp, translator)
	if !errors.Is(err, ErrUpstreamStreamIncomplete) {
		t.Fatalf("error = %v, want ErrUpstreamStreamIncomplete", err)
	}
}

func TestStreamAnthropicResponseRejectsEOFWithoutMessageStop(t *testing.T) {
	base := providers.NewAnthropicTranslator(config.ProviderConfig{APIKey: "dummy", BaseURL: "https://api.anthropic.com/v1"})
	translator := providers.NewAnthropicStreamTranslator(base)
	providerResp := &http.Response{
		StatusCode: http.StatusOK,
		Body: io.NopCloser(strings.NewReader(
			"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_partial\",\"model\":\"claude\",\"usage\":{\"input_tokens\":3}}}\n\n",
		)),
	}

	err := NewHandler().StreamAnthropicResponse(context.Background(), httptest.NewRecorder(), providerResp, translator)
	if !errors.Is(err, ErrUpstreamStreamIncomplete) {
		t.Fatalf("error = %v, want ErrUpstreamStreamIncomplete", err)
	}
}

func TestStreamNDJSONResponseRejectsEOFWithoutDone(t *testing.T) {
	base := providers.NewOllamaTranslator(config.ProviderConfig{BaseURL: "http://localhost:11434"})
	translator := providers.NewOllamaStreamTranslator(base)
	providerResp := &http.Response{
		StatusCode: http.StatusOK,
		Body: io.NopCloser(strings.NewReader(
			"{\"model\":\"qwen\",\"message\":{\"role\":\"assistant\",\"content\":\"partial\"},\"done\":false}\n",
		)),
	}

	err := NewHandler().StreamNDJSONResponse(context.Background(), httptest.NewRecorder(), providerResp, translator)
	if !errors.Is(err, ErrUpstreamStreamIncomplete) {
		t.Fatalf("error = %v, want ErrUpstreamStreamIncomplete", err)
	}
}
