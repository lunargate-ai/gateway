package streaming

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/pkg/models"
)

type wrappedTerminalTranslator struct {
	models.ProviderTranslator
	chunk *models.StreamChunk
}

func (t wrappedTerminalTranslator) ParseStreamChunk([]byte) (*models.StreamChunk, error) {
	return t.chunk, fmt.Errorf("translator terminal: %w", providers.ErrStreamDone)
}

func TestStreamHandlersRecognizeWrappedTerminalError(t *testing.T) {
	base := providers.NewOpenAITranslator(config.ProviderConfig{
		APIKey:  "dummy",
		BaseURL: "https://api.openai.com/v1",
	})
	tests := []struct {
		name string
		body string
		run  func(*Handler, http.ResponseWriter, *http.Response, models.ProviderTranslator) error
	}{
		{
			name: "sse",
			body: "data: {}\n\n",
			run: func(handler *Handler, writer http.ResponseWriter, response *http.Response, translator models.ProviderTranslator) error {
				return handler.StreamResponse(context.Background(), writer, response, translator)
			},
		},
		{
			name: "anthropic sse",
			body: "event: message_stop\ndata: {}\n\n",
			run: func(handler *Handler, writer http.ResponseWriter, response *http.Response, translator models.ProviderTranslator) error {
				return handler.StreamAnthropicResponse(context.Background(), writer, response, translator)
			},
		},
		{
			name: "ndjson",
			body: "{}\n",
			run: func(handler *Handler, writer http.ResponseWriter, response *http.Response, translator models.ProviderTranslator) error {
				return handler.StreamNDJSONResponse(context.Background(), writer, response, translator)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			providerResp := &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(strings.NewReader(tt.body)),
			}
			recorder := httptest.NewRecorder()
			translator := wrappedTerminalTranslator{ProviderTranslator: base}

			if err := tt.run(NewHandler(), recorder, providerResp, translator); err != nil {
				t.Fatalf("stream returned wrapped terminal error: %v", err)
			}
			if got := recorder.Body.String(); got != "data: [DONE]\n\n" {
				t.Fatalf("stream body = %q, want terminal frame", got)
			}
			for _, header := range []string{"Connection", "Transfer-Encoding"} {
				if got := recorder.Header().Get(header); got != "" {
					t.Fatalf("hop-by-hop header %s = %q, want empty", header, got)
				}
			}
		})
	}
}

func TestStreamResponseParsesCompleteSSEEvents(t *testing.T) {
	translator := providers.NewOpenAITranslator(config.ProviderConfig{
		APIKey:  "dummy",
		BaseURL: "https://api.openai.com/v1",
	})
	providerResp := &http.Response{
		StatusCode: http.StatusOK,
		Body: io.NopCloser(strings.NewReader(
			": keepalive\r\n\r\n" +
				"event:chat.completion.chunk\r\n" +
				"data:{\"id\":\"chatcmpl_sse\",\"object\":\"chat.completion.chunk\",\r\n" +
				": ignored within event\r\n" +
				"data: \"created\":123,\"model\":\"gpt-test\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hello\"},\"finish_reason\":null}]}\r\n\r\n" +
				"data:{\"id\":\"chatcmpl_sse\",\"object\":\"chat.completion.chunk\",\"created\":123,\"model\":\"gpt-test\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\r\n\r\n" +
				"data:[DONE]\r\n\r\n",
		)),
	}
	recorder := httptest.NewRecorder()

	if err := NewHandler().StreamResponse(context.Background(), recorder, providerResp, translator); err != nil {
		t.Fatalf("stream failed: %v", err)
	}

	frames := strings.Split(strings.TrimSpace(recorder.Body.String()), "\n\n")
	if len(frames) != 3 {
		t.Fatalf("stream frames = %d, want 3: %q", len(frames), recorder.Body.String())
	}
	var contentChunk models.StreamChunk
	if err := json.Unmarshal([]byte(strings.TrimPrefix(frames[0], "data: ")), &contentChunk); err != nil {
		t.Fatalf("decode content chunk: %v", err)
	}
	if len(contentChunk.Choices) != 1 || contentChunk.Choices[0].Delta == nil || contentChunk.Choices[0].Delta.ContentString() != "hello" {
		t.Fatalf("unexpected content chunk: %#v", contentChunk)
	}
	var terminalChunk models.StreamChunk
	if err := json.Unmarshal([]byte(strings.TrimPrefix(frames[1], "data: ")), &terminalChunk); err != nil {
		t.Fatalf("decode terminal chunk: %v", err)
	}
	if len(terminalChunk.Choices) != 1 || terminalChunk.Choices[0].FinishReason == nil || *terminalChunk.Choices[0].FinishReason != "stop" {
		t.Fatalf("unexpected terminal chunk: %#v", terminalChunk)
	}
	if frames[2] != "data: [DONE]" {
		t.Fatalf("terminal frame = %q, want data: [DONE]", frames[2])
	}
}

func TestStreamAnthropicResponseParsesCompleteSSEEvents(t *testing.T) {
	base := providers.NewAnthropicTranslator(config.ProviderConfig{
		APIKey:  "dummy",
		BaseURL: "https://api.anthropic.com/v1",
	})
	providerResp := &http.Response{
		StatusCode: http.StatusOK,
		Body: io.NopCloser(strings.NewReader(
			": keepalive\r\n\r\n" +
				"event:message_start\r\n" +
				"data:{\"message\":{\"id\":\"msg_sse\",\"model\":\"claude-test\",\"usage\":{\"input_tokens\":1}},\r\n" +
				": ignored within event\r\n" +
				"data: \"type\":\"message_start\"}\r\n\r\n" +
				"event:content_block_delta\r\n" +
				"data:{\"index\":0,\r\n" +
				"data: \"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}\r\n\r\n" +
				"event:message_stop\r\n" +
				"data:{}\r\n\r\n",
		)),
	}
	recorder := httptest.NewRecorder()

	if err := NewHandler().StreamAnthropicResponse(
		context.Background(),
		recorder,
		providerResp,
		providers.NewAnthropicStreamTranslator(base),
	); err != nil {
		t.Fatalf("stream failed: %v", err)
	}

	frames := strings.Split(strings.TrimSpace(recorder.Body.String()), "\n\n")
	if len(frames) != 3 {
		t.Fatalf("stream frames = %d, want 3: %q", len(frames), recorder.Body.String())
	}
	var contentChunk models.StreamChunk
	if err := json.Unmarshal([]byte(strings.TrimPrefix(frames[1], "data: ")), &contentChunk); err != nil {
		t.Fatalf("decode content chunk: %v", err)
	}
	if len(contentChunk.Choices) != 1 || contentChunk.Choices[0].Delta == nil || contentChunk.Choices[0].Delta.ContentString() != "hello" {
		t.Fatalf("unexpected content chunk: %#v", contentChunk)
	}
	if frames[2] != "data: [DONE]" {
		t.Fatalf("terminal frame = %q, want data: [DONE]", frames[2])
	}
}

type wrappedEOFReadCloser struct{}

func (wrappedEOFReadCloser) Read([]byte) (int, error) {
	return 0, fmt.Errorf("wrapped reader end: %w", io.EOF)
}

func (wrappedEOFReadCloser) Close() error { return nil }

func TestStreamAnthropicResponseRecognizesWrappedEOF(t *testing.T) {
	base := providers.NewAnthropicTranslator(config.ProviderConfig{APIKey: "dummy"})
	providerResp := &http.Response{
		StatusCode: http.StatusOK,
		Body:       wrappedEOFReadCloser{},
	}

	err := NewHandler().StreamAnthropicResponse(
		context.Background(),
		httptest.NewRecorder(),
		providerResp,
		providers.NewAnthropicStreamTranslator(base),
	)
	if !errors.Is(err, ErrUpstreamStreamIncomplete) {
		t.Fatalf("error = %v, want ErrUpstreamStreamIncomplete", err)
	}
}

type providerErrorTranslator struct {
	models.ProviderTranslator
	err *providers.ProviderError
}

func (t providerErrorTranslator) ParseStreamChunk([]byte) (*models.StreamChunk, error) {
	return nil, t.err
}

func TestStreamParseErrorPreservesProviderErrorForErrorsAs(t *testing.T) {
	want := &providers.ProviderError{
		StatusCode: http.StatusTooManyRequests,
		Provider:   "test-provider",
		Type:       "rate_limit_error",
		Message:    "slow down",
	}
	base := providers.NewOpenAITranslator(config.ProviderConfig{APIKey: "dummy"})
	providerResp := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader("data: {}\n\n")),
	}

	err := NewHandler().StreamResponse(
		context.Background(),
		httptest.NewRecorder(),
		providerResp,
		providerErrorTranslator{ProviderTranslator: base, err: want},
	)
	var got *providers.ProviderError
	if !errors.As(err, &got) {
		t.Fatalf("error = %v, want wrapped ProviderError", err)
	}
	if got != want {
		t.Fatalf("ProviderError = %#v, want original %#v", got, want)
	}
}

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

func TestGenericStreamHandlersRespectIncludeUsageWithoutHidingMetrics(t *testing.T) {
	base := providers.NewOpenAITranslator(config.ProviderConfig{APIKey: "dummy"})
	terminal := &models.StreamChunk{
		Choices: []models.Choice{{Index: 0}},
		Usage:   &models.Usage{PromptTokens: 3, CompletionTokens: 2, TotalTokens: 5},
	}
	tests := []struct {
		name string
		body string
		run  func(*Handler, http.ResponseWriter, *http.Response, models.ProviderTranslator, ChunkObserver, bool) error
	}{
		{
			name: "openai sse",
			body: "data: {}\n\n",
			run: func(handler *Handler, writer http.ResponseWriter, response *http.Response, translator models.ProviderTranslator, observer ChunkObserver, includeUsage bool) error {
				return handler.StreamResponseWithObserverAndUsage(
					context.Background(), writer, response, translator, observer, includeUsage,
				)
			},
		},
		{
			name: "ollama ndjson",
			body: "{}\n",
			run: func(handler *Handler, writer http.ResponseWriter, response *http.Response, translator models.ProviderTranslator, observer ChunkObserver, includeUsage bool) error {
				return handler.StreamNDJSONResponseWithObserverAndUsage(
					context.Background(), writer, response, translator, observer, includeUsage,
				)
			},
		},
	}

	for _, testCase := range tests {
		for _, includeUsage := range []bool{false, true} {
			name := map[bool]string{false: "excluded", true: "included"}[includeUsage]
			t.Run(testCase.name+"/"+name, func(t *testing.T) {
				providerResp := &http.Response{
					StatusCode: http.StatusOK,
					Body:       io.NopCloser(strings.NewReader(testCase.body)),
				}
				recorder := httptest.NewRecorder()
				observedUsage := 0
				err := testCase.run(
					NewHandler(),
					recorder,
					providerResp,
					wrappedTerminalTranslator{ProviderTranslator: base, chunk: terminal},
					func(chunk *models.StreamChunk) {
						if chunk != nil && chunk.Usage != nil {
							observedUsage++
						}
					},
					includeUsage,
				)
				if err != nil {
					t.Fatalf("stream returned error: %v", err)
				}
				if observedUsage != 1 {
					t.Fatalf("observer saw %d usage chunks, want 1", observedUsage)
				}
				gotUsage := strings.Count(recorder.Body.String(), `"usage":`)
				if includeUsage && gotUsage != 1 {
					t.Fatalf("client saw %d usage chunks, want 1: %s", gotUsage, recorder.Body.String())
				}
				if !includeUsage && gotUsage != 0 {
					t.Fatalf("client saw usage without opting in: %s", recorder.Body.String())
				}
			})
		}
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

func TestStreamAnthropicResponseRespectsIncludeUsageWithoutHidingMetrics(t *testing.T) {
	for _, includeUsage := range []bool{false, true} {
		t.Run(map[bool]string{false: "excluded", true: "included"}[includeUsage], func(t *testing.T) {
			base := providers.NewAnthropicTranslator(config.ProviderConfig{APIKey: "dummy"})
			translator := providers.NewAnthropicStreamTranslator(base)
			providerResp := &http.Response{
				StatusCode: http.StatusOK,
				Body: io.NopCloser(strings.NewReader(
					"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_usage\",\"model\":\"claude\",\"usage\":{\"input_tokens\":3}}}\n\n" +
						"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n" +
						"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
				)),
			}
			recorder := httptest.NewRecorder()
			observedUsage := 0

			err := NewHandler().StreamAnthropicResponseWithObserverAndUsage(
				context.Background(),
				recorder,
				providerResp,
				translator,
				func(chunk *models.StreamChunk) {
					if chunk.Usage != nil {
						observedUsage++
					}
				},
				includeUsage,
			)
			if err != nil {
				t.Fatalf("stream failed: %v", err)
			}
			if observedUsage != 2 {
				t.Fatalf("observer saw %d usage chunks, want 2", observedUsage)
			}

			frames := strings.Split(strings.TrimSpace(recorder.Body.String()), "\n\n")
			wantFrames := 3
			if includeUsage {
				wantFrames = 4
			}
			if len(frames) != wantFrames {
				t.Fatalf("stream frames = %d, want %d: %s", len(frames), wantFrames, recorder.Body.String())
			}
			for index, frame := range frames[:2] {
				if strings.Contains(frame, `"usage":`) {
					t.Fatalf("ordinary chunk %d contains usage: %s", index, frame)
				}
			}
			if includeUsage {
				assertCanonicalUsageTrailer(t, frames[2], 3, 2, 5)
			} else if got := strings.Count(recorder.Body.String(), `"usage":`); got != 0 {
				t.Fatalf("client saw %d usage chunks without opting in: %s", got, recorder.Body.String())
			}
			if frames[len(frames)-1] != "data: [DONE]" {
				t.Fatalf("terminal frame = %q, want data: [DONE]", frames[len(frames)-1])
			}
		})
	}
}

func TestStreamNDJSONResponseEmitsCanonicalUsageTrailer(t *testing.T) {
	for _, includeUsage := range []bool{false, true} {
		t.Run(map[bool]string{false: "excluded", true: "included"}[includeUsage], func(t *testing.T) {
			base := providers.NewOllamaTranslator(config.ProviderConfig{BaseURL: "http://localhost:11434"})
			providerResp := &http.Response{
				StatusCode: http.StatusOK,
				Body: io.NopCloser(strings.NewReader(
					"{\"model\":\"qwen\",\"message\":{\"role\":\"assistant\",\"content\":\"hello\"},\"done\":false}\n" +
						"{\"model\":\"qwen\",\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true,\"done_reason\":\"stop\",\"prompt_eval_count\":3,\"eval_count\":2}\n",
				)),
			}
			recorder := httptest.NewRecorder()
			observedUsage := 0

			err := NewHandler().StreamNDJSONResponseWithObserverAndUsage(
				context.Background(),
				recorder,
				providerResp,
				providers.NewOllamaStreamTranslator(base),
				func(chunk *models.StreamChunk) {
					if chunk.Usage != nil {
						observedUsage++
					}
				},
				includeUsage,
			)
			if err != nil {
				t.Fatalf("stream failed: %v", err)
			}
			if observedUsage != 1 {
				t.Fatalf("observer saw %d usage chunks, want 1", observedUsage)
			}

			frames := strings.Split(strings.TrimSpace(recorder.Body.String()), "\n\n")
			wantFrames := 3
			if includeUsage {
				wantFrames = 4
			}
			if len(frames) != wantFrames {
				t.Fatalf("stream frames = %d, want %d: %s", len(frames), wantFrames, recorder.Body.String())
			}
			for index, frame := range frames[:2] {
				if strings.Contains(frame, `"usage":`) {
					t.Fatalf("ordinary chunk %d contains usage: %s", index, frame)
				}
			}
			var finishChunk models.StreamChunk
			if err := json.Unmarshal([]byte(strings.TrimPrefix(frames[1], "data: ")), &finishChunk); err != nil {
				t.Fatalf("decode finish chunk: %v", err)
			}
			if len(finishChunk.Choices) != 1 || finishChunk.Choices[0].FinishReason == nil || *finishChunk.Choices[0].FinishReason != "stop" {
				t.Fatalf("unexpected finish chunk: %#v", finishChunk)
			}
			if includeUsage {
				assertCanonicalUsageTrailer(t, frames[2], 3, 2, 5)
			} else if got := strings.Count(recorder.Body.String(), `"usage":`); got != 0 {
				t.Fatalf("client saw %d usage chunks without opting in: %s", got, recorder.Body.String())
			}
			if frames[len(frames)-1] != "data: [DONE]" {
				t.Fatalf("terminal frame = %q, want data: [DONE]", frames[len(frames)-1])
			}
		})
	}
}

func assertCanonicalUsageTrailer(t *testing.T, frame string, promptTokens, completionTokens, totalTokens int) {
	t.Helper()
	var chunk models.StreamChunk
	if err := json.Unmarshal([]byte(strings.TrimPrefix(frame, "data: ")), &chunk); err != nil {
		t.Fatalf("decode usage trailer: %v", err)
	}
	if len(chunk.Choices) != 0 {
		t.Fatalf("usage trailer choices = %#v, want empty", chunk.Choices)
	}
	if chunk.Usage == nil ||
		chunk.Usage.PromptTokens != promptTokens ||
		chunk.Usage.CompletionTokens != completionTokens ||
		chunk.Usage.TotalTokens != totalTokens {
		t.Fatalf("usage trailer = %#v, want prompt=%d completion=%d total=%d", chunk.Usage, promptTokens, completionTokens, totalTokens)
	}
}

func TestStreamResponseStopsReadingAfterDownstreamFailure(t *testing.T) {
	chunk := func(content string) string {
		return "data: {\"id\":\"chatcmpl_write_failure\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-test\",\"choices\":[{\"index\":0,\"delta\":{\"content\":" + strconv.Quote(content) + "}}]}\n\n"
	}
	tests := []struct {
		name        string
		upstream    []string
		failWriteAt int
		failFlushAt int
		wantReads   int
	}{
		{
			name:        "chunk write",
			upstream:    []string{chunk("first"), chunk("second"), "data: [DONE]\n"},
			failWriteAt: 2,
			wantReads:   2,
		},
		{
			name:        "chunk flush",
			upstream:    []string{chunk("first"), chunk("second"), "data: [DONE]\n"},
			failFlushAt: 3,
			wantReads:   2,
		},
		{
			name:        "done write",
			upstream:    []string{"data: [DONE]\n\n", chunk("must not be read")},
			failWriteAt: 1,
			wantReads:   1,
		},
		{
			name:        "done flush",
			upstream:    []string{"data: [DONE]\n\n", chunk("must not be read")},
			failFlushAt: 2,
			wantReads:   1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := &stepReadCloser{chunks: tt.upstream}
			writer := newFailAfterNWriter(tt.failWriteAt, tt.failFlushAt)
			translator := providers.NewOpenAITranslator(config.ProviderConfig{APIKey: "dummy"})
			providerResp := &http.Response{StatusCode: http.StatusOK, Body: body}

			err := NewHandler().StreamResponse(context.Background(), writer, providerResp, translator)
			if !errors.Is(err, errInjectedDownstreamFailure) {
				t.Fatalf("error = %v, want injected downstream failure", err)
			}
			if body.reads != tt.wantReads {
				t.Fatalf("upstream reads = %d, want %d", body.reads, tt.wantReads)
			}
			if !body.closed {
				t.Fatal("upstream body was not closed")
			}
		})
	}
}

func TestStreamAnthropicResponseStopsReadingAfterDownstreamFailure(t *testing.T) {
	tests := []struct {
		name        string
		upstream    []string
		failWriteAt int
		failFlushAt int
	}{
		{
			name: "chunk write",
			upstream: []string{
				"event: message_start\n",
				"data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_failure\",\"model\":\"claude\",\"usage\":{\"input_tokens\":1}}}\n\n",
				"event: message_stop\n",
				"data: {\"type\":\"message_stop\"}\n\n",
			},
			failWriteAt: 1,
		},
		{
			name: "done flush",
			upstream: []string{
				"event: message_stop\n",
				"data: {\"type\":\"message_stop\"}\n\n",
				"event: message_start\n",
			},
			failFlushAt: 2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := &stepReadCloser{chunks: tt.upstream}
			writer := newFailAfterNWriter(tt.failWriteAt, tt.failFlushAt)
			base := providers.NewAnthropicTranslator(config.ProviderConfig{APIKey: "dummy"})
			providerResp := &http.Response{StatusCode: http.StatusOK, Body: body}

			err := NewHandler().StreamAnthropicResponse(
				context.Background(),
				writer,
				providerResp,
				providers.NewAnthropicStreamTranslator(base),
			)
			if !errors.Is(err, errInjectedDownstreamFailure) {
				t.Fatalf("error = %v, want injected downstream failure", err)
			}
			if body.reads != 2 {
				t.Fatalf("upstream reads = %d, want 2", body.reads)
			}
			if !body.closed {
				t.Fatal("upstream body was not closed")
			}
		})
	}
}

var errInjectedDownstreamFailure = errors.New("injected downstream failure")

type failAfterNWriter struct {
	header      http.Header
	body        bytes.Buffer
	writes      int
	flushes     int
	failWriteAt int
	failFlushAt int
}

func newFailAfterNWriter(failWriteAt int, failFlushAt int) *failAfterNWriter {
	return &failAfterNWriter{
		header:      make(http.Header),
		failWriteAt: failWriteAt,
		failFlushAt: failFlushAt,
	}
}

func (w *failAfterNWriter) Header() http.Header {
	return w.header
}

func (w *failAfterNWriter) WriteHeader(int) {}

func (w *failAfterNWriter) Write(payload []byte) (int, error) {
	w.writes++
	if w.failWriteAt > 0 && w.writes == w.failWriteAt {
		return 0, errInjectedDownstreamFailure
	}
	return w.body.Write(payload)
}

func (w *failAfterNWriter) Flush() {
	_ = w.FlushError()
}

func (w *failAfterNWriter) FlushError() error {
	w.flushes++
	if w.failFlushAt > 0 && w.flushes == w.failFlushAt {
		return errInjectedDownstreamFailure
	}
	return nil
}

type stepReadCloser struct {
	chunks []string
	reads  int
	closed bool
}

func (r *stepReadCloser) Read(payload []byte) (int, error) {
	if r.closed {
		return 0, io.ErrClosedPipe
	}
	if r.reads >= len(r.chunks) {
		return 0, io.EOF
	}
	chunk := r.chunks[r.reads]
	r.reads++
	if len(chunk) > len(payload) {
		return 0, errors.New("test chunk exceeds read buffer")
	}
	return copy(payload, chunk), nil
}

func (r *stepReadCloser) Close() error {
	r.closed = true
	return nil
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

func TestStreamNDJSONResponseStopsReadingAfterDownstreamFailure(t *testing.T) {
	chunk := func(content string, done bool) string {
		return "{\"model\":\"qwen\",\"message\":{\"role\":\"assistant\",\"content\":" + strconv.Quote(content) + "},\"done\":" + strconv.FormatBool(done) + "}\n"
	}
	tests := []struct {
		name        string
		upstream    []string
		failWriteAt int
		failFlushAt int
		wantReads   int
	}{
		{
			name:        "chunk write",
			upstream:    []string{chunk("first", false), chunk("second", false), chunk("", true)},
			failWriteAt: 2,
			wantReads:   2,
		},
		{
			name:        "chunk flush",
			upstream:    []string{chunk("first", false), chunk("second", false), chunk("", true)},
			failFlushAt: 3,
			wantReads:   2,
		},
		{
			name:        "done write",
			upstream:    []string{chunk("", true), chunk("must not be read", false)},
			failWriteAt: 2,
			wantReads:   1,
		},
		{
			name:        "done flush",
			upstream:    []string{chunk("", true), chunk("must not be read", false)},
			failFlushAt: 3,
			wantReads:   1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := &stepReadCloser{chunks: tt.upstream}
			writer := newFailAfterNWriter(tt.failWriteAt, tt.failFlushAt)
			base := providers.NewOllamaTranslator(config.ProviderConfig{BaseURL: "http://localhost:11434"})
			providerResp := &http.Response{StatusCode: http.StatusOK, Body: body}

			err := NewHandler().StreamNDJSONResponse(
				context.Background(),
				writer,
				providerResp,
				providers.NewOllamaStreamTranslator(base),
			)
			if !errors.Is(err, errInjectedDownstreamFailure) {
				t.Fatalf("error = %v, want injected downstream failure", err)
			}
			if body.reads != tt.wantReads {
				t.Fatalf("upstream reads = %d, want %d", body.reads, tt.wantReads)
			}
			if !body.closed {
				t.Fatal("upstream body was not closed")
			}
		})
	}
}

func TestStreamNDJSONResponseClosesUpstreamAfterHeaderFlushFailure(t *testing.T) {
	body := &stepReadCloser{chunks: []string{"must not be read"}}
	writer := newFailAfterNWriter(0, 1)
	base := providers.NewOllamaTranslator(config.ProviderConfig{BaseURL: "http://localhost:11434"})
	providerResp := &http.Response{StatusCode: http.StatusOK, Body: body}

	err := NewHandler().StreamNDJSONResponse(
		context.Background(),
		writer,
		providerResp,
		providers.NewOllamaStreamTranslator(base),
	)
	if !errors.Is(err, errInjectedDownstreamFailure) {
		t.Fatalf("error = %v, want injected downstream failure", err)
	}
	if body.reads != 0 {
		t.Fatalf("upstream reads = %d, want 0", body.reads)
	}
	if !body.closed {
		t.Fatal("upstream body was not closed")
	}
}
