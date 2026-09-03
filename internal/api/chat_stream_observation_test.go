package api

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/observability"
	"github.com/lunargate-ai/gateway/pkg/models"
	"github.com/prometheus/client_golang/prometheus/testutil"
)

func TestChatStreamObservationDisabledDoesNotRetainResponse(t *testing.T) {
	finishReason := "tool_calls"
	index := 0
	chunk := &models.StreamChunk{Choices: []models.Choice{{
		Delta: &models.Message{
			Content:          strings.Repeat("response", 1024),
			ReasoningContent: strings.Repeat("reasoning", 1024),
			ToolCalls: []models.ToolCall{{
				Index: &index,
				ID:    "call-secret",
				Type:  "function",
				Function: models.ToolCallFunction{
					Name:      "private_tool",
					Arguments: strings.Repeat("private-arguments", 1024),
				},
			}},
		},
		FinishReason: &finishReason,
	}}}

	observation := newChatStreamObservation(false)
	if !observation.observe(chunk) {
		t.Fatal("disabled observation did not report content for timing")
	}
	assertEmptyChatStreamObservation(t, observation)
	if response := observation.collectorResponse("request-id", "openai/gpt-test", models.TokenUsage{}); response != nil {
		t.Fatalf("disabled observation returned collector response: %#v", response)
	}

	observation = newChatStreamObservation(true)
	observation.observe(chunk)
	if observation.text.Len() == 0 || len(observation.toolCallOrder) == 0 {
		t.Fatal("enabled observation did not retain fixture")
	}
	observation.disable()
	assertEmptyChatStreamObservation(t, observation)
}

func TestChatStreamCollectorOffPreservesClientAndUsage(t *testing.T) {
	contentFrame := chatStreamObservationFrame(t, "chatcmpl-no-collector", "hello", nil, nil, nil)
	usageFrame := chatStreamObservationFrame(t, "chatcmpl-no-collector", "", nil, nil, chatStreamObservationUsage())

	response, metrics := runChatStreamObservationRequest(t, nil, func(w io.Writer) {
		_, _ = io.WriteString(w, contentFrame)
		_, _ = io.WriteString(w, usageFrame)
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	})

	assertCompleteObservedChatStream(t, response, "hello")
	assertObservedChatPrometheusUsage(t, metrics)
}

func TestChatStreamShareResponsesFalseDoesNotCaptureBody(t *testing.T) {
	contentFrame := chatStreamObservationFrame(t, "chatcmpl-sharing-off", "private-response", nil, nil, nil)
	usageFrame := chatStreamObservationFrame(t, "chatcmpl-sharing-off", "", nil, nil, chatStreamObservationUsage())
	capture := newCollectorCapture(t, true, false)

	response, metrics := runChatStreamObservationRequest(t, capture.client, func(w io.Writer) {
		_, _ = io.WriteString(w, contentFrame)
		_, _ = io.WriteString(w, usageFrame)
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	})

	assertCompleteObservedChatStream(t, response, "private-response")
	assertObservedChatPrometheusUsage(t, metrics)
	_, metric, requestLog := capture.waitForRequestEvents(t)
	assertObservedChatCollectorUsageAndTiming(t, metric)
	if _, exists := requestLog["response"]; exists {
		t.Fatalf("share_responses=false exported response: %#v", requestLog["response"])
	}
}

func TestChatStreamObservationBodyLimitOmitsCollectorPayloadWithoutChangingClient(t *testing.T) {
	const contentFrameCount = maxObservedChatStreamBodyBytes/(1<<20) + 1
	content := strings.Repeat("x", 1<<20)
	contentFrame := chatStreamObservationFrame(t, "chatcmpl-body-limit", content, nil, nil, nil)
	usageFrame := chatStreamObservationFrame(t, "chatcmpl-body-limit", "", nil, nil, chatStreamObservationUsage())
	capture := newCollectorCapture(t, false, true)

	response, metrics := runChatStreamObservationRequest(t, capture.client, func(w io.Writer) {
		for range contentFrameCount {
			_, _ = io.WriteString(w, contentFrame)
		}
		_, _ = io.WriteString(w, usageFrame)
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	})

	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body suffix=%q", response.Code, observedChatStreamSuffix(response.Body.String()))
	}
	body := response.Body.String()
	if got := strings.Count(body, `"content":"`); got != contentFrameCount {
		t.Fatalf("client received %d content frames, want %d", got, contentFrameCount)
	}
	assertCompleteObservedChatStream(t, response, content[:32])
	assertObservedChatPrometheusUsage(t, metrics)

	_, metric, requestLog := capture.waitForRequestEvents(t)
	assertObservedChatCollectorUsageAndTiming(t, metric)
	assertOmittedChatStreamCollectorResponse(t, requestLog, chatStreamBodyLimitReason)
}

func TestChatStreamObservationToolLimitOmitsCollectorPayloadWithoutChangingClient(t *testing.T) {
	toolCalls := make([]models.ToolCall, 0, maxObservedChatStreamToolCalls+1)
	for i := 0; i <= maxObservedChatStreamToolCalls; i++ {
		index := i
		toolCalls = append(toolCalls, models.ToolCall{
			Index: &index,
			ID:    "call-" + strconv.Itoa(i),
			Type:  "function",
			Function: models.ToolCallFunction{
				Name:      "tool_" + strconv.Itoa(i),
				Arguments: `{"value":` + strconv.Itoa(i) + `}`,
			},
		})
	}
	finishReason := "tool_calls"
	toolFrame := chatStreamObservationFrame(t, "chatcmpl-tool-limit", "", toolCalls, &finishReason, nil)
	usageFrame := chatStreamObservationFrame(t, "chatcmpl-tool-limit", "", nil, nil, chatStreamObservationUsage())
	capture := newCollectorCapture(t, false, true)

	response, metrics := runChatStreamObservationRequest(t, capture.client, func(w io.Writer) {
		_, _ = io.WriteString(w, toolFrame)
		_, _ = io.WriteString(w, usageFrame)
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	})

	assertCompleteObservedChatStream(t, response, `"id":"call-128"`)
	assertObservedChatPrometheusUsage(t, metrics)
	_, metric, requestLog := capture.waitForRequestEvents(t)
	assertObservedChatCollectorUsageAndTiming(t, metric)
	assertOmittedChatStreamCollectorResponse(t, requestLog, chatStreamToolLimitReason)
}

func assertEmptyChatStreamObservation(t *testing.T, observation *chatStreamObservation) {
	t.Helper()
	if observation.text.Len() != 0 || observation.reasoning.Len() != 0 || observation.capturedBytes != 0 {
		t.Fatalf("observation retained response strings: text=%d reasoning=%d bytes=%d", observation.text.Len(), observation.reasoning.Len(), observation.capturedBytes)
	}
	if observation.toolCallByKey != nil || observation.toolCallOrder != nil || observation.finishReason != nil {
		t.Fatalf("observation retained structured response: tools=%d order=%d finish=%v", len(observation.toolCallByKey), len(observation.toolCallOrder), observation.finishReason)
	}
}

func runChatStreamObservationRequest(
	t *testing.T,
	collector *observability.CollectorClient,
	writeStream func(io.Writer),
) (*httptest.ResponseRecorder, *observability.Metrics) {
	t.Helper()
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		writeStream(w)
	}))
	t.Cleanup(upstream.Close)

	handler, metrics := newObservedOpenAIHandler(
		t,
		upstream.URL,
		config.TargetConfig{Provider: "openai", Model: "gpt-test", Weight: 1},
		collector,
		config.CacheConfig{Enabled: false},
	)
	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-test","messages":[{"role":"user","content":"hello"}],"stream":true,"stream_options":{"include_usage":true}}`),
	)
	response := httptest.NewRecorder()
	handler.ChatCompletions(response, request)
	return response, metrics
}

func chatStreamObservationFrame(
	t *testing.T,
	id string,
	content string,
	toolCalls []models.ToolCall,
	finishReason *string,
	usage *models.Usage,
) string {
	t.Helper()
	choices := make([]models.Choice, 0, 1)
	if content != "" || len(toolCalls) > 0 || finishReason != nil {
		delta := &models.Message{}
		if content != "" {
			delta.Content = content
		}
		if len(toolCalls) > 0 {
			delta.ToolCalls = toolCalls
		}
		choices = append(choices, models.Choice{Index: 0, Delta: delta, FinishReason: finishReason})
	}
	encoded, err := json.Marshal(models.StreamChunk{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: 1,
		Model:   "gpt-test",
		Choices: choices,
		Usage:   usage,
	})
	if err != nil {
		t.Fatalf("encode stream fixture: %v", err)
	}
	return "data: " + string(encoded) + "\n\n"
}

func chatStreamObservationUsage() *models.Usage {
	return &models.Usage{PromptTokens: 3, CompletionTokens: 5, TotalTokens: 8}
}

func assertCompleteObservedChatStream(t *testing.T, response *httptest.ResponseRecorder, expected string) {
	t.Helper()
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body suffix=%q", response.Code, observedChatStreamSuffix(response.Body.String()))
	}
	body := response.Body.String()
	if !strings.Contains(body, expected) {
		t.Fatalf("client stream does not contain %q; body suffix=%q", expected, observedChatStreamSuffix(body))
	}
	if !strings.Contains(body, `"prompt_tokens":3`) || !strings.Contains(body, `"completion_tokens":5`) {
		t.Fatalf("client stream lost usage; body suffix=%q", observedChatStreamSuffix(body))
	}
	if !strings.HasSuffix(body, "data: [DONE]\n\n") {
		t.Fatalf("client stream is incomplete; body suffix=%q", observedChatStreamSuffix(body))
	}
}

func assertObservedChatPrometheusUsage(t *testing.T, metrics *observability.Metrics) {
	t.Helper()
	if got := testutil.ToFloat64(metrics.TokensTotal.WithLabelValues("openai", "gpt-test", "input")); got != 3 {
		t.Fatalf("input token metric = %v, want 3", got)
	}
	if got := testutil.ToFloat64(metrics.TokensTotal.WithLabelValues("openai", "gpt-test", "output")); got != 5 {
		t.Fatalf("output token metric = %v, want 5", got)
	}
}

func assertObservedChatCollectorUsageAndTiming(t *testing.T, metric map[string]interface{}) {
	t.Helper()
	if got := metric["tokens_input"]; got != float64(3) {
		t.Fatalf("collector input tokens = %#v, want 3", got)
	}
	if got := metric["tokens_output"]; got != float64(5) {
		t.Fatalf("collector output tokens = %#v, want 5", got)
	}
	if _, ok := metric["ttft_ms"]; !ok {
		t.Fatalf("collector metric has no ttft_ms: %#v", metric)
	}
	if _, ok := metric["ttlt_ms"]; !ok {
		t.Fatalf("collector metric has no ttlt_ms: %#v", metric)
	}
}

func assertOmittedChatStreamCollectorResponse(t *testing.T, requestLog map[string]interface{}, reason string) {
	t.Helper()
	response, ok := requestLog["response"].(map[string]interface{})
	if !ok {
		t.Fatalf("request-log response = %#v, want omission object", requestLog["response"])
	}
	if response["object"] != "lunargate.chat.completion.observation" || response["response_omitted"] != true || response["truncated"] != true {
		t.Fatalf("collector response is not explicitly omitted/truncated: %#v", response)
	}
	if got := response["truncation_reason"]; got != reason {
		t.Fatalf("truncation reason = %#v, want %q", got, reason)
	}
	if _, exists := response["choices"]; exists {
		t.Fatalf("omitted collector response contains partial choices: %#v", response)
	}
	limits, ok := response["observation_limit"].(map[string]interface{})
	if !ok || limits["max_bytes"] != float64(maxObservedChatStreamBodyBytes) || limits["max_tool_calls"] != float64(maxObservedChatStreamToolCalls) {
		t.Fatalf("observation limits = %#v", response["observation_limit"])
	}
	usage, ok := response["usage"].(map[string]interface{})
	if !ok || usage["prompt_tokens"] != float64(3) || usage["completion_tokens"] != float64(5) || usage["total_tokens"] != float64(8) {
		t.Fatalf("omitted response usage = %#v", response["usage"])
	}
}

func observedChatStreamSuffix(body string) string {
	const maxSuffix = 512
	if len(body) <= maxSuffix {
		return body
	}
	return body[len(body)-maxSuffix:]
}
