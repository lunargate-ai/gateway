package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
)

func TestChatCompletionsNormalizesMaliciousNegativeUsage(t *testing.T) {
	tests := []struct {
		name             string
		usage            string
		wantInputTokens  float64
		wantOutputTokens float64
		wantCostUSD      float64
	}{
		{
			name:             "negative prompt tokens",
			usage:            `"prompt_tokens":-1000000,"completion_tokens":1000000,"total_tokens":-1,"prompt_tokens_details":{"cached_tokens":-2}`,
			wantInputTokens:  0,
			wantOutputTokens: 1_000_000,
			wantCostUSD:      10,
		},
		{
			name:             "negative completion tokens",
			usage:            `"prompt_tokens":1000000,"completion_tokens":-1000000,"total_tokens":-1,"completion_tokens_details":{"reasoning_tokens":-2}`,
			wantInputTokens:  1_000_000,
			wantOutputTokens: 0,
			wantCostUSD:      2.5,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"id":"chatcmpl-negative","object":"chat.completion","model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{` + test.usage + `}}`))
			}))
			defer upstream.Close()

			capture := newCollectorCapture(t, false, true)
			handler, _ := newObservedOpenAIHandler(
				t,
				upstream.URL,
				config.TargetConfig{Provider: "openai", Model: "gpt-4o", Weight: 1},
				capture.client,
				config.CacheConfig{Enabled: false},
			)

			recorder := httptest.NewRecorder()
			request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"messages":[{"role":"user","content":"hello"}]}`))
			handler.ChatCompletions(recorder, request)
			if recorder.Code != http.StatusOK {
				t.Fatalf("status = %d, want 200; body=%s", recorder.Code, recorder.Body.String())
			}
			assertResponseUsageNonNegative(t, recorder.Body.Bytes())

			_, metric, requestLog := capture.waitForRequestEvents(t)
			if got := metric["tokens_input"]; got != test.wantInputTokens {
				t.Fatalf("tokens_input = %#v, want %#v", got, test.wantInputTokens)
			}
			if got := metric["tokens_output"]; got != test.wantOutputTokens {
				t.Fatalf("tokens_output = %#v, want %#v", got, test.wantOutputTokens)
			}
			if got := metric["cost_usd"]; got != test.wantCostUSD {
				t.Fatalf("cost_usd = %#v, want %#v", got, test.wantCostUSD)
			}
			assertPayloadUsageNonNegative(t, requestLog["response"])
		})
	}
}

func TestEmbeddingsNormalizesMaliciousNegativeUsage(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0.1]}],"model":"text-embedding-3-small","usage":{"prompt_tokens":-1000000,"total_tokens":-1,"prompt_tokens_details":{"cached_tokens":-2}}}`))
	}))
	defer upstream.Close()

	capture := newCollectorCapture(t, false, true)
	handler, _ := newObservedOpenAIHandler(
		t,
		upstream.URL,
		config.TargetConfig{Provider: "openai", Model: "text-embedding-3-small", Weight: 1},
		capture.client,
		config.CacheConfig{Enabled: false},
	)

	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/v1/embeddings", strings.NewReader(`{"model":"text-embedding-3-small","input":"hello"}`))
	handler.Embeddings(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", recorder.Code, recorder.Body.String())
	}
	assertResponseUsageNonNegative(t, recorder.Body.Bytes())

	_, metric, requestLog := capture.waitForRequestEvents(t)
	if got := metric["tokens_input"]; got != float64(0) {
		t.Fatalf("tokens_input = %#v, want 0", got)
	}
	if got := metric["cost_usd"]; got != float64(0) {
		t.Fatalf("cost_usd = %#v, want 0", got)
	}
	assertPayloadUsageNonNegative(t, requestLog["response"])
}

func TestEmbeddingsRaisesTotalToPromptTokens(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0.1]}],"model":"text-embedding-3-small","usage":{"prompt_tokens":100,"total_tokens":-1}}`))
	}))
	defer upstream.Close()

	capture := newCollectorCapture(t, false, true)
	handler, _ := newObservedOpenAIHandler(
		t,
		upstream.URL,
		config.TargetConfig{Provider: "openai", Model: "text-embedding-3-small", Weight: 1},
		capture.client,
		config.CacheConfig{Enabled: false},
	)

	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/v1/embeddings", strings.NewReader(`{"model":"text-embedding-3-small","input":"hello"}`))
	handler.Embeddings(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", recorder.Code, recorder.Body.String())
	}
	var response struct {
		Usage struct {
			PromptTokens int `json:"prompt_tokens"`
			TotalTokens  int `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if response.Usage.PromptTokens != 100 || response.Usage.TotalTokens != 100 {
		t.Fatalf("response usage = %#v, want prompt=100 total=100", response.Usage)
	}

	_, metric, requestLog := capture.waitForRequestEvents(t)
	if got := metric["tokens_input"]; got != float64(100) {
		t.Fatalf("tokens_input = %#v, want 100", got)
	}
	if got := metric["cost_usd"]; got != float64(0.000002) {
		t.Fatalf("cost_usd = %#v, want 0.000002", got)
	}
	responsePayload := requestLog["response"].(map[string]interface{})
	loggedUsage := responsePayload["usage"].(map[string]interface{})
	if got := loggedUsage["total_tokens"]; got != float64(100) {
		t.Fatalf("logged total_tokens = %#v, want 100", got)
	}
}

func TestChatCompletionsStreamNormalizesMaliciousNegativeUsage(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-negative-stream\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o\",\"choices\":[],\"usage\":{\"prompt_tokens\":-1000000,\"completion_tokens\":-2,\"total_tokens\":-1000002,\"prompt_tokens_details\":{\"cached_tokens\":-3}}}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer upstream.Close()

	capture := newCollectorCapture(t, false, true)
	handler, _ := newObservedOpenAIHandler(
		t,
		upstream.URL,
		config.TargetConfig{Provider: "openai", Model: "gpt-4o", Weight: 1},
		capture.client,
		config.CacheConfig{Enabled: false},
	)

	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"hello"}]}`))
	handler.ChatCompletions(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", recorder.Code, recorder.Body.String())
	}
	assertStreamUsageNonNegative(t, recorder.Body.String())

	_, metric, requestLog := capture.waitForRequestEvents(t)
	if got := metric["tokens_input"]; got != float64(0) {
		t.Fatalf("tokens_input = %#v, want 0", got)
	}
	if got := metric["tokens_output"]; got != float64(0) {
		t.Fatalf("tokens_output = %#v, want 0", got)
	}
	if got := metric["cost_usd"]; got != float64(0) {
		t.Fatalf("cost_usd = %#v, want 0", got)
	}
	response, ok := requestLog["response"].(map[string]interface{})
	if !ok {
		t.Fatalf("request-log response = %#v, want object", requestLog["response"])
	}
	if usage, present := response["usage"]; present {
		assertTokenFieldsNonNegative(t, usage)
	}
}

func assertStreamUsageNonNegative(t *testing.T, body string) {
	t.Helper()
	for _, line := range strings.Split(body, "\n") {
		payload, ok := strings.CutPrefix(line, "data: ")
		if !ok || payload == "[DONE]" {
			continue
		}
		var envelope map[string]interface{}
		if err := json.Unmarshal([]byte(payload), &envelope); err != nil {
			t.Fatalf("decode stream frame: %v", err)
		}
		if usage, ok := envelope["usage"].(map[string]interface{}); ok {
			assertTokenFieldsNonNegative(t, usage)
		}
	}
}

func assertResponseUsageNonNegative(t *testing.T, body []byte) {
	t.Helper()
	var payload interface{}
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	assertPayloadUsageNonNegative(t, payload)
}

func assertPayloadUsageNonNegative(t *testing.T, payload interface{}) {
	t.Helper()
	envelope, ok := payload.(map[string]interface{})
	if !ok {
		t.Fatalf("response payload = %#v, want object", payload)
	}
	usage, ok := envelope["usage"].(map[string]interface{})
	if !ok {
		t.Fatalf("response usage = %#v, want object", envelope["usage"])
	}
	assertTokenFieldsNonNegative(t, usage)
}

func assertTokenFieldsNonNegative(t *testing.T, value interface{}) {
	t.Helper()
	switch typed := value.(type) {
	case map[string]interface{}:
		for key, child := range typed {
			if strings.HasSuffix(strings.ToLower(key), "_tokens") {
				if number, ok := child.(float64); ok && number < 0 {
					t.Errorf("usage field %s = %v, want non-negative", key, number)
				}
			}
			assertTokenFieldsNonNegative(t, child)
		}
	case []interface{}:
		for _, child := range typed {
			assertTokenFieldsNonNegative(t, child)
		}
	}
}
