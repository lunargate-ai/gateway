package api

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestChatCompletionsPreservesAnthropicRefusal(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" {
			t.Errorf("upstream path = %q, want /v1/messages", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"msg_refusal","type":"message","role":"assistant","model":"claude-opus-5",
			"content":[
				{"type":"text","text":"partial visible answer"},
				{"type":"tool_use","id":"toolu_partial","name":"unsafe_partial_tool","input":{"value":1}}
			],
			"stop_reason":"refusal",
			"stop_details":{"type":"refusal","category":"cyber","explanation":"This request was declined."},
			"usage":{"input_tokens":3,"output_tokens":4}
		}`)
	}))
	defer upstream.Close()

	handler := newUpstreamErrorTestHandler(t, map[string]config.ProviderConfig{
		"anthropic-primary": {Type: "anthropic", APIKey: "dummy", BaseURL: upstream.URL},
	}, config.RouteConfig{
		Name:    "chat",
		Match:   config.MatchConfig{Path: "/v1/chat/completions"},
		Targets: []config.TargetConfig{{Provider: "anthropic-primary", Model: "claude-opus-5", Weight: 1}},
	}, config.RetryConfig{Enabled: false})

	recorder := httptest.NewRecorder()
	handler.ChatCompletions(recorder, httptest.NewRequest(
		http.MethodPost,
		"/v1/chat/completions",
		bytes.NewBufferString(`{"model":"claude-opus-5","messages":[{"role":"user","content":"hello"}]}`),
	))
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, body=%s", recorder.Code, recorder.Body.String())
	}

	var response models.UnifiedResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	choice := response.Choices[0]
	if choice.FinishReason == nil || *choice.FinishReason != "content_filter" {
		t.Fatalf("finish reason = %#v, want content_filter", choice.FinishReason)
	}
	if choice.Message.Refusal != "This request was declined." {
		t.Fatalf("refusal = %q", choice.Message.Refusal)
	}
	if choice.Message.ContentString() != "" {
		t.Fatalf("refusal retained partial content = %#v", choice.Message.Content)
	}
	if len(choice.Message.ToolCalls) != 0 {
		t.Fatalf("refusal retained partial tool calls = %#v", choice.Message.ToolCalls)
	}
	if bytes.Contains(recorder.Body.Bytes(), []byte("partial visible answer")) ||
		bytes.Contains(recorder.Body.Bytes(), []byte("unsafe_partial_tool")) {
		t.Fatalf("response leaked unusable refusal output: %s", recorder.Body.String())
	}
}
