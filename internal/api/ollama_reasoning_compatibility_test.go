package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/internal/routing"
	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestChatCompletionsRejectsUnsupportedOllamaReasoningBeforeUpstream(t *testing.T) {
	var upstreamCalls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		upstreamCalls.Add(1)
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer upstream.Close()

	handler := newUpstreamErrorTestHandler(t, map[string]config.ProviderConfig{
		"ollama-primary": {Type: "ollama", BaseURL: upstream.URL},
	}, config.RouteConfig{
		Name:    "chat",
		Match:   config.MatchConfig{Path: "/v1/chat/completions"},
		Targets: []config.TargetConfig{{Provider: "ollama-primary", Model: "qwen3.5", Weight: 1}},
	}, config.RetryConfig{Enabled: false})

	tests := []struct {
		name      string
		body      string
		wantParam string
	}{
		{
			name:      "top-level minimal",
			body:      `{"model":"qwen3.5","messages":[{"role":"user","content":"hi"}],"reasoning_effort":"minimal"}`,
			wantParam: "reasoning_effort",
		},
		{
			name:      "nested max",
			body:      `{"model":"qwen3.5","messages":[{"role":"user","content":"hi"}],"reasoning":{"effort":"max"}}`,
			wantParam: "reasoning.effort",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			recorder := httptest.NewRecorder()
			handler.ChatCompletions(recorder, httptest.NewRequest(
				http.MethodPost,
				"/v1/chat/completions",
				bytes.NewBufferString(test.body),
			))
			if recorder.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400; body=%s", recorder.Code, recorder.Body.String())
			}
			var response models.ErrorResponse
			if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
				t.Fatalf("decode error response: %v", err)
			}
			if response.Error.Param == nil || *response.Error.Param != test.wantParam {
				t.Fatalf("error param = %#v, want %q", response.Error.Param, test.wantParam)
			}
			if response.Error.Code == nil || *response.Error.Code != "unsupported_feature" {
				t.Fatalf("error code = %#v, want unsupported_feature", response.Error.Code)
			}
		})
	}
	if calls := upstreamCalls.Load(); calls != 0 {
		t.Fatalf("upstream calls = %d, want 0", calls)
	}
}

func TestCompatibleChatFallbacksDropOllamaForUnsupportedReasoningEffort(t *testing.T) {
	handler := &Handler{registry: providers.NewRegistry(map[string]config.ProviderConfig{
		"ollama-backup": {Type: "ollama"},
		"openai-backup": {Type: "openai"},
	})}
	request := &models.UnifiedRequest{
		SourceRequestType: requestTypeChatCompletions,
		RawJSON: json.RawMessage(`{
			"model":"model","messages":[{"role":"user","content":"hi"}],
			"reasoning":{"effort":"minimal"}
		}`),
		ReasoningEffort: "minimal",
		Messages:        []models.Message{{Role: "user", Content: "hi"}},
	}

	got := handler.compatibleChatFallbacks([]routing.Target{
		{Provider: "ollama-backup", Model: "qwen3.5"},
		{Provider: "openai-backup", Model: "gpt-5.4"},
	}, request)
	if len(got) != 1 || got[0].Provider != "openai-backup" {
		t.Fatalf("compatible fallbacks = %#v, want only openai-backup", got)
	}
}
