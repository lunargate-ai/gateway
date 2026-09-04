package api

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/internal/routing"
	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestChatCompletionsRejectsLossyNestedControlBeforeUpstream(t *testing.T) {
	for _, providerType := range []string{"anthropic", "ollama"} {
		t.Run(providerType, func(t *testing.T) {
			var upstreamCalls atomic.Int32
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				upstreamCalls.Add(1)
				w.WriteHeader(http.StatusInternalServerError)
			}))
			defer upstream.Close()

			providerID := providerType + "-primary"
			handler := newUpstreamErrorTestHandler(t, map[string]config.ProviderConfig{
				providerID: {Type: providerType, APIKey: "dummy", BaseURL: upstream.URL},
			}, config.RouteConfig{
				Name:    "chat",
				Match:   config.MatchConfig{Path: "/v1/chat/completions"},
				Targets: []config.TargetConfig{{Provider: providerID, Model: "model", Weight: 1}},
			}, config.RetryConfig{Enabled: false})

			recorder := httptest.NewRecorder()
			handler.ChatCompletions(recorder, httptest.NewRequest(
				http.MethodPost,
				"/v1/chat/completions",
				bytes.NewBufferString(`{"model":"model","messages":[{"role":"user","content":"hi"}],"stream_options":{"include_obfuscation":true}}`),
			))

			if recorder.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400; body=%s", recorder.Code, recorder.Body.String())
			}
			var response models.ErrorResponse
			if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
				t.Fatalf("decode error response: %v", err)
			}
			if response.Error.Param == nil || *response.Error.Param != "stream_options.include_obfuscation" {
				t.Fatalf("error param = %#v, want stream_options.include_obfuscation", response.Error.Param)
			}
			if calls := upstreamCalls.Load(); calls != 0 {
				t.Fatalf("upstream calls = %d, want 0", calls)
			}
		})
	}
}

func TestCompatibleChatFallbacksDropTargetsWithLossyNestedControls(t *testing.T) {
	handler := &Handler{registry: providers.NewRegistry(map[string]config.ProviderConfig{
		"anthropic-backup": {Type: "anthropic"},
		"ollama-backup":    {Type: "ollama"},
		"openai-backup":    {Type: "openai"},
	})}
	request := &models.UnifiedRequest{
		SourceRequestType: requestTypeChatCompletions,
		RawJSON:           json.RawMessage(`{"model":"model","messages":[],"reasoning":{"summary":"auto"}}`),
	}
	fallbacks := []routing.Target{
		{Provider: "anthropic-backup", Model: "claude"},
		{Provider: "ollama-backup", Model: "qwen"},
		{Provider: "openai-backup", Model: "gpt"},
	}

	got := handler.compatibleChatFallbacks(fallbacks, request)
	if len(got) != 1 || got[0].Provider != "openai-backup" {
		t.Fatalf("compatible fallbacks = %#v, want only openai-backup", got)
	}

	err := handler.validateChatCompatibility(routing.Target{Provider: "anthropic-backup"}, request)
	var compatibilityErr *models.CompatibilityError
	if !errors.As(err, &compatibilityErr) || compatibilityErr.Field != "reasoning.summary" {
		t.Fatalf("primary compatibility error = %#v, want reasoning.summary", err)
	}
}

func TestChatCompletionsRejectsLossyNestedMessageBeforeUpstream(t *testing.T) {
	for _, providerType := range []string{"anthropic", "ollama"} {
		t.Run(providerType, func(t *testing.T) {
			var upstreamCalls atomic.Int32
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				upstreamCalls.Add(1)
				w.WriteHeader(http.StatusInternalServerError)
			}))
			defer upstream.Close()

			providerID := providerType + "-primary"
			handler := newUpstreamErrorTestHandler(t, map[string]config.ProviderConfig{
				providerID: {Type: providerType, APIKey: "dummy", BaseURL: upstream.URL},
			}, config.RouteConfig{
				Name:    "chat",
				Match:   config.MatchConfig{Path: "/v1/chat/completions"},
				Targets: []config.TargetConfig{{Provider: providerID, Model: "model", Weight: 1}},
			}, config.RetryConfig{Enabled: false})

			recorder := httptest.NewRecorder()
			handler.ChatCompletions(recorder, httptest.NewRequest(
				http.MethodPost,
				"/v1/chat/completions",
				bytes.NewBufferString(`{"model":"model","messages":[{"role":"assistant","audio":{"id":"audio_1"}}]}`),
			))

			if recorder.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400; body=%s", recorder.Code, recorder.Body.String())
			}
			var response models.ErrorResponse
			if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
				t.Fatalf("decode error response: %v", err)
			}
			if response.Error.Param == nil || *response.Error.Param != "messages[0].audio" {
				t.Fatalf("error param = %#v, want messages[0].audio", response.Error.Param)
			}
			if calls := upstreamCalls.Load(); calls != 0 {
				t.Fatalf("upstream calls = %d, want 0", calls)
			}
		})
	}
}

func TestCompatibleChatFallbacksDropTargetsWithLossyNestedMessages(t *testing.T) {
	handler := &Handler{registry: providers.NewRegistry(map[string]config.ProviderConfig{
		"anthropic-backup": {Type: "anthropic"},
		"ollama-backup":    {Type: "ollama"},
		"openai-backup":    {Type: "openai"},
	})}
	request := &models.UnifiedRequest{
		SourceRequestType: requestTypeChatCompletions,
		RawJSON: json.RawMessage(`{
			"model":"model",
			"messages":[{"role":"user","content":[{"type":"text","text":"hi","prompt_cache_breakpoint":{"type":"ephemeral"}}]}]
		}`),
	}

	got := handler.compatibleChatFallbacks([]routing.Target{
		{Provider: "anthropic-backup", Model: "claude"},
		{Provider: "ollama-backup", Model: "qwen"},
		{Provider: "openai-backup", Model: "gpt"},
	}, request)
	if len(got) != 1 || got[0].Provider != "openai-backup" {
		t.Fatalf("compatible fallbacks = %#v, want only openai-backup", got)
	}
}

func TestChatCompletionsRejectsLossyNestedToolBeforeUpstream(t *testing.T) {
	for _, providerType := range []string{"anthropic", "ollama"} {
		t.Run(providerType, func(t *testing.T) {
			var upstreamCalls atomic.Int32
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				upstreamCalls.Add(1)
				w.WriteHeader(http.StatusInternalServerError)
			}))
			defer upstream.Close()

			providerID := providerType + "-primary"
			handler := newUpstreamErrorTestHandler(t, map[string]config.ProviderConfig{
				providerID: {Type: providerType, APIKey: "dummy", BaseURL: upstream.URL},
			}, config.RouteConfig{
				Name:    "chat",
				Match:   config.MatchConfig{Path: "/v1/chat/completions"},
				Targets: []config.TargetConfig{{Provider: providerID, Model: "model", Weight: 1}},
			}, config.RetryConfig{Enabled: false})

			recorder := httptest.NewRecorder()
			handler.ChatCompletions(recorder, httptest.NewRequest(
				http.MethodPost,
				"/v1/chat/completions",
				bytes.NewBufferString(`{
					"model":"model",
					"messages":[{"role":"user","content":"hi"}],
					"tools":[{"type":"function","function":{"name":"lookup","future":"x"}}]
				}`),
			))

			if recorder.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400; body=%s", recorder.Code, recorder.Body.String())
			}
			var response models.ErrorResponse
			if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
				t.Fatalf("decode error response: %v", err)
			}
			if response.Error.Param == nil || *response.Error.Param != "tools[0].function.future" {
				t.Fatalf("error param = %#v, want tools[0].function.future", response.Error.Param)
			}
			if calls := upstreamCalls.Load(); calls != 0 {
				t.Fatalf("upstream calls = %d, want 0", calls)
			}
		})
	}
}

func TestCompatibleChatFallbacksRespectFunctionStrictness(t *testing.T) {
	handler := &Handler{registry: providers.NewRegistry(map[string]config.ProviderConfig{
		"anthropic-backup": {Type: "anthropic"},
		"ollama-backup":    {Type: "ollama"},
		"openai-backup":    {Type: "openai"},
	})}
	strict := true
	request := &models.UnifiedRequest{
		SourceRequestType: requestTypeChatCompletions,
		RawJSON: json.RawMessage(`{
			"model":"model",
			"messages":[{"role":"user","content":"hi"}],
			"tools":[{"type":"function","function":{"name":"lookup","strict":true}}]
		}`),
		Messages: []models.Message{{Role: "user", Content: "hi"}},
		Tools: []models.Tool{{Type: "function", Function: models.ToolFunction{
			Name: "lookup", Strict: &strict,
		}}},
	}

	got := handler.compatibleChatFallbacks([]routing.Target{
		{Provider: "anthropic-backup", Model: "claude"},
		{Provider: "ollama-backup", Model: "qwen"},
		{Provider: "openai-backup", Model: "gpt"},
	}, request)
	if len(got) != 2 || got[0].Provider != "anthropic-backup" || got[1].Provider != "openai-backup" {
		t.Fatalf("compatible fallbacks = %#v, want Anthropic and OpenAI", got)
	}
}

func TestChatCompletionsRejectsLossyToolChoiceShapeBeforeUpstream(t *testing.T) {
	for _, providerType := range []string{"anthropic", "ollama"} {
		t.Run(providerType, func(t *testing.T) {
			var upstreamCalls atomic.Int32
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				upstreamCalls.Add(1)
				w.WriteHeader(http.StatusInternalServerError)
			}))
			defer upstream.Close()

			providerID := providerType + "-primary"
			handler := newUpstreamErrorTestHandler(t, map[string]config.ProviderConfig{
				providerID: {Type: providerType, APIKey: "dummy", BaseURL: upstream.URL},
			}, config.RouteConfig{
				Name:    "chat",
				Match:   config.MatchConfig{Path: "/v1/chat/completions"},
				Targets: []config.TargetConfig{{Provider: providerID, Model: "model", Weight: 1}},
			}, config.RetryConfig{Enabled: false})

			recorder := httptest.NewRecorder()
			handler.ChatCompletions(recorder, httptest.NewRequest(
				http.MethodPost,
				"/v1/chat/completions",
				bytes.NewBufferString(`{
					"model":"model",
					"messages":[{"role":"user","content":"hi"}],
					"tool_choice":{"type":"auto","function":{"name":"lookup"}}
				}`),
			))

			if recorder.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400; body=%s", recorder.Code, recorder.Body.String())
			}
			var response models.ErrorResponse
			if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
				t.Fatalf("decode error response: %v", err)
			}
			if response.Error.Param == nil || *response.Error.Param != "tool_choice.function" {
				t.Fatalf("error param = %#v, want tool_choice.function", response.Error.Param)
			}
			if calls := upstreamCalls.Load(); calls != 0 {
				t.Fatalf("upstream calls = %d, want 0", calls)
			}
		})
	}
}

func TestCompatibleChatFallbacksDropTranslatedTargetsForLossyToolChoiceShape(t *testing.T) {
	handler := &Handler{registry: providers.NewRegistry(map[string]config.ProviderConfig{
		"anthropic-backup": {Type: "anthropic"},
		"ollama-backup":    {Type: "ollama"},
		"openai-backup":    {Type: "openai"},
	})}
	request := &models.UnifiedRequest{
		SourceRequestType: requestTypeChatCompletions,
		RawJSON: json.RawMessage(`{
			"model":"model","messages":[{"role":"user","content":"hi"}],
			"tool_choice":{"type":"auto","function":{"name":"lookup"}}
		}`),
		Messages: []models.Message{{Role: "user", Content: "hi"}},
		ToolChoice: map[string]interface{}{
			"type":     "auto",
			"function": map[string]interface{}{"name": "lookup"},
		},
	}

	got := handler.compatibleChatFallbacks([]routing.Target{
		{Provider: "anthropic-backup", Model: "claude"},
		{Provider: "ollama-backup", Model: "qwen"},
		{Provider: "openai-backup", Model: "gpt"},
	}, request)
	if len(got) != 1 || got[0].Provider != "openai-backup" {
		t.Fatalf("compatible fallbacks = %#v, want only OpenAI", got)
	}
}

func TestChatCompletionsRejectsConflictingSchemaAnnotationBeforeUpstream(t *testing.T) {
	for _, providerType := range []string{"anthropic", "ollama"} {
		t.Run(providerType, func(t *testing.T) {
			var upstreamCalls atomic.Int32
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				upstreamCalls.Add(1)
				w.WriteHeader(http.StatusInternalServerError)
			}))
			defer upstream.Close()

			providerID := providerType + "-primary"
			providerConfig := config.ProviderConfig{Type: providerType, APIKey: "dummy", BaseURL: upstream.URL}
			if providerType == "anthropic" {
				providerConfig.Capabilities.StructuredOutputs = true
			}
			handler := newUpstreamErrorTestHandler(t, map[string]config.ProviderConfig{
				providerID: providerConfig,
			}, config.RouteConfig{
				Name:    "chat",
				Match:   config.MatchConfig{Path: "/v1/chat/completions"},
				Targets: []config.TargetConfig{{Provider: providerID, Model: "model", Weight: 1}},
			}, config.RetryConfig{Enabled: false})

			recorder := httptest.NewRecorder()
			handler.ChatCompletions(recorder, httptest.NewRequest(
				http.MethodPost,
				"/v1/chat/completions",
				bytes.NewBufferString(`{
					"model":"model",
					"messages":[{"role":"user","content":"hi"}],
					"response_format":{"type":"json_schema","json_schema":{
						"name":"wrapper","schema":{"type":"object","title":"schema"},"strict":true
					}}
				}`),
			))

			if recorder.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400; body=%s", recorder.Code, recorder.Body.String())
			}
			var response models.ErrorResponse
			if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
				t.Fatalf("decode error response: %v", err)
			}
			if response.Error.Param == nil || *response.Error.Param != "response_format.json_schema.name" {
				t.Fatalf("error param = %#v, want response_format.json_schema.name", response.Error.Param)
			}
			if calls := upstreamCalls.Load(); calls != 0 {
				t.Fatalf("upstream calls = %d, want 0", calls)
			}
		})
	}
}

func TestCompatibleChatFallbacksDropConflictingSchemaAnnotations(t *testing.T) {
	handler := &Handler{registry: providers.NewRegistry(map[string]config.ProviderConfig{
		"anthropic-backup": {Type: "anthropic", Capabilities: config.ProviderCapabilities{StructuredOutputs: true}},
		"ollama-backup":    {Type: "ollama"},
		"openai-backup":    {Type: "openai"},
	})}
	strict := true
	request := &models.UnifiedRequest{
		SourceRequestType: requestTypeChatCompletions,
		RawJSON: json.RawMessage(`{
			"model":"model","messages":[],
			"response_format":{"type":"json_schema","json_schema":{"name":"wrapper","schema":{"type":"object","title":"schema"},"strict":true}}
		}`),
		ResponseFormat: &models.ResponseFormat{Type: "json_schema", JSONSchema: &models.JSONSchemaResponseFormat{
			Name: "wrapper", Schema: map[string]interface{}{"type": "object", "title": "schema"}, Strict: &strict,
		}},
	}

	got := handler.compatibleChatFallbacks([]routing.Target{
		{Provider: "anthropic-backup", Model: "claude"},
		{Provider: "ollama-backup", Model: "qwen"},
		{Provider: "openai-backup", Model: "gpt"},
	}, request)
	if len(got) != 1 || got[0].Provider != "openai-backup" {
		t.Fatalf("compatible fallbacks = %#v, want only OpenAI", got)
	}
}

func TestChatCompletionsRejectsInvalidAnthropicImageBeforeUpstream(t *testing.T) {
	var upstreamCalls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		upstreamCalls.Add(1)
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer upstream.Close()

	handler := newUpstreamErrorTestHandler(t, map[string]config.ProviderConfig{
		"anthropic-primary": {Type: "anthropic", APIKey: "dummy", BaseURL: upstream.URL},
	}, config.RouteConfig{
		Name:    "chat",
		Match:   config.MatchConfig{Path: "/v1/chat/completions"},
		Targets: []config.TargetConfig{{Provider: "anthropic-primary", Model: "claude", Weight: 1}},
	}, config.RetryConfig{Enabled: false})

	recorder := httptest.NewRecorder()
	handler.ChatCompletions(recorder, httptest.NewRequest(
		http.MethodPost,
		"/v1/chat/completions",
		bytes.NewBufferString(`{
			"model":"claude",
			"messages":[{"role":"user","content":[{"type":"image_url","image_url":true}]}]
		}`),
	))

	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400; body=%s", recorder.Code, recorder.Body.String())
	}
	var response models.ErrorResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode error response: %v", err)
	}
	if response.Error.Param == nil || *response.Error.Param != "messages[0].content[0].image_url" {
		t.Fatalf("error param = %#v, want messages[0].content[0].image_url", response.Error.Param)
	}
	if calls := upstreamCalls.Load(); calls != 0 {
		t.Fatalf("upstream calls = %d, want 0", calls)
	}
}

func TestChatCompletionsRejectsEmptyAnthropicMessageBeforeUpstream(t *testing.T) {
	var upstreamCalls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		upstreamCalls.Add(1)
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer upstream.Close()

	handler := newUpstreamErrorTestHandler(t, map[string]config.ProviderConfig{
		"anthropic-primary": {Type: "anthropic", APIKey: "dummy", BaseURL: upstream.URL},
	}, config.RouteConfig{
		Name:    "chat",
		Match:   config.MatchConfig{Path: "/v1/chat/completions"},
		Targets: []config.TargetConfig{{Provider: "anthropic-primary", Model: "claude", Weight: 1}},
	}, config.RetryConfig{Enabled: false})

	recorder := httptest.NewRecorder()
	handler.ChatCompletions(recorder, httptest.NewRequest(
		http.MethodPost,
		"/v1/chat/completions",
		bytes.NewBufferString(`{"model":"claude","messages":[{"role":"user","content":null}]}`),
	))

	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400; body=%s", recorder.Code, recorder.Body.String())
	}
	var response models.ErrorResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode error response: %v", err)
	}
	if response.Error.Param == nil || *response.Error.Param != "messages[0].content" {
		t.Fatalf("error param = %#v, want messages[0].content", response.Error.Param)
	}
	if calls := upstreamCalls.Load(); calls != 0 {
		t.Fatalf("upstream calls = %d, want 0", calls)
	}
}

func TestCompatibleChatFallbacksDropAnthropicForEmptyMessage(t *testing.T) {
	handler := &Handler{registry: providers.NewRegistry(map[string]config.ProviderConfig{
		"anthropic-backup": {Type: "anthropic"},
		"ollama-backup":    {Type: "ollama"},
		"openai-backup":    {Type: "openai"},
	})}
	request := &models.UnifiedRequest{
		SourceRequestType: requestTypeChatCompletions,
		RawJSON:           json.RawMessage(`{"model":"model","messages":[{"role":"user","content":null}]}`),
		Messages:          []models.Message{{Role: "user"}},
	}

	got := handler.compatibleChatFallbacks([]routing.Target{
		{Provider: "anthropic-backup", Model: "claude"},
		{Provider: "ollama-backup", Model: "qwen"},
		{Provider: "openai-backup", Model: "gpt"},
	}, request)
	if len(got) != 2 || got[0].Provider != "ollama-backup" || got[1].Provider != "openai-backup" {
		t.Fatalf("compatible fallbacks = %#v, want Ollama and OpenAI", got)
	}
}
