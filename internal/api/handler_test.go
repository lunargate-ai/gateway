package api

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/middleware"
	"github.com/lunargate-ai/gateway/internal/observability"
	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/internal/resilience"
	"github.com/lunargate-ai/gateway/internal/routing"
	"github.com/lunargate-ai/gateway/internal/streaming"
	"github.com/lunargate-ai/gateway/pkg/models"

	"github.com/prometheus/client_golang/prometheus"
)

func TestChatCompletions_RequestBodyTooLarge_Returns413(t *testing.T) {
	reg := providers.NewRegistry(map[string]config.ProviderConfig{})
	router := routing.NewEngine(config.RoutingConfig{DefaultStrategy: "weighted", Routes: []config.RouteConfig{}})
	retrier := resilience.NewRetrier(config.RetryConfig{Enabled: false})
	cbm := resilience.NewCircuitBreakerManager()
	fb := resilience.NewFallbackExecutor(retrier, cbm)
	cache := middleware.NewCache(config.CacheConfig{Enabled: false})
	streamer := streaming.NewHandler()
	metrics := observability.NewMetricsWithRegisterer(prometheus.NewRegistry())
	h := NewHandler(reg, router, fb, cache, streamer, metrics, nil, nil, nil)

	bigContent := string(bytes.Repeat([]byte("a"), (10<<20)+1024))
	payload := []byte(`{"model":"mock-gpt","messages":[{"role":"user","content":"` + bigContent + `"}]}`)
	req := httptest.NewRequest(http.MethodPost, "http://example.com/v1/chat/completions", bytes.NewReader(payload))
	rec := httptest.NewRecorder()

	h.ChatCompletions(rec, req)

	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("expected status %d, got %d", http.StatusRequestEntityTooLarge, rec.Code)
	}
}

func TestChatCompletions_ProviderErrorPassthrough(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":{"message":"bad key","type":"invalid_api_key"}}`))
	}))
	defer upstream.Close()

	providerID := "openai"
	cfgProviders := map[string]config.ProviderConfig{
		providerID: {Type: "openai", APIKey: "dummy", BaseURL: upstream.URL},
	}
	reg := providers.NewRegistry(cfgProviders)

	router := routing.NewEngine(config.RoutingConfig{
		DefaultStrategy: "weighted",
		Routes: []config.RouteConfig{
			{
				Name:    "default",
				Match:   config.MatchConfig{Path: "*"},
				Targets: []config.TargetConfig{{Provider: providerID, Model: "gpt-4-turbo", Weight: 1}},
			},
		},
	})

	retrier := resilience.NewRetrier(config.RetryConfig{Enabled: true, MaxAttempts: 1, RetryableErrors: []int{429, 500, 502, 503, 504}})
	cbm := resilience.NewCircuitBreakerManager()
	fb := resilience.NewFallbackExecutor(retrier, cbm)
	cache := middleware.NewCache(config.CacheConfig{Enabled: false})
	streamer := streaming.NewHandler()
	metrics := observability.NewMetricsWithRegisterer(prometheus.NewRegistry())
	h := NewHandler(reg, router, fb, cache, streamer, metrics, nil, nil, nil)

	payload := models.UnifiedRequest{Model: "gpt-4-turbo", Messages: []models.Message{{Role: "user", Content: "hi"}}}
	b, _ := json.Marshal(payload)
	req := httptest.NewRequest(http.MethodPost, "http://example.com/v1/chat/completions", bytes.NewReader(b))
	rec := httptest.NewRecorder()

	h.ChatCompletions(rec, req)

	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("expected status %d, got %d", http.StatusUnauthorized, rec.Code)
	}

	var resp models.ErrorResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to unmarshal response: %v", err)
	}
	if resp.Error.Type != "invalid_api_key" {
		t.Fatalf("expected error type %q, got %q", "invalid_api_key", resp.Error.Type)
	}
	if resp.Error.Message != "bad key" {
		t.Fatalf("expected error message %q, got %q", "bad key", resp.Error.Message)
	}
}

func TestChatCompletions_SetsTimingHeaders(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"mock-gpt","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	defer upstream.Close()

	providerID := "openai"
	cfgProviders := map[string]config.ProviderConfig{
		providerID: {Type: "openai", APIKey: "dummy", BaseURL: upstream.URL},
	}
	reg := providers.NewRegistry(cfgProviders)

	router := routing.NewEngine(config.RoutingConfig{
		DefaultStrategy: "weighted",
		Routes: []config.RouteConfig{
			{
				Name:    "default",
				Match:   config.MatchConfig{Path: "*"},
				Targets: []config.TargetConfig{{Provider: providerID, Model: "mock-gpt", Weight: 1}},
			},
		},
	})

	retrier := resilience.NewRetrier(config.RetryConfig{Enabled: false})
	cbm := resilience.NewCircuitBreakerManager()
	fb := resilience.NewFallbackExecutor(retrier, cbm)
	cache := middleware.NewCache(config.CacheConfig{Enabled: false})
	streamer := streaming.NewHandler()
	metrics := observability.NewMetricsWithRegisterer(prometheus.NewRegistry())
	h := NewHandler(reg, router, fb, cache, streamer, metrics, nil, nil, nil)

	payload := models.UnifiedRequest{Model: "mock-gpt", Messages: []models.Message{{Role: "user", Content: "hi"}}}
	b, _ := json.Marshal(payload)
	req := httptest.NewRequest(http.MethodPost, "http://example.com/v1/chat/completions", bytes.NewReader(b))
	rec := httptest.NewRecorder()

	h.ChatCompletions(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}
	if rec.Header().Get("X-LunarGate-Overhead-Duration-Ms") == "" {
		t.Fatalf("expected X-LunarGate-Overhead-Duration-Ms header to be set")
	}
	if rec.Header().Get("X-LunarGate-Latency-Ms") == "" {
		t.Fatalf("expected X-LunarGate-Latency-Ms header to be set")
	}
}

func TestResponses_MapsToChatCompletions(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"mock-gpt","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`))
	}))
	defer upstream.Close()

	providerID := "openai"
	cfgProviders := map[string]config.ProviderConfig{
		providerID: {Type: "openai", APIKey: "dummy", BaseURL: upstream.URL},
	}
	reg := providers.NewRegistry(cfgProviders)

	router := routing.NewEngine(config.RoutingConfig{
		DefaultStrategy: "weighted",
		Routes: []config.RouteConfig{
			{
				Name:    "default",
				Match:   config.MatchConfig{Path: "*"},
				Targets: []config.TargetConfig{{Provider: providerID, Model: "mock-gpt", Weight: 1}},
			},
		},
	})

	retrier := resilience.NewRetrier(config.RetryConfig{Enabled: false})
	cbm := resilience.NewCircuitBreakerManager()
	fb := resilience.NewFallbackExecutor(retrier, cbm)
	cache := middleware.NewCache(config.CacheConfig{Enabled: false})
	streamer := streaming.NewHandler()
	metrics := observability.NewMetricsWithRegisterer(prometheus.NewRegistry())
	h := NewHandler(reg, router, fb, cache, streamer, metrics, nil, nil, nil)

	payload := []byte(`{"model":"lunargate/auto","input":[{"role":"user","content":[{"type":"input_text","text":"Say hi"}]}]}`)
	req := httptest.NewRequest(http.MethodPost, "http://example.com/v1/responses", bytes.NewReader(payload))
	rec := httptest.NewRecorder()

	h.Responses(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}

	var out models.ResponsesResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("failed to unmarshal responses payload: %v", err)
	}
	if out.Object != "response" {
		t.Fatalf("expected response object, got %q", out.Object)
	}
	if out.OutputText != "ok" {
		t.Fatalf("expected output_text %q, got %q", "ok", out.OutputText)
	}
	if rec.Header().Get("X-LunarGate-Provider") == "" {
		t.Fatalf("expected X-LunarGate-Provider header to be set")
	}
}

func TestResponses_RoutesViaChatPathWhenOnlyChatRouteConfigured(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"mock-gpt","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	defer upstream.Close()

	providerID := "openai"
	cfgProviders := map[string]config.ProviderConfig{
		providerID: {Type: "openai", APIKey: "dummy", BaseURL: upstream.URL},
	}
	reg := providers.NewRegistry(cfgProviders)

	router := routing.NewEngine(config.RoutingConfig{
		DefaultStrategy: "weighted",
		Routes: []config.RouteConfig{
			{
				Name:    "chat-only-route",
				Match:   config.MatchConfig{Path: "/v1/chat/completions"},
				Targets: []config.TargetConfig{{Provider: providerID, Model: "mock-gpt", Weight: 1}},
			},
		},
	})

	retrier := resilience.NewRetrier(config.RetryConfig{Enabled: false})
	cbm := resilience.NewCircuitBreakerManager()
	fb := resilience.NewFallbackExecutor(retrier, cbm)
	cache := middleware.NewCache(config.CacheConfig{Enabled: false})
	streamer := streaming.NewHandler()
	metrics := observability.NewMetricsWithRegisterer(prometheus.NewRegistry())
	h := NewHandler(reg, router, fb, cache, streamer, metrics, nil, nil, nil)

	payload := []byte(`{"model":"lunargate/auto","input":[{"role":"user","content":[{"type":"input_text","text":"Say hi"}]}]}`)
	req := httptest.NewRequest(http.MethodPost, "http://example.com/v1/responses", bytes.NewReader(payload))
	rec := httptest.NewRecorder()

	h.Responses(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}
	if got := rec.Header().Get("X-LunarGate-Route"); got != "chat-only-route" {
		t.Fatalf("expected routed via chat-only-route, got %q", got)
	}

	var out models.ResponsesResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("failed to unmarshal responses payload: %v", err)
	}
	if out.OutputText != "ok" {
		t.Fatalf("expected output_text %q, got %q", "ok", out.OutputText)
	}
}

func TestResponses_RoutesViaResponsesPathWhenResponsesRouteConfigured(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-2","object":"chat.completion","created":1,"model":"mock-gpt","choices":[{"index":0,"message":{"role":"assistant","content":"ok-responses-route"},"finish_reason":"stop"}]}`))
	}))
	defer upstream.Close()

	providerID := "openai"
	cfgProviders := map[string]config.ProviderConfig{
		providerID: {Type: "openai", APIKey: "dummy", BaseURL: upstream.URL},
	}
	reg := providers.NewRegistry(cfgProviders)

	router := routing.NewEngine(config.RoutingConfig{
		DefaultStrategy: "weighted",
		Routes: []config.RouteConfig{
			{
				Name:    "responses-only-route",
				Match:   config.MatchConfig{Path: "/v1/responses"},
				Targets: []config.TargetConfig{{Provider: providerID, Model: "mock-gpt", Weight: 1}},
			},
		},
	})

	retrier := resilience.NewRetrier(config.RetryConfig{Enabled: false})
	cbm := resilience.NewCircuitBreakerManager()
	fb := resilience.NewFallbackExecutor(retrier, cbm)
	cache := middleware.NewCache(config.CacheConfig{Enabled: false})
	streamer := streaming.NewHandler()
	metrics := observability.NewMetricsWithRegisterer(prometheus.NewRegistry())
	h := NewHandler(reg, router, fb, cache, streamer, metrics, nil, nil, nil)

	payload := []byte(`{"model":"lunargate/auto","input":[{"role":"user","content":[{"type":"input_text","text":"Say hi"}]}]}`)
	req := httptest.NewRequest(http.MethodPost, "http://example.com/v1/responses", bytes.NewReader(payload))
	rec := httptest.NewRecorder()

	h.Responses(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}
	if got := rec.Header().Get("X-LunarGate-Route"); got != "responses-only-route" {
		t.Fatalf("expected routed via responses-only-route, got %q", got)
	}

	var out models.ResponsesResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("failed to unmarshal responses payload: %v", err)
	}
	if out.OutputText != "ok-responses-route" {
		t.Fatalf("expected output_text %q, got %q", "ok-responses-route", out.OutputText)
	}
}

func TestResponses_StreamPassthrough(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"mock-gpt\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer upstream.Close()

	providerID := "openai"
	cfgProviders := map[string]config.ProviderConfig{
		providerID: {Type: "openai", APIKey: "dummy", BaseURL: upstream.URL},
	}
	reg := providers.NewRegistry(cfgProviders)

	router := routing.NewEngine(config.RoutingConfig{
		DefaultStrategy: "weighted",
		Routes: []config.RouteConfig{
			{
				Name:    "responses-default",
				Match:   config.MatchConfig{Path: "/v1/responses"},
				Targets: []config.TargetConfig{{Provider: providerID, Model: "mock-gpt", Weight: 1}},
			},
		},
	})

	retrier := resilience.NewRetrier(config.RetryConfig{Enabled: false})
	cbm := resilience.NewCircuitBreakerManager()
	fb := resilience.NewFallbackExecutor(retrier, cbm)
	cache := middleware.NewCache(config.CacheConfig{Enabled: false})
	streamer := streaming.NewHandler()
	metrics := observability.NewMetricsWithRegisterer(prometheus.NewRegistry())
	h := NewHandler(reg, router, fb, cache, streamer, metrics, nil, nil, nil)

	payload := []byte(`{"model":"lunargate/auto","stream":true,"input":[{"role":"user","content":[{"type":"input_text","text":"Say hi"}]}]}`)
	req := httptest.NewRequest(http.MethodPost, "http://example.com/v1/responses", bytes.NewReader(payload))
	rec := httptest.NewRecorder()

	h.Responses(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}
	body := rec.Body.String()
	if !strings.Contains(body, "data: ") {
		t.Fatalf("expected streaming body to contain SSE data, got %q", body)
	}
	if !strings.Contains(body, `"type":"response.completed"`) {
		t.Fatalf("expected responses stream to emit response.completed event, got %q", body)
	}
	if strings.Contains(body, "streaming responses are not supported yet") {
		t.Fatalf("unexpected legacy streaming error in body")
	}
}

func TestResponses_StreamToolCallLifecycle(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		chunk1 := map[string]interface{}{
			"id":      "chatcmpl-tool",
			"object":  "chat.completion.chunk",
			"created": 1,
			"model":   "mock-gpt",
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"delta": map[string]interface{}{
						"tool_calls": []map[string]interface{}{
							{
								"index": 0,
								"id":    "call_1",
								"type":  "function",
								"function": map[string]interface{}{
									"name":      "exec_command",
									"arguments": "{\"cmd\":\"pwd\"",
								},
							},
						},
					},
					"finish_reason": nil,
				},
			},
		}
		chunk1Bytes, _ := json.Marshal(chunk1)
		_, _ = w.Write([]byte("data: " + string(chunk1Bytes) + "\n\n"))

		chunk2 := map[string]interface{}{
			"id":      "chatcmpl-tool",
			"object":  "chat.completion.chunk",
			"created": 1,
			"model":   "mock-gpt",
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"delta": map[string]interface{}{
						"tool_calls": []map[string]interface{}{
							{
								"index": 0,
								"id":    "call_1",
								"type":  "function",
								"function": map[string]interface{}{
									"arguments": "}",
								},
							},
						},
					},
					"finish_reason": "tool_calls",
				},
			},
		}
		chunk2Bytes, _ := json.Marshal(chunk2)
		_, _ = w.Write([]byte("data: " + string(chunk2Bytes) + "\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer upstream.Close()

	providerID := "openai"
	cfgProviders := map[string]config.ProviderConfig{
		providerID: {Type: "openai", APIKey: "dummy", BaseURL: upstream.URL},
	}
	reg := providers.NewRegistry(cfgProviders)

	router := routing.NewEngine(config.RoutingConfig{
		DefaultStrategy: "weighted",
		Routes: []config.RouteConfig{
			{
				Name:    "responses-default",
				Match:   config.MatchConfig{Path: "/v1/responses"},
				Targets: []config.TargetConfig{{Provider: providerID, Model: "mock-gpt", Weight: 1}},
			},
		},
	})

	retrier := resilience.NewRetrier(config.RetryConfig{Enabled: false})
	cbm := resilience.NewCircuitBreakerManager()
	fb := resilience.NewFallbackExecutor(retrier, cbm)
	cache := middleware.NewCache(config.CacheConfig{Enabled: false})
	streamer := streaming.NewHandler()
	metrics := observability.NewMetricsWithRegisterer(prometheus.NewRegistry())
	h := NewHandler(reg, router, fb, cache, streamer, metrics, nil, nil, nil)

	payload := []byte(`{"model":"lunargate/auto","stream":true,"input":[{"role":"user","content":[{"type":"input_text","text":"run pwd"}]}]}`)
	req := httptest.NewRequest(http.MethodPost, "http://example.com/v1/responses", bytes.NewReader(payload))
	rec := httptest.NewRecorder()

	h.Responses(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}
	body := rec.Body.String()
	if !strings.Contains(body, `"type":"response.function_call_arguments.delta"`) {
		t.Fatalf("expected function_call_arguments.delta event, got %q", body)
	}
	if !strings.Contains(body, `"type":"response.function_call_arguments.done"`) {
		t.Fatalf("expected function_call_arguments.done event, got %q", body)
	}
	if !strings.Contains(body, `"type":"response.output_item.done"`) {
		t.Fatalf("expected output_item.done event, got %q", body)
	}
	if !strings.Contains(body, `"type":"response.completed"`) {
		t.Fatalf("expected response.completed event, got %q", body)
	}
}

func TestResponses_StreamMultipleToolCallsLifecycle(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		chunk1 := map[string]interface{}{
			"id":      "chatcmpl-tool-multi",
			"object":  "chat.completion.chunk",
			"created": 1,
			"model":   "mock-gpt",
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"delta": map[string]interface{}{
						"tool_calls": []map[string]interface{}{
							{
								"index": 0,
								"id":    "call_1",
								"type":  "function",
								"function": map[string]interface{}{
									"name":      "exec_a",
									"arguments": "{\"x\":1",
								},
							},
							{
								"index": 1,
								"id":    "call_2",
								"type":  "function",
								"function": map[string]interface{}{
									"name":      "exec_b",
									"arguments": "{\"y\":2",
								},
							},
						},
					},
					"finish_reason": nil,
				},
			},
		}
		chunk1Bytes, _ := json.Marshal(chunk1)
		_, _ = w.Write([]byte("data: " + string(chunk1Bytes) + "\n\n"))

		chunk2 := map[string]interface{}{
			"id":      "chatcmpl-tool-multi",
			"object":  "chat.completion.chunk",
			"created": 1,
			"model":   "mock-gpt",
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"delta": map[string]interface{}{
						"tool_calls": []map[string]interface{}{
							{
								"index": 0,
								"id":    "call_1",
								"type":  "function",
								"function": map[string]interface{}{
									"arguments": "}",
								},
							},
							{
								"index": 1,
								"id":    "call_2",
								"type":  "function",
								"function": map[string]interface{}{
									"arguments": "}",
								},
							},
						},
					},
					"finish_reason": "tool_calls",
				},
			},
		}
		chunk2Bytes, _ := json.Marshal(chunk2)
		_, _ = w.Write([]byte("data: " + string(chunk2Bytes) + "\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer upstream.Close()

	providerID := "openai"
	cfgProviders := map[string]config.ProviderConfig{
		providerID: {Type: "openai", APIKey: "dummy", BaseURL: upstream.URL},
	}
	reg := providers.NewRegistry(cfgProviders)

	router := routing.NewEngine(config.RoutingConfig{
		DefaultStrategy: "weighted",
		Routes: []config.RouteConfig{
			{
				Name:    "responses-default",
				Match:   config.MatchConfig{Path: "/v1/responses"},
				Targets: []config.TargetConfig{{Provider: providerID, Model: "mock-gpt", Weight: 1}},
			},
		},
	})

	retrier := resilience.NewRetrier(config.RetryConfig{Enabled: false})
	cbm := resilience.NewCircuitBreakerManager()
	fb := resilience.NewFallbackExecutor(retrier, cbm)
	cache := middleware.NewCache(config.CacheConfig{Enabled: false})
	streamer := streaming.NewHandler()
	metrics := observability.NewMetricsWithRegisterer(prometheus.NewRegistry())
	h := NewHandler(reg, router, fb, cache, streamer, metrics, nil, nil, nil)

	payload := []byte(`{"model":"lunargate/auto","stream":true,"input":[{"role":"user","content":[{"type":"input_text","text":"run tools"}]}]}`)
	req := httptest.NewRequest(http.MethodPost, "http://example.com/v1/responses", bytes.NewReader(payload))
	rec := httptest.NewRecorder()

	h.Responses(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}
	body := rec.Body.String()
	if strings.Count(body, `"type":"response.function_call_arguments.done"`) != 2 {
		t.Fatalf("expected two function_call_arguments.done events, got %q", body)
	}
	if !strings.Contains(body, `"call_id":"call_1"`) || !strings.Contains(body, `"call_id":"call_2"`) {
		t.Fatalf("expected both call_ids in stream output, got %q", body)
	}
	if !strings.Contains(body, `"type":"response.completed"`) {
		t.Fatalf("expected response.completed event, got %q", body)
	}
}

func TestCopyHeaders_PreservesExistingDestinationHeaders(t *testing.T) {
	dst := http.Header{}
	dst.Set("X-Keep", "keep")

	src := http.Header{}
	src.Set("X-Keep", "replace")
	src.Set("X-New", "new-value")
	src.Set("Content-Length", "123")

	copyHeaders(dst, src)

	if got := dst.Get("X-Keep"); got != "keep" {
		t.Fatalf("expected existing destination header to be preserved, got %q", got)
	}
	if got := dst.Get("X-New"); got != "new-value" {
		t.Fatalf("expected new source header to be copied, got %q", got)
	}
	if got := dst.Get("Content-Length"); got != "" {
		t.Fatalf("expected Content-Length to be skipped, got %q", got)
	}
}

func TestChatCompletions_UpstreamRequestTypeResponses_UsesResponsesEndpoint(t *testing.T) {
	var capturedPath string
	var capturedBody map[string]interface{}
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedPath = r.URL.Path
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("failed to read upstream body: %v", err)
		}
		if err := json.Unmarshal(body, &capturedBody); err != nil {
			t.Fatalf("failed to decode upstream body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"resp_1","object":"response","created_at":1,"status":"completed","model":"gpt-5.3-codex","output":[],"output_text":"ok","usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`))
	}))
	defer upstream.Close()

	providerID := "openai"
	reg := providers.NewRegistry(map[string]config.ProviderConfig{
		providerID: {Type: "openai", APIKey: "dummy", BaseURL: upstream.URL + "/v1"},
	})
	router := routing.NewEngine(config.RoutingConfig{
		DefaultStrategy: "weighted",
		Routes: []config.RouteConfig{{
			Name:  "responses-upstream-route",
			Match: config.MatchConfig{Path: "*"},
			Targets: []config.TargetConfig{{
				Provider:            providerID,
				Model:               "gpt-5.3-codex",
				Weight:              1,
				UpstreamRequestType: "responses",
			}},
		}},
	})
	retrier := resilience.NewRetrier(config.RetryConfig{Enabled: false})
	cbm := resilience.NewCircuitBreakerManager()
	fb := resilience.NewFallbackExecutor(retrier, cbm)
	cache := middleware.NewCache(config.CacheConfig{Enabled: false})
	streamer := streaming.NewHandler()
	metrics := observability.NewMetricsWithRegisterer(prometheus.NewRegistry())
	h := NewHandler(reg, router, fb, cache, streamer, metrics, nil, nil, nil)

	payload := []byte(`{"model":"lunargate/auto","messages":[{"role":"user","content":"hello"}],"tools":[{"type":"function","function":{"name":"exec_command","description":"run command","parameters":{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}}}],"tool_choice":"auto"}`)
	req := httptest.NewRequest(http.MethodPost, "http://example.com/v1/chat/completions", bytes.NewReader(payload))
	rec := httptest.NewRecorder()

	h.ChatCompletions(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, rec.Code)
	}
	if capturedPath != "/v1/responses" {
		t.Fatalf("expected upstream path /v1/responses, got %q", capturedPath)
	}
	if capturedBody == nil {
		t.Fatalf("expected captured upstream payload")
	}
	if _, ok := capturedBody["input"]; !ok {
		t.Fatalf("expected responses upstream payload with input")
	}
	if _, ok := capturedBody["messages"]; ok {
		t.Fatalf("did not expect chat-completions messages field in responses payload")
	}
	if choice, _ := capturedBody["tool_choice"].(string); choice != "auto" {
		t.Fatalf("expected responses upstream tool_choice=auto to be preserved, got %q", choice)
	}
}
