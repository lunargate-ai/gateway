package api

import (
	"bytes"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/resilience"
)

type streamingStatusProtocol struct {
	name        string
	providerCfg config.ProviderConfig
	validStream string
}

func streamingStatusProtocols() []streamingStatusProtocol {
	return []streamingStatusProtocol{
		{
			name:        "openai_sse",
			providerCfg: config.ProviderConfig{Type: "openai", APIKey: "dummy"},
			validStream: "data: {\"id\":\"chatcmpl-fallback\",\"object\":\"chat.completion.chunk\",\"model\":\"fallback-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"fallback\"},\"finish_reason\":\"stop\"}]}\n\n" +
				"data: [DONE]\n\n",
		},
		{
			name:        "anthropic_sse",
			providerCfg: config.ProviderConfig{Type: "anthropic", APIKey: "dummy"},
			validStream: "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg-fallback\",\"model\":\"fallback-model\",\"usage\":{\"input_tokens\":1}}}\n\n" +
				"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"fallback\"}}\n\n" +
				"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n" +
				"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
		},
		{
			name:        "ollama_ndjson",
			providerCfg: config.ProviderConfig{Type: "ollama"},
			validStream: "{\"model\":\"fallback-model\",\"message\":{\"role\":\"assistant\",\"content\":\"fallback\"},\"done\":false}\n" +
				"{\"model\":\"fallback-model\",\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true,\"done_reason\":\"stop\"}\n",
		},
	}
}

func newStreamingStatusHandler(
	t *testing.T,
	protocol streamingStatusProtocol,
	withFallback bool,
	primaryTransport http.RoundTripper,
	fallbackTransport http.RoundTripper,
) (*Handler, *resilience.CircuitBreakerManager) {
	t.Helper()
	primaryCfg := protocol.providerCfg
	primaryCfg.BaseURL = "https://primary.example/v1"
	providerConfigs := map[string]config.ProviderConfig{"primary": primaryCfg}
	var fallbackTargets []config.TargetConfig
	if withFallback {
		fallbackCfg := protocol.providerCfg
		fallbackCfg.BaseURL = "https://fallback.example/v1"
		providerConfigs["fallback"] = fallbackCfg
		fallbackTargets = []config.TargetConfig{{Provider: "fallback", Model: "fallback-model", Weight: 1}}
	}
	handler, breaker, _ := newResilienceClassificationHandler(
		t,
		providerConfigs,
		config.RouteConfig{
			Name:     "stream-status",
			Match:    config.MatchConfig{Path: "/v1/chat/completions"},
			Targets:  []config.TargetConfig{{Provider: "primary", Model: "primary-model", Weight: 1}},
			Fallback: fallbackTargets,
		},
		config.RetryConfig{
			Enabled:         true,
			MaxAttempts:     2,
			InitialDelay:    0,
			MaxDelay:        0,
			Multiplier:      1,
			RetryableErrors: []int{http.StatusTooManyRequests, http.StatusInternalServerError},
		},
	)
	handler.UpdateProviderConfigs(providerConfigs)
	setProviderTransportForTest(t, handler, "primary", primaryTransport)
	if withFallback {
		setProviderTransportForTest(t, handler, "fallback", fallbackTransport)
	}
	return handler, breaker
}

func performStreamingStatusRequest(handler *Handler) *httptest.ResponseRecorder {
	recorder := httptest.NewRecorder()
	handler.ChatCompletions(recorder, httptest.NewRequest(
		http.MethodPost,
		"/v1/chat/completions",
		strings.NewReader(`{"messages":[{"role":"user","content":"hello"}],"stream":true}`),
	))
	return recorder
}

func TestStreamingProviderInvalidSuccessStatusesUseRetryFallbackWithoutRedirect(t *testing.T) {
	statuses := []int{http.StatusCreated, http.StatusNoContent, http.StatusFound}
	for _, protocol := range streamingStatusProtocols() {
		for _, status := range statuses {
			statusName := http.StatusText(status)
			t.Run(protocol.name+"/"+statusName+"/fallback", func(t *testing.T) {
				var primaryCalls, primaryClosed, fallbackCalls, fallbackClosed, redirectCalls atomic.Int32
				primaryTransport := protocolResponseTransport(
					status,
					func() io.ReadCloser { return io.NopCloser(strings.NewReader(protocol.validStream)) },
					&primaryCalls,
					&primaryClosed,
					&redirectCalls,
				)
				fallbackTransport := protocolResponseTransport(
					http.StatusOK,
					func() io.ReadCloser { return io.NopCloser(strings.NewReader(protocol.validStream)) },
					&fallbackCalls,
					&fallbackClosed,
					&redirectCalls,
				)
				handler, breaker := newStreamingStatusHandler(t, protocol, true, primaryTransport, fallbackTransport)
				recorder := performStreamingStatusRequest(handler)

				if recorder.Code != http.StatusOK || !bytes.Contains(recorder.Body.Bytes(), []byte("fallback")) {
					t.Fatalf("fallback response = status %d body %s", recorder.Code, recorder.Body.String())
				}
				if primaryCalls.Load() != 2 || primaryClosed.Load() != 2 {
					t.Fatalf("primary calls/closed = %d/%d, want 2/2", primaryCalls.Load(), primaryClosed.Load())
				}
				if fallbackCalls.Load() != 1 || fallbackClosed.Load() != 1 {
					t.Fatalf("fallback calls/closed = %d/%d, want 1/1", fallbackCalls.Load(), fallbackClosed.Load())
				}
				if redirectCalls.Load() != 0 {
					t.Fatalf("redirect calls = %d, want 0", redirectCalls.Load())
				}
				if failures := breaker.Get("primary").Counts().TotalFailures; failures != 1 {
					t.Fatalf("primary breaker failures = %d, want 1", failures)
				}
			})

			t.Run(protocol.name+"/"+statusName+"/no_fallback", func(t *testing.T) {
				var calls, closed, redirectCalls atomic.Int32
				transport := protocolResponseTransport(
					status,
					func() io.ReadCloser { return io.NopCloser(strings.NewReader(protocol.validStream)) },
					&calls,
					&closed,
					&redirectCalls,
				)
				handler, breaker := newStreamingStatusHandler(t, protocol, false, transport, nil)
				recorder := performStreamingStatusRequest(handler)

				assertProtocolErrorType(t, recorder, "invalid_response_status")
				if calls.Load() != 2 || closed.Load() != 2 {
					t.Fatalf("calls/closed = %d/%d, want 2/2", calls.Load(), closed.Load())
				}
				if redirectCalls.Load() != 0 {
					t.Fatalf("redirect calls = %d, want 0", redirectCalls.Load())
				}
				if failures := breaker.Get("primary").Counts().TotalFailures; failures != 1 {
					t.Fatalf("primary breaker failures = %d, want 1", failures)
				}
			})
		}
	}
}
