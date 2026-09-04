package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestChatCompletionsPreservesRawOpenAIErrorAndSafeHeaders(t *testing.T) {
	body := []byte(" \n{\"error\":{\"message\":\"bad key\",\"type\":\"invalid_request_error\",\"param\":\"api_key\",\"code\":\"invalid_api_key\",\"future\":7},\"future_top\":true}\n")
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		w.Header().Set("X-OpenAI-Request-ID", "req_upstream")
		w.Header().Set("Retry-After", "7")
		w.Header().Set("Set-Cookie", "session=secret")
		w.Header().Set("Content-Encoding", "identity")
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write(body)
	}))
	defer upstream.Close()

	handler := newUpstreamErrorTestHandler(t, map[string]config.ProviderConfig{
		"openai": {Type: "openai", APIKey: "dummy", BaseURL: upstream.URL},
	}, config.RouteConfig{
		Name:    "chat",
		Match:   config.MatchConfig{Path: "/v1/chat/completions"},
		Targets: []config.TargetConfig{{Provider: "openai", Model: "gpt-test", Weight: 1}},
	}, config.RetryConfig{Enabled: true, MaxAttempts: 2, RetryableErrors: []int{http.StatusTooManyRequests}})

	recorder := httptest.NewRecorder()
	handler.ChatCompletions(recorder, chatErrorRequest(false))

	if recorder.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want 401; body=%s", recorder.Code, recorder.Body.String())
	}
	if !bytes.Equal(recorder.Body.Bytes(), body) {
		t.Fatalf("raw error changed\n got: %q\nwant: %q", recorder.Body.Bytes(), body)
	}
	if got := recorder.Header().Get("X-OpenAI-Request-ID"); got != "req_upstream" {
		t.Fatalf("safe request ID header = %q", got)
	}
	if got := recorder.Header().Get("Retry-After"); got != "7" {
		t.Fatalf("safe retry header = %q", got)
	}
	for _, key := range []string{"Set-Cookie", "Content-Encoding", "Content-Length", "Transfer-Encoding", "Connection"} {
		if got := recorder.Header().Values(key); len(got) != 0 {
			t.Fatalf("unsafe header %s leaked: %v", key, got)
		}
	}
}

func TestChatCompletionsRetryExhaustionPreservesFinalOpenAIError(t *testing.T) {
	for _, status := range []int{http.StatusTooManyRequests, http.StatusServiceUnavailable} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			var calls atomic.Int32
			first := []byte(`{"error":{"message":"first attempt","type":"server_error"},"attempt":1}`)
			final := []byte(fmt.Sprintf(`{"error":{"message":"final attempt","type":"%s","param":null,"code":"final_code"},"attempt":2}`, upstreamErrorTypeForStatus(status)))
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				attempt := calls.Add(1)
				w.Header().Set("Content-Type", "application/json")
				w.Header().Set("X-Upstream-Attempt", fmt.Sprintf("%d", attempt))
				w.WriteHeader(status)
				if attempt == 1 {
					_, _ = w.Write(first)
					return
				}
				_, _ = w.Write(final)
			}))
			defer upstream.Close()

			handler := newUpstreamErrorTestHandler(t, map[string]config.ProviderConfig{
				"openai": {Type: "openai", APIKey: "dummy", BaseURL: upstream.URL},
			}, config.RouteConfig{
				Name:    "chat",
				Match:   config.MatchConfig{Path: "/v1/chat/completions"},
				Targets: []config.TargetConfig{{Provider: "openai", Model: "gpt-test", Weight: 1}},
			}, config.RetryConfig{
				Enabled:         true,
				MaxAttempts:     2,
				Multiplier:      1,
				RetryableErrors: []int{status},
			})

			recorder := httptest.NewRecorder()
			handler.ChatCompletions(recorder, chatErrorRequest(false))

			if calls.Load() != 2 {
				t.Fatalf("upstream calls = %d, want 2", calls.Load())
			}
			if recorder.Code != status {
				t.Fatalf("status = %d, want %d", recorder.Code, status)
			}
			if !bytes.Equal(recorder.Body.Bytes(), final) {
				t.Fatalf("did not preserve only final error\n got: %s\nwant: %s", recorder.Body.Bytes(), final)
			}
			if got := recorder.Header().Get("X-Upstream-Attempt"); got != "2" {
				t.Fatalf("upstream attempt header = %q, want final attempt", got)
			}
		})
	}
}

func TestChatCompletionsFallbackExposesOnlyFinalFailure(t *testing.T) {
	primaryBody := []byte(`{"error":{"message":"primary secret","type":"server_error"}}`)
	primary := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("X-Upstream-Target", "primary")
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write(primaryBody)
	}))
	defer primary.Close()

	finalBody := []byte(`{"error":{"message":"final unavailable","type":"server_error","code":"fallback_final"},"target":"fallback"}`)
	fallback := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Upstream-Target", "fallback")
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write(finalBody)
	}))
	defer fallback.Close()

	handler := newUpstreamErrorTestHandler(t, map[string]config.ProviderConfig{
		"primary":  {Type: "openai", APIKey: "dummy", BaseURL: primary.URL},
		"fallback": {Type: "openai", APIKey: "dummy", BaseURL: fallback.URL},
	}, config.RouteConfig{
		Name:     "chat",
		Match:    config.MatchConfig{Path: "/v1/chat/completions"},
		Targets:  []config.TargetConfig{{Provider: "primary", Model: "gpt-primary", Weight: 1}},
		Fallback: []config.TargetConfig{{Provider: "fallback", Model: "gpt-fallback", Weight: 1}},
	}, config.RetryConfig{
		Enabled:         true,
		MaxAttempts:     1,
		RetryableErrors: []int{http.StatusInternalServerError, http.StatusServiceUnavailable},
	})

	recorder := httptest.NewRecorder()
	handler.ChatCompletions(recorder, chatErrorRequest(false))

	if recorder.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503", recorder.Code)
	}
	if !bytes.Equal(recorder.Body.Bytes(), finalBody) {
		t.Fatalf("wrong fallback error\n got: %s\nwant: %s", recorder.Body.Bytes(), finalBody)
	}
	if bytes.Contains(recorder.Body.Bytes(), []byte("primary secret")) {
		t.Fatal("primary failure leaked into final response")
	}
	if got := recorder.Header().Get("X-Upstream-Target"); got != "fallback" {
		t.Fatalf("upstream target header = %q, want fallback", got)
	}
}

func TestChatCompletionsNormalizesAbacusVendorError(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"success":false,"error":"Invalid model: missing","errorType":"UserFeedbackError"}`))
	}))
	defer upstream.Close()

	handler := newUpstreamErrorTestHandler(t, map[string]config.ProviderConfig{
		"abacus": {Type: "openai", APIKey: "dummy", BaseURL: upstream.URL},
	}, config.RouteConfig{
		Name:    "chat",
		Match:   config.MatchConfig{Path: "/v1/chat/completions"},
		Targets: []config.TargetConfig{{Provider: "abacus", Model: "missing", Weight: 1}},
	}, config.RetryConfig{Enabled: false})

	recorder := httptest.NewRecorder()
	handler.ChatCompletions(recorder, chatErrorRequest(false))

	assertNormalizedUpstreamError(t, recorder, http.StatusBadRequest, "Invalid model: missing", "invalid_request_error", "UserFeedbackError")
	if bytes.Contains(recorder.Body.Bytes(), []byte(`"success"`)) || bytes.Contains(recorder.Body.Bytes(), []byte(`"errorType"`)) {
		t.Fatalf("vendor top-level shape leaked: %s", recorder.Body.String())
	}
}

func TestChatCompletionsTruncatedErrorUsesSyntheticEnvelope(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error":{"message":"`))
		_, _ = w.Write(bytes.Repeat([]byte("s"), upstreamErrorBodyLimit+128))
	}))
	defer upstream.Close()

	handler := newUpstreamErrorTestHandler(t, map[string]config.ProviderConfig{
		"openai": {Type: "openai", APIKey: "dummy", BaseURL: upstream.URL},
	}, config.RouteConfig{
		Name:    "chat",
		Match:   config.MatchConfig{Path: "/v1/chat/completions"},
		Targets: []config.TargetConfig{{Provider: "openai", Model: "gpt-test", Weight: 1}},
	}, config.RetryConfig{Enabled: true, MaxAttempts: 1, RetryableErrors: []int{http.StatusInternalServerError}})

	recorder := httptest.NewRecorder()
	handler.ChatCompletions(recorder, chatErrorRequest(false))

	assertNormalizedUpstreamError(t, recorder, http.StatusInternalServerError, "upstream request failed with status 500", "server_error", "")
	if recorder.Body.Len() > 512 {
		t.Fatalf("synthetic error unexpectedly large: %d bytes", recorder.Body.Len())
	}
}

func TestEmbeddingsPreservesRawOpenAIError(t *testing.T) {
	body := []byte(`{"error":{"message":"bad dimensions","type":"invalid_request_error","param":"dimensions","code":"invalid_dimensions","future":{"x":1}},"request_id":"upstream"}`)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-OpenAI-Request-ID", "embed_req")
		w.Header().Set("Set-Cookie", "secret=value")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write(body)
	}))
	defer upstream.Close()

	handler := newUpstreamErrorTestHandler(t, map[string]config.ProviderConfig{
		"openai": {Type: "openai", APIKey: "dummy", BaseURL: upstream.URL},
	}, config.RouteConfig{
		Name:    "embeddings",
		Match:   config.MatchConfig{Path: "/v1/embeddings"},
		Targets: []config.TargetConfig{{Provider: "openai", Model: "text-embedding-test", Weight: 1}},
	}, config.RetryConfig{Enabled: false})

	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/v1/embeddings", strings.NewReader(`{"model":"openai/text-embedding-test","input":"hello"}`))
	handler.Embeddings(recorder, request)

	if recorder.Code != http.StatusBadRequest || !bytes.Equal(recorder.Body.Bytes(), body) {
		t.Fatalf("embeddings error = %d %q, want exact %q", recorder.Code, recorder.Body.Bytes(), body)
	}
	if recorder.Header().Get("X-OpenAI-Request-ID") != "embed_req" {
		t.Fatalf("safe embeddings header missing: %v", recorder.Header())
	}
	if recorder.Header().Get("Set-Cookie") != "" {
		t.Fatalf("unsafe embeddings cookie leaked: %v", recorder.Header().Values("Set-Cookie"))
	}
}

func TestResponsesPreservesRawNativeOpenAIError(t *testing.T) {
	body := []byte(`{"error":{"message":"response rejected","type":"invalid_request_error","param":"input","code":"bad_input"},"future":true}`)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/responses" {
			t.Errorf("upstream path = %q, want /v1/responses", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write(body)
	}))
	defer upstream.Close()

	handler := newUpstreamErrorTestHandler(t, map[string]config.ProviderConfig{
		"openai": {Type: "openai", APIKey: "dummy", BaseURL: upstream.URL + "/v1"},
	}, config.RouteConfig{
		Name:  "responses",
		Match: config.MatchConfig{Path: "/v1/responses"},
		Targets: []config.TargetConfig{{
			Provider: "openai", Model: "gpt-test", Weight: 1, UpstreamRequestType: "responses",
		}},
	}, config.RetryConfig{Enabled: false})

	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"model":"openai/gpt-test","input":"hello"}`))
	handler.Responses(recorder, request)

	if recorder.Code != http.StatusBadRequest || !bytes.Equal(recorder.Body.Bytes(), body) {
		t.Fatalf("responses error = %d %q, want exact %q", recorder.Code, recorder.Body.Bytes(), body)
	}
}

func TestStreamingErrorBeforeHeadersPreservesRawOpenAIEnvelope(t *testing.T) {
	body := []byte(`{"error":{"message":"stream rejected","type":"invalid_request_error","param":"stream","code":"bad_stream"},"future":true}`)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-OpenAI-Request-ID", "stream_req")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write(body)
	}))
	defer upstream.Close()

	handler := newUpstreamErrorTestHandler(t, map[string]config.ProviderConfig{
		"openai": {Type: "openai", APIKey: "dummy", BaseURL: upstream.URL},
	}, config.RouteConfig{
		Name:    "chat",
		Match:   config.MatchConfig{Path: "/v1/chat/completions"},
		Targets: []config.TargetConfig{{Provider: "openai", Model: "gpt-test", Weight: 1}},
	}, config.RetryConfig{Enabled: false})

	recorder := httptest.NewRecorder()
	handler.ChatCompletions(recorder, chatErrorRequest(true))

	if recorder.Code != http.StatusBadRequest || !bytes.Equal(recorder.Body.Bytes(), body) {
		t.Fatalf("stream error = %d %q, want exact %q", recorder.Code, recorder.Body.Bytes(), body)
	}
	if got := recorder.Header().Get("Content-Type"); got != "application/json" {
		t.Fatalf("content type = %q, want application/json", got)
	}
	if got := recorder.Header().Get("X-OpenAI-Request-ID"); got != "stream_req" {
		t.Fatalf("safe stream header = %q", got)
	}
}

func TestUpstreamHTTPErrorWriteFiltersConnectionNamedHeaders(t *testing.T) {
	failure := newUpstreamHTTPError(http.StatusBadGateway, http.Header{
		"Connection":       []string{"X-Hop-Secret"},
		"X-Hop-Secret":     []string{"do-not-copy"},
		"X-Safe-Upstream":  []string{"copy-me"},
		"Proxy-Connection": []string{"close"},
	}, []byte(`{"error":"vendor"}`), false, "anthropic")
	recorder := httptest.NewRecorder()
	failure.write(recorder)

	if got := recorder.Header().Get("X-Safe-Upstream"); got != "copy-me" {
		t.Fatalf("safe header = %q", got)
	}
	for _, key := range []string{"Connection", "X-Hop-Secret", "Proxy-Connection"} {
		if got := recorder.Header().Values(key); len(got) != 0 {
			t.Fatalf("unsafe header %s leaked: %v", key, got)
		}
	}
}

func newUpstreamErrorTestHandler(
	t *testing.T,
	providers map[string]config.ProviderConfig,
	route config.RouteConfig,
	retry config.RetryConfig,
) *Handler {
	t.Helper()
	handler, _, _ := newResilienceClassificationHandler(t, providers, route, retry)
	return handler
}

func chatErrorRequest(stream bool) *http.Request {
	body := fmt.Sprintf(`{"messages":[{"role":"user","content":"hello"}],"stream":%t}`, stream)
	return httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
}

func assertNormalizedUpstreamError(t *testing.T, recorder *httptest.ResponseRecorder, status int, message, errType, code string) {
	t.Helper()
	if recorder.Code != status {
		t.Fatalf("status = %d, want %d; body=%s", recorder.Code, status, recorder.Body.String())
	}
	var response models.ErrorResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode normalized error: %v; body=%s", err, recorder.Body.String())
	}
	if response.Error.Message != message || response.Error.Type != errType {
		t.Fatalf("error = %#v, want message=%q type=%q", response.Error, message, errType)
	}
	if code == "" {
		if response.Error.Code != nil {
			t.Fatalf("error code = %#v, want nil", response.Error.Code)
		}
	} else if response.Error.Code == nil || *response.Error.Code != code {
		t.Fatalf("error code = %#v, want %q", response.Error.Code, code)
	}
}
