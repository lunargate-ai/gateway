package api

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
)

func TestResponsesCompactSelectsCanonicalProviderWithoutFallback(t *testing.T) {
	var wrongCalls atomic.Int32
	wrong := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		wrongCalls.Add(1)
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer wrong.Close()

	var selectedCalls atomic.Int32
	var selectedBody []byte
	selected := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		selectedCalls.Add(1)
		selectedBody, _ = io.ReadAll(r.Body)
		if r.Method != http.MethodPost || r.URL.Path != "/v1/responses/compact" {
			t.Errorf("request = %s %s", r.Method, r.URL.Path)
		}
		if r.URL.RawQuery != "future=true" {
			t.Errorf("raw query = %q", r.URL.RawQuery)
		}
		if r.Header.Get("Authorization") != "Bearer selected-secret" {
			t.Errorf("Authorization = %q", r.Header.Get("Authorization"))
		}
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-OpenAI-Request-ID", "req_compact")
		w.Header().Set("Set-Cookie", "do-not-forward=true")
		w.WriteHeader(http.StatusMultiStatus)
		_, _ = io.WriteString(w, `{"id":"resp_compacted","object":"response.compaction","future_response":{"kept":true}}`)
	}))
	defer selected.Close()

	router, _, cache := newNativeLifecycleRouterFromConfigs(t, map[string]config.ProviderConfig{
		"wrong": {
			Type: "openai", APIKey: "wrong-secret", BaseURL: wrong.URL + "/v1", DefaultModel: "gpt-wrong",
			Capabilities: config.ProviderCapabilities{ResponseCompaction: true},
		},
		"selected": {
			Type: "openai", APIKey: "selected-secret", BaseURL: selected.URL + "/v1", DefaultModel: "gpt-selected",
			Capabilities: config.ProviderCapabilities{ResponseCompaction: true},
		},
	})
	defer cache.Stop()

	body := []byte(`{"model":"selected/gpt-selected","input":[{"role":"user","content":"hello","future_item":9007199254740993}],"future_request":{"enabled":true}}`)
	request := httptest.NewRequest(http.MethodPost, "/v1/responses/compact?future=true", bytes.NewReader(body))
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	router.ServeHTTP(response, request)

	if response.Code != http.StatusMultiStatus {
		t.Fatalf("status = %d, want 207; body=%s", response.Code, response.Body.String())
	}
	if got := selectedCalls.Load(); got != 1 {
		t.Fatalf("selected provider calls = %d, want one", got)
	}
	if got := wrongCalls.Load(); got != 0 {
		t.Fatalf("wrong/fallback provider calls = %d, want zero", got)
	}
	var upstream map[string]json.RawMessage
	if err := json.Unmarshal(selectedBody, &upstream); err != nil {
		t.Fatalf("decode upstream body: %v", err)
	}
	if got := parseJSONStringRaw(upstream["model"]); got != "gpt-selected" {
		t.Fatalf("upstream model = %q", got)
	}
	if !bytes.Contains(upstream["input"], []byte("9007199254740993")) || len(upstream["future_request"]) == 0 {
		t.Fatalf("additive request fields changed: %s", selectedBody)
	}
	if got := response.Body.String(); got != `{"id":"resp_compacted","object":"response.compaction","future_response":{"kept":true}}` {
		t.Fatalf("raw response changed: %q", got)
	}
	if response.Header().Get("X-LunarGate-Provider") != "selected" || response.Header().Get("X-OpenAI-Request-ID") != "req_compact" {
		t.Fatalf("response headers = %#v", response.Header())
	}
	if got := response.Header().Values("Set-Cookie"); len(got) != 0 {
		t.Fatalf("unsafe cookie leaked: %q", got)
	}
}

func TestResponsesInputTokensUsesOnlyCapableProviderAndPreservesError(t *testing.T) {
	var calls atomic.Int32
	var upstreamBody string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		body, _ := io.ReadAll(r.Body)
		upstreamBody = string(body)
		if r.URL.Path != "/v1/responses/input_tokens" {
			t.Errorf("path = %q", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Retry-After", "9")
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = io.WriteString(w, `{"error":{"message":"temporarily unavailable","type":"server_error","code":"overloaded"},"future_error":true}`)
	}))
	defer upstream.Close()

	router, _, cache := newNativeLifecycleRouterFromConfigs(t, map[string]config.ProviderConfig{
		"native": {
			Type: "openai", APIKey: "secret", BaseURL: upstream.URL + "/v1", DefaultModel: "gpt-native",
			Capabilities: config.ProviderCapabilities{ResponseInputTokens: true},
		},
		"disabled": {
			Type: "openai", APIKey: "disabled", BaseURL: upstream.URL + "/disabled", DefaultModel: "gpt-disabled",
		},
	})
	defer cache.Stop()

	body := ` { "model": "gpt-native", "input": "hello", "future": 9007199254740993 } `
	response := performLifecycleRequest(t, router, http.MethodPost, "/v1/responses/input_tokens", []byte(body))
	if response.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503; body=%s", response.Code, response.Body.String())
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("upstream calls = %d, want exactly one", got)
	}
	if upstreamBody != body {
		t.Fatalf("request body changed\n got: %q\nwant: %q", upstreamBody, body)
	}
	if !strings.Contains(response.Body.String(), `"future_error":true`) {
		t.Fatalf("raw upstream error changed: %s", response.Body.String())
	}
	if response.Header().Get("Retry-After") != "9" {
		t.Fatalf("Retry-After = %q", response.Header().Get("Retry-After"))
	}
}

func TestResponseOperationsRejectAmbiguousOrUnsupportedProvider(t *testing.T) {
	testCases := []struct {
		name      string
		path      string
		header    string
		providers map[string]config.ProviderCapabilities
		wantCode  string
	}{
		{
			name: "ambiguous compaction", path: "/v1/responses/compact",
			providers: map[string]config.ProviderCapabilities{
				"one": {ResponseCompaction: true},
				"two": {ResponseCompaction: true},
			},
			wantCode: "ambiguous_provider",
		},
		{
			name: "explicit unsupported provider", path: "/v1/responses/input_tokens", header: "disabled",
			providers: map[string]config.ProviderCapabilities{
				"disabled": {},
			},
			wantCode: "unsupported_feature",
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			var calls atomic.Int32
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				calls.Add(1)
			}))
			defer upstream.Close()
			router, _, cache := newNativeLifecycleRouter(t, upstream.URL+"/v1", testCase.providers)
			defer cache.Stop()
			request := httptest.NewRequest(http.MethodPost, testCase.path, strings.NewReader(`{"model":"gpt-native","input":"hello"}`))
			if testCase.header != "" {
				request.Header.Set("X-LunarGate-Provider", testCase.header)
			}
			response := httptest.NewRecorder()
			router.ServeHTTP(response, request)
			if response.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400; body=%s", response.Code, response.Body.String())
			}
			assertLifecycleError(t, response.Body.Bytes(), "provider", testCase.wantCode)
			if got := calls.Load(); got != 0 {
				t.Fatalf("upstream calls = %d, want zero", got)
			}
		})
	}
}

func TestResponseOperationBodyLimit(t *testing.T) {
	var calls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
	}))
	defer upstream.Close()
	router, _, cache := newNativeLifecycleRouter(t, upstream.URL+"/v1", map[string]config.ProviderCapabilities{
		"native": {ResponseCompaction: true},
	})
	defer cache.Stop()

	oversized := `{"input":"` + strings.Repeat("x", int(maxRequestBodyBytes)) + `"}`
	response := performLifecycleRequest(t, router, http.MethodPost, "/v1/responses/compact", []byte(oversized))
	if response.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("status = %d, want 413; body=%s", response.Code, response.Body.String())
	}
	if got := calls.Load(); got != 0 {
		t.Fatalf("upstream calls = %d, want zero", got)
	}
}
