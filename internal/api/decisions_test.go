package api

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/middleware"
	"github.com/lunargate-ai/gateway/internal/observability"
	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/internal/resilience"
	"github.com/lunargate-ai/gateway/internal/routing"
	"github.com/prometheus/client_golang/prometheus"
)

const apiDecisionBody = `{"model":"lunargate/auto","input":"billing","questions":[{"type":"predicate","name":"billing","instructions":"Is this about billing?"}]}`
const apiLunaResponse = `{"model":"gpt-6-luna","answers":[{"type":"predicate","name":"billing","probability":0.9}],"usage":{"input_tokens":10,"output_tokens":0,"total_tokens":10},"additive":true}`

func newDecisionsTestHandler(t *testing.T, configs map[string]config.ProviderConfig, targets, fallbacks []config.TargetConfig, cacheEnabled bool) (*Handler, *resilience.CircuitBreakerManager) {
	t.Helper()
	reg := providers.NewRegistry(configs)
	routes := routing.NewEngine(config.RoutingConfig{DefaultStrategy: "round-robin", Routes: []config.RouteConfig{{Name: "decisions", Match: config.MatchConfig{Path: "*"}, Targets: targets, Fallback: fallbacks}}})
	cache := middleware.NewCache(config.CacheConfig{Enabled: cacheEnabled, TTL: time.Minute, MaxSize: 100})
	t.Cleanup(cache.Stop)
	breakers := resilience.NewCircuitBreakerManager()
	h := NewHandler(reg, routes, resilience.NewFallbackExecutor(resilience.NewRetrier(config.RetryConfig{Enabled: false}), breakers), cache, nil, observability.NewMetricsWithRegisterer(prometheus.NewRegistry()), nil, nil, nil)
	return h, breakers
}
func decisionAPIRequest(h *Handler, body string) *httptest.ResponseRecorder {
	w := httptest.NewRecorder()
	h.withRuntime((*Handler).Decisions)(w, httptest.NewRequest(http.MethodPost, "/v1/decisions", strings.NewReader(body)))
	return w
}

func TestDecisionsFiltersBeforeSelectionAndKeepsNativeCacheEnvelope(t *testing.T) {
	var calls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		if r.URL.Path != "/decisions" {
			t.Errorf("wrong native endpoint: %s", r.URL.Path)
		}
		_, _ = io.WriteString(w, apiLunaResponse)
	}))
	defer upstream.Close()
	configs := map[string]config.ProviderConfig{"openai": {Type: "openai", BaseURL: upstream.URL, Capabilities: config.ProviderCapabilities{Decisions: true, DecisionModels: []string{"gpt-6-luna"}}}}
	h, _ := newDecisionsTestHandler(t, configs, []config.TargetConfig{{Provider: "openai", Model: "gpt-4o", Weight: 1}, {Provider: "openai", Model: "gpt-6-luna", Weight: 1}}, nil, true)
	for i := 0; i < 10; i++ {
		w := decisionAPIRequest(h, apiDecisionBody)
		if w.Code != 200 || !bytes.Contains(w.Body.Bytes(), []byte(`"additive":true`)) {
			t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
		}
		if w.Header().Get("X-LunarGate-Model") != "openai/gpt-6-luna" {
			t.Fatalf("incorrect effective model: %v", w.Header())
		}
	}
	if calls.Load() != 1 {
		t.Fatalf("cache/native routing calls=%d", calls.Load())
	}
	w := decisionAPIRequest(h, strings.Replace(apiDecisionBody, `"lunargate/auto"`, `"openai/gpt-4o"`, 1))
	if w.Code != 400 || calls.Load() != 1 {
		t.Fatalf("unsupported model made an upstream call: %d %s", w.Code, w.Body.String())
	}
}

func TestDecisionsNativeCrossProviderFallbackAndAliasCacheBypass(t *testing.T) {
	var luna, jev atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/decisions" {
			luna.Add(1)
			w.WriteHeader(503)
			_, _ = io.WriteString(w, `{"error":{"message":"temporarily unavailable","type":"upstream_error"}}`)
			return
		}
		if r.URL.Path != "/systemone" {
			t.Errorf("emulated or wrong endpoint: %s", r.URL.Path)
		}
		jev.Add(1)
		_, _ = io.WriteString(w, `{"model":"jev-1.14.0","answers":{"q0":{"type":"noul","noul":0.8}},"usage":{"input_tokens":12,"output_tokens":1}}`)
	}))
	defer upstream.Close()
	configs := map[string]config.ProviderConfig{
		"openai":   {Type: "openai", BaseURL: upstream.URL, Capabilities: config.ProviderCapabilities{Decisions: true, DecisionModels: []string{"gpt-6-luna"}}},
		"typesafe": {Type: "typesafe", BaseURL: upstream.URL, Capabilities: config.ProviderCapabilities{Decisions: true, DecisionModels: []string{"jev-latest"}}},
	}
	h, _ := newDecisionsTestHandler(t, configs, []config.TargetConfig{{Provider: "openai", Model: "gpt-6-luna", Weight: 1}}, []config.TargetConfig{{Provider: "typesafe", Model: "jev-latest", Weight: 1}, {Provider: "openai", Model: "gpt-4o", Weight: 1}}, true)
	for i := 0; i < 2; i++ {
		w := decisionAPIRequest(h, apiDecisionBody)
		if w.Code != 200 || w.Header().Get("X-LunarGate-Provider") != "typesafe" || w.Header().Get("X-LunarGate-Upstream-Request-Type") != "systemone" || !bytes.Contains(w.Body.Bytes(), []byte(`"model":"jev-1.14.0"`)) {
			t.Fatalf("native fallback failed: %d %s %v", w.Code, w.Body.String(), w.Header())
		}
	}
	if luna.Load() != 2 || jev.Load() != 2 {
		t.Fatalf("fallback poisoned primary cache: luna=%d jev=%d", luna.Load(), jev.Load())
	}
	h, _ = newDecisionsTestHandler(t, configs, []config.TargetConfig{{Provider: "typesafe", Model: "jev-latest", Weight: 1}}, nil, true)
	for i := 0; i < 2; i++ {
		if w := decisionAPIRequest(h, apiDecisionBody); w.Code != 200 {
			t.Fatal(w.Body.String())
		}
	}
	if jev.Load() != 4 {
		t.Fatal("mutable Jev alias was cached")
	}
}

func TestTypeSafeGenerativeRequestsFailWithoutIOOrBreakerFailures(t *testing.T) {
	var calls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { calls.Add(1); w.WriteHeader(500) }))
	defer upstream.Close()
	configs := map[string]config.ProviderConfig{"typesafe": {Type: "typesafe", BaseURL: upstream.URL, DefaultModel: "jev-latest", Capabilities: config.ProviderCapabilities{Decisions: true, DecisionModels: []string{"jev-latest"}, Conversations: true, ResponsesLifecycle: true, ResponseCancellation: true, ResponseCompaction: true, ResponseInputTokens: true}}}
	h, breakers := newDecisionsTestHandler(t, configs, []config.TargetConfig{{Provider: "typesafe", Model: "jev-latest", Weight: 1}}, nil, false)
	for _, test := range []struct {
		path, body string
		handler    func(http.ResponseWriter, *http.Request)
	}{
		{"/v1/chat/completions", `{"model":"typesafe/jev-latest","messages":[{"role":"user","content":"hello"}]}`, h.ChatCompletions},
		{"/v1/responses", `{"model":"typesafe/jev-latest","input":"hello"}`, h.Responses},
		{"/v1/embeddings", `{"model":"typesafe/jev-latest","input":"hello"}`, h.Embeddings},
		{"/v1/conversations", `{}`, h.CreateConversation},
		{"/v1/responses/input_tokens", `{"model":"typesafe/jev-latest","input":"hello"}`, h.CountResponseInputTokens},
	} {
		w := httptest.NewRecorder()
		request := httptest.NewRequest(http.MethodPost, test.path, strings.NewReader(test.body))
		request.Header.Set("X-LunarGate-Provider", "typesafe")
		test.handler(w, request)
		if w.Code != 400 {
			t.Fatalf("%s returned %d %s", test.path, w.Code, w.Body.String())
		}
	}
	if calls.Load() != 0 || breakers.Get("typesafe").Counts().TotalFailures != 0 {
		t.Fatal("unsupported API reached upstream or breaker")
	}
}

func TestDecisionsDefaultModelAndCapabilitiesReload(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { _, _ = io.WriteString(w, apiLunaResponse) }))
	defer upstream.Close()
	configs := map[string]config.ProviderConfig{"openai": {Type: "openai", BaseURL: upstream.URL, DefaultModel: "gpt-6-luna", Capabilities: config.ProviderCapabilities{Decisions: true, DecisionModels: []string{"gpt-6-luna"}}}}
	h, _ := newDecisionsTestHandler(t, configs, []config.TargetConfig{{Provider: "openai", Weight: 1}}, nil, false)
	if w := decisionAPIRequest(h, apiDecisionBody); w.Code != 200 {
		t.Fatal(w.Body.String())
	}
	if caps := h.RuntimeDecisionCapabilities(); len(caps.Models) != 1 || len(caps.Routes) != 1 {
		t.Fatalf("missing hello capabilities: %#v", caps)
	}
	cfg := configs["openai"]
	cfg.Capabilities.Decisions = false
	configs["openai"] = cfg
	h.UpdateProviderConfigs(configs)
	if w := decisionAPIRequest(h, apiDecisionBody); w.Code != 400 {
		t.Fatalf("reload didn't disable Decisions: %d %s", w.Code, w.Body.String())
	}
	if caps := h.RuntimeDecisionCapabilities(); len(caps.Models) != 0 || len(caps.Routes) != 0 {
		t.Fatal("stale hello capabilities")
	}
}

func TestDecisionsRejectStreamingAndKeepUpstreamErrors(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(429)
		_, _ = io.WriteString(w, `{"error":{"message":"limited","type":"rate_limit_error","future":true}}`)
	}))
	defer upstream.Close()
	h, _ := newDecisionsTestHandler(t, map[string]config.ProviderConfig{"openai": {Type: "openai", BaseURL: upstream.URL, Capabilities: config.ProviderCapabilities{Decisions: true, DecisionModels: []string{"gpt-6-luna"}}}}, []config.TargetConfig{{Provider: "openai", Model: "gpt-6-luna", Weight: 1}}, nil, false)
	if w := decisionAPIRequest(h, strings.Replace(apiDecisionBody, `"input"`, `"stream":true,"input"`, 1)); w.Code != 400 {
		t.Fatal("stream accepted")
	}
	w := decisionAPIRequest(h, apiDecisionBody)
	var body map[string]interface{}
	_ = json.Unmarshal(w.Body.Bytes(), &body)
	if w.Code != 429 || !bytes.Contains(w.Body.Bytes(), []byte(`"future":true`)) {
		t.Fatalf("upstream error was rewritten: %d %s", w.Code, w.Body.String())
	}
	if w.Header().Get("X-LunarGate-Model") != "openai/gpt-6-luna" || w.Header().Get("X-LunarGate-Upstream-Request-Type") != "decisions" {
		t.Fatalf("error lost actual upstream identity: %v", w.Header())
	}
}

func TestDecisionsOnlyPinnedJevModelsUseCache(t *testing.T) {
	h, _ := newDecisionsTestHandler(t, map[string]config.ProviderConfig{"native": {Type: "typesafe"}}, nil, nil, true)
	request := httptest.NewRequest("POST", "/v1/decisions", nil)
	for _, model := range []string{"jev-latest", "jev-preview", "jev-another-alias", "jev-1.14"} {
		if h.decisionCacheAllowed(routing.Target{Provider: "native"}, model, request) {
			t.Fatalf("mutable or unknown alias cached: %s", model)
		}
	}
	if !h.decisionCacheAllowed(routing.Target{Provider: "native"}, "jev-1.14.0", request) {
		t.Fatal("pinned Jev model cannot use exact-match cache")
	}
}

func TestDecisionsDuplicateNamesFilterLunaBeforeWeightedSelection(t *testing.T) {
	var calls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		if r.URL.Path != "/systemone" {
			t.Errorf("duplicate names reached Luna: %s", r.URL.Path)
		}
		_, _ = io.WriteString(w, `{"model":"jev-1.14.0","answers":{"q0":{"type":"noul","noul":0.5},"q1":{"type":"noul","noul":0.4}},"usage":{"input_tokens":12,"output_tokens":1}}`)
	}))
	defer upstream.Close()
	configs := map[string]config.ProviderConfig{
		"openai":   {Type: "openai", BaseURL: upstream.URL, Capabilities: config.ProviderCapabilities{Decisions: true, DecisionModels: []string{"gpt-6-luna"}}},
		"typesafe": {Type: "typesafe", BaseURL: upstream.URL, Capabilities: config.ProviderCapabilities{Decisions: true, DecisionModels: []string{"jev-latest"}}},
	}
	h, _ := newDecisionsTestHandler(t, configs, []config.TargetConfig{{Provider: "openai", Model: "gpt-6-luna", Weight: 100}, {Provider: "typesafe", Model: "jev-latest", Weight: 1}}, nil, false)
	body := `{"model":"lunargate/auto","input":"synthetic","questions":[{"type":"predicate","name":"same","instructions":"A?"},{"type":"predicate","name":"same","instructions":"B?"}]}`
	for i := 0; i < 4; i++ {
		if response := decisionAPIRequest(h, body); response.Code != 200 || response.Header().Get("X-LunarGate-Provider") != "typesafe" {
			t.Fatalf("incorrect duplicate-name routing: %d %s", response.Code, response.Body.String())
		}
	}
	if calls.Load() != 4 {
		t.Fatal("unexpected native call count")
	}
}

func TestDecisionsCollectorPrivacyNativeUsageAndCacheHit(t *testing.T) {
	for _, sharing := range []bool{false, true} {
		t.Run(strconv.FormatBool(sharing), func(t *testing.T) {
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, _ = io.WriteString(w, strings.Replace(apiLunaResponse, `"input_tokens":10`, `"input_tokens":10,"input_tokens_details":{"cached_tokens":4,"cache_write_tokens":2}`, 1))
			}))
			defer upstream.Close()
			capture := newCollectorCapture(t, sharing, sharing)
			configs := map[string]config.ProviderConfig{"openai": {Type: "openai", BaseURL: upstream.URL, Capabilities: config.ProviderCapabilities{Decisions: true, DecisionModels: []string{"gpt-6-luna"}}}}
			h, _ := newDecisionsTestHandler(t, configs, []config.TargetConfig{{Provider: "openai", Model: "gpt-6-luna", Weight: 1}}, nil, true)
			h.collector = capture.client
			for i := 0; i < 2; i++ {
				w := decisionAPIRequest(h, apiDecisionBody)
				if w.Code != 200 {
					t.Fatal(w.Body.String())
				}
				trace, metric, entry := capture.waitForTraceAndMetric(t)
				assertCapturedTraceRequestTypes(t, trace, "decisions", "decisions")
				assertCapturedRequestTypes(t, metric, "decisions", "decisions")
				if metric["tokens_input"] != float64(10) {
					t.Fatalf("wrong usage %#v", metric)
				}
				if i == 0 && (metric["tokens_input_cached"] != float64(4) || metric["tokens_input_cache_write"] != float64(2)) {
					t.Fatalf("lost native cache usage: %#v", metric)
				}
				if metric["cache_hit"] != (i == 1) {
					t.Fatal("wrong cache metadata")
				}
				if !sharing && entry != nil {
					t.Fatal("metrics-only mode exported content")
				}
				if sharing {
					if entry == nil {
						t.Fatal("missing native content log")
					}
					assertCapturedRequestTypes(t, entry, "decisions", "decisions")
					assertCapturedSharedPayloads(t, entry, true, true)
				}
			}
		})
	}
}
