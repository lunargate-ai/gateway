package api

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/health"
	"github.com/lunargate-ai/gateway/internal/middleware"
	"github.com/lunargate-ai/gateway/internal/observability"
	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/internal/resilience"
	"github.com/lunargate-ai/gateway/internal/routing"
	"github.com/lunargate-ai/gateway/internal/streaming"
	"github.com/prometheus/client_golang/prometheus"
)

func TestNativeResponsesLifecycleBindsCreateAndProxiesReads(t *testing.T) {
	var calls atomic.Int32
	var mu sync.Mutex
	seen := make([]string, 0, 3)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		if got := r.Header.Get("Authorization"); got != "Bearer provider-secret" {
			t.Errorf("Authorization = %q", got)
		}
		mu.Lock()
		seen = append(seen, r.Method+" "+r.URL.RequestURI())
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-OpenAI-Request-ID", "req_native_lifecycle")
		w.Header().Set("Set-Cookie", "provider-secret=leak")
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/v1/responses":
			w.WriteHeader(http.StatusOK)
			_, _ = io.WriteString(w, `{"id":"resp_native_lifecycle","object":"response","status":"completed","model":"gpt-native","output":[],"future_field":{"kept":true}}`)
		case r.Method == http.MethodGet && r.URL.Path == "/v1/responses/resp_native_lifecycle":
			w.WriteHeader(http.StatusAccepted)
			_, _ = io.WriteString(w, `{"id":"resp_native_lifecycle","object":"response","status":"completed","future_retrieve":7}`)
		case r.Method == http.MethodGet && r.URL.Path == "/v1/responses/resp_native_lifecycle/input_items":
			_, _ = io.WriteString(w, `{"object":"list","data":[],"has_more":false,"future_list":"kept"}`)
		default:
			t.Errorf("unexpected upstream request %s %s", r.Method, r.URL.RequestURI())
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer upstream.Close()

	router, handler, cache := newNativeLifecycleRouter(t, upstream.URL+"/v1", map[string]config.ProviderCapabilities{
		"native": {ResponsesLifecycle: true},
	})
	defer cache.Stop()

	create := performLifecycleRequest(t, router, http.MethodPost, "/v1/responses", []byte(`{"model":"native/gpt-native","input":"hello"}`))
	if create.Code != http.StatusOK {
		t.Fatalf("create status = %d, want 200; body=%s", create.Code, create.Body.String())
	}
	if _, _, ok := handler.responsesState.getCompleted("resp_native_lifecycle"); ok {
		t.Fatal("native response was retained in the local emulation store")
	}
	binding, ok := handler.responseBindings.get("resp_native_lifecycle")
	if !ok {
		t.Fatal("native response provider binding was not retained")
	}
	if binding.Provider != "native" || binding.Route != "responses" || binding.Model != "native/gpt-native" || binding.UpstreamRequestType != requestTypeResponses {
		t.Fatalf("binding = %#v", binding)
	}

	retrieve := performLifecycleRequest(t, router, http.MethodGet, "/v1/responses/resp_native_lifecycle?include%5B%5D=reasoning.encrypted_content", nil)
	if retrieve.Code != http.StatusAccepted {
		t.Fatalf("retrieve status = %d, want 202; body=%s", retrieve.Code, retrieve.Body.String())
	}
	if got := retrieve.Body.String(); got != `{"id":"resp_native_lifecycle","object":"response","status":"completed","future_retrieve":7}` {
		t.Fatalf("retrieve body changed: %q", got)
	}
	if got := retrieve.Header().Get("X-OpenAI-Request-ID"); got != "req_native_lifecycle" {
		t.Fatalf("safe upstream header = %q", got)
	}
	if got := retrieve.Header().Get("X-LunarGate-Provider"); got != "native" {
		t.Fatalf("provider header = %q", got)
	}
	if got := retrieve.Header().Values("Set-Cookie"); len(got) != 0 {
		t.Fatalf("unsafe upstream cookies leaked: %q", got)
	}

	items := performLifecycleRequest(t, router, http.MethodGet, "/v1/responses/resp_native_lifecycle/input_items?limit=17&order=asc", nil)
	if items.Code != http.StatusOK || !strings.Contains(items.Body.String(), `"future_list":"kept"`) {
		t.Fatalf("input_items = %d %s", items.Code, items.Body.String())
	}

	mu.Lock()
	gotSeen := append([]string(nil), seen...)
	mu.Unlock()
	wantSeen := []string{
		"POST /v1/responses",
		"GET /v1/responses/resp_native_lifecycle?include%5B%5D=reasoning.encrypted_content",
		"GET /v1/responses/resp_native_lifecycle/input_items?limit=17&order=asc",
	}
	if fmt.Sprint(gotSeen) != fmt.Sprint(wantSeen) {
		t.Fatalf("upstream requests = %v, want %v", gotSeen, wantSeen)
	}
	if got := calls.Load(); got != 3 {
		t.Fatalf("upstream calls = %d, want 3", got)
	}
}

func TestNativeResponsesLifecycleRequiresBindingOrExplicitProvider(t *testing.T) {
	var calls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		_, _ = io.WriteString(w, `{"id":"resp_external","object":"response","status":"completed"}`)
	}))
	defer upstream.Close()

	router, _, cache := newNativeLifecycleRouter(t, upstream.URL+"/v1", map[string]config.ProviderCapabilities{
		"native": {ResponsesLifecycle: true},
	})
	defer cache.Stop()

	missing := performLifecycleRequest(t, router, http.MethodGet, "/v1/responses/resp_external", nil)
	if missing.Code != http.StatusNotFound {
		t.Fatalf("missing binding status = %d, want 404; body=%s", missing.Code, missing.Body.String())
	}
	if got := calls.Load(); got != 0 {
		t.Fatalf("unbound request made %d upstream calls", got)
	}

	explicitRequest := httptest.NewRequest(http.MethodGet, "/v1/responses/resp_external", nil)
	explicitRequest.Header.Set("X-LunarGate-Provider", "native")
	explicit := httptest.NewRecorder()
	router.ServeHTTP(explicit, explicitRequest)
	if explicit.Code != http.StatusOK {
		t.Fatalf("explicit provider status = %d, want 200; body=%s", explicit.Code, explicit.Body.String())
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("explicit provider upstream calls = %d, want 1", got)
	}
}

func TestNativeResponsesStreamRetainsNonCompletedOwnerBinding(t *testing.T) {
	for _, status := range []string{"incomplete", "failed"} {
		t.Run(status, func(t *testing.T) {
			responseID := "resp_native_" + status
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = fmt.Fprintf(
					w,
					"event: response.%s\ndata: {\"type\":\"response.%s\",\"response\":{\"id\":%q,\"object\":\"response\",\"status\":%q,\"model\":\"gpt-native\",\"output\":[]}}\n\n",
					status,
					status,
					responseID,
					status,
				)
			}))
			defer upstream.Close()

			router, handler, cache := newNativeLifecycleRouter(t, upstream.URL+"/v1", map[string]config.ProviderCapabilities{
				"native": {ResponsesLifecycle: true},
			})
			defer cache.Stop()

			created := performLifecycleRequest(t, router, http.MethodPost, "/v1/responses", []byte(`{"model":"native/gpt-native","input":"hello","stream":true}`))
			if created.Code != http.StatusOK {
				t.Fatalf("stream create status = %d, want 200; body=%s", created.Code, created.Body.String())
			}
			if _, ok := handler.responseBindings.get(responseID); !ok {
				t.Fatalf("%s native terminal response owner was not retained", status)
			}
			if _, _, ok := handler.responsesState.getCompleted(responseID); ok {
				t.Fatalf("%s native response was put in local completed state", status)
			}
		})
	}
}

func TestNativeResponsesStoreFalseRetainsNoLifecycleState(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"resp_native_stateless","object":"response","status":"completed","model":"gpt-native","output":[]}`)
	}))
	defer upstream.Close()

	router, handler, cache := newNativeLifecycleRouter(t, upstream.URL+"/v1", map[string]config.ProviderCapabilities{
		"native": {ResponsesLifecycle: true},
	})
	defer cache.Stop()
	created := performLifecycleRequest(t, router, http.MethodPost, "/v1/responses", []byte(`{"model":"native/gpt-native","input":"hello","store":false}`))
	if created.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", created.Code, created.Body.String())
	}
	if _, ok := handler.responseBindings.get("resp_native_stateless"); ok {
		t.Fatal("store:false native response retained an owner binding")
	}
	if _, _, ok := handler.responsesState.getCompleted("resp_native_stateless"); ok {
		t.Fatal("store:false native response retained local lifecycle state")
	}
}

func TestNativeResponsesRetrieveStreamsRawSSEAndEscapesResponseID(t *testing.T) {
	const rawStream = ": keepalive\n\nevent: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp space\",\"object\":\"response\",\"status\":\"completed\",\"future_field\":true}}\n\n"
	var escapedPath string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		escapedPath = r.URL.EscapedPath()
		w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
		w.WriteHeader(http.StatusAccepted)
		_, _ = io.WriteString(w, rawStream)
	}))
	defer upstream.Close()

	router, handler, cache := newNativeLifecycleRouter(t, upstream.URL+"/v1", map[string]config.ProviderCapabilities{
		"native": {ResponsesLifecycle: true},
	})
	defer cache.Stop()
	binding := mustResponseBinding(t, handler, "native")
	binding.Route = "responses"
	binding.Model = "native/gpt-native"
	handler.responseBindings.put("resp space", binding)

	retrieve := performLifecycleRequest(t, router, http.MethodGet, "/v1/responses/resp%20space", nil)
	if retrieve.Code != http.StatusAccepted {
		t.Fatalf("retrieve status = %d, want 202; body=%s", retrieve.Code, retrieve.Body.String())
	}
	if got := retrieve.Body.String(); got != rawStream {
		t.Fatalf("native retrieve stream changed\n got: %q\nwant: %q", got, rawStream)
	}
	if escapedPath != "/v1/responses/resp%20space" {
		t.Fatalf("upstream escaped path = %q", escapedPath)
	}
}

func TestNativeResponsesDeleteReleasesBindingOnlyAfterUpstreamSuccess(t *testing.T) {
	testCases := []struct {
		name        string
		status      int
		wantBinding bool
	}{
		{name: "success", status: http.StatusOK, wantBinding: false},
		{name: "upstream failure", status: http.StatusTooManyRequests, wantBinding: true},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			var calls atomic.Int32
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				if r.Method != http.MethodDelete || r.URL.Path != "/v1/responses/resp_delete" {
					t.Errorf("upstream request = %s %s", r.Method, r.URL.Path)
				}
				w.Header().Set("Content-Type", "application/json")
				w.Header().Set("Retry-After", "4")
				w.WriteHeader(testCase.status)
				if testCase.status < http.StatusMultipleChoices {
					_, _ = io.WriteString(w, `{"id":"resp_delete","object":"response.deleted","deleted":true,"future_delete":"kept"}`)
					return
				}
				_, _ = io.WriteString(w, `{"error":{"message":"busy","type":"rate_limit_error","code":"rate_limit"},"future_error":true}`)
			}))
			defer upstream.Close()

			router, handler, cache := newNativeLifecycleRouter(t, upstream.URL+"/v1", map[string]config.ProviderCapabilities{
				"native": {ResponsesLifecycle: true},
			})
			defer cache.Stop()
			handler.responseBindings.put("resp_delete", mustResponseBinding(t, handler, "native"))

			deleted := performLifecycleRequest(t, router, http.MethodDelete, "/v1/responses/resp_delete", nil)
			if deleted.Code != testCase.status {
				t.Fatalf("delete status = %d, want %d; body=%s", deleted.Code, testCase.status, deleted.Body.String())
			}
			if got := calls.Load(); got != 1 {
				t.Fatalf("upstream calls = %d, want exactly one", got)
			}
			if _, ok := handler.responseBindings.get("resp_delete"); ok != testCase.wantBinding {
				t.Fatalf("binding retained = %v, want %v", ok, testCase.wantBinding)
			}
			if deleted.Header().Get("Retry-After") != "4" {
				t.Fatalf("Retry-After = %q", deleted.Header().Get("Retry-After"))
			}
			if !strings.Contains(deleted.Body.String(), "future_") {
				t.Fatalf("raw upstream envelope changed: %s", deleted.Body.String())
			}
		})
	}
}

func TestNativeResponsesCancelIsSingleAttemptAndKeepsBinding(t *testing.T) {
	var calls atomic.Int32
	var requestBody string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		body, _ := io.ReadAll(r.Body)
		requestBody = string(body)
		if r.Method != http.MethodPost || r.URL.Path != "/v1/responses/resp_cancel/cancel" {
			t.Errorf("upstream request = %s %s", r.Method, r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusConflict)
		_, _ = io.WriteString(w, `{"error":{"message":"already terminal","type":"invalid_request_error"},"future_error":"kept"}`)
	}))
	defer upstream.Close()

	router, handler, cache := newNativeLifecycleRouter(t, upstream.URL+"/v1", map[string]config.ProviderCapabilities{
		"native": {ResponsesLifecycle: true, ResponseCancellation: true},
	})
	defer cache.Stop()
	handler.responseBindings.put("resp_cancel", mustResponseBinding(t, handler, "native"))

	request := httptest.NewRequest(http.MethodPost, "/v1/responses/resp_cancel/cancel?future=true", strings.NewReader(`{"reason":"client_request"}`))
	request.Header.Set("Content-Type", "application/json")
	cancelled := httptest.NewRecorder()
	router.ServeHTTP(cancelled, request)
	if cancelled.Code != http.StatusConflict {
		t.Fatalf("cancel status = %d, want 409; body=%s", cancelled.Code, cancelled.Body.String())
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("upstream calls = %d, want exactly one", got)
	}
	if requestBody != `{"reason":"client_request"}` {
		t.Fatalf("cancel body changed: %q", requestBody)
	}
	if _, ok := handler.responseBindings.get("resp_cancel"); !ok {
		t.Fatal("cancel removed the owner binding")
	}
	if !strings.Contains(cancelled.Body.String(), `"future_error":"kept"`) {
		t.Fatalf("raw cancel error changed: %s", cancelled.Body.String())
	}
}

func TestLocalResponsesCancelFailsExplicitlyWithoutUpstreamCall(t *testing.T) {
	router, _, calls, closeTest := newResponsesLifecycleTestRouter(t)
	defer closeTest()
	created := performLifecycleRequest(t, router, http.MethodPost, "/v1/responses", []byte(`{"model":"mock-gpt","input":"hello"}`))
	responseID := lifecycleStringField(t, decodeLifecycleObject(t, created.Body.Bytes()), "id")

	cancelled := performLifecycleRequest(t, router, http.MethodPost, "/v1/responses/"+responseID+"/cancel", []byte(`{}`))
	if cancelled.Code != http.StatusBadRequest {
		t.Fatalf("local cancel status = %d, want 400; body=%s", cancelled.Code, cancelled.Body.String())
	}
	assertLifecycleError(t, cancelled.Body.Bytes(), "response_id", "unsupported_feature")
	if got := calls.Load(); got != 1 {
		t.Fatalf("upstream calls = %d, want only the create call", got)
	}
}

func TestResponseBindingStoreExpiresAndEvicts(t *testing.T) {
	store := newResponseBindingStore(5 * time.Millisecond)
	store.maxEntries = 1
	store.put("resp_first", responseBinding{Provider: "first", AccountFingerprint: "first-account"})
	store.put("resp_second", responseBinding{Provider: "second", AccountFingerprint: "second-account"})
	if _, ok := store.get("resp_first"); ok {
		t.Fatal("oldest binding was not evicted")
	}
	if binding, ok := store.get("resp_second"); !ok || binding.Provider != "second" {
		t.Fatalf("newest binding = %#v, %v", binding, ok)
	}
	time.Sleep(10 * time.Millisecond)
	if _, ok := store.get("resp_second"); ok {
		t.Fatal("expired binding was retained")
	}
}

func TestResponseBindingStoreRejectsOversizedEntry(t *testing.T) {
	store := newResponseBindingStore(time.Hour)
	store.maxBytes = 32
	store.put("resp_large", responseBinding{Provider: "provider", Model: strings.Repeat("m", 64), AccountFingerprint: "account"})
	if _, ok := store.get("resp_large"); ok {
		t.Fatal("oversized binding was retained")
	}
}

func TestBoundResponseBindingRejectsChangedProviderAccountWithoutUpstreamCall(t *testing.T) {
	var upstreamCalls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		upstreamCalls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"resp_account","object":"response","status":"completed"}`)
	}))
	defer upstream.Close()

	original := config.ProviderConfig{
		Type:         "openai",
		APIKey:       "original-secret",
		BaseURL:      "https://original.example/v1",
		Organization: "org-original",
		DefaultModel: "gpt-native",
		Capabilities: config.ProviderCapabilities{ResponsesLifecycle: true},
	}
	router, handler, cache := newNativeLifecycleRouterFromConfigs(t, map[string]config.ProviderConfig{"native": original})
	defer cache.Stop()
	binding := mustResponseBinding(t, handler, "native")
	if strings.Contains(binding.AccountFingerprint, original.APIKey) || len(binding.AccountFingerprint) != sha256.Size*2 {
		t.Fatalf("unsafe account fingerprint %q", binding.AccountFingerprint)
	}
	handler.responseBindings.put("resp_account", binding)

	changed := original
	changed.APIKey = "rotated-secret"
	changed.BaseURL = upstream.URL + "/v1"
	changed.Organization = "org-rotated"
	changedConfigs := map[string]config.ProviderConfig{"native": changed}
	handler.registry.UpdateProvidersConfig(changedConfigs)
	handler.UpdateProviderConfigs(changedConfigs)

	response := performLifecycleRequest(t, router, http.MethodGet, "/v1/responses/resp_account", nil)
	if response.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400; body=%s", response.Code, response.Body.String())
	}
	assertLifecycleError(t, response.Body.Bytes(), "provider", "provider_binding_stale")
	if got := upstreamCalls.Load(); got != 0 {
		t.Fatalf("stale binding made %d upstream calls", got)
	}
	for _, secret := range []string{original.APIKey, changed.APIKey, binding.AccountFingerprint} {
		if strings.Contains(response.Body.String(), secret) {
			t.Fatalf("binding error leaked account identity: %s", response.Body.String())
		}
	}
}

func newNativeLifecycleRouter(
	t *testing.T,
	baseURL string,
	capabilities map[string]config.ProviderCapabilities,
) (http.Handler, *Handler, *middleware.Cache) {
	t.Helper()
	providerConfigs := make(map[string]config.ProviderConfig, len(capabilities))
	for provider, providerCapabilities := range capabilities {
		providerConfigs[provider] = config.ProviderConfig{
			Type:         "openai",
			APIKey:       "provider-secret",
			BaseURL:      baseURL,
			DefaultModel: "gpt-native",
			Capabilities: providerCapabilities,
		}
	}
	return newNativeLifecycleRouterFromConfigs(t, providerConfigs)
}

func mustResponseBinding(t *testing.T, handler *Handler, provider string) responseBinding {
	t.Helper()
	fingerprint, ok := handler.responseAccountFingerprint(provider)
	if !ok {
		t.Fatalf("provider %q has no account fingerprint", provider)
	}
	return responseBinding{
		Provider:            provider,
		UpstreamRequestType: requestTypeResponses,
		AccountFingerprint:  fingerprint,
	}
}

func newNativeLifecycleRouterFromConfigs(
	t *testing.T,
	providerConfigs map[string]config.ProviderConfig,
) (http.Handler, *Handler, *middleware.Cache) {
	t.Helper()
	targets := make([]config.TargetConfig, 0, len(providerConfigs))
	for provider, providerConfig := range providerConfigs {
		targets = append(targets, config.TargetConfig{
			Provider:            provider,
			Model:               providerConfig.DefaultModel,
			Weight:              1,
			UpstreamRequestType: requestTypeResponses,
		})
	}
	registry := providers.NewRegistry(providerConfigs)
	routingEngine := routing.NewEngine(config.RoutingConfig{
		DefaultStrategy: "weighted",
		Routes: []config.RouteConfig{{
			Name:    "responses",
			Match:   config.MatchConfig{Path: "/v1/responses"},
			Targets: targets,
		}},
	})
	cache := middleware.NewCache(config.CacheConfig{Enabled: false})
	handler := NewHandler(
		registry,
		routingEngine,
		resilience.NewFallbackExecutor(
			resilience.NewRetrier(config.RetryConfig{Enabled: false}),
			resilience.NewCircuitBreakerManager(),
		),
		cache,
		streaming.NewHandler(),
		observability.NewMetricsWithRegisterer(prometheus.NewRegistry()),
		nil,
		nil,
		nil,
	)
	handler.UpdateProviderConfigs(providerConfigs)
	return NewRouter(handler, nil, nil, health.NewChecker("test")), handler, cache
}

func decodeNativeLifecycleBody(t *testing.T, body string) map[string]interface{} {
	t.Helper()
	var decoded map[string]interface{}
	if err := json.Unmarshal([]byte(body), &decoded); err != nil {
		t.Fatalf("decode lifecycle body: %v", err)
	}
	return decoded
}
