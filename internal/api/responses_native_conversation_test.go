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

func TestResponsesPinsNativeConversationProviderAndPreservesPayload(t *testing.T) {
	const (
		conversationID = "conv_native_response"
		requestBody    = `{
			"model":"gpt-native",
			"conversation":{"id":"conv_native_response","future_conversation":{"keep":true}},
			"input":[{"role":"user","content":[{"type":"input_text","text":"hello"}]}],
			"store":false,
			"future_top":{"large_integer":9007199254740993}
		}`
		responseBody = "{\n  \"id\":\"resp_native_conversation\",\"object\":\"response\",\"created_at\":1,\"status\":\"completed\",\"model\":\"gpt-native\",\"output\":[],\"future_response\":{\"kept\":true}\n}\n"
	)

	var alphaCalls atomic.Int32
	var betaCalls atomic.Int32
	var alphaRequest []byte
	var alphaAuthorization string
	alpha := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		alphaCalls.Add(1)
		var err error
		alphaRequest, err = io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read alpha request: %v", err)
			return
		}
		alphaAuthorization = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Upstream-Trace", "alpha-trace")
		w.Header().Set("Set-Cookie", "provider-secret=blocked")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, responseBody)
	}))
	defer alpha.Close()
	beta := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		betaCalls.Add(1)
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer beta.Close()

	providerConfigs := map[string]config.ProviderConfig{
		"alpha": {
			Type:         "openai",
			APIKey:       "alpha-secret",
			BaseURL:      alpha.URL + "/v1",
			DefaultModel: "gpt-native",
			Capabilities: config.ProviderCapabilities{Conversations: true, ResponsesLifecycle: true},
		},
		"beta": {
			Type:         "openai",
			APIKey:       "beta-secret",
			BaseURL:      beta.URL + "/v1",
			DefaultModel: "gpt-native",
			Capabilities: config.ProviderCapabilities{Conversations: true, ResponsesLifecycle: true},
		},
	}
	router, handler, cache := newNativeLifecycleRouterFromConfigs(t, providerConfigs)
	defer cache.Stop()
	binding, err := handler.validateConversationProvider("alpha")
	if err != nil {
		t.Fatal(err)
	}
	if !handler.conversationBindings.put(conversationID, binding) {
		t.Fatal("failed to retain native conversation binding")
	}

	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/v1/responses?include%5B%5D=reasoning.encrypted_content", strings.NewReader(requestBody))
	router.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusOK || recorder.Body.String() != responseBody {
		t.Fatalf("response = %d %q, want preserved %d %q", recorder.Code, recorder.Body.String(), http.StatusOK, responseBody)
	}
	if alphaCalls.Load() != 1 || betaCalls.Load() != 0 {
		t.Fatalf("provider calls: alpha=%d beta=%d", alphaCalls.Load(), betaCalls.Load())
	}
	if alphaAuthorization != "Bearer alpha-secret" {
		t.Fatalf("alpha authorization = %q", alphaAuthorization)
	}
	if recorder.Header().Get("X-LunarGate-Provider") != "alpha" || recorder.Header().Get("X-Upstream-Trace") != "alpha-trace" {
		t.Fatalf("response headers = %#v", recorder.Header())
	}
	if recorder.Header().Get("Set-Cookie") != "" {
		t.Fatal("unsafe upstream cookie was forwarded")
	}

	var upstreamPayload map[string]json.RawMessage
	if err := json.Unmarshal(alphaRequest, &upstreamPayload); err != nil {
		t.Fatalf("decode upstream payload: %v; body=%s", err, alphaRequest)
	}
	var conversation map[string]interface{}
	if err := json.Unmarshal(upstreamPayload["conversation"], &conversation); err != nil {
		t.Fatalf("decode upstream conversation: %v", err)
	}
	if conversation["id"] != conversationID || conversation["future_conversation"] == nil {
		t.Fatalf("upstream conversation = %#v", conversation)
	}
	if string(upstreamPayload["store"]) != "false" {
		t.Fatalf("upstream store = %s, want false", upstreamPayload["store"])
	}
	if !bytes.Contains(upstreamPayload["future_top"], []byte("9007199254740993")) {
		t.Fatalf("future request field changed: %s", upstreamPayload["future_top"])
	}
	if _, _, stored := handler.responsesState.getCompleted("resp_native_conversation"); stored {
		t.Fatal("store:false native response was retained locally")
	}
	if _, stored := handler.responseBindings.get("resp_native_conversation"); stored {
		t.Fatal("store:false native response binding was retained")
	}
	if _, local := handler.conversationsState.get(conversationID); local {
		t.Fatal("native conversation was copied into local conversation state")
	}
}

func TestResponsesAllowsExplicitNativeConversationRecovery(t *testing.T) {
	var calls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		var payload map[string]json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		got, err := parseResponsesConversationID(payload["conversation"])
		if err != nil {
			t.Errorf("parse conversation: %v", err)
			return
		}
		if got != "conv_external" {
			t.Errorf("conversation = %q", got)
		}
		_, _ = io.WriteString(w, `{"id":"resp_explicit","object":"response","status":"completed","model":"gpt-native","output":[]}`)
	}))
	defer upstream.Close()
	router, _, cache := newNativeLifecycleRouter(t, upstream.URL+"/v1", map[string]config.ProviderCapabilities{
		"native": {Conversations: true, ResponsesLifecycle: true},
	})
	defer cache.Stop()

	request := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"model":"gpt-native","conversation":"conv_external","input":"hello","store":false}`))
	request.Header.Set("X-LunarGate-Provider", "native")
	recorder := httptest.NewRecorder()
	router.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK || calls.Load() != 1 {
		t.Fatalf("explicit recovery = %d %s, calls=%d", recorder.Code, recorder.Body.String(), calls.Load())
	}
}

func TestResponsesRejectsUnknownNativeConversationWithoutUpstream(t *testing.T) {
	var calls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer upstream.Close()
	router, _, cache := newNativeLifecycleRouter(t, upstream.URL+"/v1", map[string]config.ProviderCapabilities{
		"native": {Conversations: true, ResponsesLifecycle: true},
	})
	defer cache.Stop()

	recorder := performLifecycleRequest(t, router, http.MethodPost, "/v1/responses", []byte(`{"model":"gpt-native","conversation":"conv_unknown","input":"hello"}`))
	assertConversationError(t, recorder, http.StatusNotFound, "conversation_id", "conversation_not_found")
	if calls.Load() != 0 {
		t.Fatalf("unknown conversation caused %d upstream calls", calls.Load())
	}
}

func TestResponsesNativeConversationRequiresNativeResponsesCapability(t *testing.T) {
	var calls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer upstream.Close()
	router, handler, cache := newNativeLifecycleRouter(t, upstream.URL+"/v1", map[string]config.ProviderCapabilities{
		"native": {Conversations: true},
	})
	defer cache.Stop()
	binding, err := handler.validateConversationProvider("native")
	if err != nil {
		t.Fatal(err)
	}
	handler.conversationBindings.put("conv_no_responses", binding)

	recorder := performLifecycleRequest(t, router, http.MethodPost, "/v1/responses", []byte(`{"model":"gpt-native","conversation":"conv_no_responses","input":"hello"}`))
	assertConversationError(t, recorder, http.StatusBadRequest, "conversation", "unsupported_feature")
	if calls.Load() != 0 {
		t.Fatalf("missing capability caused %d upstream calls", calls.Load())
	}
}

func TestResponsesNativeConversationRejectsConflictingModelProvider(t *testing.T) {
	var calls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer upstream.Close()
	providerConfigs := map[string]config.ProviderConfig{
		"alpha": {
			Type:         "openai",
			APIKey:       "alpha-secret",
			BaseURL:      upstream.URL + "/v1",
			DefaultModel: "gpt-native",
			Capabilities: config.ProviderCapabilities{Conversations: true, ResponsesLifecycle: true},
		},
		"beta": {
			Type:         "openai",
			APIKey:       "beta-secret",
			BaseURL:      upstream.URL + "/v1",
			DefaultModel: "gpt-native",
			Capabilities: config.ProviderCapabilities{Conversations: true, ResponsesLifecycle: true},
		},
	}
	router, handler, cache := newNativeLifecycleRouterFromConfigs(t, providerConfigs)
	defer cache.Stop()
	binding, err := handler.validateConversationProvider("alpha")
	if err != nil {
		t.Fatal(err)
	}
	handler.conversationBindings.put("conv_alpha", binding)

	recorder := performLifecycleRequest(t, router, http.MethodPost, "/v1/responses", []byte(`{"model":"beta/gpt-native","conversation":"conv_alpha","input":"hello"}`))
	assertConversationError(t, recorder, http.StatusBadRequest, "model", "invalid_value")
	if calls.Load() != 0 {
		t.Fatalf("conflicting model caused %d upstream calls", calls.Load())
	}
}

func TestResponsesNativeConversationRejectsStaleProviderAccount(t *testing.T) {
	var calls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer upstream.Close()
	providerConfig := config.ProviderConfig{
		Type:         "openai",
		APIKey:       "first-secret",
		BaseURL:      upstream.URL + "/v1",
		DefaultModel: "gpt-native",
		Capabilities: config.ProviderCapabilities{Conversations: true, ResponsesLifecycle: true},
	}
	router, handler, cache := newNativeLifecycleRouterFromConfigs(t, map[string]config.ProviderConfig{"native": providerConfig})
	defer cache.Stop()
	binding, err := handler.validateConversationProvider("native")
	if err != nil {
		t.Fatal(err)
	}
	handler.conversationBindings.put("conv_stale", binding)

	providerConfig.APIKey = "second-secret"
	updated := map[string]config.ProviderConfig{"native": providerConfig}
	if !handler.registry.UpdateProvidersConfig(updated) {
		t.Fatal("failed to update provider registry")
	}
	handler.UpdateProviderConfigs(updated)

	recorder := performLifecycleRequest(t, router, http.MethodPost, "/v1/responses", []byte(`{"model":"gpt-native","conversation":"conv_stale","input":"hello"}`))
	assertConversationError(t, recorder, http.StatusBadRequest, "provider", "provider_binding_stale")
	if calls.Load() != 0 {
		t.Fatalf("stale provider binding caused %d upstream calls", calls.Load())
	}
}

func TestResponsesNativeConversationRejectsTranslatedTarget(t *testing.T) {
	var calls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer upstream.Close()
	handler, cache := newNativeContinuationTestHandler(t, upstream.URL+"/v1", requestTypeChatCompletions)
	defer cache.Stop()
	providerConfigs := map[string]config.ProviderConfig{
		"openai": {
			Type:         "openai",
			APIKey:       "dummy",
			BaseURL:      upstream.URL + "/v1",
			DefaultModel: "gpt-5.4",
			Capabilities: config.ProviderCapabilities{Conversations: true, ResponsesLifecycle: true},
		},
	}
	if !handler.registry.UpdateProvidersConfig(providerConfigs) {
		t.Fatal("failed to update provider registry")
	}
	handler.UpdateProviderConfigs(providerConfigs)
	binding, err := handler.validateConversationProvider("openai")
	if err != nil {
		t.Fatal(err)
	}
	handler.conversationBindings.put("conv_translated", binding)

	recorder := httptest.NewRecorder()
	handler.Responses(recorder, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"model":"gpt-5.4","conversation":"conv_translated","input":"hello"}`)))
	assertConversationError(t, recorder, http.StatusBadRequest, "conversation", "unsupported_feature")
	if calls.Load() != 0 {
		t.Fatalf("translated target caused %d upstream calls", calls.Load())
	}
}
