package api

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sort"
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

type observedChatLifecycleRequest struct {
	method        string
	requestURI    string
	body          string
	authorization string
	organization  string
	accept        string
	beta          string
	idempotency   string
	xAPIKey       string
	cookie        string
}

func TestStoredChatCompletionsLifecycleProxiesRawCRUDListAndMessages(t *testing.T) {
	var mu sync.Mutex
	observed := make([]observedChatLifecycleRequest, 0, 6)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read upstream body: %v", err)
		}
		mu.Lock()
		observed = append(observed, observedChatLifecycleRequest{
			method:        r.Method,
			requestURI:    r.URL.RequestURI(),
			body:          string(body),
			authorization: r.Header.Get("Authorization"),
			organization:  r.Header.Get("OpenAI-Organization"),
			accept:        r.Header.Get("Accept"),
			beta:          r.Header.Get("OpenAI-Beta"),
			idempotency:   r.Header.Get("Idempotency-Key"),
			xAPIKey:       r.Header.Get("X-Api-Key"),
			cookie:        r.Header.Get("Cookie"),
		})
		mu.Unlock()

		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Upstream-Trace", "trace-kept")
		w.Header().Set("Set-Cookie", "provider-session=must-not-leak")
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/v1/chat/completions":
			_, _ = io.WriteString(w, `{"id":"chatcmpl_owner","object":"chat.completion","created":1,"model":"gpt-native","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"future_create":{"kept":true}}`)
		case r.Method == http.MethodGet && r.URL.Path == "/v1/chat/completions":
			_, _ = io.WriteString(w, `{"object":"list","data":[],"first_id":null,"last_id":null,"has_more":false,"future_list":"kept"}`)
		case r.Method == http.MethodGet && r.URL.Path == "/v1/chat/completions/chatcmpl_owner":
			w.WriteHeader(http.StatusAccepted)
			_, _ = io.WriteString(w, `{"id":"chatcmpl_owner","object":"chat.completion","future_retrieve":7}`)
		case r.Method == http.MethodPost && r.URL.Path == "/v1/chat/completions/chatcmpl_owner":
			_, _ = io.WriteString(w, `{"id":"chatcmpl_owner","object":"chat.completion","metadata":{"updated":true},"future_update":"kept"}`)
		case r.Method == http.MethodGet && r.URL.Path == "/v1/chat/completions/chatcmpl_owner/messages":
			_, _ = io.WriteString(w, `{"object":"list","data":[],"has_more":false,"future_messages":"kept"}`)
		case r.Method == http.MethodDelete && r.URL.Path == "/v1/chat/completions/chatcmpl_owner":
			_, _ = io.WriteString(w, `{"id":"chatcmpl_owner","object":"chat.completion.deleted","deleted":true,"future_delete":"kept"}`)
		default:
			t.Errorf("unexpected upstream request %s %s", r.Method, r.URL.RequestURI())
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer upstream.Close()

	configs := map[string]config.ProviderConfig{
		"native": {
			Type:         "openai",
			APIKey:       "provider-secret",
			BaseURL:      upstream.URL + "/v1",
			Organization: "provider-org",
			DefaultModel: "gpt-native",
			Capabilities: config.ProviderCapabilities{ChatCompletionsLifecycle: true},
		},
	}
	router, handler, cache := newStoredChatLifecycleRouterFromConfigs(t, configs)
	defer cache.Stop()

	createBody := `{"model":"native/gpt-native","messages":[{"role":"user","content":"hello"}],"store":true,"future_create_option":{"kept":true}}`
	created := performStoredChatLifecycleRequest(t, router, http.MethodPost, "/v1/chat/completions", createBody, nil)
	if created.Code != http.StatusOK || !strings.Contains(created.Body.String(), `"future_create":{"kept":true}`) {
		t.Fatalf("create = %d %s", created.Code, created.Body.String())
	}
	binding, ok := handler.chatCompletionBindings.get("chatcmpl_owner")
	if !ok {
		t.Fatal("stored Chat Completion owner binding was not retained")
	}
	if binding.Provider != "native" || binding.Route != "chat" || binding.Model != "native/gpt-native" {
		t.Fatalf("owner binding = %#v", binding)
	}

	commonHeaders := map[string]string{
		"Authorization":       "Bearer client-secret",
		"OpenAI-Organization": "client-org",
		"X-Api-Key":           "client-api-secret",
		"Cookie":              "client-session=secret",
		"Accept":              "application/vnd.openai+json",
		"OpenAI-Beta":         "stored-chat=v1",
		"Idempotency-Key":     "idem-lifecycle-1",
	}

	listed := performStoredChatLifecycleRequest(
		t,
		router,
		http.MethodGet,
		"/v1/chat/completions?after=chatcmpl%2Fcursor&limit=17&metadata%5Bteam%5D=blue&order=asc",
		`{"future_list_body":true}`,
		commonHeaders,
	)
	if listed.Code != http.StatusOK || !strings.Contains(listed.Body.String(), `"future_list":"kept"`) {
		t.Fatalf("list = %d %s", listed.Code, listed.Body.String())
	}

	retrieved := performStoredChatLifecycleRequest(t, router, http.MethodGet, "/v1/chat/completions/chatcmpl_owner?future=a%2Fb", "", commonHeaders)
	if retrieved.Code != http.StatusAccepted || retrieved.Body.String() != `{"id":"chatcmpl_owner","object":"chat.completion","future_retrieve":7}` {
		t.Fatalf("retrieve = %d %q", retrieved.Code, retrieved.Body.String())
	}
	if retrieved.Header().Get("X-Upstream-Trace") != "trace-kept" {
		t.Fatalf("safe upstream response header = %q", retrieved.Header().Get("X-Upstream-Trace"))
	}
	if cookies := retrieved.Header().Values("Set-Cookie"); len(cookies) != 0 {
		t.Fatalf("unsafe upstream response cookies leaked: %q", cookies)
	}

	updateBody := "{\n  \"metadata\": {\"team\":\"blue\"}, \"future_update\": 9007199254740993\n}"
	updated := performStoredChatLifecycleRequest(t, router, http.MethodPost, "/v1/chat/completions/chatcmpl_owner?audit=true", updateBody, commonHeaders)
	if updated.Code != http.StatusOK || !strings.Contains(updated.Body.String(), `"future_update":"kept"`) {
		t.Fatalf("update = %d %s", updated.Code, updated.Body.String())
	}

	messagesBody := `{"future_messages_body":"kept"}`
	messages := performStoredChatLifecycleRequest(t, router, http.MethodGet, "/v1/chat/completions/chatcmpl_owner/messages?after=msg%2Fone&limit=3&order=desc", messagesBody, commonHeaders)
	if messages.Code != http.StatusOK || !strings.Contains(messages.Body.String(), `"future_messages":"kept"`) {
		t.Fatalf("messages = %d %s", messages.Code, messages.Body.String())
	}

	deleteBody := `{"hard":false}`
	deleted := performStoredChatLifecycleRequest(t, router, http.MethodDelete, "/v1/chat/completions/chatcmpl_owner?audit=delete", deleteBody, commonHeaders)
	if deleted.Code != http.StatusOK || !strings.Contains(deleted.Body.String(), `"future_delete":"kept"`) {
		t.Fatalf("delete = %d %s", deleted.Code, deleted.Body.String())
	}
	if _, ok := handler.chatCompletionBindings.get("chatcmpl_owner"); ok {
		t.Fatal("successful delete retained the owner binding")
	}

	mu.Lock()
	got := append([]observedChatLifecycleRequest(nil), observed...)
	mu.Unlock()
	if len(got) != 6 {
		t.Fatalf("upstream requests = %d, want 6: %#v", len(got), got)
	}
	wantMethodsAndURIs := []string{
		"POST /v1/chat/completions",
		"GET /v1/chat/completions?after=chatcmpl%2Fcursor&limit=17&metadata%5Bteam%5D=blue&order=asc",
		"GET /v1/chat/completions/chatcmpl_owner?future=a%2Fb",
		"POST /v1/chat/completions/chatcmpl_owner?audit=true",
		"GET /v1/chat/completions/chatcmpl_owner/messages?after=msg%2Fone&limit=3&order=desc",
		"DELETE /v1/chat/completions/chatcmpl_owner?audit=delete",
	}
	for i := range got {
		if actual := got[i].method + " " + got[i].requestURI; actual != wantMethodsAndURIs[i] {
			t.Errorf("upstream request %d = %q, want %q", i, actual, wantMethodsAndURIs[i])
		}
		if got[i].authorization != "Bearer provider-secret" || got[i].organization != "provider-org" {
			t.Errorf("request %d credentials = Authorization %q, organization %q", i, got[i].authorization, got[i].organization)
		}
		if got[i].xAPIKey != "" || got[i].cookie != "" {
			t.Errorf("request %d leaked inbound credentials: x-api-key=%q cookie=%q", i, got[i].xAPIKey, got[i].cookie)
		}
	}
	for i := 1; i < len(got); i++ {
		if got[i].accept != commonHeaders["Accept"] || got[i].beta != commonHeaders["OpenAI-Beta"] || got[i].idempotency != commonHeaders["Idempotency-Key"] {
			t.Errorf("request %d safe headers = accept %q beta %q idempotency %q", i, got[i].accept, got[i].beta, got[i].idempotency)
		}
	}
	if got[1].body != `{"future_list_body":true}` || got[2].body != "" || got[3].body != updateBody || got[4].body != messagesBody || got[5].body != deleteBody {
		t.Fatalf("lifecycle request bodies changed: %#v", got)
	}
}

func TestStoredChatCompletionsListProviderSelectionIsFailClosed(t *testing.T) {
	var alphaCalls atomic.Int32
	alpha := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		alphaCalls.Add(1)
		_, _ = io.WriteString(w, `{"object":"list","data":[],"provider":"alpha"}`)
	}))
	defer alpha.Close()
	var betaCalls atomic.Int32
	beta := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		betaCalls.Add(1)
		_, _ = io.WriteString(w, `{"object":"list","data":[],"provider":"beta"}`)
	}))
	defer beta.Close()

	configs := map[string]config.ProviderConfig{
		"alpha": storedChatProviderConfig(alpha.URL+"/v1", true),
		"beta":  storedChatProviderConfig(beta.URL+"/v1", true),
	}
	router, _, cache := newStoredChatLifecycleRouterFromConfigs(t, configs)
	defer cache.Stop()

	ambiguous := performStoredChatLifecycleRequest(t, router, http.MethodGet, "/v1/chat/completions", "", nil)
	if ambiguous.Code != http.StatusBadRequest {
		t.Fatalf("ambiguous list status = %d; body=%s", ambiguous.Code, ambiguous.Body.String())
	}
	assertLifecycleError(t, ambiguous.Body.Bytes(), "provider", "ambiguous_provider")
	if alphaCalls.Load() != 0 || betaCalls.Load() != 0 {
		t.Fatalf("ambiguous selection made upstream calls: alpha=%d beta=%d", alphaCalls.Load(), betaCalls.Load())
	}

	explicit := performStoredChatLifecycleRequest(t, router, http.MethodGet, "/v1/chat/completions", "", map[string]string{"X-LunarGate-Provider": "beta"})
	if explicit.Code != http.StatusOK || !strings.Contains(explicit.Body.String(), `"provider":"beta"`) {
		t.Fatalf("explicit list = %d %s", explicit.Code, explicit.Body.String())
	}
	if alphaCalls.Load() != 0 || betaCalls.Load() != 1 {
		t.Fatalf("explicit selection calls: alpha=%d beta=%d", alphaCalls.Load(), betaCalls.Load())
	}

	t.Run("capability disabled", func(t *testing.T) {
		disabledConfigs := map[string]config.ProviderConfig{"disabled": storedChatProviderConfig(alpha.URL+"/v1", false)}
		disabledRouter, _, disabledCache := newStoredChatLifecycleRouterFromConfigs(t, disabledConfigs)
		defer disabledCache.Stop()
		response := performStoredChatLifecycleRequest(t, disabledRouter, http.MethodGet, "/v1/chat/completions", "", map[string]string{"X-LunarGate-Provider": "disabled"})
		if response.Code != http.StatusBadRequest {
			t.Fatalf("status = %d; body=%s", response.Code, response.Body.String())
		}
		assertLifecycleError(t, response.Body.Bytes(), "provider", "unsupported_feature")
	})

	t.Run("non OpenAI transport", func(t *testing.T) {
		anthropicConfig := storedChatProviderConfig(alpha.URL+"/v1", true)
		anthropicConfig.Type = "anthropic"
		anthropicConfigs := map[string]config.ProviderConfig{"anthropic": anthropicConfig}
		anthropicRouter, _, anthropicCache := newStoredChatLifecycleRouterFromConfigs(t, anthropicConfigs)
		defer anthropicCache.Stop()
		response := performStoredChatLifecycleRequest(t, anthropicRouter, http.MethodGet, "/v1/chat/completions", "", map[string]string{"X-LunarGate-Provider": "anthropic"})
		if response.Code != http.StatusBadRequest {
			t.Fatalf("status = %d; body=%s", response.Code, response.Body.String())
		}
		assertLifecycleError(t, response.Body.Bytes(), "provider", "unsupported_feature")
	})
}

func TestStoredChatCompletionIDRequiresOwnerBindingOrExplicitProvider(t *testing.T) {
	var alphaCalls atomic.Int32
	var alphaEscapedPath atomic.Value
	alphaEscapedPath.Store("")
	alpha := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		alphaCalls.Add(1)
		alphaEscapedPath.Store(r.URL.EscapedPath())
		_, _ = fmt.Fprintf(w, `{"id":%q,"object":"chat.completion","provider":"alpha"}`, strings.TrimPrefix(r.URL.Path, "/v1/chat/completions/"))
	}))
	defer alpha.Close()
	var betaCalls atomic.Int32
	beta := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		betaCalls.Add(1)
		_, _ = fmt.Fprintf(w, `{"id":%q,"object":"chat.completion","provider":"beta"}`, strings.TrimPrefix(r.URL.Path, "/v1/chat/completions/"))
	}))
	defer beta.Close()

	configs := map[string]config.ProviderConfig{
		"alpha": storedChatProviderConfig(alpha.URL+"/v1", true),
		"beta":  storedChatProviderConfig(beta.URL+"/v1", true),
	}
	router, handler, cache := newStoredChatLifecycleRouterFromConfigs(t, configs)
	defer cache.Stop()

	missing := performStoredChatLifecycleRequest(t, router, http.MethodGet, "/v1/chat/completions/chatcmpl_external", "", nil)
	if missing.Code != http.StatusNotFound {
		t.Fatalf("unbound status = %d; body=%s", missing.Code, missing.Body.String())
	}
	assertLifecycleError(t, missing.Body.Bytes(), "completion_id", "completion_not_found")
	if alphaCalls.Load() != 0 || betaCalls.Load() != 0 {
		t.Fatalf("unbound request made upstream calls: alpha=%d beta=%d", alphaCalls.Load(), betaCalls.Load())
	}

	binding := mustChatCompletionBinding(t, handler, "alpha")
	if !handler.chatCompletionBindings.put("chatcmpl space", binding) {
		t.Fatal("failed to seed owner binding")
	}
	mismatch := performStoredChatLifecycleRequest(t, router, http.MethodGet, "/v1/chat/completions/chatcmpl%20space", "", map[string]string{"X-LunarGate-Provider": "beta"})
	if mismatch.Code != http.StatusBadRequest {
		t.Fatalf("mismatched owner status = %d; body=%s", mismatch.Code, mismatch.Body.String())
	}
	assertLifecycleError(t, mismatch.Body.Bytes(), "provider", "invalid_value")
	if alphaCalls.Load() != 0 || betaCalls.Load() != 0 {
		t.Fatalf("mismatched owner made upstream calls: alpha=%d beta=%d", alphaCalls.Load(), betaCalls.Load())
	}

	bound := performStoredChatLifecycleRequest(t, router, http.MethodGet, "/v1/chat/completions/chatcmpl%20space", "", nil)
	if bound.Code != http.StatusOK || !strings.Contains(bound.Body.String(), `"provider":"alpha"`) {
		t.Fatalf("bound retrieve = %d %s", bound.Code, bound.Body.String())
	}
	if got := alphaEscapedPath.Load().(string); got != "/v1/chat/completions/chatcmpl%20space" {
		t.Fatalf("upstream escaped path = %q", got)
	}

	now := time.Unix(1_700_000_000, 0)
	expiringStore := newChatCompletionBindingStore(time.Minute)
	expiringStore.now = func() time.Time { return now }
	handler.chatCompletionBindings = expiringStore
	if !expiringStore.put("chatcmpl_expired", mustChatCompletionBinding(t, handler, "alpha")) {
		t.Fatal("failed to seed expiring binding")
	}
	now = now.Add(time.Minute)
	explicit := performStoredChatLifecycleRequest(t, router, http.MethodGet, "/v1/chat/completions/chatcmpl_expired", "", map[string]string{"X-LunarGate-Provider": "beta"})
	if explicit.Code != http.StatusOK || !strings.Contains(explicit.Body.String(), `"provider":"beta"`) {
		t.Fatalf("expired explicit retrieve = %d %s", explicit.Code, explicit.Body.String())
	}
	if alphaCalls.Load() != 1 || betaCalls.Load() != 1 {
		t.Fatalf("owner selection calls: alpha=%d beta=%d", alphaCalls.Load(), betaCalls.Load())
	}
}

func TestStoredChatCompletionBindingRejectsChangedProviderAccount(t *testing.T) {
	var upstreamCalls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		upstreamCalls.Add(1)
		_, _ = io.WriteString(w, `{"id":"chatcmpl_account","object":"chat.completion"}`)
	}))
	defer upstream.Close()

	original := storedChatProviderConfig("https://original.example/v1", true)
	original.APIKey = "original-secret"
	original.Organization = "org-original"
	router, handler, cache := newStoredChatLifecycleRouterFromConfigs(t, map[string]config.ProviderConfig{"native": original})
	defer cache.Stop()
	binding := mustChatCompletionBinding(t, handler, "native")
	assertChatCompletionBindingFingerprintIsOpaque(t, binding, original.APIKey)
	if !handler.chatCompletionBindings.put("chatcmpl_account", binding) {
		t.Fatal("failed to seed account binding")
	}

	changed := original
	changed.APIKey = "rotated-secret"
	changed.Organization = "org-rotated"
	changed.BaseURL = upstream.URL + "/v1"
	changedConfigs := map[string]config.ProviderConfig{"native": changed}
	if !handler.registry.UpdateProvidersConfig(changedConfigs) {
		t.Fatal("rotated provider config was rejected")
	}
	handler.UpdateProviderConfigs(changedConfigs)

	response := performStoredChatLifecycleRequest(t, router, http.MethodGet, "/v1/chat/completions/chatcmpl_account", "", nil)
	if response.Code != http.StatusBadRequest {
		t.Fatalf("stale binding status = %d; body=%s", response.Code, response.Body.String())
	}
	assertLifecycleError(t, response.Body.Bytes(), "provider", "provider_binding_stale")
	if upstreamCalls.Load() != 0 {
		t.Fatalf("stale binding made %d upstream calls", upstreamCalls.Load())
	}
	for _, secret := range []string{original.APIKey, changed.APIKey, binding.AccountFingerprint} {
		if strings.Contains(response.Body.String(), secret) {
			t.Fatalf("binding error leaked account identity: %s", response.Body.String())
		}
	}
}

func TestStoredChatCompletionDeleteFailureRetainsBindingAndDoesNotRedirect(t *testing.T) {
	var redirectTargetCalls atomic.Int32
	var redirectTargetAuthorization atomic.Value
	redirectTargetAuthorization.Store("")
	redirectTarget := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		redirectTargetCalls.Add(1)
		redirectTargetAuthorization.Store(r.Header.Get("Authorization"))
		_, _ = io.WriteString(w, `{"id":"chatcmpl_delete","deleted":true}`)
	}))
	defer redirectTarget.Close()

	var sourceCalls atomic.Int32
	source := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sourceCalls.Add(1)
		if got := r.Header.Get("Authorization"); got != "Bearer provider-secret" {
			t.Errorf("source Authorization = %q", got)
		}
		if r.URL.Query().Get("failure") == "rate_limit" {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = io.WriteString(w, `{"error":{"message":"busy","type":"rate_limit_error"},"future_failure":"kept"}`)
			return
		}
		w.Header().Set("Location", redirectTarget.URL+"/v1/chat/completions/chatcmpl_delete")
		w.WriteHeader(http.StatusTemporaryRedirect)
		_, _ = io.WriteString(w, `{"future_redirect":"kept"}`)
	}))
	defer source.Close()

	router, handler, cache := newStoredChatLifecycleRouterFromConfigs(t, map[string]config.ProviderConfig{
		"native": storedChatProviderConfig(source.URL+"/v1", true),
	})
	defer cache.Stop()
	if !handler.chatCompletionBindings.put("chatcmpl_delete", mustChatCompletionBinding(t, handler, "native")) {
		t.Fatal("failed to seed delete binding")
	}

	failed := performStoredChatLifecycleRequest(t, router, http.MethodDelete, "/v1/chat/completions/chatcmpl_delete?failure=rate_limit", `{"hard":true}`, nil)
	if failed.Code != http.StatusTooManyRequests || !strings.Contains(failed.Body.String(), `"future_failure":"kept"`) {
		t.Fatalf("failed delete = %d %s", failed.Code, failed.Body.String())
	}
	if sourceCalls.Load() != 1 {
		t.Fatalf("failed delete source calls = %d, want exactly one", sourceCalls.Load())
	}
	if _, ok := handler.chatCompletionBindings.get("chatcmpl_delete"); !ok {
		t.Fatal("rate-limited delete removed owner binding")
	}

	response := performStoredChatLifecycleRequest(t, router, http.MethodDelete, "/v1/chat/completions/chatcmpl_delete", `{"hard":true}`, nil)
	if response.Code != http.StatusTemporaryRedirect || !strings.Contains(response.Body.String(), `"future_redirect":"kept"`) {
		t.Fatalf("delete redirect = %d %s", response.Code, response.Body.String())
	}
	if sourceCalls.Load() != 2 || redirectTargetCalls.Load() != 0 {
		t.Fatalf("redirect calls: source=%d target=%d", sourceCalls.Load(), redirectTargetCalls.Load())
	}
	if got := redirectTargetAuthorization.Load().(string); got != "" {
		t.Fatalf("redirect target received Authorization %q", got)
	}
	if _, ok := handler.chatCompletionBindings.get("chatcmpl_delete"); !ok {
		t.Fatal("failed delete removed owner binding")
	}
}

func TestStoredChatCompletionLifecycleRejectsInvalidEntityEnvelope(t *testing.T) {
	const completionID = "chatcmpl_expected"
	endpoints := []struct {
		name   string
		method string
		body   string
	}{
		{name: "retrieve", method: http.MethodGet},
		{name: "update", method: http.MethodPost, body: `{}`},
		{name: "delete", method: http.MethodDelete},
	}
	commonInvalid := []struct {
		name string
		body string
	}{
		{name: "missing id", body: `{"object":"chat.completion","future_secret":true}`},
		{name: "non-string id", body: `{"id":7,"object":"chat.completion","future_secret":true}`},
		{name: "padded id", body: `{"id":" chatcmpl_expected","object":"chat.completion","future_secret":true}`},
		{name: "mismatched id", body: `{"id":"chatcmpl_other","object":"chat.completion","future_secret":true}`},
		{name: "missing object", body: `{"id":"chatcmpl_expected","future_secret":true}`},
		{name: "wrong object", body: `{"id":"chatcmpl_expected","object":"response","future_secret":true}`},
		{name: "multiple values", body: `{"id":"chatcmpl_expected","object":"chat.completion"}{"future_secret":true}`},
	}

	for _, endpoint := range endpoints {
		cases := append([]struct {
			name string
			body string
		}(nil), commonInvalid...)
		if endpoint.method == http.MethodDelete {
			cases = []struct {
				name string
				body string
			}{
				{name: "wrong object", body: `{"id":"chatcmpl_expected","object":"chat.completion","deleted":true,"future_secret":true}`},
				{name: "missing deleted", body: `{"id":"chatcmpl_expected","object":"chat.completion.deleted","future_secret":true}`},
				{name: "deleted false", body: `{"id":"chatcmpl_expected","object":"chat.completion.deleted","deleted":false,"future_secret":true}`},
				{name: "non-boolean deleted", body: `{"id":"chatcmpl_expected","object":"chat.completion.deleted","deleted":"true","future_secret":true}`},
				{name: "mismatched id", body: `{"id":"chatcmpl_other","object":"chat.completion.deleted","deleted":true,"future_secret":true}`},
			}
		}

		for _, testCase := range cases {
			t.Run(endpoint.name+"/"+testCase.name, func(t *testing.T) {
				upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
					w.Header().Set("Content-Type", "application/json")
					w.WriteHeader(http.StatusAccepted)
					_, _ = io.WriteString(w, testCase.body)
				}))
				defer upstream.Close()

				router, handler, cache := newStoredChatLifecycleRouterFromConfigs(t, map[string]config.ProviderConfig{
					"native": storedChatProviderConfig(upstream.URL+"/v1", true),
				})
				defer cache.Stop()
				if !handler.chatCompletionBindings.put(completionID, mustChatCompletionBinding(t, handler, "native")) {
					t.Fatal("failed to seed owner binding")
				}

				response := performStoredChatLifecycleRequest(
					t,
					router,
					endpoint.method,
					"/v1/chat/completions/"+completionID,
					endpoint.body,
					nil,
				)
				if response.Code != http.StatusBadGateway {
					t.Fatalf("status = %d, want 502; body=%s", response.Code, response.Body.String())
				}
				if contentType := response.Header().Get("Content-Type"); !strings.HasPrefix(contentType, "application/json") {
					t.Fatalf("Content-Type = %q, want application/json", contentType)
				}
				if strings.Contains(response.Body.String(), "future_secret") || !json.Valid(response.Body.Bytes()) {
					t.Fatalf("invalid lifecycle response leaked downstream: %q", response.Body.String())
				}
				if _, ok := handler.chatCompletionBindings.get(completionID); !ok {
					t.Fatal("invalid lifecycle response removed the owner binding")
				}
			})
		}
	}
}

func TestStoredChatCompletionLifecycleBoundsEntityResponseBody(t *testing.T) {
	const completionID = "chatcmpl_expected"
	oversized := `{"id":"chatcmpl_expected","object":"chat.completion","padding":"` +
		strings.Repeat("x", maxNativeLifecycleResponseBytes) + `"}`
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, oversized)
	}))
	defer upstream.Close()

	router, handler, cache := newStoredChatLifecycleRouterFromConfigs(t, map[string]config.ProviderConfig{
		"native": storedChatProviderConfig(upstream.URL+"/v1", true),
	})
	defer cache.Stop()
	if !handler.chatCompletionBindings.put(completionID, mustChatCompletionBinding(t, handler, "native")) {
		t.Fatal("failed to seed owner binding")
	}

	response := performStoredChatLifecycleRequest(t, router, http.MethodGet, "/v1/chat/completions/"+completionID, "", nil)
	if response.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502; body=%s", response.Code, response.Body.String())
	}
	if strings.Contains(response.Body.String(), "padding") || !json.Valid(response.Body.Bytes()) {
		t.Fatalf("oversized lifecycle response leaked downstream: %q", response.Body.String())
	}
}

func TestStoredChatCompletionNonStreamCreateBindingPolicy(t *testing.T) {
	tests := []struct {
		name         string
		storeField   string
		capability   bool
		status       int
		wantStatus   int
		responseBody string
		completionID string
		wantBinding  bool
	}{
		{
			name:         "store true",
			storeField:   `,"store":true`,
			capability:   true,
			status:       http.StatusOK,
			responseBody: `{"id":"chatcmpl_stored","object":"chat.completion","created":1,"model":"gpt-native","choices":[]}`,
			completionID: "chatcmpl_stored",
			wantBinding:  true,
		},
		{
			name:         "store false",
			storeField:   `,"store":false`,
			capability:   true,
			status:       http.StatusOK,
			responseBody: `{"id":"chatcmpl_stateless","object":"chat.completion","created":1,"model":"gpt-native","choices":[]}`,
			completionID: "chatcmpl_stateless",
		},
		{
			name:         "missing upstream ID",
			storeField:   `,"store":true`,
			capability:   true,
			status:       http.StatusOK,
			wantStatus:   http.StatusBadGateway,
			responseBody: `{"object":"chat.completion","created":1,"model":"gpt-native","choices":[]}`,
		},
		{
			name:         "padded upstream ID",
			storeField:   `,"store":true`,
			capability:   true,
			status:       http.StatusOK,
			wantStatus:   http.StatusBadGateway,
			responseBody: `{"id":" chatcmpl_padded ","object":"chat.completion","created":1,"model":"gpt-native","choices":[]}`,
			completionID: "chatcmpl_padded",
		},
		{
			name:         "non-string upstream ID",
			storeField:   `,"store":true`,
			capability:   true,
			status:       http.StatusOK,
			wantStatus:   http.StatusBadGateway,
			responseBody: `{"id":7,"object":"chat.completion","created":1,"model":"gpt-native","choices":[]}`,
		},
		{
			name:         "missing upstream object",
			storeField:   `,"store":true`,
			capability:   true,
			status:       http.StatusOK,
			wantStatus:   http.StatusBadGateway,
			responseBody: `{"id":"chatcmpl_missing_object","created":1,"model":"gpt-native","choices":[]}`,
			completionID: "chatcmpl_missing_object",
		},
		{
			name:         "padded upstream object",
			storeField:   `,"store":true`,
			capability:   true,
			status:       http.StatusOK,
			wantStatus:   http.StatusBadGateway,
			responseBody: `{"id":"chatcmpl_padded_object","object":" chat.completion ","created":1,"model":"gpt-native","choices":[]}`,
			completionID: "chatcmpl_padded_object",
		},
		{
			name:         "wrong upstream object",
			storeField:   `,"store":true`,
			capability:   true,
			status:       http.StatusOK,
			wantStatus:   http.StatusBadGateway,
			responseBody: `{"id":"chatcmpl_wrong_object","object":"response","created":1,"model":"gpt-native","choices":[]}`,
			completionID: "chatcmpl_wrong_object",
		},
		{
			name:         "store false keeps compatibility normalization",
			storeField:   `,"store":false`,
			capability:   true,
			status:       http.StatusOK,
			responseBody: `{"created":1,"model":"gpt-native","choices":[]}`,
		},
		{
			name:         "store omitted",
			capability:   true,
			status:       http.StatusOK,
			responseBody: `{"id":"chatcmpl_default","object":"chat.completion","created":1,"model":"gpt-native","choices":[]}`,
			completionID: "chatcmpl_default",
		},
		{
			name:         "capability disabled",
			storeField:   `,"store":true`,
			status:       http.StatusOK,
			responseBody: `{"id":"chatcmpl_disabled","object":"chat.completion","created":1,"model":"gpt-native","choices":[]}`,
			completionID: "chatcmpl_disabled",
		},
		{
			name:         "failed create",
			storeField:   `,"store":true`,
			capability:   true,
			status:       http.StatusInternalServerError,
			responseBody: `{"error":{"message":"failed","type":"server_error"}}`,
			completionID: "chatcmpl_failed",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var calls atomic.Int32
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				calls.Add(1)
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(test.status)
				_, _ = io.WriteString(w, test.responseBody)
			}))
			defer upstream.Close()

			router, handler, cache := newStoredChatLifecycleRouterFromConfigs(t, map[string]config.ProviderConfig{
				"native": storedChatProviderConfig(upstream.URL+"/v1", test.capability),
			})
			defer cache.Stop()
			requestBody := `{"model":"native/gpt-native","messages":[{"role":"user","content":"hello"}]` + test.storeField + `}`
			response := performStoredChatLifecycleRequest(t, router, http.MethodPost, "/v1/chat/completions", requestBody, nil)
			wantStatus := test.wantStatus
			if wantStatus == 0 {
				wantStatus = test.status
			}
			if response.Code != wantStatus {
				t.Fatalf("create status = %d, want %d; body=%s", response.Code, wantStatus, response.Body.String())
			}
			if calls.Load() != 1 {
				t.Fatalf("upstream calls = %d, want one", calls.Load())
			}
			_, bound := handler.chatCompletionBindings.get(test.completionID)
			if bound != test.wantBinding {
				t.Fatalf("binding retained = %v, want %v", bound, test.wantBinding)
			}
		})
	}
}

func TestStoredChatCompletionStreamBindingRequiresSuccessfulTerminal(t *testing.T) {
	stableStream := func(id string) string {
		return fmt.Sprintf(
			"data: {\"id\":%q,\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-native\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\n"+
				"data: {\"id\":%q,\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-native\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
			id,
			id,
		)
	}
	tests := []struct {
		name        string
		stream      string
		store       bool
		capability  bool
		completion  string
		wantStatus  int
		wantBinding bool
	}{
		{
			name:        "complete stable stream",
			stream:      stableStream("chatcmpl_streamed") + "data: [DONE]\n\n",
			store:       true,
			capability:  true,
			completion:  "chatcmpl_streamed",
			wantBinding: true,
		},
		{
			name:       "missing done frame",
			stream:     stableStream("chatcmpl_incomplete"),
			store:      true,
			capability: true,
			completion: "chatcmpl_incomplete",
		},
		{
			name: "conflicting completion IDs",
			stream: stableStream("chatcmpl_first") +
				"data: {\"id\":\"chatcmpl_second\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-native\",\"choices\":[]}\n\n" +
				"data: [DONE]\n\n",
			store:      true,
			capability: true,
			completion: "chatcmpl_first",
		},
		{
			name:       "store false",
			stream:     stableStream("chatcmpl_stream_false") + "data: [DONE]\n\n",
			capability: true,
			completion: "chatcmpl_stream_false",
		},
		{
			name:       "capability disabled",
			stream:     stableStream("chatcmpl_stream_disabled") + "data: [DONE]\n\n",
			store:      true,
			completion: "chatcmpl_stream_disabled",
		},
		{
			name:       "padded upstream ID",
			stream:     stableStream(" chatcmpl_stream_padded ") + "data: [DONE]\n\n",
			store:      true,
			capability: true,
			completion: "chatcmpl_stream_padded",
			wantStatus: http.StatusBadGateway,
		},
		{
			name: "missing upstream ID",
			stream: "data: {\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-native\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\n" +
				"data: [DONE]\n\n",
			store:      true,
			capability: true,
			wantStatus: http.StatusBadGateway,
		},
		{
			name: "missing upstream object",
			stream: "data: {\"id\":\"chatcmpl_stream_missing_object\",\"created\":1,\"model\":\"gpt-native\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\n" +
				"data: [DONE]\n\n",
			store:      true,
			capability: true,
			completion: "chatcmpl_stream_missing_object",
			wantStatus: http.StatusBadGateway,
		},
		{
			name: "padded upstream object",
			stream: "data: {\"id\":\"chatcmpl_stream_padded_object\",\"object\":\" chat.completion.chunk \",\"created\":1,\"model\":\"gpt-native\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\n" +
				"data: [DONE]\n\n",
			store:      true,
			capability: true,
			completion: "chatcmpl_stream_padded_object",
			wantStatus: http.StatusBadGateway,
		},
		{
			name: "wrong upstream object",
			stream: "data: {\"id\":\"chatcmpl_stream_wrong_object\",\"object\":\"response\",\"created\":1,\"model\":\"gpt-native\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\n" +
				"data: [DONE]\n\n",
			store:      true,
			capability: true,
			completion: "chatcmpl_stream_wrong_object",
			wantStatus: http.StatusBadGateway,
		},
		{
			name: "store false keeps compatibility normalization",
			stream: "data: {\"created\":1,\"model\":\"gpt-native\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\n" +
				"data: [DONE]\n\n",
			capability: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = io.WriteString(w, test.stream)
			}))
			defer upstream.Close()

			router, handler, cache := newStoredChatLifecycleRouterFromConfigs(t, map[string]config.ProviderConfig{
				"native": storedChatProviderConfig(upstream.URL+"/v1", test.capability),
			})
			defer cache.Stop()
			requestBody := fmt.Sprintf(
				`{"model":"native/gpt-native","messages":[{"role":"user","content":"hello"}],"stream":true,"store":%t}`,
				test.store,
			)
			response := performStoredChatLifecycleRequest(t, router, http.MethodPost, "/v1/chat/completions", requestBody, nil)
			wantStatus := test.wantStatus
			if wantStatus == 0 {
				wantStatus = http.StatusOK
			}
			if response.Code != wantStatus {
				t.Fatalf("stream status = %d, want %d; body=%s", response.Code, wantStatus, response.Body.String())
			}
			_, bound := handler.chatCompletionBindings.get(test.completion)
			if bound != test.wantBinding {
				t.Fatalf("binding retained = %v, want %v; stream=%s", bound, test.wantBinding, response.Body.String())
			}
		})
	}
}

func TestStoredChatCompletionOwnerCollisionFailsClosed(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(map[bool]string{false: "non-stream", true: "stream"}[stream], func(t *testing.T) {
			const completionID = "chatcmpl_shared"
			var alphaCalls atomic.Int32
			var betaCalls atomic.Int32
			newUpstream := func(calls *atomic.Int32) *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					calls.Add(1)
					if r.Method != http.MethodPost || r.URL.Path != "/v1/chat/completions" {
						t.Errorf("unexpected lifecycle request %s %s", r.Method, r.URL.Path)
						w.WriteHeader(http.StatusInternalServerError)
						return
					}
					if stream {
						w.Header().Set("Content-Type", "text/event-stream")
						_, _ = fmt.Fprintf(w,
							"data: {\"id\":%q,\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-native\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\n"+
								"data: {\"id\":%q,\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-native\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"+
								"data: [DONE]\n\n",
							completionID,
							completionID,
						)
						return
					}
					w.Header().Set("Content-Type", "application/json")
					_, _ = fmt.Fprintf(w, `{"id":%q,"object":"chat.completion","created":1,"model":"gpt-native","choices":[]}`, completionID)
				}))
			}

			alpha := newUpstream(&alphaCalls)
			defer alpha.Close()
			beta := newUpstream(&betaCalls)
			defer beta.Close()
			router, handler, cache := newStoredChatLifecycleRouterFromConfigs(t, map[string]config.ProviderConfig{
				"alpha": storedChatProviderConfig(alpha.URL+"/v1", true),
				"beta":  storedChatProviderConfig(beta.URL+"/v1", true),
			})
			defer cache.Stop()

			streamField := ""
			if stream {
				streamField = `,"stream":true`
			}
			body := `{"model":"gpt-native","messages":[{"role":"user","content":"hello"}],"store":true` + streamField + `}`
			for _, provider := range []string{"alpha", "beta"} {
				created := performStoredChatLifecycleRequest(t, router, http.MethodPost, "/v1/chat/completions", body, map[string]string{
					"X-LunarGate-Provider": provider,
				})
				if created.Code != http.StatusOK {
					t.Fatalf("provider %s create status = %d; body=%s", provider, created.Code, created.Body.String())
				}
			}
			if _, lookup := handler.chatCompletionBindings.lookup(completionID); lookup != ownerLookupConflict {
				t.Fatalf("owner lookup = %v, want conflict", lookup)
			}

			implicit := performStoredChatLifecycleRequest(t, router, http.MethodGet, "/v1/chat/completions/"+completionID, "", nil)
			if implicit.Code != http.StatusBadRequest {
				t.Fatalf("implicit lifecycle status = %d, want 400; body=%s", implicit.Code, implicit.Body.String())
			}
			assertLifecycleError(t, implicit.Body.Bytes(), "completion_id", "provider_binding_conflict")
			if alphaCalls.Load() != 1 || betaCalls.Load() != 1 {
				t.Fatalf("upstream calls after conflict: alpha=%d beta=%d, want one create each", alphaCalls.Load(), betaCalls.Load())
			}
		})
	}
}

func TestResponsesTranslatedToChatDoNotBindHiddenChatCompletionID(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("upstream path = %q", r.URL.Path)
		}
		_, _ = io.WriteString(w, `{"id":"chatcmpl_hidden","object":"chat.completion","created":1,"model":"gpt-native","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer upstream.Close()

	configs := map[string]config.ProviderConfig{
		"native": storedChatProviderConfig(upstream.URL+"/v1", true),
	}
	router, handler, cache := newStoredChatLifecycleRouter(t, configs, "/v1/responses", requestTypeChatCompletions)
	defer cache.Stop()
	response := performStoredChatLifecycleRequest(t, router, http.MethodPost, "/v1/responses", `{"model":"native/gpt-native","input":"hello","store":true}`, nil)
	if response.Code != http.StatusOK {
		t.Fatalf("translated Responses create = %d; body=%s", response.Code, response.Body.String())
	}
	if _, ok := handler.chatCompletionBindings.get("chatcmpl_hidden"); ok {
		t.Fatal("Responses-to-Chat translation bound an upstream ID that is hidden from the client")
	}
}

func storedChatProviderConfig(baseURL string, lifecycle bool) config.ProviderConfig {
	return config.ProviderConfig{
		Type:         "openai",
		APIKey:       "provider-secret",
		BaseURL:      baseURL,
		DefaultModel: "gpt-native",
		Capabilities: config.ProviderCapabilities{ChatCompletionsLifecycle: lifecycle},
	}
}

func newStoredChatLifecycleRouterFromConfigs(
	t *testing.T,
	providerConfigs map[string]config.ProviderConfig,
) (http.Handler, *Handler, *middleware.Cache) {
	return newStoredChatLifecycleRouter(t, providerConfigs, "/v1/chat/completions", requestTypeChatCompletions)
}

func newStoredChatLifecycleRouter(
	t *testing.T,
	providerConfigs map[string]config.ProviderConfig,
	routePath string,
	upstreamRequestType string,
) (http.Handler, *Handler, *middleware.Cache) {
	t.Helper()
	providerIDs := make([]string, 0, len(providerConfigs))
	for provider := range providerConfigs {
		providerIDs = append(providerIDs, provider)
	}
	sort.Strings(providerIDs)
	targets := make([]config.TargetConfig, 0, len(providerIDs))
	for _, provider := range providerIDs {
		providerConfig := providerConfigs[provider]
		targets = append(targets, config.TargetConfig{
			Provider:            provider,
			Model:               providerConfig.DefaultModel,
			Weight:              1,
			UpstreamRequestType: upstreamRequestType,
		})
	}
	registry := providers.NewRegistry(providerConfigs)
	routingEngine := routing.NewEngine(config.RoutingConfig{
		DefaultStrategy: "weighted",
		Routes: []config.RouteConfig{{
			Name:    "chat",
			Match:   config.MatchConfig{Path: routePath},
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

func performStoredChatLifecycleRequest(
	t *testing.T,
	handler http.Handler,
	method string,
	path string,
	body string,
	headers map[string]string,
) *httptest.ResponseRecorder {
	t.Helper()
	request := httptest.NewRequest(method, path, strings.NewReader(body))
	for name, value := range headers {
		request.Header.Set(name, value)
	}
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, request)
	return recorder
}

func mustChatCompletionBinding(t *testing.T, handler *Handler, provider string) chatCompletionBinding {
	t.Helper()
	binding, err := handler.validateChatCompletionProvider(provider)
	if err != nil {
		t.Fatalf("provider %q binding: %v", provider, err)
	}
	return binding
}

func assertChatCompletionBindingFingerprintIsOpaque(t *testing.T, binding chatCompletionBinding, secret string) {
	t.Helper()
	if strings.Contains(binding.AccountFingerprint, secret) || len(binding.AccountFingerprint) != sha256.Size*2 {
		t.Fatalf("unsafe account fingerprint %q", binding.AccountFingerprint)
	}
}
