package api

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/lunargate-ai/gateway/internal/config"
)

type capturedNativeConversationRequest struct {
	method         string
	path           string
	rawQuery       string
	body           string
	authorization  string
	accept         string
	openAIBeta     string
	idempotencyKey string
}

func TestNativeConversationsCRUDPreservesEnvelopeAndBinding(t *testing.T) {
	const (
		createBody   = `{"metadata":{"nested":{"future":true}},"future_top":[1,2,3]}`
		createResult = "{\n  \"id\": \"conv_native_crud\", \"object\": \"conversation\", \"future\": {\"kept\": true}\n}\n"
		updateBody   = `{"metadata":{"nested":{"stage":"development"}},"future_update":true}`
		updateResult = `{"id":"conv_native_crud","object":"conversation","future_update":"kept"}`
		getResult    = `{"id":"conv_native_crud","object":"conversation","future_get":"kept"}`
		deleteResult = `{"id":"conv_native_crud","object":"conversation.deleted","deleted":true,"future_delete":"kept"}`
	)

	var mu sync.Mutex
	calls := make([]capturedNativeConversationRequest, 0, 4)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read upstream body: %v", err)
			return
		}
		mu.Lock()
		calls = append(calls, capturedNativeConversationRequest{
			method:         r.Method,
			path:           r.URL.Path,
			rawQuery:       r.URL.RawQuery,
			body:           string(body),
			authorization:  r.Header.Get("Authorization"),
			accept:         r.Header.Get("Accept"),
			openAIBeta:     r.Header.Get("OpenAI-Beta"),
			idempotencyKey: r.Header.Get("Idempotency-Key"),
		})
		mu.Unlock()

		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Upstream-Trace", "trace-native")
		w.Header().Set("Set-Cookie", "secret=blocked")
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/v1/conversations":
			w.WriteHeader(http.StatusCreated)
			_, _ = io.WriteString(w, createResult)
		case r.Method == http.MethodGet && r.URL.Path == "/v1/conversations/conv_native_crud":
			_, _ = io.WriteString(w, getResult)
		case r.Method == http.MethodPost && r.URL.Path == "/v1/conversations/conv_native_crud":
			_, _ = io.WriteString(w, updateResult)
		case r.Method == http.MethodDelete && r.URL.Path == "/v1/conversations/conv_native_crud":
			_, _ = io.WriteString(w, deleteResult)
		default:
			t.Errorf("unexpected upstream request: %s %s", r.Method, r.URL.RequestURI())
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer upstream.Close()

	router, handler, cache := newNativeLifecycleRouter(t, upstream.URL+"/v1", map[string]config.ProviderCapabilities{
		"native": {Conversations: true},
	})
	defer cache.Stop()

	create := performNativeConversationRequest(
		t,
		router,
		http.MethodPost,
		"/v1/conversations?include%5B%5D=reasoning.encrypted_content&future=a%2Fb",
		createBody,
		map[string]string{
			"Accept":          "application/json",
			"OpenAI-Beta":     "responses=v1",
			"Idempotency-Key": "conversation-create-1",
		},
	)
	if create.Code != http.StatusCreated || create.Body.String() != createResult {
		t.Fatalf("create = %d %q", create.Code, create.Body.String())
	}
	assertNativeConversationResponseHeaders(t, create, "native")
	if binding, ok := handler.conversationBindings.get("conv_native_crud"); !ok || binding.Provider != "native" {
		t.Fatalf("create binding = %#v, ok = %v", binding, ok)
	}

	get := performNativeConversationRequest(t, router, http.MethodGet, "/v1/conversations/conv_native_crud?include%5B%5D=reasoning.encrypted_content", "", nil)
	if get.Code != http.StatusOK || get.Body.String() != getResult {
		t.Fatalf("get = %d %q", get.Code, get.Body.String())
	}
	assertNativeConversationResponseHeaders(t, get, "native")

	update := performNativeConversationRequest(t, router, http.MethodPost, "/v1/conversations/conv_native_crud?future=keep", updateBody, nil)
	if update.Code != http.StatusOK || update.Body.String() != updateResult {
		t.Fatalf("update = %d %q", update.Code, update.Body.String())
	}

	deleted := performNativeConversationRequest(t, router, http.MethodDelete, "/v1/conversations/conv_native_crud?future=keep", "", nil)
	if deleted.Code != http.StatusOK || deleted.Body.String() != deleteResult {
		t.Fatalf("delete = %d %q", deleted.Code, deleted.Body.String())
	}
	if _, ok := handler.conversationBindings.get("conv_native_crud"); ok {
		t.Fatal("successful delete retained native binding")
	}

	mu.Lock()
	defer mu.Unlock()
	if len(calls) != 4 {
		t.Fatalf("upstream calls = %d, want 4", len(calls))
	}
	wants := []capturedNativeConversationRequest{
		{method: http.MethodPost, path: "/v1/conversations", rawQuery: "include%5B%5D=reasoning.encrypted_content&future=a%2Fb", body: createBody},
		{method: http.MethodGet, path: "/v1/conversations/conv_native_crud", rawQuery: "include%5B%5D=reasoning.encrypted_content"},
		{method: http.MethodPost, path: "/v1/conversations/conv_native_crud", rawQuery: "future=keep", body: updateBody},
		{method: http.MethodDelete, path: "/v1/conversations/conv_native_crud", rawQuery: "future=keep"},
	}
	for index, call := range calls {
		want := wants[index]
		if call.method != want.method || call.path != want.path || call.rawQuery != want.rawQuery || call.body != want.body {
			t.Errorf("call %d = %#v, want transport %#v", index, call, want)
		}
		if call.authorization != "Bearer provider-secret" {
			t.Errorf("call %d authorization was not provider credential", index)
		}
	}
	if calls[0].accept != "application/json" || calls[0].openAIBeta != "responses=v1" || calls[0].idempotencyKey != "conversation-create-1" {
		t.Fatalf("forwarded create headers = %#v", calls[0])
	}
}

func TestNativeConversationCreateBindsOnlySuccessfulValidBoundedResponses(t *testing.T) {
	tests := []struct {
		name          string
		status        int
		body          string
		wantStatus    int
		wantRawBody   bool
		wantBindingID string
	}{
		{
			name:          "successful valid response",
			status:        http.StatusOK,
			body:          `{"id":"conv_valid","object":"conversation"}`,
			wantStatus:    http.StatusOK,
			wantRawBody:   true,
			wantBindingID: "conv_valid",
		},
		{
			name:        "successful noncanonical nonempty ID stays lossless but unbound",
			status:      http.StatusOK,
			body:        `{"id":"third_party_conversation","object":"conversation"}`,
			wantStatus:  http.StatusOK,
			wantRawBody: true,
		},
		{
			name:        "upstream rejection remains provider response",
			status:      http.StatusBadRequest,
			body:        `{"id":"conv_rejected","error":{"message":"bad"}}`,
			wantStatus:  http.StatusBadRequest,
			wantRawBody: true,
		},
		{name: "missing ID", status: http.StatusOK, body: `{"object":"conversation"}`, wantStatus: http.StatusBadGateway},
		{name: "blank ID", status: http.StatusOK, body: `{"id":"  ","object":"conversation"}`, wantStatus: http.StatusBadGateway},
		{name: "non-string ID", status: http.StatusOK, body: `{"id":7,"object":"conversation"}`, wantStatus: http.StatusBadGateway},
		{name: "missing object", status: http.StatusOK, body: `{"id":"conv_missing_object"}`, wantStatus: http.StatusBadGateway},
		{name: "wrong object", status: http.StatusOK, body: `{"id":"conv_wrong_object","object":"response"}`, wantStatus: http.StatusBadGateway},
		{name: "invalid JSON", status: http.StatusOK, body: `{"id":"conv_truncated"`, wantStatus: http.StatusBadGateway},
		{name: "trailing JSON", status: http.StatusOK, body: `{"id":"conv_first","object":"conversation"} {}`, wantStatus: http.StatusBadGateway},
		{name: "JSON array", status: http.StatusOK, body: `[{"id":"conv_array","object":"conversation"}]`, wantStatus: http.StatusBadGateway},
		{
			name:       "response exceeds validation bound",
			status:     http.StatusOK,
			body:       `{"id":"conv_large","object":"conversation","padding":"` + strings.Repeat("x", maxNativeConversationResponseBytes) + `"}`,
			wantStatus: http.StatusBadGateway,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(test.status)
				_, _ = io.WriteString(w, test.body)
			}))
			defer upstream.Close()
			router, handler, cache := newNativeLifecycleRouter(t, upstream.URL+"/v1", map[string]config.ProviderCapabilities{
				"native": {Conversations: true},
			})
			defer cache.Stop()

			response := performNativeConversationRequest(t, router, http.MethodPost, "/v1/conversations", `{}`, nil)
			if response.Code != test.wantStatus {
				t.Fatalf("response status = %d, want %d; body=%s", response.Code, test.wantStatus, response.Body.String())
			}
			if test.wantRawBody {
				if response.Body.String() != test.body {
					t.Fatalf("response body = %q, want raw %q", response.Body.String(), test.body)
				}
			} else {
				assertInvalidNativeConversationResponse(t, response)
			}
			if test.wantBindingID != "" {
				if _, bound := handler.conversationBindings.get(test.wantBindingID); !bound {
					t.Fatalf("expected binding for %q", test.wantBindingID)
				}
				return
			}
			for _, conversationID := range []string{
				"third_party_conversation",
				"conv_rejected",
				"conv_missing_object",
				"conv_wrong_object",
				"conv_truncated",
				"conv_first",
				"conv_array",
				"conv_large",
			} {
				if _, bound := handler.conversationBindings.get(conversationID); bound {
					t.Fatalf("invalid response retained binding for %q", conversationID)
				}
			}
		})
	}
}

func TestNativeConversationCreateAcceptsExactResponseSizeLimit(t *testing.T) {
	prefix := `{"id":"conv_exact_limit","object":"conversation","padding":"`
	suffix := `"}`
	body := prefix + strings.Repeat("x", maxNativeConversationResponseBytes-len(prefix)-len(suffix)) + suffix
	if len(body) != maxNativeConversationResponseBytes {
		t.Fatalf("fixture size = %d, want %d", len(body), maxNativeConversationResponseBytes)
	}

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, body)
	}))
	defer upstream.Close()
	router, handler, cache := newNativeLifecycleRouter(t, upstream.URL+"/v1", map[string]config.ProviderCapabilities{
		"native": {Conversations: true},
	})
	defer cache.Stop()

	response := performNativeConversationRequest(t, router, http.MethodPost, "/v1/conversations", `{}`, nil)
	if response.Code != http.StatusOK || response.Body.String() != body {
		t.Fatalf("response = %d with %d bytes, want 200 with %d raw bytes", response.Code, response.Body.Len(), len(body))
	}
	if _, bound := handler.conversationBindings.get("conv_exact_limit"); !bound {
		t.Fatal("exact-limit valid response did not retain binding")
	}
}

func TestNativeConversationReadAndUpdateRejectInvalidSuccessResponses(t *testing.T) {
	const conversationID = "conv_expected"
	tests := []struct {
		name   string
		method string
		body   string
	}{
		{name: "get missing ID", method: http.MethodGet, body: `{"object":"conversation"}`},
		{name: "get mismatched ID", method: http.MethodGet, body: `{"id":"conv_other","object":"conversation"}`},
		{name: "get whitespace-padded ID", method: http.MethodGet, body: `{"id":" conv_expected ","object":"conversation"}`},
		{name: "get wrong object", method: http.MethodGet, body: `{"id":"conv_expected","object":"response"}`},
		{name: "get trailing object", method: http.MethodGet, body: `{"id":"conv_expected","object":"conversation"} {}`},
		{name: "update missing ID", method: http.MethodPost, body: `{"object":"conversation"}`},
		{name: "update mismatched ID", method: http.MethodPost, body: `{"id":"conv_other","object":"conversation"}`},
		{name: "update wrong object", method: http.MethodPost, body: `{"id":"conv_expected","object":"conversation.updated"}`},
		{
			name:   "update response above limit",
			method: http.MethodPost,
			body:   `{"id":"conv_expected","object":"conversation","padding":"` + strings.Repeat("x", maxNativeConversationResponseBytes) + `"}`,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, test.body)
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
			handler.conversationBindings.put(conversationID, binding)

			requestBody := ""
			if test.method == http.MethodPost {
				requestBody = `{"metadata":{"stage":"test"}}`
			}
			response := performNativeConversationRequest(
				t,
				router,
				test.method,
				"/v1/conversations/"+conversationID,
				requestBody,
				nil,
			)
			assertInvalidNativeConversationResponse(t, response)
			if _, bound := handler.conversationBindings.get(conversationID); !bound {
				t.Fatal("invalid read/update response removed native binding")
			}
		})
	}
}

func TestNativeConversationDeleteValidatesConfirmationBeforeRemovingBinding(t *testing.T) {
	const conversationID = "conv_delete_contract"
	tests := []struct {
		name string
		body string
	}{
		{name: "missing ID", body: `{"object":"conversation.deleted","deleted":true}`},
		{name: "mismatched ID", body: `{"id":"conv_other","object":"conversation.deleted","deleted":true}`},
		{name: "wrong object", body: `{"id":"conv_delete_contract","object":"conversation","deleted":true}`},
		{name: "missing deleted", body: `{"id":"conv_delete_contract","object":"conversation.deleted"}`},
		{name: "deleted false", body: `{"id":"conv_delete_contract","object":"conversation.deleted","deleted":false}`},
		{name: "deleted non-boolean", body: `{"id":"conv_delete_contract","object":"conversation.deleted","deleted":"true"}`},
		{name: "trailing object", body: `{"id":"conv_delete_contract","object":"conversation.deleted","deleted":true} {}`},
		{
			name: "response above limit",
			body: `{"id":"conv_delete_contract","object":"conversation.deleted","deleted":true,"padding":"` +
				strings.Repeat("x", maxNativeConversationResponseBytes) + `"}`,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, test.body)
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
			handler.conversationBindings.put(conversationID, binding)

			response := performNativeConversationRequest(
				t,
				router,
				http.MethodDelete,
				"/v1/conversations/"+conversationID,
				"",
				nil,
			)
			assertInvalidNativeConversationResponse(t, response)
			if _, bound := handler.conversationBindings.get(conversationID); !bound {
				t.Fatal("invalid delete confirmation removed native binding")
			}
		})
	}
}

func TestNativeConversationOwnerCollisionFailsClosed(t *testing.T) {
	const conversationID = "conv_shared"
	var alphaCalls atomic.Int32
	var betaCalls atomic.Int32
	newUpstream := func(calls *atomic.Int32) *httptest.Server {
		return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			calls.Add(1)
			if r.Method != http.MethodPost || r.URL.Path != "/v1/conversations" {
				t.Errorf("unexpected lifecycle request %s %s", r.Method, r.URL.Path)
				w.WriteHeader(http.StatusInternalServerError)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"id":"`+conversationID+`","object":"conversation"}`)
		}))
	}

	alpha := newUpstream(&alphaCalls)
	defer alpha.Close()
	beta := newUpstream(&betaCalls)
	defer beta.Close()
	router, handler, cache := newNativeLifecycleRouterFromConfigs(t, map[string]config.ProviderConfig{
		"alpha": {
			Type:         "openai",
			APIKey:       "alpha-secret",
			BaseURL:      alpha.URL + "/v1",
			DefaultModel: "gpt-native",
			Capabilities: config.ProviderCapabilities{Conversations: true},
		},
		"beta": {
			Type:         "openai",
			APIKey:       "beta-secret",
			BaseURL:      beta.URL + "/v1",
			DefaultModel: "gpt-native",
			Capabilities: config.ProviderCapabilities{Conversations: true},
		},
	})
	defer cache.Stop()

	for _, provider := range []string{"alpha", "beta"} {
		created := performNativeConversationRequest(t, router, http.MethodPost, "/v1/conversations", `{}`, map[string]string{
			"X-LunarGate-Provider": provider,
		})
		if created.Code != http.StatusOK {
			t.Fatalf("provider %s create status = %d; body=%s", provider, created.Code, created.Body.String())
		}
	}
	if _, lookup := handler.conversationBindings.lookup(conversationID); lookup != ownerLookupConflict {
		t.Fatalf("owner lookup = %v, want conflict", lookup)
	}

	implicit := performNativeConversationRequest(t, router, http.MethodGet, "/v1/conversations/"+conversationID, "", nil)
	assertConversationError(t, implicit, http.StatusBadRequest, "conversation_id", "provider_binding_conflict")
	if alphaCalls.Load() != 1 || betaCalls.Load() != 1 {
		t.Fatalf("upstream calls after conflict: alpha=%d beta=%d, want one create each", alphaCalls.Load(), betaCalls.Load())
	}
}

func TestNativeConversationExistingOwnerResolution(t *testing.T) {
	var calls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		_, _ = io.WriteString(w, `{"id":"conv_external","object":"conversation","native":true}`)
	}))
	defer upstream.Close()
	router, handler, cache := newNativeLifecycleRouter(t, upstream.URL+"/v1", map[string]config.ProviderCapabilities{
		"native": {Conversations: true},
	})
	defer cache.Stop()

	missing := performNativeConversationRequest(t, router, http.MethodGet, "/v1/conversations/conv_external", "", nil)
	assertConversationError(t, missing, http.StatusNotFound, "conversation_id", "conversation_not_found")
	if calls.Load() != 0 {
		t.Fatalf("unknown conversation caused %d implicit upstream calls", calls.Load())
	}

	explicit := performNativeConversationRequest(t, router, http.MethodGet, "/v1/conversations/conv_external?include%5B%5D=x", "", map[string]string{
		"X-LunarGate-Provider": "native",
	})
	if explicit.Code != http.StatusOK || !strings.Contains(explicit.Body.String(), `"native":true`) {
		t.Fatalf("explicit recovery = %d %s", explicit.Code, explicit.Body.String())
	}
	if calls.Load() != 1 {
		t.Fatalf("explicit recovery calls = %d, want 1", calls.Load())
	}

	local, err := handler.conversationsState.create(map[string]string{"owner": "local"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	localResponse := performNativeConversationRequest(t, router, http.MethodGet, "/v1/conversations/"+local.ID, "", map[string]string{
		"X-LunarGate-Provider": "native",
	})
	if localResponse.Code != http.StatusOK || !strings.Contains(localResponse.Body.String(), `"owner":"local"`) {
		t.Fatalf("local precedence = %d %s", localResponse.Code, localResponse.Body.String())
	}
	if calls.Load() != 1 {
		t.Fatalf("local conversation unexpectedly reached upstream: %d calls", calls.Load())
	}
}

func TestNativeConversationDeleteRetainsBindingOnFailure(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = io.WriteString(w, `{"error":{"message":"retry later"}}`)
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
	handler.conversationBindings.put("conv_delete", binding)

	response := performNativeConversationRequest(t, router, http.MethodDelete, "/v1/conversations/conv_delete", "", nil)
	if response.Code != http.StatusServiceUnavailable {
		t.Fatalf("delete status = %d, body = %s", response.Code, response.Body.String())
	}
	if _, ok := handler.conversationBindings.get("conv_delete"); !ok {
		t.Fatal("failed delete released native binding")
	}
}

func TestNativeConversationItemsPreserveEnvelopeAndQuery(t *testing.T) {
	const (
		createBody   = `{"items":[{"type":"future_item","payload":{"keep":true}}],"future_create":true}`
		createResult = `{"object":"list","data":[{"id":"item_native","type":"future_item","future":true}],"future_list":"kept"}`
		listResult   = `{"object":"list","data":[{"id":"item_native","type":"future_item"}],"has_more":false,"future_list":"kept"}`
		getResult    = `{"id":"item_native","type":"future_item","future_get":"kept"}`
		deleteResult = `{"id":"conv_native_items","object":"conversation","future_delete":"kept"}`
	)

	var mu sync.Mutex
	calls := make([]capturedNativeConversationRequest, 0, 4)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read item body: %v", err)
			return
		}
		mu.Lock()
		calls = append(calls, capturedNativeConversationRequest{
			method:        r.Method,
			path:          r.URL.Path,
			rawQuery:      r.URL.RawQuery,
			body:          string(body),
			authorization: r.Header.Get("Authorization"),
		})
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/v1/conversations/conv_native_items/items":
			_, _ = io.WriteString(w, createResult)
		case r.Method == http.MethodGet && r.URL.Path == "/v1/conversations/conv_native_items/items":
			_, _ = io.WriteString(w, listResult)
		case r.Method == http.MethodGet && r.URL.Path == "/v1/conversations/conv_native_items/items/item_native":
			_, _ = io.WriteString(w, getResult)
		case r.Method == http.MethodDelete && r.URL.Path == "/v1/conversations/conv_native_items/items/item_native":
			_, _ = io.WriteString(w, deleteResult)
		default:
			t.Errorf("unexpected native items request: %s %s", r.Method, r.URL.RequestURI())
			w.WriteHeader(http.StatusNotFound)
		}
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
	handler.conversationBindings.put("conv_native_items", binding)

	created := performNativeConversationRequest(
		t,
		router,
		http.MethodPost,
		"/v1/conversations/conv_native_items/items?include%5B%5D=reasoning.encrypted_content&future=a%2Fb",
		createBody,
		nil,
	)
	if created.Code != http.StatusOK || created.Body.String() != createResult {
		t.Fatalf("create items = %d %q", created.Code, created.Body.String())
	}

	listed := performNativeConversationRequest(
		t,
		router,
		http.MethodGet,
		"/v1/conversations/conv_native_items/items?after=item_previous&limit=17&order=asc&include%5B%5D=message.input_image.image_url",
		"",
		nil,
	)
	if listed.Code != http.StatusOK || listed.Body.String() != listResult {
		t.Fatalf("list items = %d %q", listed.Code, listed.Body.String())
	}

	item := performNativeConversationRequest(t, router, http.MethodGet, "/v1/conversations/conv_native_items/items/item_native?include=future", "", nil)
	if item.Code != http.StatusOK || item.Body.String() != getResult {
		t.Fatalf("get item = %d %q", item.Code, item.Body.String())
	}

	deleted := performNativeConversationRequest(t, router, http.MethodDelete, "/v1/conversations/conv_native_items/items/item_native?include=future", "", nil)
	if deleted.Code != http.StatusOK || deleted.Body.String() != deleteResult {
		t.Fatalf("delete item = %d %q", deleted.Code, deleted.Body.String())
	}
	if _, ok := handler.conversationBindings.get("conv_native_items"); !ok {
		t.Fatal("item deletion released conversation binding")
	}

	mu.Lock()
	defer mu.Unlock()
	wants := []capturedNativeConversationRequest{
		{method: http.MethodPost, path: "/v1/conversations/conv_native_items/items", rawQuery: "include%5B%5D=reasoning.encrypted_content&future=a%2Fb", body: createBody},
		{method: http.MethodGet, path: "/v1/conversations/conv_native_items/items", rawQuery: "after=item_previous&limit=17&order=asc&include%5B%5D=message.input_image.image_url"},
		{method: http.MethodGet, path: "/v1/conversations/conv_native_items/items/item_native", rawQuery: "include=future"},
		{method: http.MethodDelete, path: "/v1/conversations/conv_native_items/items/item_native", rawQuery: "include=future"},
	}
	if len(calls) != len(wants) {
		t.Fatalf("item calls = %d, want %d", len(calls), len(wants))
	}
	for index, call := range calls {
		want := wants[index]
		if call.method != want.method || call.path != want.path || call.rawQuery != want.rawQuery || call.body != want.body {
			t.Errorf("item call %d = %#v, want %#v", index, call, want)
		}
		if call.authorization != "Bearer provider-secret" {
			t.Errorf("item call %d did not use provider credentials", index)
		}
	}
}

func TestNativeConversationItemsRejectInvalidSuccessResponses(t *testing.T) {
	const conversationID = "conv_items_contract"
	tests := []struct {
		name        string
		method      string
		path        string
		requestBody string
		response    string
	}{
		{
			name:        "create missing list object",
			method:      http.MethodPost,
			path:        "/v1/conversations/conv_items_contract/items",
			requestBody: `{"items":[{"type":"future_item","payload":{"keep":true}}]}`,
			response:    `{"data":[{"id":"item_expected","type":"future_item"}]}`,
		},
		{
			name:        "create wrong object",
			method:      http.MethodPost,
			path:        "/v1/conversations/conv_items_contract/items",
			requestBody: `{"items":[{"type":"future_item"}]}`,
			response:    `{"object":"conversation","data":[]}`,
		},
		{
			name:        "create trailing JSON",
			method:      http.MethodPost,
			path:        "/v1/conversations/conv_items_contract/items",
			requestBody: `{"items":[{"type":"future_item"}]}`,
			response:    `{"object":"list","data":[]} {}`,
		},
		{
			name:        "create response above limit",
			method:      http.MethodPost,
			path:        "/v1/conversations/conv_items_contract/items",
			requestBody: `{"items":[{"type":"future_item"}]}`,
			response: `{"object":"list","padding":"` +
				strings.Repeat("x", maxNativeConversationResponseBytes) + `"}`,
		},
		{
			name:     "list missing list object",
			method:   http.MethodGet,
			path:     "/v1/conversations/conv_items_contract/items",
			response: `{"data":[],"has_more":false}`,
		},
		{
			name:     "list wrong object",
			method:   http.MethodGet,
			path:     "/v1/conversations/conv_items_contract/items",
			response: `{"object":"conversation.item.list","data":[]}`,
		},
		{
			name:     "list is array instead of object",
			method:   http.MethodGet,
			path:     "/v1/conversations/conv_items_contract/items",
			response: `[{"id":"item_expected","type":"future_item"}]`,
		},
		{
			name:     "get missing item ID",
			method:   http.MethodGet,
			path:     "/v1/conversations/conv_items_contract/items/item_expected",
			response: `{"type":"future_item"}`,
		},
		{
			name:     "get non-string item ID",
			method:   http.MethodGet,
			path:     "/v1/conversations/conv_items_contract/items/item_expected",
			response: `{"id":7,"type":"future_item"}`,
		},
		{
			name:     "get whitespace-padded item ID",
			method:   http.MethodGet,
			path:     "/v1/conversations/conv_items_contract/items/item_expected",
			response: `{"id":" item_expected ","type":"future_item"}`,
		},
		{
			name:     "get mismatched item ID",
			method:   http.MethodGet,
			path:     "/v1/conversations/conv_items_contract/items/item_expected",
			response: `{"id":"item_other","type":"future_item"}`,
		},
		{
			name:     "delete item missing conversation ID",
			method:   http.MethodDelete,
			path:     "/v1/conversations/conv_items_contract/items/item_expected",
			response: `{"object":"conversation"}`,
		},
		{
			name:     "delete item returns item ID",
			method:   http.MethodDelete,
			path:     "/v1/conversations/conv_items_contract/items/item_expected",
			response: `{"id":"item_expected","object":"conversation"}`,
		},
		{
			name:     "delete item mismatched conversation ID",
			method:   http.MethodDelete,
			path:     "/v1/conversations/conv_items_contract/items/item_expected",
			response: `{"id":"conv_other","object":"conversation"}`,
		},
		{
			name:     "delete item wrong object",
			method:   http.MethodDelete,
			path:     "/v1/conversations/conv_items_contract/items/item_expected",
			response: `{"id":"conv_items_contract","object":"conversation.deleted"}`,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, test.response)
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
			if !handler.conversationBindings.put(conversationID, binding) {
				t.Fatal("failed to seed native conversation binding")
			}

			response := performNativeConversationRequest(
				t,
				router,
				test.method,
				test.path,
				test.requestBody,
				nil,
			)
			assertInvalidNativeConversationResponse(t, response)
			retained, ok := handler.conversationBindings.get(conversationID)
			if !ok || !sameConversationBindingOwner(retained, binding) {
				t.Fatalf("invalid item response changed binding: got %#v, ok=%v; want %#v", retained, ok, binding)
			}
			if _, local := handler.conversationsState.get(conversationID); local {
				t.Fatal("native item request mutated local conversation state")
			}
		})
	}
}

func TestNativeConversationItemsAcceptExactResponseSizeLimit(t *testing.T) {
	const conversationID = "conv_items_exact_limit"
	prefix := `{"object":"list","padding":"`
	suffix := `"}`
	body := prefix + strings.Repeat("x", maxNativeConversationResponseBytes-len(prefix)-len(suffix)) + suffix
	if len(body) != maxNativeConversationResponseBytes {
		t.Fatalf("fixture size = %d, want %d", len(body), maxNativeConversationResponseBytes)
	}

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, body)
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
	if !handler.conversationBindings.put(conversationID, binding) {
		t.Fatal("failed to seed native conversation binding")
	}

	response := performNativeConversationRequest(
		t,
		router,
		http.MethodGet,
		"/v1/conversations/"+conversationID+"/items",
		"",
		nil,
	)
	if response.Code != http.StatusOK || response.Body.String() != body {
		t.Fatalf("response = %d with %d bytes, want 200 with %d raw bytes", response.Code, response.Body.Len(), len(body))
	}
	retained, ok := handler.conversationBindings.get(conversationID)
	if !ok || !sameConversationBindingOwner(retained, binding) {
		t.Fatalf("valid item list changed binding: got %#v, ok=%v; want %#v", retained, ok, binding)
	}
}

func TestNativeConversationItemsPreserveUpstreamErrors(t *testing.T) {
	const (
		conversationID = "conv_items_error"
		upstreamBody   = `{"error":{"message":"rate limited","future":{"retry":true}}}`
	)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = io.WriteString(w, upstreamBody)
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
	if !handler.conversationBindings.put(conversationID, binding) {
		t.Fatal("failed to seed native conversation binding")
	}

	response := performNativeConversationRequest(
		t,
		router,
		http.MethodPost,
		"/v1/conversations/"+conversationID+"/items",
		`{"items":[{"type":"future_item"}]}`,
		nil,
	)
	if response.Code != http.StatusTooManyRequests || response.Body.String() != upstreamBody {
		t.Fatalf("response = %d %q, want raw upstream 429", response.Code, response.Body.String())
	}
	retained, ok := handler.conversationBindings.get(conversationID)
	if !ok || !sameConversationBindingOwner(retained, binding) {
		t.Fatalf("upstream error changed binding: got %#v, ok=%v; want %#v", retained, ok, binding)
	}
}

func TestNativeConversationTransportIsSingleHopAndSanitized(t *testing.T) {
	var redirectedCalls atomic.Int32
	redirectTarget := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		redirectedCalls.Add(1)
	}))
	defer redirectTarget.Close()
	redirectSource := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Location", redirectTarget.URL+"/secret-target")
		w.WriteHeader(http.StatusTemporaryRedirect)
	}))
	defer redirectSource.Close()

	router, _, cache := newNativeLifecycleRouter(t, redirectSource.URL+"/v1", map[string]config.ProviderCapabilities{
		"native": {Conversations: true},
	})
	response := performNativeConversationRequest(t, router, http.MethodPost, "/v1/conversations", `{}`, nil)
	cache.Stop()
	if response.Code != http.StatusTemporaryRedirect || redirectedCalls.Load() != 0 {
		t.Fatalf("redirect response = %d, target calls = %d", response.Code, redirectedCalls.Load())
	}
	if response.Header().Get("Location") != redirectTarget.URL+"/secret-target" {
		t.Fatalf("location = %q", response.Header().Get("Location"))
	}

	providerConfigs := map[string]config.ProviderConfig{
		"native": {
			Type:         "openai",
			APIKey:       "provider-secret-must-not-leak",
			BaseURL:      "http://127.0.0.1:1/private-upstream",
			DefaultModel: "gpt-native",
			Timeout:      time.Second,
			Capabilities: config.ProviderCapabilities{Conversations: true},
		},
	}
	failingRouter, _, failingCache := newNativeLifecycleRouterFromConfigs(t, providerConfigs)
	defer failingCache.Stop()
	failure := performNativeConversationRequest(t, failingRouter, http.MethodPost, "/v1/conversations", `{}`, nil)
	if failure.Code != http.StatusBadGateway {
		t.Fatalf("transport failure = %d %s", failure.Code, failure.Body.String())
	}
	for _, secret := range []string{"127.0.0.1", "private-upstream", "provider-secret-must-not-leak"} {
		if strings.Contains(failure.Body.String(), secret) {
			t.Fatalf("client error leaked %q: %s", secret, failure.Body.String())
		}
	}
}

func performNativeConversationRequest(
	t *testing.T,
	handler http.Handler,
	method string,
	path string,
	body string,
	headers map[string]string,
) *httptest.ResponseRecorder {
	t.Helper()
	request := httptest.NewRequest(method, path, bytes.NewBufferString(body))
	if body != "" {
		request.Header.Set("Content-Type", "application/json")
	}
	for key, value := range headers {
		request.Header.Set(key, value)
	}
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	return response
}

func assertNativeConversationResponseHeaders(t *testing.T, response *httptest.ResponseRecorder, provider string) {
	t.Helper()
	if response.Header().Get("X-LunarGate-Provider") != provider {
		t.Fatalf("provider header = %q", response.Header().Get("X-LunarGate-Provider"))
	}
	if response.Header().Get("X-Upstream-Trace") != "trace-native" {
		t.Fatalf("safe upstream header = %q", response.Header().Get("X-Upstream-Trace"))
	}
	if response.Header().Get("Set-Cookie") != "" {
		t.Fatal("unsafe Set-Cookie header was forwarded")
	}
}

func assertInvalidNativeConversationResponse(t *testing.T, response *httptest.ResponseRecorder) {
	t.Helper()
	if response.Code != http.StatusBadGateway {
		t.Fatalf("response status = %d, want 502; body=%s", response.Code, response.Body.String())
	}
	var envelope struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
		} `json:"error"`
	}
	if err := json.Unmarshal(response.Body.Bytes(), &envelope); err != nil {
		t.Fatalf("decode sanitized error %q: %v", response.Body.String(), err)
	}
	if envelope.Error.Type != "provider_error" ||
		envelope.Error.Message != "native conversation provider returned an invalid response object" {
		t.Fatalf("unexpected sanitized error: %#v", envelope.Error)
	}
}
