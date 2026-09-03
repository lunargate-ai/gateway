package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/lunargate-ai/gateway/internal/config"
)

func TestOwnerBindingStoresRejectNonCanonicalResourceIDs(t *testing.T) {
	responseStore := newResponseBindingStore(time.Hour)
	responseOwner := responseBinding{Provider: "provider", AccountFingerprint: "account"}
	conversationStore := newConversationBindingStore(time.Hour)
	conversationOwner := conversationBinding{Provider: "provider", AccountFingerprint: "account"}
	chatStore := newChatCompletionBindingStore(time.Hour)
	chatOwner := chatCompletionBinding{Provider: "provider", AccountFingerprint: "account"}

	tests := []struct {
		name     string
		prefix   string
		claim    func(string) ownerClaimResult
		put      func(string) bool
		lookup   func(string) ownerLookupResult
		deleteID func(string) bool
	}{
		{
			name:   "responses",
			prefix: "resp_",
			claim: func(id string) ownerClaimResult {
				return responseStore.claim(id, responseOwner)
			},
			put: func(id string) bool {
				return responseStore.put(id, responseOwner)
			},
			lookup: func(id string) ownerLookupResult {
				_, result := responseStore.lookup(id)
				return result
			},
			deleteID: func(id string) bool {
				return responseStore.deleteIfOwned(id, responseOwner)
			},
		},
		{
			name:   "conversations",
			prefix: "conv_",
			claim: func(id string) ownerClaimResult {
				return conversationStore.claim(id, conversationOwner)
			},
			put: func(id string) bool {
				return conversationStore.put(id, conversationOwner)
			},
			lookup: func(id string) ownerLookupResult {
				_, result := conversationStore.lookup(id)
				return result
			},
			deleteID: func(id string) bool {
				return conversationStore.deleteIfOwned(id, conversationOwner)
			},
		},
		{
			name:   "stored chat completions",
			prefix: "chatcmpl_",
			claim: func(id string) ownerClaimResult {
				return chatStore.claim(id, chatOwner)
			},
			put: func(id string) bool {
				return chatStore.put(id, chatOwner)
			},
			lookup: func(id string) ownerLookupResult {
				_, result := chatStore.lookup(id)
				return result
			},
			deleteID: func(id string) bool {
				return chatStore.deleteIfOwned(id, chatOwner)
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			canonical := test.prefix + "canonical"
			padded := " " + canonical + " "
			internalSpace := test.prefix + "internal space"

			if got := test.claim(padded); got != ownerClaimUnavailable {
				t.Fatalf("padded claim = %v, want unavailable", got)
			}
			if test.put(padded) {
				t.Fatal("padded put succeeded")
			}
			if !test.put(canonical) {
				t.Fatal("canonical put failed")
			}
			if got := test.lookup(padded); got != ownerLookupMissing {
				t.Fatalf("padded lookup = %v, want missing", got)
			}
			if test.deleteID(padded) {
				t.Fatal("padded ID deleted the canonical binding")
			}
			if got := test.lookup(canonical); got != ownerLookupBound {
				t.Fatalf("canonical lookup after padded delete = %v, want bound", got)
			}
			if !test.put(internalSpace) || test.lookup(internalSpace) != ownerLookupBound {
				t.Fatal("internal whitespace was not preserved as part of the opaque ID")
			}
		})
	}
}

func TestConversationURLResourceIDsRejectSurroundingWhitespace(t *testing.T) {
	handler := &Handler{conversationsState: newConversationStateStore(time.Hour)}
	router := conversationTestRouter(handler)
	created := performConversationRequest(
		t,
		router,
		http.MethodPost,
		"/v1/conversations",
		`{"metadata":{"stable":"yes"},"items":[{"type":"message","id":"msg_target","role":"user","content":"one"}]}`,
	)
	if created.Code != http.StatusOK {
		t.Fatalf("create status = %d; body=%s", created.Code, created.Body.String())
	}
	var conversation conversationObject
	decodeConversationResponse(t, created, &conversation)

	conversationPaths := []struct {
		name   string
		method string
		path   string
		body   string
	}{
		{name: "get", method: http.MethodGet, path: "/v1/conversations/%20" + conversation.ID},
		{name: "update", method: http.MethodPost, path: "/v1/conversations/" + conversation.ID + "%20", body: `{"metadata":{"stable":"no"}}`},
		{name: "delete", method: http.MethodDelete, path: "/v1/conversations/%20" + conversation.ID},
		{name: "create items", method: http.MethodPost, path: "/v1/conversations/" + conversation.ID + "%20/items", body: `{"items":[{"type":"message","role":"user","content":"must not be stored"}]}`},
		{name: "list items", method: http.MethodGet, path: "/v1/conversations/%20" + conversation.ID + "/items"},
		{name: "get item", method: http.MethodGet, path: "/v1/conversations/" + conversation.ID + "%20/items/msg_target"},
		{name: "delete item", method: http.MethodDelete, path: "/v1/conversations/%20" + conversation.ID + "/items/msg_target"},
	}
	for _, test := range conversationPaths {
		t.Run("conversation id/"+test.name, func(t *testing.T) {
			response := performConversationRequest(t, router, test.method, test.path, test.body)
			assertConversationError(t, response, http.StatusBadRequest, "conversation_id", "invalid_value")
		})
	}

	for _, test := range []struct {
		name   string
		method string
		path   string
	}{
		{name: "get item", method: http.MethodGet, path: "/v1/conversations/" + conversation.ID + "/items/%20msg_target"},
		{name: "delete item", method: http.MethodDelete, path: "/v1/conversations/" + conversation.ID + "/items/msg_target%20"},
	} {
		t.Run("item id/"+test.name, func(t *testing.T) {
			response := performConversationRequest(t, router, test.method, test.path, "")
			assertConversationError(t, response, http.StatusBadRequest, "item_id", "invalid_value")
		})
	}

	paddedCursor := performConversationRequest(
		t,
		router,
		http.MethodGet,
		"/v1/conversations/"+conversation.ID+"/items?after="+url.QueryEscape(" msg_target "),
		"",
	)
	assertConversationError(t, paddedCursor, http.StatusBadRequest, "after", "invalid_value")

	stored, ok := handler.conversationsState.get(conversation.ID)
	if !ok || stored.Metadata["stable"] != "yes" {
		t.Fatalf("padded request mutated conversation: %#v, ok=%v", stored, ok)
	}
	items, ok := handler.conversationsState.getItems(conversation.ID)
	if !ok || len(items) != 1 || conversationItemID(items[0]) != "msg_target" {
		t.Fatalf("padded request mutated items: %#v, ok=%v", items, ok)
	}
}

func TestStoredChatCompletionURLIDRejectsSurroundingWhitespace(t *testing.T) {
	var upstreamCalls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		upstreamCalls.Add(1)
		_, _ = w.Write([]byte(`{"id":"chatcmpl_target","object":"chat.completion"}`))
	}))
	defer upstream.Close()

	router, handler, cache := newStoredChatLifecycleRouterFromConfigs(t, map[string]config.ProviderConfig{
		"native": storedChatProviderConfig(upstream.URL+"/v1", true),
	})
	defer cache.Stop()
	binding := mustChatCompletionBinding(t, handler, "native")
	if !handler.chatCompletionBindings.put("chatcmpl_target", binding) {
		t.Fatal("failed to seed stored Chat Completion binding")
	}

	for _, test := range []struct {
		name   string
		method string
		path   string
		body   string
	}{
		{name: "retrieve", method: http.MethodGet, path: "/v1/chat/completions/%20chatcmpl_target"},
		{name: "update", method: http.MethodPost, path: "/v1/chat/completions/chatcmpl_target%20", body: `{}`},
		{name: "delete", method: http.MethodDelete, path: "/v1/chat/completions/%20chatcmpl_target"},
		{name: "messages", method: http.MethodGet, path: "/v1/chat/completions/chatcmpl_target%20/messages"},
	} {
		t.Run(test.name, func(t *testing.T) {
			response := performStoredChatLifecycleRequest(t, router, test.method, test.path, test.body, nil)
			if response.Code != http.StatusBadRequest {
				t.Fatalf("status = %d; body=%s", response.Code, response.Body.String())
			}
			assertLifecycleError(t, response.Body.Bytes(), "completion_id", "invalid_value")
		})
	}

	if upstreamCalls.Load() != 0 {
		t.Fatalf("padded IDs made %d upstream calls", upstreamCalls.Load())
	}
	if _, ok := handler.chatCompletionBindings.get("chatcmpl_target"); !ok {
		t.Fatal("padded delete removed the canonical binding")
	}
}

func TestResponseURLIDRejectsSurroundingWhitespaceWithoutMutation(t *testing.T) {
	router, handler, upstreamCalls, closeTest := newResponsesLifecycleTestRouter(t)
	defer closeTest()
	created := performLifecycleRequest(
		t,
		router,
		http.MethodPost,
		"/v1/responses",
		[]byte(`{"model":"mock-gpt","store":true,"input":"stable input"}`),
	)
	if created.Code != http.StatusOK {
		t.Fatalf("create status = %d; body=%s", created.Code, created.Body.String())
	}
	responseID := lifecycleStringField(t, decodeLifecycleObject(t, created.Body.Bytes()), "id")

	for _, test := range []struct {
		name   string
		method string
		path   string
	}{
		{name: "retrieve", method: http.MethodGet, path: "/v1/responses/%20" + responseID},
		{name: "delete", method: http.MethodDelete, path: "/v1/responses/" + responseID + "%20"},
		{name: "cancel", method: http.MethodPost, path: "/v1/responses/%20" + responseID + "/cancel"},
		{name: "input items", method: http.MethodGet, path: "/v1/responses/" + responseID + "%20/input_items"},
	} {
		t.Run(test.name, func(t *testing.T) {
			response := performLifecycleRequest(t, router, test.method, test.path, nil)
			if response.Code != http.StatusBadRequest {
				t.Fatalf("status = %d; body=%s", response.Code, response.Body.String())
			}
			assertLifecycleError(t, response.Body.Bytes(), "response_id", "invalid_value")
		})
	}

	inputItems := performLifecycleRequest(
		t,
		router,
		http.MethodGet,
		"/v1/responses/"+responseID+"/input_items",
		nil,
	)
	items := lifecycleData(t, decodeLifecycleObject(t, inputItems.Body.Bytes()))
	if len(items) != 1 {
		t.Fatalf("input items = %#v", items)
	}
	itemID := lifecycleStringField(t, items[0], "id")
	paddedCursor := performLifecycleRequest(
		t,
		router,
		http.MethodGet,
		"/v1/responses/"+responseID+"/input_items?after="+url.QueryEscape(" "+itemID+" "),
		nil,
	)
	if paddedCursor.Code != http.StatusBadRequest {
		t.Fatalf("padded cursor status = %d; body=%s", paddedCursor.Code, paddedCursor.Body.String())
	}
	assertLifecycleError(t, paddedCursor.Body.Bytes(), "after", "invalid_value")

	canonical := performLifecycleRequest(t, router, http.MethodGet, "/v1/responses/"+responseID, nil)
	if canonical.Code != http.StatusOK {
		t.Fatalf("canonical response was mutated: %d %s", canonical.Code, canonical.Body.String())
	}
	if upstreamCalls.Load() != 1 {
		t.Fatalf("padded lifecycle requests changed upstream call count to %d", upstreamCalls.Load())
	}
	if _, _, ok := handler.responsesState.getCompleted(responseID); !ok {
		t.Fatal("padded lifecycle request removed canonical local state")
	}
}

func TestConversationItemURLPreservesInternalWhitespace(t *testing.T) {
	handler := &Handler{conversationsState: newConversationStateStore(time.Hour)}
	router := conversationTestRouter(handler)
	created := performConversationRequest(
		t,
		router,
		http.MethodPost,
		"/v1/conversations",
		`{"items":[{"type":"message","id":"msg internal space","role":"user","content":"one"}]}`,
	)
	var conversation conversationObject
	decodeConversationResponse(t, created, &conversation)

	item := performConversationRequest(
		t,
		router,
		http.MethodGet,
		"/v1/conversations/"+conversation.ID+"/items/msg%20internal%20space",
		"",
	)
	if item.Code != http.StatusOK {
		t.Fatalf("internal-space item status = %d; body=%s", item.Code, item.Body.String())
	}
}

func TestParseResponsesConversationIDRejectsOnlyBoundaryWhitespace(t *testing.T) {
	for _, raw := range []string{`" conv_target "`, `{"id":" conv_target "}`} {
		if id, err := parseResponsesConversationID([]byte(raw)); err == nil {
			t.Fatalf("parseResponsesConversationID(%s) = %q, want error", raw, id)
		}
	}
	for _, raw := range []string{`"conv internal space"`, `{"id":"conv internal space"}`} {
		if id, err := parseResponsesConversationID([]byte(raw)); err != nil || id != "conv internal space" {
			t.Fatalf("parseResponsesConversationID(%s) = %q, %v", raw, id, err)
		}
	}
}

func TestResponsesConversationBodyRejectsPaddedIDBeforeHistoryLookup(t *testing.T) {
	router, handler, upstreamCalls, closeTest := newResponsesLifecycleTestRouter(t)
	defer closeTest()
	created := performLifecycleRequest(
		t,
		router,
		http.MethodPost,
		"/v1/conversations",
		[]byte(`{"items":[{"type":"message","id":"msg_history","role":"user","content":"private history"}]}`),
	)
	if created.Code != http.StatusOK {
		t.Fatalf("conversation create status = %d; body=%s", created.Code, created.Body.String())
	}
	conversationID := lifecycleStringField(t, decodeLifecycleObject(t, created.Body.Bytes()), "id")

	for _, conversation := range []string{
		fmt.Sprintf("%q", " "+conversationID+" "),
		fmt.Sprintf(`{"id":%q}`, " "+conversationID+" "),
	} {
		response := performLifecycleRequest(
			t,
			router,
			http.MethodPost,
			"/v1/responses",
			[]byte(fmt.Sprintf(`{"model":"mock-gpt","conversation":%s,"input":"new input"}`, conversation)),
		)
		if response.Code != http.StatusBadRequest {
			t.Fatalf("conversation=%s status = %d; body=%s", conversation, response.Code, response.Body.String())
		}
		assertLifecycleError(t, response.Body.Bytes(), "conversation", "invalid_value")
	}

	items, ok := handler.conversationsState.getItems(conversationID)
	if !ok || len(items) != 1 || conversationItemID(items[0]) != "msg_history" {
		t.Fatalf("padded conversation ID mutated history: %#v, ok=%v", items, ok)
	}
	if upstreamCalls.Load() != 0 {
		t.Fatalf("padded conversation ID made %d upstream calls", upstreamCalls.Load())
	}
}

func TestResponsesPreviousResponseIDRejectsBoundaryWhitespaceWithoutLookup(t *testing.T) {
	router, handler, upstreamCalls, closeTest := newResponsesLifecycleTestRouter(t)
	defer closeTest()
	created := performLifecycleRequest(
		t,
		router,
		http.MethodPost,
		"/v1/responses",
		[]byte(`{"model":"mock-gpt","store":true,"input":"stable input"}`),
	)
	if created.Code != http.StatusOK {
		t.Fatalf("create status = %d; body=%s", created.Code, created.Body.String())
	}
	responseID := lifecycleStringField(t, decodeLifecycleObject(t, created.Body.Bytes()), "id")

	for _, invalidID := range []string{" " + responseID + " ", "", "   "} {
		response := performLifecycleRequest(
			t,
			router,
			http.MethodPost,
			"/v1/responses",
			[]byte(fmt.Sprintf(`{"model":"mock-gpt","previous_response_id":%q,"input":"must not continue"}`, invalidID)),
		)
		if response.Code != http.StatusBadRequest {
			t.Fatalf("previous_response_id=%q status = %d; body=%s", invalidID, response.Code, response.Body.String())
		}
		assertLifecycleError(t, response.Body.Bytes(), "previous_response_id", "invalid_value")
	}

	if upstreamCalls.Load() != 1 {
		t.Fatalf("invalid previous_response_id changed upstream call count to %d", upstreamCalls.Load())
	}
	if _, _, ok := handler.responsesState.getCompleted(responseID); !ok {
		t.Fatal("invalid previous_response_id removed canonical continuation state")
	}
}

func TestResponsesConversationRejectsPaddedPreviousResponseIDBeforeHistoryLookup(t *testing.T) {
	router, handler, upstreamCalls, closeTest := newResponsesLifecycleTestRouter(t)
	defer closeTest()
	created := performLifecycleRequest(
		t,
		router,
		http.MethodPost,
		"/v1/conversations",
		[]byte(`{"items":[{"type":"message","id":"msg_history","role":"user","content":"private history"}]}`),
	)
	if created.Code != http.StatusOK {
		t.Fatalf("conversation create status = %d; body=%s", created.Code, created.Body.String())
	}
	conversationID := lifecycleStringField(t, decodeLifecycleObject(t, created.Body.Bytes()), "id")

	response := performLifecycleRequest(
		t,
		router,
		http.MethodPost,
		"/v1/responses",
		[]byte(fmt.Sprintf(
			`{"model":"mock-gpt","conversation":%q,"previous_response_id":" resp_target ","input":"must not continue"}`,
			conversationID,
		)),
	)
	if response.Code != http.StatusBadRequest {
		t.Fatalf("status = %d; body=%s", response.Code, response.Body.String())
	}
	assertLifecycleError(t, response.Body.Bytes(), "previous_response_id", "invalid_value")

	items, ok := handler.conversationsState.getItems(conversationID)
	if !ok || len(items) != 1 || conversationItemID(items[0]) != "msg_history" {
		t.Fatalf("invalid previous_response_id mutated conversation history: %#v, ok=%v", items, ok)
	}
	if upstreamCalls.Load() != 0 {
		t.Fatalf("invalid previous_response_id made %d upstream calls", upstreamCalls.Load())
	}
}

func TestResponsesWebSocketPreviousResponseIDIsExactAndOpaque(t *testing.T) {
	for _, rawID := range []string{`" resp_target "`, `""`, `"   "`, `42`} {
		request, err := parseResponsesWebSocketCreateRequest([]byte(
			`{"type":"response.create","previous_response_id":` + rawID + `,"input":"continue"}`,
		))
		if request != nil {
			t.Fatalf("previous_response_id=%s request = %#v, want nil", rawID, request)
		}
		eventErr, ok := err.(*responsesWebSocketEventError)
		if !ok || eventErr.status != http.StatusBadRequest || eventErr.param != "previous_response_id" || eventErr.code != "invalid_value" {
			t.Fatalf("previous_response_id=%s error = %#v, want invalid_value", rawID, err)
		}
	}

	internalID := "resp_internal space"
	request, err := parseResponsesWebSocketCreateRequest([]byte(
		`{"type":"response.create","previous_response_id":"` + internalID + `","input":"continue"}`,
	))
	if err != nil || request.previousResponseID != internalID {
		t.Fatalf("internal-space ID parsed as %#v, %v", request, err)
	}
	nullRequest, err := parseResponsesWebSocketCreateRequest([]byte(
		`{"type":"response.create","previous_response_id":null,"input":"continue"}`,
	))
	if err != nil || nullRequest.previousResponseID != "" {
		t.Fatalf("null optional ID parsed as %#v, %v", nullRequest, err)
	}
}

func TestResponsesWebSocketStateNeverAliasesPaddedResourceID(t *testing.T) {
	canonicalID := "resp_target"
	payload := map[string]json.RawMessage{
		"model": json.RawMessage(`"mock-gpt"`),
		"input": json.RawMessage(`"stable input"`),
	}
	session := &responsesWebSocketSession{
		cachedStates:        make(map[string]*responsesWebSocketCachedState),
		maxCachedStateBytes: 1024,
	}
	if err := session.cacheState(canonicalID, payload); err != nil {
		t.Fatalf("cache canonical state: %v", err)
	}

	_, resolveErr := session.resolveCreatePayload(&responsesWebSocketCreateRequest{
		previousResponseID: " " + canonicalID + " ",
		payload:            map[string]json.RawMessage{"input": json.RawMessage(`"must not merge"`)},
		generate:           true,
	})
	eventErr, ok := resolveErr.(*responsesWebSocketEventError)
	if !ok || eventErr.status != http.StatusBadRequest || eventErr.code != "invalid_value" {
		t.Fatalf("padded resolve error = %#v, want invalid_value", resolveErr)
	}
	session.evictState(" " + canonicalID + " ")
	if session.cachedStates[canonicalID] == nil {
		t.Fatal("padded eviction removed canonical websocket state")
	}

	for _, invalidID := range []string{" " + canonicalID + " ", ""} {
		cacheErr := session.cacheState(invalidID, payload)
		if cacheErr == nil || cacheErr.status != http.StatusBadGateway || cacheErr.code != "invalid_response_id" {
			t.Fatalf("cache invalid ID %q error = %#v, want invalid_response_id", invalidID, cacheErr)
		}
		if session.cachedStates[canonicalID] == nil {
			t.Fatalf("cache invalid ID %q removed canonical state", invalidID)
		}
	}

	internalID := "resp_internal space"
	internalSession := &responsesWebSocketSession{
		cachedStates:        make(map[string]*responsesWebSocketCachedState),
		maxCachedStateBytes: 1024,
	}
	if err := internalSession.cacheState(internalID, payload); err != nil {
		t.Fatalf("cache internal-space state: %v", err)
	}
	resolved, err := internalSession.resolveCreatePayload(&responsesWebSocketCreateRequest{
		previousResponseID: internalID,
		payload:            map[string]json.RawMessage{"input": json.RawMessage(`"continue"`)},
		generate:           true,
	})
	if err != nil || resolved == nil {
		t.Fatalf("resolve internal-space state = %#v, %v", resolved, err)
	}
	internalSession.evictState(internalID)
	if len(internalSession.cachedStates) != 0 {
		t.Fatalf("internal-space state was not evicted exactly: %#v", internalSession.cachedStates)
	}
}

func TestNativeLifecycleCursorsRejectBoundaryWhitespaceBeforeUpstream(t *testing.T) {
	t.Run("response input items", func(t *testing.T) {
		var upstreamCalls atomic.Int32
		upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			upstreamCalls.Add(1)
			_, _ = w.Write([]byte(`{"object":"list","data":[],"has_more":false}`))
		}))
		defer upstream.Close()

		router, handler, cache := newNativeLifecycleRouter(
			t,
			upstream.URL+"/v1",
			map[string]config.ProviderCapabilities{
				"native": {ResponsesLifecycle: true},
			},
		)
		defer cache.Stop()
		binding := mustResponseBinding(t, handler, "native")
		if !handler.responseBindings.put("resp_native_cursor", binding) {
			t.Fatal("failed to seed native response binding")
		}

		response := performLifecycleRequest(
			t,
			router,
			http.MethodGet,
			"/v1/responses/resp_native_cursor/input_items?after="+url.QueryEscape(" item_target "),
			nil,
		)
		if response.Code != http.StatusBadRequest {
			t.Fatalf("status = %d; body=%s", response.Code, response.Body.String())
		}
		assertLifecycleError(t, response.Body.Bytes(), "after", "invalid_value")
		if upstreamCalls.Load() != 0 {
			t.Fatalf("padded native response cursor made %d upstream calls", upstreamCalls.Load())
		}
	})

	t.Run("conversation items", func(t *testing.T) {
		var upstreamCalls atomic.Int32
		upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			upstreamCalls.Add(1)
			_, _ = w.Write([]byte(`{"object":"list","data":[],"has_more":false}`))
		}))
		defer upstream.Close()

		router, handler, cache := newNativeLifecycleRouter(
			t,
			upstream.URL+"/v1",
			map[string]config.ProviderCapabilities{
				"native": {Conversations: true},
			},
		)
		defer cache.Stop()
		binding, err := handler.validateConversationProvider("native")
		if err != nil {
			t.Fatalf("native conversation binding: %v", err)
		}
		if !handler.conversationBindings.put("conv_native_cursor", binding) {
			t.Fatal("failed to seed native conversation binding")
		}

		response := performLifecycleRequest(
			t,
			router,
			http.MethodGet,
			"/v1/conversations/conv_native_cursor/items?after="+url.QueryEscape(" item_target "),
			nil,
		)
		assertConversationError(t, response, http.StatusBadRequest, "after", "invalid_value")
		if upstreamCalls.Load() != 0 {
			t.Fatalf("padded native conversation cursor made %d upstream calls", upstreamCalls.Load())
		}
	})

	t.Run("stored chat lists", func(t *testing.T) {
		var upstreamCalls atomic.Int32
		upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			upstreamCalls.Add(1)
			_, _ = w.Write([]byte(`{"object":"list","data":[],"has_more":false}`))
		}))
		defer upstream.Close()

		router, handler, cache := newStoredChatLifecycleRouterFromConfigs(t, map[string]config.ProviderConfig{
			"native": storedChatProviderConfig(upstream.URL+"/v1", true),
		})
		defer cache.Stop()
		binding := mustChatCompletionBinding(t, handler, "native")
		if !handler.chatCompletionBindings.put("chatcmpl_native_cursor", binding) {
			t.Fatal("failed to seed native Chat Completion binding")
		}

		paths := []string{
			"/v1/chat/completions?after=" + url.QueryEscape(" chatcmpl_target "),
			"/v1/chat/completions/chatcmpl_native_cursor/messages?after=" + url.QueryEscape(" msg_target "),
		}
		for _, path := range paths {
			response := performLifecycleRequest(t, router, http.MethodGet, path, nil)
			if response.Code != http.StatusBadRequest {
				t.Fatalf("path=%q status = %d; body=%s", path, response.Code, response.Body.String())
			}
			assertLifecycleError(t, response.Body.Bytes(), "after", "invalid_value")
		}
		if upstreamCalls.Load() != 0 {
			t.Fatalf("padded native Chat cursor made %d upstream calls", upstreamCalls.Load())
		}
		if _, ok := handler.chatCompletionBindings.get("chatcmpl_native_cursor"); !ok {
			t.Fatal("padded messages cursor removed the canonical completion binding")
		}
	})
}

func TestTranslatedResponseIDNeverCanonicalizesPaddedUpstreamID(t *testing.T) {
	if got := translatedResponseID("resp_exact"); got != "resp_exact" {
		t.Fatalf("exact translated response ID = %q", got)
	}
	if got := translatedResponseID("resp_internal space"); got != "resp_internal space" {
		t.Fatalf("internal-space translated response ID = %q", got)
	}
	if got := translatedResponseID(" resp_existing "); got == "resp_existing" || !strings.HasPrefix(got, "resp_") {
		t.Fatalf("padded upstream ID was canonicalized or not replaced: %q", got)
	}
}
