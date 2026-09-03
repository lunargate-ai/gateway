package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestConversationsCRUDAndPagination(t *testing.T) {
	handler := &Handler{conversationsState: newConversationStateStore(time.Hour)}
	router := conversationTestRouter(handler)

	create := performConversationRequest(t, router, http.MethodPost, "/v1/conversations", `{
		"metadata":{"tenant":"test"},
		"items":[{"role":"user","content":"first","custom_field":"preserved"}]
	}`)
	if create.Code != http.StatusOK {
		t.Fatalf("create status = %d, body = %s", create.Code, create.Body.String())
	}
	var conversation conversationObject
	decodeConversationResponse(t, create, &conversation)
	if !strings.HasPrefix(conversation.ID, "conv_") || conversation.Object != "conversation" {
		t.Fatalf("unexpected conversation: %#v", conversation)
	}
	if conversation.Metadata["tenant"] != "test" {
		t.Fatalf("metadata = %#v", conversation.Metadata)
	}

	createdItems := performConversationRequest(t, router, http.MethodPost, "/v1/conversations/"+conversation.ID+"/items", `{
		"items":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"second"}]}]
	}`)
	if createdItems.Code != http.StatusOK {
		t.Fatalf("create items status = %d, body = %s", createdItems.Code, createdItems.Body.String())
	}
	var createdList conversationItemList
	decodeConversationResponse(t, createdItems, &createdList)
	if len(createdList.Data) != 1 || createdList.FirstID == nil || createdList.LastID == nil {
		t.Fatalf("created list = %#v", createdList)
	}
	secondID := conversationItemID(createdList.Data[0])
	if !strings.HasPrefix(secondID, "msg_") || parseJSONStringRaw(createdList.Data[0]["status"]) != "completed" {
		t.Fatalf("created item = %s", mustMarshalForTest(t, createdList.Data[0]))
	}

	ascending := performConversationRequest(t, router, http.MethodGet, "/v1/conversations/"+conversation.ID+"/items?order=asc&limit=1", "")
	var firstPage conversationItemList
	decodeConversationResponse(t, ascending, &firstPage)
	if len(firstPage.Data) != 1 || !firstPage.HasMore || firstPage.FirstID == nil {
		t.Fatalf("first page = %#v", firstPage)
	}
	firstID := *firstPage.FirstID
	if parseJSONStringRaw(firstPage.Data[0]["custom_field"]) != "preserved" {
		t.Fatalf("additive item field was not preserved: %s", mustMarshalForTest(t, firstPage.Data[0]))
	}

	secondPageRecorder := performConversationRequest(t, router, http.MethodGet, "/v1/conversations/"+conversation.ID+"/items?order=asc&limit=1&after="+firstID, "")
	var secondPage conversationItemList
	decodeConversationResponse(t, secondPageRecorder, &secondPage)
	if len(secondPage.Data) != 1 || secondPage.HasMore || conversationItemID(secondPage.Data[0]) != secondID {
		t.Fatalf("second page = %#v", secondPage)
	}

	retrieveItem := performConversationRequest(t, router, http.MethodGet, "/v1/conversations/"+conversation.ID+"/items/"+secondID, "")
	var item map[string]json.RawMessage
	decodeConversationResponse(t, retrieveItem, &item)
	if conversationItemID(item) != secondID {
		t.Fatalf("retrieved item ID = %q", conversationItemID(item))
	}

	deletedItem := performConversationRequest(t, router, http.MethodDelete, "/v1/conversations/"+conversation.ID+"/items/"+secondID, "")
	if deletedItem.Code != http.StatusOK {
		t.Fatalf("delete item status = %d, body = %s", deletedItem.Code, deletedItem.Body.String())
	}
	notFoundItem := performConversationRequest(t, router, http.MethodGet, "/v1/conversations/"+conversation.ID+"/items/"+secondID, "")
	assertConversationError(t, notFoundItem, http.StatusNotFound, "item_id", "conversation_item_not_found")

	updated := performConversationRequest(t, router, http.MethodPost, "/v1/conversations/"+conversation.ID, `{"metadata":{"stage":"development"}}`)
	var updatedConversation conversationObject
	decodeConversationResponse(t, updated, &updatedConversation)
	if len(updatedConversation.Metadata) != 1 || updatedConversation.Metadata["stage"] != "development" {
		t.Fatalf("updated metadata = %#v", updatedConversation.Metadata)
	}

	retrieved := performConversationRequest(t, router, http.MethodGet, "/v1/conversations/"+conversation.ID, "")
	if retrieved.Code != http.StatusOK {
		t.Fatalf("retrieve status = %d, body = %s", retrieved.Code, retrieved.Body.String())
	}

	deleted := performConversationRequest(t, router, http.MethodDelete, "/v1/conversations/"+conversation.ID, "")
	var deletedObject conversationDeletedObject
	decodeConversationResponse(t, deleted, &deletedObject)
	if deletedObject.ID != conversation.ID || deletedObject.Object != "conversation.deleted" || !deletedObject.Deleted {
		t.Fatalf("deleted object = %#v", deletedObject)
	}
	notFound := performConversationRequest(t, router, http.MethodGet, "/v1/conversations/"+conversation.ID, "")
	assertConversationError(t, notFound, http.StatusNotFound, "conversation_id", "conversation_not_found")
}

func TestConversationRequestValidation(t *testing.T) {
	handler := &Handler{conversationsState: newConversationStateStore(time.Hour)}
	router := conversationTestRouter(handler)
	existing, err := handler.conversationsState.create(nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	tooManyItems := make([]string, maxConversationItemsPerRequest+1)
	for index := range tooManyItems {
		tooManyItems[index] = `{"role":"user","content":"x"}`
	}
	tooManyMetadata := make([]string, maxConversationMetadataPairs+1)
	for index := range tooManyMetadata {
		tooManyMetadata[index] = fmt.Sprintf(`"key%d":"value"`, index)
	}

	tests := []struct {
		name   string
		method string
		path   string
		body   string
		param  string
		code   string
	}{
		{name: "too many initial items", method: http.MethodPost, path: "/v1/conversations", body: `{"items":[` + strings.Join(tooManyItems, ",") + `]}`, param: "items", code: "array_above_max_length"},
		{name: "metadata values must be strings", method: http.MethodPost, path: "/v1/conversations", body: `{"metadata":{"number":1}}`, param: "metadata", code: "invalid_metadata"},
		{name: "too many metadata properties", method: http.MethodPost, path: "/v1/conversations", body: `{"metadata":{` + strings.Join(tooManyMetadata, ",") + `}}`, param: "metadata", code: "invalid_metadata"},
		{name: "invalid item", method: http.MethodPost, path: "/v1/conversations", body: `{"items":[{"content":"missing role and type"}]}`, param: "items[0].type", code: "invalid_conversation_item"},
		{name: "unknown create field", method: http.MethodPost, path: "/v1/conversations", body: `{"future_field":true}`, param: "", code: ""},
		{name: "unknown items field", method: http.MethodPost, path: "/v1/conversations/" + existing.ID + "/items", body: `{"items":[],"future_field":true}`, param: "", code: ""},
		{name: "unknown update field", method: http.MethodPost, path: "/v1/conversations/" + existing.ID, body: `{"metadata":{},"future_field":true}`, param: "", code: ""},
		{name: "invalid json", method: http.MethodPost, path: "/v1/conversations", body: `{`, param: "", code: ""},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			recorder := performConversationRequest(t, router, test.method, test.path, test.body)
			if test.param == "" && test.code == "" {
				if recorder.Code != http.StatusBadRequest {
					t.Fatalf("status = %d, body = %s", recorder.Code, recorder.Body.String())
				}
				return
			}
			assertConversationError(t, recorder, http.StatusBadRequest, test.param, test.code)
		})
	}
}

func TestConversationItemIdentityValidation(t *testing.T) {
	handler := &Handler{conversationsState: newConversationStateStore(time.Hour)}
	router := conversationTestRouter(handler)

	tests := []struct {
		name  string
		body  string
		param string
		code  string
	}{
		{name: "unresolvable item reference", body: `{"items":[{"type":"item_reference","id":"item_existing"}]}`, param: "items[0].type", code: "unsupported_feature"},
		{name: "null id", body: `{"items":[{"type":"message","role":"user","id":null,"content":"hello"}]}`, param: "items[0].id", code: "invalid_value"},
		{name: "numeric id", body: `{"items":[{"type":"message","role":"user","id":42,"content":"hello"}]}`, param: "items[0].id", code: "invalid_value"},
		{name: "empty id", body: `{"items":[{"type":"message","role":"user","id":"  ","content":"hello"}]}`, param: "items[0].id", code: "invalid_value"},
		{name: "duplicate request id", body: `{"items":[{"type":"message","role":"user","id":"msg_same","content":"one"},{"type":"message","role":"user","id":"msg_same","content":"two"}]}`, param: "items[1].id", code: "duplicate_item_id"},
	}
	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			recorder := performConversationRequest(t, router, http.MethodPost, "/v1/conversations", testCase.body)
			assertConversationError(t, recorder, http.StatusBadRequest, testCase.param, testCase.code)
		})
	}

	created := performConversationRequest(t, router, http.MethodPost, "/v1/conversations", `{"items":[{"type":"message","role":"user","id":"msg_stable","content":"one"}]}`)
	if created.Code != http.StatusOK {
		t.Fatalf("create status = %d, body = %s", created.Code, created.Body.String())
	}
	var conversation conversationObject
	decodeConversationResponse(t, created, &conversation)
	duplicate := performConversationRequest(t, router, http.MethodPost, "/v1/conversations/"+conversation.ID+"/items", `{"items":[{"type":"message","role":"assistant","id":"msg_stable","content":"two"}]}`)
	assertConversationError(t, duplicate, http.StatusBadRequest, "items", "duplicate_item_id")
	items, ok := handler.conversationsState.getItems(conversation.ID)
	if !ok || len(items) != 1 || conversationItemID(items[0]) != "msg_stable" {
		t.Fatalf("duplicate ID mutated stored items: ok=%v items=%s", ok, mustMarshalForTest(t, items))
	}
}

func TestLocalConversationRejectsUnsupportedQueryWithoutMutation(t *testing.T) {
	handler := &Handler{conversationsState: newConversationStateStore(time.Hour)}
	router := conversationTestRouter(handler)
	created := performConversationRequest(t, router, http.MethodPost, "/v1/conversations", `{"items":[{"role":"user","content":"one"}]}`)
	var conversation conversationObject
	decodeConversationResponse(t, created, &conversation)

	tests := []struct {
		name   string
		method string
		path   string
		body   string
		param  string
		code   string
	}{
		{name: "create include", method: http.MethodPost, path: "/v1/conversations?include%5B%5D=reasoning.encrypted_content", body: `{}`, param: "include", code: "unsupported_feature"},
		{name: "retrieve unknown", method: http.MethodGet, path: "/v1/conversations/" + conversation.ID + "?future=true", param: "future", code: "unknown_parameter"},
		{name: "update include", method: http.MethodPost, path: "/v1/conversations/" + conversation.ID + "?include=reasoning.encrypted_content", body: `{"metadata":{}}`, param: "include", code: "unsupported_feature"},
		{name: "delete unknown", method: http.MethodDelete, path: "/v1/conversations/" + conversation.ID + "?future=true", param: "future", code: "unknown_parameter"},
		{name: "create items include", method: http.MethodPost, path: "/v1/conversations/" + conversation.ID + "/items?include%5B%5D=message.output_text.logprobs", body: `{"items":[{"role":"user","content":"two"}]}`, param: "include", code: "unsupported_feature"},
		{name: "list items include", method: http.MethodGet, path: "/v1/conversations/" + conversation.ID + "/items?order=asc&include=message.input_image.image_url", param: "include", code: "unsupported_feature"},
		{name: "list items bracketed cursor", method: http.MethodGet, path: "/v1/conversations/" + conversation.ID + "/items?after%5B%5D=item_missing", param: "after[]", code: "unknown_parameter"},
		{name: "get item unknown", method: http.MethodGet, path: "/v1/conversations/" + conversation.ID + "/items/item_missing?future=true", param: "future", code: "unknown_parameter"},
		{name: "delete item include", method: http.MethodDelete, path: "/v1/conversations/" + conversation.ID + "/items/item_missing?include=x", param: "include", code: "unsupported_feature"},
	}

	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			recorder := performConversationRequest(t, router, testCase.method, testCase.path, testCase.body)
			assertConversationError(t, recorder, http.StatusBadRequest, testCase.param, testCase.code)
		})
	}

	items, ok := handler.conversationsState.getItems(conversation.ID)
	if !ok || len(items) != 1 {
		t.Fatalf("unsupported query mutated conversation items: ok=%v items=%d", ok, len(items))
	}
}

func TestConversationListValidation(t *testing.T) {
	handler := &Handler{conversationsState: newConversationStateStore(time.Hour)}
	router := conversationTestRouter(handler)
	create := performConversationRequest(t, router, http.MethodPost, "/v1/conversations", `{"items":[{"role":"user","content":"one"}]}`)
	var conversation conversationObject
	decodeConversationResponse(t, create, &conversation)

	tests := []struct {
		query string
		param string
		code  string
	}{
		{query: "order=sideways", param: "order", code: "invalid_value"},
		{query: "limit=0", param: "limit", code: "invalid_value"},
		{query: "limit=101", param: "limit", code: "invalid_value"},
		{query: "after=item_missing", param: "after", code: "invalid_cursor"},
	}
	for _, test := range tests {
		recorder := performConversationRequest(t, router, http.MethodGet, "/v1/conversations/"+conversation.ID+"/items?"+test.query, "")
		assertConversationError(t, recorder, http.StatusBadRequest, test.param, test.code)
	}
}

func TestConversationStateBoundsExpiryAndCloning(t *testing.T) {
	now := time.Unix(1000, 0)
	store := newConversationStateStore(time.Minute)
	store.now = func() time.Time { return now }
	store.maxEntries = 2

	first, err := store.create(map[string]string{"key": "one"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	second, err := store.create(nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := store.get(first.ID); !ok {
		t.Fatal("expected first conversation")
	}
	third, err := store.create(nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := store.get(second.ID); ok {
		t.Fatal("least recently used conversation was not evicted")
	}
	if _, ok := store.get(first.ID); !ok {
		t.Fatal("recently used conversation was evicted")
	}

	item := map[string]json.RawMessage{
		"id":      json.RawMessage(`"msg_original"`),
		"type":    json.RawMessage(`"message"`),
		"content": json.RawMessage(`"before"`),
	}
	if _, err := store.addItems(third.ID, []map[string]json.RawMessage{item}); err != nil {
		t.Fatal(err)
	}
	item["content"] = json.RawMessage(`"after"`)
	stored, err := store.getItem(third.ID, "msg_original")
	if err != nil {
		t.Fatal(err)
	}
	if parseJSONStringRaw(stored["content"]) != "before" {
		t.Fatalf("stored item mutated through caller alias: %s", stored["content"])
	}
	stored["content"] = json.RawMessage(`"mutated"`)
	storedAgain, err := store.getItem(third.ID, "msg_original")
	if err != nil || parseJSONStringRaw(storedAgain["content"]) != "before" {
		t.Fatalf("stored item mutated through return alias: %v, %s", err, storedAgain["content"])
	}

	now = now.Add(time.Minute)
	if _, ok := store.get(third.ID); ok {
		t.Fatal("expired conversation remained available")
	}
	if store.totalBytes < 0 {
		t.Fatalf("totalBytes = %d", store.totalBytes)
	}
}

func TestConversationStateConcurrentAccess(t *testing.T) {
	store := newConversationStateStore(time.Hour)
	conversation, err := store.create(nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	var wg sync.WaitGroup
	for worker := 0; worker < 16; worker++ {
		worker := worker
		wg.Add(1)
		go func() {
			defer wg.Done()
			for index := 0; index < 25; index++ {
				itemID := fmt.Sprintf("item_%d_%d", worker, index)
				item := map[string]json.RawMessage{
					"id":   mustJSONRawString(itemID),
					"type": json.RawMessage(`"message"`),
				}
				_, _ = store.addItems(conversation.ID, []map[string]json.RawMessage{item})
				_, _ = store.getItem(conversation.ID, itemID)
				_, _ = store.get(conversation.ID)
			}
		}()
	}
	wg.Wait()

	items, err := store.listItems(conversation.ID, "", "asc", 100)
	if err != nil {
		t.Fatal(err)
	}
	if len(items.Data) != 100 || !items.HasMore {
		t.Fatalf("first page length = %d, has_more = %t", len(items.Data), items.HasMore)
	}
}

func conversationTestRouter(handler *Handler) http.Handler {
	router := chi.NewRouter()
	router.Route("/v1", func(router chi.Router) {
		router.Post("/conversations", handler.CreateConversation)
		router.Get("/conversations/{conversation_id}", handler.GetConversation)
		router.Post("/conversations/{conversation_id}", handler.UpdateConversation)
		router.Delete("/conversations/{conversation_id}", handler.DeleteConversation)
		router.Post("/conversations/{conversation_id}/items", handler.CreateConversationItems)
		router.Get("/conversations/{conversation_id}/items", handler.ListConversationItems)
		router.Get("/conversations/{conversation_id}/items/{item_id}", handler.GetConversationItem)
		router.Delete("/conversations/{conversation_id}/items/{item_id}", handler.DeleteConversationItem)
	})
	return router
}

func performConversationRequest(t *testing.T, handler http.Handler, method, path, body string) *httptest.ResponseRecorder {
	t.Helper()
	request := httptest.NewRequest(method, path, bytes.NewBufferString(body))
	if body != "" {
		request.Header.Set("Content-Type", "application/json")
	}
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, request)
	return recorder
}

func decodeConversationResponse(t *testing.T, recorder *httptest.ResponseRecorder, dst interface{}) {
	t.Helper()
	if err := json.Unmarshal(recorder.Body.Bytes(), dst); err != nil {
		t.Fatalf("decode response %q: %v", recorder.Body.String(), err)
	}
}

func assertConversationError(t *testing.T, recorder *httptest.ResponseRecorder, status int, param, code string) {
	t.Helper()
	if recorder.Code != status {
		t.Fatalf("status = %d, want %d, body = %s", recorder.Code, status, recorder.Body.String())
	}
	var response models.ErrorResponse
	decodeConversationResponse(t, recorder, &response)
	if param != "" && (response.Error.Param == nil || *response.Error.Param != param) {
		t.Fatalf("param = %#v, want %q", response.Error.Param, param)
	}
	if code != "" && (response.Error.Code == nil || *response.Error.Code != code) {
		t.Fatalf("code = %#v, want %q", response.Error.Code, code)
	}
}

func mustMarshalForTest(t *testing.T, value interface{}) string {
	t.Helper()
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
}
