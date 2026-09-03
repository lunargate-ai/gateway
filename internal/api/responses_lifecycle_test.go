package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/lunargate-ai/gateway/internal/health"
)

func TestLocalResponsesLifecycleNonStream(t *testing.T) {
	router, handler, calls, closeTest := newResponsesLifecycleTestRouter(t)
	defer closeTest()

	create := performLifecycleRequest(t, router, http.MethodPost, "/v1/responses", []byte(`{
		"model":"mock-gpt",
		"store":true,
		"input":[
			{"role":"user","content":"first"},
			{"type":"message","role":"user","content":[{"type":"input_text","text":"second"}]}
		]
	}`))
	if create.Code != http.StatusOK {
		t.Fatalf("create status = %d, want 200; body=%s", create.Code, create.Body.String())
	}
	created := decodeLifecycleObject(t, create.Body.Bytes())
	responseID := lifecycleStringField(t, created, "id")

	retrieve := performLifecycleRequest(t, router, http.MethodGet, "/v1/responses/"+responseID, nil)
	if retrieve.Code != http.StatusOK {
		t.Fatalf("retrieve status = %d, want 200; body=%s", retrieve.Code, retrieve.Body.String())
	}
	retrieved := decodeLifecycleObject(t, retrieve.Body.Bytes())
	if !reflect.DeepEqual(retrieved, created) {
		t.Fatalf("retrieved response = %#v, want created response %#v", retrieved, created)
	}
	for _, testCase := range []struct {
		query string
		param string
	}{
		{query: "include%5B%5D=reasoning.encrypted_content", param: "include"},
		{query: "include_obfuscation=false", param: "include_obfuscation"},
		{query: "starting_after=7", param: "starting_after"},
	} {
		unsupported := performLifecycleRequest(
			t,
			router,
			http.MethodGet,
			"/v1/responses/"+responseID+"?"+testCase.query,
			nil,
		)
		if unsupported.Code != http.StatusBadRequest {
			t.Fatalf("retrieve query %q status = %d, want 400; body=%s", testCase.query, unsupported.Code, unsupported.Body.String())
		}
		assertLifecycleError(t, unsupported.Body.Bytes(), testCase.param, "unsupported_feature")
	}

	inputItems := performLifecycleRequest(
		t,
		router,
		http.MethodGet,
		"/v1/responses/"+responseID+"/input_items",
		nil,
	)
	if inputItems.Code != http.StatusOK {
		t.Fatalf("input_items status = %d, want 200; body=%s", inputItems.Code, inputItems.Body.String())
	}
	defaultPage := decodeLifecycleObject(t, inputItems.Body.Bytes())
	if defaultPage["object"] != "list" || defaultPage["has_more"] != false {
		t.Fatalf("input_items envelope = %#v", defaultPage)
	}
	defaultData := lifecycleData(t, defaultPage)
	if len(defaultData) != 2 {
		t.Fatalf("input_items count = %d, want 2", len(defaultData))
	}
	if got := lifecycleInputText(t, defaultData[0]); got != "second" {
		t.Fatalf("default descending first item text = %q, want second", got)
	}
	if got := lifecycleInputText(t, defaultData[1]); got != "first" {
		t.Fatalf("default descending second item text = %q, want first", got)
	}
	for index, item := range defaultData {
		if got := lifecycleStringField(t, item, "type"); got != "message" {
			t.Fatalf("item %d type = %q, want message", index, got)
		}
		if got := lifecycleStringField(t, item, "id"); !strings.HasPrefix(got, "msg_") {
			t.Fatalf("item %d id = %q, want generated msg_ id", index, got)
		}
	}

	firstPageRecorder := performLifecycleRequest(
		t,
		router,
		http.MethodGet,
		"/v1/responses/"+responseID+"/input_items?order=asc&limit=1",
		nil,
	)
	if firstPageRecorder.Code != http.StatusOK {
		t.Fatalf("first page status = %d, want 200; body=%s", firstPageRecorder.Code, firstPageRecorder.Body.String())
	}
	firstPage := decodeLifecycleObject(t, firstPageRecorder.Body.Bytes())
	firstData := lifecycleData(t, firstPage)
	if len(firstData) != 1 || lifecycleInputText(t, firstData[0]) != "first" {
		t.Fatalf("first ascending page = %#v, want first input", firstData)
	}
	if firstPage["has_more"] != true {
		t.Fatalf("first page has_more = %#v, want true", firstPage["has_more"])
	}
	firstID := lifecycleStringField(t, firstData[0], "id")
	if firstPage["first_id"] != firstID || firstPage["last_id"] != firstID {
		t.Fatalf("first page cursors = %#v/%#v, want %q", firstPage["first_id"], firstPage["last_id"], firstID)
	}

	secondPageRecorder := performLifecycleRequest(
		t,
		router,
		http.MethodGet,
		"/v1/responses/"+responseID+"/input_items?order=asc&limit=1&after="+url.QueryEscape(firstID),
		nil,
	)
	if secondPageRecorder.Code != http.StatusOK {
		t.Fatalf("second page status = %d, want 200; body=%s", secondPageRecorder.Code, secondPageRecorder.Body.String())
	}
	secondPage := decodeLifecycleObject(t, secondPageRecorder.Body.Bytes())
	secondData := lifecycleData(t, secondPage)
	if len(secondData) != 1 || lifecycleInputText(t, secondData[0]) != "second" {
		t.Fatalf("second ascending page = %#v, want second input", secondData)
	}
	if secondPage["has_more"] != false {
		t.Fatalf("second page has_more = %#v, want false", secondPage["has_more"])
	}

	invalidLimit := performLifecycleRequest(
		t,
		router,
		http.MethodGet,
		"/v1/responses/"+responseID+"/input_items?limit=0",
		nil,
	)
	if invalidLimit.Code != http.StatusBadRequest {
		t.Fatalf("invalid limit status = %d, want 400", invalidLimit.Code)
	}
	assertLifecycleError(t, invalidLimit.Body.Bytes(), "limit", "invalid_value")

	for _, includeQuery := range []string{
		"include=message.input_image.image_url",
		"include%5B%5D=message.output_text.logprobs",
	} {
		included := performLifecycleRequest(
			t,
			router,
			http.MethodGet,
			"/v1/responses/"+responseID+"/input_items?"+includeQuery,
			nil,
		)
		if included.Code != http.StatusBadRequest {
			t.Fatalf("include query %q status = %d, want 400; body=%s", includeQuery, included.Code, included.Body.String())
		}
		assertLifecycleError(t, included.Body.Bytes(), "include", "unsupported_feature")
	}

	deleted := performLifecycleRequest(t, router, http.MethodDelete, "/v1/responses/"+responseID, nil)
	if deleted.Code != http.StatusOK {
		t.Fatalf("delete status = %d, want 200; body=%s", deleted.Code, deleted.Body.String())
	}
	if got := decodeLifecycleObject(t, deleted.Body.Bytes()); !reflect.DeepEqual(got, map[string]interface{}{
		"id":      responseID,
		"object":  "response",
		"deleted": true,
	}) {
		t.Fatalf("delete body = %#v", got)
	}
	if _, ok := handler.responsesState.get(responseID); ok {
		t.Fatalf("deleted response %q remains in continuation state", responseID)
	}

	for _, path := range []string{
		"/v1/responses/" + responseID,
		"/v1/responses/" + responseID + "/input_items",
	} {
		notFound := performLifecycleRequest(t, router, http.MethodGet, path, nil)
		if notFound.Code != http.StatusNotFound {
			t.Fatalf("GET %s status = %d, want 404", path, notFound.Code)
		}
		assertLifecycleError(t, notFound.Body.Bytes(), "response_id", "response_not_found")
	}
	deleteAgain := performLifecycleRequest(t, router, http.MethodDelete, "/v1/responses/"+responseID, nil)
	if deleteAgain.Code != http.StatusNotFound {
		t.Fatalf("second delete status = %d, want 404", deleteAgain.Code)
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("upstream calls = %d, want one create call", got)
	}
}

func TestLocalResponsesLifecycleStoresCompletedStreamOnlyWhenEnabled(t *testing.T) {
	router, handler, _, closeTest := newResponsesLifecycleTestRouter(t)
	defer closeTest()

	stored := performLifecycleRequest(t, router, http.MethodPost, "/v1/responses", []byte(`{
		"model":"mock-gpt",
		"stream":true,
		"store":true,
		"input":"stream input"
	}`))
	if stored.Code != http.StatusOK {
		t.Fatalf("stream create status = %d, want 200; body=%s", stored.Code, stored.Body.String())
	}
	storedID := lifecycleCompletedStreamResponseID(t, stored.Body.String())
	retrieve := performLifecycleRequest(t, router, http.MethodGet, "/v1/responses/"+storedID, nil)
	if retrieve.Code != http.StatusOK {
		t.Fatalf("stream retrieve status = %d, want 200; body=%s", retrieve.Code, retrieve.Body.String())
	}
	retrieved := decodeLifecycleObject(t, retrieve.Body.Bytes())
	if retrieved["status"] != "completed" || retrieved["output_text"] != "streamed answer" {
		t.Fatalf("retrieved streamed response = %#v", retrieved)
	}
	items := performLifecycleRequest(t, router, http.MethodGet, "/v1/responses/"+storedID+"/input_items", nil)
	if items.Code != http.StatusOK {
		t.Fatalf("stream input_items status = %d, want 200; body=%s", items.Code, items.Body.String())
	}
	data := lifecycleData(t, decodeLifecycleObject(t, items.Body.Bytes()))
	if len(data) != 1 || lifecycleInputText(t, data[0]) != "stream input" {
		t.Fatalf("stream input_items = %#v", data)
	}

	statelessStream := performLifecycleRequest(t, router, http.MethodPost, "/v1/responses", []byte(`{
		"model":"mock-gpt",
		"stream":true,
		"store":false,
		"input":"do not retain stream"
	}`))
	if statelessStream.Code != http.StatusOK {
		t.Fatalf("stateless stream status = %d, want 200; body=%s", statelessStream.Code, statelessStream.Body.String())
	}
	statelessStreamID := lifecycleCompletedStreamResponseID(t, statelessStream.Body.String())
	assertResponseAbsentFromLifecycle(t, router, handler, statelessStreamID)

	statelessNonStream := performLifecycleRequest(t, router, http.MethodPost, "/v1/responses", []byte(`{
		"model":"mock-gpt",
		"store":false,
		"input":"do not retain response"
	}`))
	if statelessNonStream.Code != http.StatusOK {
		t.Fatalf("stateless response status = %d, want 200; body=%s", statelessNonStream.Code, statelessNonStream.Body.String())
	}
	statelessNonStreamID := lifecycleStringField(t, decodeLifecycleObject(t, statelessNonStream.Body.Bytes()), "id")
	assertResponseAbsentFromLifecycle(t, router, handler, statelessNonStreamID)
}

func TestLocalResponsesLifecycleEvictsOldestCompletedResponse(t *testing.T) {
	router, handler, _, closeTest := newResponsesLifecycleTestRouter(t)
	defer closeTest()
	handler.responsesState.maxEntries = 1

	first := performLifecycleRequest(t, router, http.MethodPost, "/v1/responses", []byte(`{
		"model":"mock-gpt","store":true,"input":"first"
	}`))
	second := performLifecycleRequest(t, router, http.MethodPost, "/v1/responses", []byte(`{
		"model":"mock-gpt","store":true,"input":"second"
	}`))
	if first.Code != http.StatusOK || second.Code != http.StatusOK {
		t.Fatalf("create statuses = %d/%d, want 200/200", first.Code, second.Code)
	}
	firstID := lifecycleStringField(t, decodeLifecycleObject(t, first.Body.Bytes()), "id")
	secondID := lifecycleStringField(t, decodeLifecycleObject(t, second.Body.Bytes()), "id")

	firstRetrieve := performLifecycleRequest(t, router, http.MethodGet, "/v1/responses/"+firstID, nil)
	if firstRetrieve.Code != http.StatusNotFound {
		t.Fatalf("evicted response status = %d, want 404", firstRetrieve.Code)
	}
	secondRetrieve := performLifecycleRequest(t, router, http.MethodGet, "/v1/responses/"+secondID, nil)
	if secondRetrieve.Code != http.StatusOK {
		t.Fatalf("newest response status = %d, want 200; body=%s", secondRetrieve.Code, secondRetrieve.Body.String())
	}
}

func TestResponsesStateStoreCountsCompletedLifecycleBytes(t *testing.T) {
	store := newResponsesStateStore(time.Hour)
	requestPayload := map[string]json.RawMessage{
		"model": json.RawMessage(`"mock-gpt"`),
		"input": json.RawMessage(`"hello"`),
	}
	completed := map[string]interface{}{
		"id":          "resp_byte_limit",
		"object":      "response",
		"status":      "completed",
		"model":       "mock-gpt",
		"output":      []interface{}{},
		"output_text": strings.Repeat("x", 128),
	}
	continuation := withCompletedResponseHistory(requestPayload, completed)
	response, err := json.Marshal(completed)
	if err != nil {
		t.Fatalf("encode response: %v", err)
	}
	items, err := responsesStateInputItems("resp_byte_limit", requestPayload["input"])
	if err != nil {
		t.Fatalf("prepare input items: %v", err)
	}
	fullSize := responsesStateEntrySize("resp_byte_limit", continuation, response, items)
	if fullSize <= responsesStatePayloadSize("resp_byte_limit", continuation) {
		t.Fatal("completed response and input items were not added to state byte accounting")
	}
	store.maxBytes = fullSize - 1
	store.putCompleted("resp_byte_limit", requestPayload, completed)

	if _, ok := store.get("resp_byte_limit"); ok {
		t.Fatal("oversized completed state was retained")
	}
	if _, _, ok := store.getCompleted("resp_byte_limit"); ok {
		t.Fatal("oversized completed lifecycle response was retained")
	}
}

func TestResponsesStateStoreConcurrentLifecycleAccess(t *testing.T) {
	store := newResponsesStateStore(time.Hour)
	requestPayload := map[string]json.RawMessage{
		"model": json.RawMessage(`"mock-gpt"`),
		"input": json.RawMessage(`"hello"`),
	}

	var workers sync.WaitGroup
	for worker := 0; worker < 8; worker++ {
		worker := worker
		workers.Add(1)
		go func() {
			defer workers.Done()
			for iteration := 0; iteration < 200; iteration++ {
				responseID := fmt.Sprintf("resp_%d", iteration%16)
				store.putCompleted(responseID, requestPayload, map[string]interface{}{
					"id":          responseID,
					"object":      "response",
					"status":      "completed",
					"model":       "mock-gpt",
					"output":      []interface{}{},
					"output_text": fmt.Sprintf("%d-%d", worker, iteration),
				})
				_, _, _ = store.getCompleted(responseID)
				if iteration%7 == 0 {
					store.delete(responseID)
				}
			}
		}()
	}
	workers.Wait()

	store.mu.Lock()
	defer store.mu.Unlock()
	calculated := 0
	for responseID, entry := range store.entries {
		calculated += responsesStateEntrySize(responseID, entry.payload, entry.response, entry.inputItems)
	}
	if store.totalBytes != calculated {
		t.Fatalf("totalBytes = %d, calculated = %d", store.totalBytes, calculated)
	}
	if store.totalBytes > store.maxBytes {
		t.Fatalf("totalBytes = %d exceeds maxBytes = %d", store.totalBytes, store.maxBytes)
	}
}

func newResponsesLifecycleTestRouter(t *testing.T) (http.Handler, *Handler, *atomic.Int32, func()) {
	t.Helper()
	var calls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read upstream request: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		var payload map[string]interface{}
		if err := json.Unmarshal(body, &payload); err != nil {
			t.Errorf("decode upstream request: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}

		call := calls.Add(1)
		responseID := fmt.Sprintf("chatcmpl-lifecycle-%d", call)
		if stream, _ := payload["stream"].(bool); stream {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = fmt.Fprintf(w, "data: {\"id\":%q,\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"mock-gpt\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"streamed answer\"},\"finish_reason\":null}]}\n\n", responseID)
			_, _ = fmt.Fprintf(w, "data: {\"id\":%q,\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"mock-gpt\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n", responseID)
			_, _ = io.WriteString(w, "data: [DONE]\n\n")
			return
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(w, `{"id":%q,"object":"chat.completion","created":1,"model":"mock-gpt","choices":[{"index":0,"message":{"role":"assistant","content":"non-stream answer"},"finish_reason":"stop"}]}`, responseID)
	}))

	handler := newResponsesWebSocketTestHandler(upstream.URL)
	router := NewRouter(handler, nil, nil, health.NewChecker("test"))
	return router, handler, &calls, func() {
		handler.cache.Stop()
		upstream.Close()
	}
}

func performLifecycleRequest(t *testing.T, handler http.Handler, method string, path string, body []byte) *httptest.ResponseRecorder {
	t.Helper()
	request := httptest.NewRequest(method, path, bytes.NewReader(body))
	if body != nil {
		request.Header.Set("Content-Type", "application/json")
	}
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, request)
	return recorder
}

func decodeLifecycleObject(t *testing.T, body []byte) map[string]interface{} {
	t.Helper()
	var decoded map[string]interface{}
	if err := json.Unmarshal(body, &decoded); err != nil {
		t.Fatalf("decode JSON object %q: %v", string(body), err)
	}
	return decoded
}

func lifecycleStringField(t *testing.T, object map[string]interface{}, field string) string {
	t.Helper()
	value, ok := object[field].(string)
	if !ok || strings.TrimSpace(value) == "" {
		t.Fatalf("field %q = %#v, want non-empty string", field, object[field])
	}
	return value
}

func lifecycleData(t *testing.T, page map[string]interface{}) []map[string]interface{} {
	t.Helper()
	rawData, ok := page["data"].([]interface{})
	if !ok {
		t.Fatalf("data = %#v, want array", page["data"])
	}
	data := make([]map[string]interface{}, 0, len(rawData))
	for index, rawItem := range rawData {
		item, ok := rawItem.(map[string]interface{})
		if !ok {
			t.Fatalf("data[%d] = %#v, want object", index, rawItem)
		}
		data = append(data, item)
	}
	return data
}

func lifecycleInputText(t *testing.T, item map[string]interface{}) string {
	t.Helper()
	content, ok := item["content"].([]interface{})
	if !ok || len(content) == 0 {
		t.Fatalf("item content = %#v, want non-empty array", item["content"])
	}
	part, ok := content[0].(map[string]interface{})
	if !ok {
		t.Fatalf("item content[0] = %#v, want object", content[0])
	}
	text, ok := part["text"].(string)
	if !ok {
		t.Fatalf("item text = %#v, want string", part["text"])
	}
	return text
}

func lifecycleCompletedStreamResponseID(t *testing.T, body string) string {
	t.Helper()
	for _, event := range decodeSSEEvents(t, body) {
		if event["type"] != "response.completed" {
			continue
		}
		response, ok := event["response"].(map[string]interface{})
		if !ok {
			t.Fatalf("response.completed response = %#v, want object", event["response"])
		}
		return lifecycleStringField(t, response, "id")
	}
	t.Fatal("response.completed event not found")
	return ""
}

func assertResponseAbsentFromLifecycle(t *testing.T, router http.Handler, handler *Handler, responseID string) {
	t.Helper()
	if _, ok := handler.responsesState.get(responseID); ok {
		t.Fatalf("store:false response %q was retained in continuation state", responseID)
	}
	retrieve := performLifecycleRequest(t, router, http.MethodGet, "/v1/responses/"+responseID, nil)
	if retrieve.Code != http.StatusNotFound {
		t.Fatalf("store:false response %q retrieve status = %d, want 404", responseID, retrieve.Code)
	}
}

func assertLifecycleError(t *testing.T, body []byte, wantParam string, wantCode string) {
	t.Helper()
	decoded := decodeLifecycleObject(t, body)
	errorObject, ok := decoded["error"].(map[string]interface{})
	if !ok {
		t.Fatalf("error body = %#v, want error object", decoded)
	}
	if errorObject["param"] != wantParam || errorObject["code"] != wantCode {
		t.Fatalf("error param/code = %#v/%#v, want %q/%q", errorObject["param"], errorObject["code"], wantParam, wantCode)
	}
}
