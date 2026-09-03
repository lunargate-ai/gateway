package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestResponsesWebSocketSession_CacheKeepsOnlyCurrentState(t *testing.T) {
	session := &responsesWebSocketSession{
		cachedStates:        make(map[string]*responsesWebSocketCachedState),
		maxCachedStateBytes: 1024,
	}
	first := map[string]json.RawMessage{"model": json.RawMessage(`"first"`)}
	second := map[string]json.RawMessage{"model": json.RawMessage(`"second"`)}

	if err := session.cacheState("resp_first", first); err != nil {
		t.Fatalf("cache first state: %v", err)
	}
	if err := session.cacheState("resp_second", second); err != nil {
		t.Fatalf("cache second state: %v", err)
	}
	if len(session.cachedStates) != responsesWebSocketMaxCachedStates {
		t.Fatalf("cached state count = %d, want %d", len(session.cachedStates), responsesWebSocketMaxCachedStates)
	}
	if session.cachedStates["resp_first"] != nil || session.cachedStates["resp_second"] == nil {
		t.Fatalf("cached states = %#v, want only current response", session.cachedStates)
	}
	wantBytes, ok := responsesWebSocketCachedStateSize("resp_second", second, session.cachedStateLimit())
	if !ok || session.cachedStateBytes != wantBytes {
		t.Fatalf("cached bytes = %d, want %d", session.cachedStateBytes, wantBytes)
	}

	session.evictState("resp_second")
	if len(session.cachedStates) != 0 || session.cachedStateBytes != 0 {
		t.Fatalf("eviction left cached state: count=%d bytes=%d", len(session.cachedStates), session.cachedStateBytes)
	}
}

func TestResponsesWebSocketSession_CacheStateByteBoundary(t *testing.T) {
	const limit = 128
	const responseID = "resp_boundary"
	session := &responsesWebSocketSession{
		cachedStates:        make(map[string]*responsesWebSocketCachedState),
		maxCachedStateBytes: limit,
	}
	fixedBytes := 2*len(responseID) + len("input")
	exactValueBytes := limit - fixedBytes
	exact := map[string]json.RawMessage{
		"input": sizedJSONString(t, exactValueBytes),
	}

	if err := session.cacheState(responseID, exact); err != nil {
		t.Fatalf("cache exact boundary: %v", err)
	}
	if session.cachedStateBytes != limit {
		t.Fatalf("cached bytes = %d, want exact limit %d", session.cachedStateBytes, limit)
	}

	over := map[string]json.RawMessage{
		"input": sizedJSONString(t, exactValueBytes+1),
	}
	err := session.cacheState(responseID, over)
	if err == nil || err.code != "state_too_large" || err.status != http.StatusRequestEntityTooLarge {
		t.Fatalf("over-limit cache error = %#v, want state_too_large", err)
	}
	if len(session.cachedStates) != 0 || session.cachedStateBytes != 0 {
		t.Fatalf("over-limit cache retained stale state: count=%d bytes=%d", len(session.cachedStates), session.cachedStateBytes)
	}
}

func TestResponsesWebSocketSession_OversizedContinuationClearsCache(t *testing.T) {
	const responseID = "resp_previous"
	base, err := normalizeResponsesWebSocketPayload(map[string]json.RawMessage{
		"model": json.RawMessage(`"gpt"`),
		"input": json.RawMessage(`"first turn"`),
	})
	if err != nil {
		t.Fatalf("normalize base: %v", err)
	}
	delta := map[string]json.RawMessage{"input": json.RawMessage(`"second turn"`)}
	merged, err := mergeResponsesWebSocketPayloads(base, delta)
	if err != nil {
		t.Fatalf("merge fixture: %v", err)
	}
	baseBytes, ok := responsesWebSocketCachedStateSize(responseID, base, 1<<20)
	if !ok {
		t.Fatal("base fixture unexpectedly oversized")
	}
	mergedBytes, ok := responsesWebSocketCachedStateSize(responseID, merged, 1<<20)
	if !ok || mergedBytes <= baseBytes {
		t.Fatalf("merged fixture bytes = %d, want greater than base %d", mergedBytes, baseBytes)
	}

	session := &responsesWebSocketSession{
		cachedStates:        make(map[string]*responsesWebSocketCachedState),
		maxCachedStateBytes: mergedBytes - 1,
	}
	if cacheErr := session.cacheState(responseID, base); cacheErr != nil {
		t.Fatalf("cache base: %v", cacheErr)
	}
	_, resolveErr := session.resolveCreatePayload(&responsesWebSocketCreateRequest{
		previousResponseID: responseID,
		payload:            delta,
		generate:           true,
	})
	eventErr, ok := resolveErr.(*responsesWebSocketEventError)
	if !ok || eventErr.code != "state_too_large" {
		t.Fatalf("resolve error = %#v, want state_too_large", resolveErr)
	}
	if len(session.cachedStates) != 0 || session.cachedStateBytes != 0 {
		t.Fatalf("failed continuation retained cache: count=%d bytes=%d", len(session.cachedStates), session.cachedStateBytes)
	}
}

func TestResponsesWebSocketProxy_ReplacesOversizedTerminalWithStateError(t *testing.T) {
	type result struct {
		err         error
		terminalErr *responsesWebSocketEventError
		cacheCount  int
		cacheBytes  int
	}
	resultCh := make(chan result, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := responsesWebSocketUpgrader.Upgrade(w, r, nil)
		if err != nil {
			resultCh <- result{err: err}
			return
		}
		defer conn.Close()

		session := &responsesWebSocketSession{
			conn:                conn,
			cachedStates:        make(map[string]*responsesWebSocketCachedState),
			maxCachedStateBytes: 128,
		}
		if cacheErr := session.cacheState("resp_old", map[string]json.RawMessage{"model": json.RawMessage(`"gpt"`)}); cacheErr != nil {
			resultCh <- result{err: cacheErr}
			return
		}
		proxy := newResponsesWebSocketProxy(session)
		proxy.cacheBasePayload = map[string]json.RawMessage{"model": json.RawMessage(`"gpt"`)}
		terminal, err := json.Marshal(map[string]interface{}{
			"type": "response.completed",
			"response": map[string]interface{}{
				"id":     "resp_large",
				"status": "completed",
				"output": []interface{}{map[string]interface{}{
					"type": "message",
					"role": "assistant",
					"content": []interface{}{map[string]interface{}{
						"type": "output_text",
						"text": strings.Repeat("x", 256),
					}},
				}},
			},
		})
		if err == nil {
			err = proxy.sendEvent(terminal)
		}
		resultCh <- result{
			err:         err,
			terminalErr: proxy.terminalError,
			cacheCount:  len(session.cachedStates),
			cacheBytes:  session.cachedStateBytes,
		}
	}))
	defer server.Close()

	conn := mustDialResponsesWebSocket(t, server.URL)
	defer conn.Close()
	event := readResponsesWebSocketEvent(t, conn)
	if got, _ := event["type"].(string); got != "error" {
		t.Fatalf("event type = %q, want error", got)
	}
	errorPayload, _ := event["error"].(map[string]interface{})
	if got, _ := errorPayload["code"].(string); got != "state_too_large" {
		t.Fatalf("error code = %q, want state_too_large", got)
	}

	got := <-resultCh
	if got.err != nil {
		t.Fatalf("proxy send: %v", got.err)
	}
	if got.terminalErr == nil || got.terminalErr.code != "state_too_large" {
		t.Fatalf("terminal error = %#v, want state_too_large", got.terminalErr)
	}
	if got.cacheCount != 0 || got.cacheBytes != 0 {
		t.Fatalf("oversized terminal retained cache: count=%d bytes=%d", got.cacheCount, got.cacheBytes)
	}
}

func TestResponsesWebSocketProxy_ClearsCachedTerminalAfterWriteFailure(t *testing.T) {
	session := &responsesWebSocketSession{
		cachedStates:        make(map[string]*responsesWebSocketCachedState),
		maxCachedStateBytes: 1024,
	}
	proxy := newResponsesWebSocketProxy(session)
	proxy.cacheBasePayload = map[string]json.RawMessage{"model": json.RawMessage(`"gpt"`)}
	terminal := []byte(`{"type":"response.completed","response":{"id":"resp_write_failed","status":"completed","output":[]}}`)

	if err := proxy.sendEvent(terminal); err == nil {
		t.Fatal("sendEvent with closed session succeeded")
	}
	if proxy.stateCached {
		t.Fatal("failed terminal write remained marked cached")
	}
	if len(session.cachedStates) != 0 || session.cachedStateBytes != 0 {
		t.Fatalf("failed terminal write retained cache: count=%d bytes=%d", len(session.cachedStates), session.cachedStateBytes)
	}
}

func sizedJSONString(t *testing.T, size int) json.RawMessage {
	t.Helper()
	if size < 2 {
		t.Fatalf("JSON string size %d is too small", size)
	}
	return json.RawMessage(`"` + strings.Repeat("x", size-2) + `"`)
}
