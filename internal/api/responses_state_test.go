package api

import (
	"encoding/json"
	"fmt"
	"sync"
	"testing"
	"time"
)

func TestResponsesStateStoreEvictsOldestAtEntryLimit(t *testing.T) {
	store := newResponsesStateStore(time.Hour)
	store.maxEntries = 2
	store.put("resp_1", stateTestPayload("one"))
	store.put("resp_2", stateTestPayload("two"))
	store.put("resp_3", stateTestPayload("three"))

	if _, ok := store.get("resp_1"); ok {
		t.Fatal("oldest response was not evicted")
	}
	for _, responseID := range []string{"resp_2", "resp_3"} {
		if _, ok := store.get(responseID); !ok {
			t.Fatalf("expected %s to remain", responseID)
		}
	}
}

func TestResponsesStateStoreEnforcesByteBudget(t *testing.T) {
	store := newResponsesStateStore(time.Hour)
	first := stateTestPayload("first payload")
	second := stateTestPayload("second payload")
	store.maxBytes = responsesStatePayloadSize("resp_1", first) + responsesStatePayloadSize("resp_2", second) - 1

	store.put("resp_1", first)
	store.put("resp_2", second)

	if _, ok := store.get("resp_1"); ok {
		t.Fatal("oldest response was not evicted to honor byte budget")
	}
	if _, ok := store.get("resp_2"); !ok {
		t.Fatal("newest response should fit after eviction")
	}
	if store.totalBytes > store.maxBytes {
		t.Fatalf("totalBytes = %d exceeds maxBytes = %d", store.totalBytes, store.maxBytes)
	}
}

func TestResponsesStateStoreConcurrentReplacement(t *testing.T) {
	store := newResponsesStateStore(time.Hour)
	var wg sync.WaitGroup
	for worker := 0; worker < 8; worker++ {
		worker := worker
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 200; i++ {
				responseID := fmt.Sprintf("resp_%d", i%16)
				store.put(responseID, stateTestPayload(fmt.Sprintf("%d-%d", worker, i)))
				_, _ = store.get(responseID)
			}
		}()
	}
	wg.Wait()

	store.mu.Lock()
	defer store.mu.Unlock()
	if len(store.entries) > 16 {
		t.Fatalf("entries = %d, want at most 16", len(store.entries))
	}
	calculated := 0
	for _, entry := range store.entries {
		calculated += entry.size
	}
	if store.totalBytes != calculated {
		t.Fatalf("totalBytes = %d, calculated = %d", store.totalBytes, calculated)
	}
}

func stateTestPayload(value string) map[string]json.RawMessage {
	return map[string]json.RawMessage{"input": json.RawMessage(fmt.Sprintf("%q", value))}
}
