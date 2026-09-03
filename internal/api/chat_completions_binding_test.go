package api

import (
	"strings"
	"testing"
	"time"

	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestChatCompletionBindingStoreExpiresAndEvictsLeastRecentlyUsed(t *testing.T) {
	store := newChatCompletionBindingStore(time.Minute)
	store.maxEntries = 2
	now := time.Unix(1_700_000_000, 0)
	store.now = func() time.Time { return now }

	first := chatCompletionBinding{Provider: "first", AccountFingerprint: "first-account"}
	second := chatCompletionBinding{Provider: "second", AccountFingerprint: "second-account"}
	third := chatCompletionBinding{Provider: "third", AccountFingerprint: "third-account"}
	if !store.put("chatcmpl_first", first) || !store.put("chatcmpl_second", second) {
		t.Fatal("failed to seed bindings")
	}
	if _, ok := store.get("chatcmpl_first"); !ok {
		t.Fatal("failed to refresh first binding")
	}
	if !store.put("chatcmpl_third", third) {
		t.Fatal("failed to insert third binding")
	}
	if _, ok := store.get("chatcmpl_second"); ok {
		t.Fatal("least recently used binding was not evicted")
	}
	if binding, ok := store.get("chatcmpl_first"); !ok || binding.Provider != "first" {
		t.Fatalf("first binding = %#v, %v", binding, ok)
	}

	now = now.Add(time.Minute)
	if _, ok := store.get("chatcmpl_first"); ok {
		t.Fatal("binding was retained at its expiry boundary")
	}
}

func TestChatCompletionBindingStoreRejectsInvalidOrOversizedEntries(t *testing.T) {
	store := newChatCompletionBindingStore(time.Hour)
	store.maxBytes = 48

	invalid := []struct {
		id      string
		binding chatCompletionBinding
	}{
		{id: "", binding: chatCompletionBinding{Provider: "native", AccountFingerprint: "account"}},
		{id: "chatcmpl_missing_provider", binding: chatCompletionBinding{AccountFingerprint: "account"}},
		{id: "chatcmpl_missing_account", binding: chatCompletionBinding{Provider: "native"}},
		{id: "chatcmpl_large", binding: chatCompletionBinding{Provider: "native", Model: strings.Repeat("m", 64), AccountFingerprint: "account"}},
	}
	for _, testCase := range invalid {
		if store.put(testCase.id, testCase.binding) {
			t.Fatalf("accepted invalid binding %#v for %q", testCase.binding, testCase.id)
		}
	}
	if len(store.entries) != 0 || store.totalBytes != 0 {
		t.Fatalf("invalid entries changed store: entries=%d bytes=%d", len(store.entries), store.totalBytes)
	}
}

func TestChatCompletionBindingStoreDelete(t *testing.T) {
	store := newChatCompletionBindingStore(time.Hour)
	binding := chatCompletionBinding{Provider: "native", AccountFingerprint: "account"}
	if !store.put("chatcmpl_delete", binding) {
		t.Fatal("failed to seed binding")
	}
	if !store.delete("chatcmpl_delete") {
		t.Fatal("delete returned false")
	}
	if store.delete("chatcmpl_delete") {
		t.Fatal("second delete returned true")
	}
	if _, ok := store.get("chatcmpl_delete"); ok {
		t.Fatal("deleted binding remains readable")
	}
}

func TestChatCompletionStreamBindingCandidateRequiresOneStableID(t *testing.T) {
	var candidate chatCompletionStreamBindingCandidate
	candidate.observe(nil)
	candidate.observe(&models.StreamChunk{})
	candidate.observe(&models.StreamChunk{ID: " chatcmpl_stable "})
	candidate.observe(&models.StreamChunk{ID: "chatcmpl_stable"})
	if got := candidate.completionID(); got != "chatcmpl_stable" {
		t.Fatalf("stable completion ID = %q", got)
	}

	candidate.observe(&models.StreamChunk{ID: "chatcmpl_other"})
	if got := candidate.completionID(); got != "" {
		t.Fatalf("inconsistent completion ID = %q, want empty", got)
	}
}

func TestChatCompletionStreamBindingCandidateUsesRawUpstreamID(t *testing.T) {
	var candidate chatCompletionStreamBindingCandidate
	candidate.observe(&models.StreamChunk{
		ID:      "chatcmpl_normalized",
		RawJSON: []byte(`{"id":"chatcmpl_upstream"}`),
	})
	candidate.observe(&models.StreamChunk{
		ID:      "chatcmpl_normalized",
		RawJSON: []byte(`{"id":"chatcmpl_other"}`),
	})
	if got := candidate.completionID(); got != "" {
		t.Fatalf("conflicting raw completion ID = %q, want empty", got)
	}
}
