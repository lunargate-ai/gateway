package api

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestBindingStoreClaimsFailClosedOnOwnerConflict(t *testing.T) {
	t.Run("responses provider", func(t *testing.T) {
		store := newResponseBindingStore(time.Hour)
		first := responseBinding{Provider: "alpha", AccountFingerprint: "account-a", Route: "first"}
		if got := store.claim("resp_shared", first); got != ownerClaimed {
			t.Fatalf("first claim = %v, want claimed", got)
		}
		if got := store.claim("resp_shared", responseBinding{
			Provider:           "alpha",
			AccountFingerprint: "account-a",
			Route:              "first",
		}); got != ownerClaimRefreshed {
			t.Fatalf("same-owner claim = %v, want refreshed", got)
		}
		if got := store.claim("resp_shared", responseBinding{
			Provider:           "alpha",
			AccountFingerprint: "account-a",
			Route:              "other-target",
		}); got != ownerClaimConflict {
			t.Fatalf("different-target claim = %v, want conflict", got)
		}
		assertResponseOwnerConflict(t, store, "resp_shared")

		if got := store.claim("resp_provider", first); got != ownerClaimed {
			t.Fatalf("provider first claim = %v, want claimed", got)
		}
		if got := store.claim("resp_provider", responseBinding{
			Provider:           "beta",
			AccountFingerprint: "account-b",
		}); got != ownerClaimConflict {
			t.Fatalf("different-owner claim = %v, want conflict", got)
		}
		assertResponseOwnerConflict(t, store, "resp_provider")

		if got := store.claim("resp_storage_kind", responseBinding{
			Provider:           "alpha",
			AccountFingerprint: "account-a",
		}); got != ownerClaimed {
			t.Fatalf("native claim = %v, want claimed", got)
		}
		if got := store.claim("resp_storage_kind", responseBinding{
			Provider:           "alpha",
			AccountFingerprint: "account-a",
			LocalSnapshot:      true,
		}); got != ownerClaimConflict {
			t.Fatalf("native/local claim = %v, want conflict", got)
		}
		assertResponseOwnerConflict(t, store, "resp_storage_kind")
	})

	t.Run("responses account", func(t *testing.T) {
		store := newResponseBindingStore(time.Hour)
		if got := store.claim("resp_account", responseBinding{
			Provider:           "alpha",
			AccountFingerprint: "account-a",
		}); got != ownerClaimed {
			t.Fatalf("first claim = %v, want claimed", got)
		}
		if got := store.claim("resp_account", responseBinding{
			Provider:           "alpha",
			AccountFingerprint: "rotated-account",
		}); got != ownerClaimConflict {
			t.Fatalf("rotated-account claim = %v, want conflict", got)
		}
		assertResponseOwnerConflict(t, store, "resp_account")
	})

	t.Run("conversations", func(t *testing.T) {
		store := newConversationBindingStore(time.Hour)
		first := conversationBinding{Provider: "alpha", AccountFingerprint: "account-a"}
		if got := store.claim("conv_shared", first); got != ownerClaimed {
			t.Fatalf("first claim = %v, want claimed", got)
		}
		if got := store.claim("conv_shared", first); got != ownerClaimRefreshed {
			t.Fatalf("same-owner claim = %v, want refreshed", got)
		}
		if got := store.claim("conv_shared", conversationBinding{
			Provider:           "beta",
			AccountFingerprint: "account-b",
		}); got != ownerClaimConflict {
			t.Fatalf("different-owner claim = %v, want conflict", got)
		}
		if _, got := store.lookup("conv_shared"); got != ownerLookupConflict {
			t.Fatalf("lookup = %v, want conflict", got)
		}
		if _, ok := store.get("conv_shared"); ok {
			t.Fatal("legacy lookup exposed an ambiguous conversation owner")
		}
	})

	t.Run("stored chat", func(t *testing.T) {
		store := newChatCompletionBindingStore(time.Hour)
		first := chatCompletionBinding{Provider: "alpha", AccountFingerprint: "account-a", Model: "first"}
		if got := store.claim("chatcmpl_shared", first); got != ownerClaimed {
			t.Fatalf("first claim = %v, want claimed", got)
		}
		if got := store.claim("chatcmpl_shared", chatCompletionBinding{
			Provider:           "alpha",
			AccountFingerprint: "account-a",
			Model:              "other-metadata",
		}); got != ownerClaimRefreshed {
			t.Fatalf("same-owner claim = %v, want refreshed", got)
		}
		if got := store.claim("chatcmpl_shared", chatCompletionBinding{
			Provider:           "alpha",
			AccountFingerprint: "rotated-account",
		}); got != ownerClaimConflict {
			t.Fatalf("rotated-account claim = %v, want conflict", got)
		}
		if _, got := store.lookup("chatcmpl_shared"); got != ownerLookupConflict {
			t.Fatalf("lookup = %v, want conflict", got)
		}
		if _, ok := store.get("chatcmpl_shared"); ok {
			t.Fatal("legacy lookup exposed an ambiguous stored-chat owner")
		}
	})
}

func TestBindingStoreClaimsRemainAmbiguousUnderConcurrency(t *testing.T) {
	// Build each closure once; the explicit structure keeps the three stores
	// independent while exercising the same simultaneous-claim schedule.
	responseStore := newResponseBindingStore(time.Hour)
	conversationStore := newConversationBindingStore(time.Hour)
	chatStore := newChatCompletionBindingStore(time.Hour)
	tests := []struct {
		name   string
		claim  func(int) ownerClaimResult
		lookup func() ownerLookupResult
	}{
		{
			name: "responses",
			claim: func(owner int) ownerClaimResult {
				return responseStore.claim("resp_race", responseBinding{
					Provider:           claimOwnerProvider(owner),
					AccountFingerprint: claimOwnerAccount(owner),
				})
			},
			lookup: func() ownerLookupResult {
				_, result := responseStore.lookup("resp_race")
				return result
			},
		},
		{
			name: "conversations",
			claim: func(owner int) ownerClaimResult {
				return conversationStore.claim("conv_race", conversationBinding{
					Provider:           claimOwnerProvider(owner),
					AccountFingerprint: claimOwnerAccount(owner),
				})
			},
			lookup: func() ownerLookupResult {
				_, result := conversationStore.lookup("conv_race")
				return result
			},
		},
		{
			name: "stored chat",
			claim: func(owner int) ownerClaimResult {
				return chatStore.claim("chatcmpl_race", chatCompletionBinding{
					Provider:           claimOwnerProvider(owner),
					AccountFingerprint: claimOwnerAccount(owner),
				})
			},
			lookup: func() ownerLookupResult {
				_, result := chatStore.lookup("chatcmpl_race")
				return result
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			const workers = 64
			start := make(chan struct{})
			var wait sync.WaitGroup
			var retained atomic.Int32
			var conflicts atomic.Int32
			for worker := 0; worker < workers; worker++ {
				owner := worker % 2
				wait.Add(1)
				go func(owner int) {
					defer wait.Done()
					<-start
					result := test.claim(owner)
					if result.retained() {
						retained.Add(1)
					}
					if result == ownerClaimConflict {
						conflicts.Add(1)
					}
				}(owner)
			}
			close(start)
			wait.Wait()
			if retained.Load() == 0 || conflicts.Load() == 0 {
				t.Fatalf("retained=%d conflicts=%d, want both", retained.Load(), conflicts.Load())
			}
			if got := test.lookup(); got != ownerLookupConflict {
				t.Fatalf("final lookup = %v, want conflict", got)
			}
		})
	}
}

func assertResponseOwnerConflict(t *testing.T, store *responseBindingStore, responseID string) {
	t.Helper()
	if _, got := store.lookup(responseID); got != ownerLookupConflict {
		t.Fatalf("lookup = %v, want conflict", got)
	}
	if _, ok := store.get(responseID); ok {
		t.Fatal("legacy lookup exposed an ambiguous response owner")
	}
}

func claimOwnerProvider(owner int) string {
	if owner == 0 {
		return "alpha"
	}
	return "beta"
}

func claimOwnerAccount(owner int) string {
	if owner == 0 {
		return "account-a"
	}
	return "account-b"
}
