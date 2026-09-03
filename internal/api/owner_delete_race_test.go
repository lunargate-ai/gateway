package api

import (
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

func TestStoredChatCompletionDeletePreservesConcurrentConflictTombstone(t *testing.T) {
	const resourceID = "chatcmpl_delete_race"
	alpha, started, release, alphaCalls := newBlockedOwnerDeleteUpstream(t, "/v1/chat/completions/"+resourceID)
	defer alpha.Close()
	defer release()
	beta, betaCalls := newRejectingOwnerDeleteUpstream()
	defer beta.Close()

	router, handler, cache := newStoredChatLifecycleRouterFromConfigs(t, map[string]config.ProviderConfig{
		"alpha": ownerDeleteProviderConfig(alpha.URL+"/v1", "alpha-secret", config.ProviderCapabilities{ChatCompletionsLifecycle: true}),
		"beta":  ownerDeleteProviderConfig(beta.URL+"/v1", "beta-secret", config.ProviderCapabilities{ChatCompletionsLifecycle: true}),
	})
	defer cache.Stop()
	alphaBinding := mustChatCompletionBinding(t, handler, "alpha")
	if got := handler.chatCompletionBindings.claim(resourceID, alphaBinding); got != ownerClaimed {
		t.Fatalf("initial owner claim = %v, want claimed", got)
	}

	result := performOwnerDeleteAsync(router, "/v1/chat/completions/"+resourceID)
	waitForOwnerDeleteStart(t, started)
	if got := handler.chatCompletionBindings.claim(resourceID, mustChatCompletionBinding(t, handler, "beta")); got != ownerClaimConflict {
		t.Fatalf("concurrent owner claim = %v, want conflict", got)
	}
	release()
	response := waitForOwnerDeleteResult(t, result)
	if response.Code != http.StatusOK {
		t.Fatalf("delete status = %d, want 200; body=%s", response.Code, response.Body.String())
	}
	if _, lookup := handler.chatCompletionBindings.lookup(resourceID); lookup != ownerLookupConflict {
		t.Fatalf("owner lookup after delete = %v, want conflict", lookup)
	}
	if alphaCalls.Load() != 1 || betaCalls.Load() != 0 {
		t.Fatalf("upstream calls: alpha=%d beta=%d, want 1/0", alphaCalls.Load(), betaCalls.Load())
	}
}

func TestNativeConversationDeletePreservesConcurrentConflictTombstone(t *testing.T) {
	const resourceID = "conv_delete_race"
	alpha, started, release, alphaCalls := newBlockedOwnerDeleteUpstream(t, "/v1/conversations/"+resourceID)
	defer alpha.Close()
	defer release()
	beta, betaCalls := newRejectingOwnerDeleteUpstream()
	defer beta.Close()

	router, handler, cache := newNativeLifecycleRouterFromConfigs(t, map[string]config.ProviderConfig{
		"alpha": ownerDeleteProviderConfig(alpha.URL+"/v1", "alpha-secret", config.ProviderCapabilities{Conversations: true}),
		"beta":  ownerDeleteProviderConfig(beta.URL+"/v1", "beta-secret", config.ProviderCapabilities{Conversations: true}),
	})
	defer cache.Stop()
	alphaBinding, err := handler.validateConversationProvider("alpha")
	if err != nil {
		t.Fatal(err)
	}
	if got := handler.conversationBindings.claim(resourceID, alphaBinding); got != ownerClaimed {
		t.Fatalf("initial owner claim = %v, want claimed", got)
	}

	result := performOwnerDeleteAsync(router, "/v1/conversations/"+resourceID)
	waitForOwnerDeleteStart(t, started)
	betaBinding, err := handler.validateConversationProvider("beta")
	if err != nil {
		t.Fatal(err)
	}
	if got := handler.conversationBindings.claim(resourceID, betaBinding); got != ownerClaimConflict {
		t.Fatalf("concurrent owner claim = %v, want conflict", got)
	}
	release()
	response := waitForOwnerDeleteResult(t, result)
	if response.Code != http.StatusOK {
		t.Fatalf("delete status = %d, want 200; body=%s", response.Code, response.Body.String())
	}
	if _, lookup := handler.conversationBindings.lookup(resourceID); lookup != ownerLookupConflict {
		t.Fatalf("owner lookup after delete = %v, want conflict", lookup)
	}
	if alphaCalls.Load() != 1 || betaCalls.Load() != 0 {
		t.Fatalf("upstream calls: alpha=%d beta=%d, want 1/0", alphaCalls.Load(), betaCalls.Load())
	}
}

func TestNativeResponseDeletePreservesConcurrentConflictTombstone(t *testing.T) {
	const resourceID = "resp_delete_race"
	alpha, started, release, alphaCalls := newBlockedOwnerDeleteUpstream(t, "/v1/responses/"+resourceID)
	defer alpha.Close()
	defer release()
	beta, betaCalls := newRejectingOwnerDeleteUpstream()
	defer beta.Close()

	router, handler, cache := newNativeLifecycleRouterFromConfigs(t, map[string]config.ProviderConfig{
		"alpha": ownerDeleteProviderConfig(alpha.URL+"/v1", "alpha-secret", config.ProviderCapabilities{ResponsesLifecycle: true}),
		"beta":  ownerDeleteProviderConfig(beta.URL+"/v1", "beta-secret", config.ProviderCapabilities{ResponsesLifecycle: true}),
	})
	defer cache.Stop()
	alphaBinding := mustResponseBinding(t, handler, "alpha")
	if got := handler.responseBindings.claim(resourceID, alphaBinding); got != ownerClaimed {
		t.Fatalf("initial owner claim = %v, want claimed", got)
	}

	result := performOwnerDeleteAsync(router, "/v1/responses/"+resourceID)
	waitForOwnerDeleteStart(t, started)
	if got := handler.responseBindings.claim(resourceID, mustResponseBinding(t, handler, "beta")); got != ownerClaimConflict {
		t.Fatalf("concurrent owner claim = %v, want conflict", got)
	}
	release()
	response := waitForOwnerDeleteResult(t, result)
	if response.Code != http.StatusOK {
		t.Fatalf("delete status = %d, want 200; body=%s", response.Code, response.Body.String())
	}
	if _, lookup := handler.responseBindings.lookup(resourceID); lookup != ownerLookupConflict {
		t.Fatalf("owner lookup after delete = %v, want conflict", lookup)
	}
	if alphaCalls.Load() != 1 || betaCalls.Load() != 0 {
		t.Fatalf("upstream calls: alpha=%d beta=%d, want 1/0", alphaCalls.Load(), betaCalls.Load())
	}
}

func ownerDeleteProviderConfig(baseURL string, apiKey string, capabilities config.ProviderCapabilities) config.ProviderConfig {
	return config.ProviderConfig{
		Type:         "openai",
		APIKey:       apiKey,
		BaseURL:      baseURL,
		DefaultModel: "gpt-native",
		Capabilities: capabilities,
	}
}

func newBlockedOwnerDeleteUpstream(
	t *testing.T,
	expectedPath string,
) (*httptest.Server, <-chan struct{}, func(), *atomic.Int32) {
	t.Helper()
	started := make(chan struct{})
	release := make(chan struct{})
	var startOnce sync.Once
	var releaseOnce sync.Once
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		if r.Method != http.MethodDelete || r.URL.Path != expectedPath {
			t.Errorf("unexpected upstream request: %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		startOnce.Do(func() { close(started) })
		<-release
		w.Header().Set("Content-Type", "application/json")
		if conversationID := strings.TrimPrefix(expectedPath, "/v1/conversations/"); conversationID != expectedPath {
			_, _ = io.WriteString(w, `{"id":"`+conversationID+`","object":"conversation.deleted","deleted":true}`)
			return
		}
		if completionID := strings.TrimPrefix(expectedPath, "/v1/chat/completions/"); completionID != expectedPath {
			_, _ = io.WriteString(w, `{"id":"`+completionID+`","object":"chat.completion.deleted","deleted":true}`)
			return
		}
		_, _ = io.WriteString(w, `{"deleted":true}`)
	}))
	return server, started, func() { releaseOnce.Do(func() { close(release) }) }, &calls
}

func newRejectingOwnerDeleteUpstream() (*httptest.Server, *atomic.Int32) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusInternalServerError)
	}))
	return server, &calls
}

func performOwnerDeleteAsync(handler http.Handler, path string) <-chan *httptest.ResponseRecorder {
	result := make(chan *httptest.ResponseRecorder, 1)
	go func() {
		request := httptest.NewRequest(http.MethodDelete, path, nil)
		recorder := httptest.NewRecorder()
		handler.ServeHTTP(recorder, request)
		result <- recorder
	}()
	return result
}

func waitForOwnerDeleteStart(t *testing.T, started <-chan struct{}) {
	t.Helper()
	select {
	case <-started:
	case <-time.After(3 * time.Second):
		t.Fatal("timed out waiting for upstream DELETE to block")
	}
}

func waitForOwnerDeleteResult(t *testing.T, result <-chan *httptest.ResponseRecorder) *httptest.ResponseRecorder {
	t.Helper()
	select {
	case response := <-result:
		return response
	case <-time.After(3 * time.Second):
		t.Fatal("timed out waiting for upstream DELETE to complete")
		return nil
	}
}
