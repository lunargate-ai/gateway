package api

import (
	"bytes"
	"encoding/json"
	"fmt"
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

func TestResponseOwnerConflictDiscardsLocalSnapshot(t *testing.T) {
	providerConfigs := map[string]config.ProviderConfig{
		"alpha": {
			Type:         "openai",
			APIKey:       "alpha-secret",
			BaseURL:      "http://alpha.invalid/v1",
			DefaultModel: "gpt-native",
			Capabilities: config.ProviderCapabilities{ResponsesLifecycle: true},
		},
		"beta": {
			Type:         "openai",
			APIKey:       "beta-secret",
			BaseURL:      "http://beta.invalid/v1",
			DefaultModel: "gpt-native",
			Capabilities: config.ProviderCapabilities{ResponsesLifecycle: true},
		},
	}
	_, handler, cache := newNativeLifecycleRouterFromConfigs(t, providerConfigs)
	defer cache.Stop()

	requestPayload := map[string]json.RawMessage{
		"model": json.RawMessage(`"gpt-native"`),
		"input": json.RawMessage(`"hello"`),
	}
	completed := map[string]interface{}{
		"id":          "resp_shared",
		"object":      "response",
		"status":      "completed",
		"model":       "gpt-native",
		"output":      []interface{}{},
		"output_text": "alpha",
	}
	if got := handler.retainLocalResponseSnapshot(
		"resp_shared",
		responseOwnerTestHeaders("alpha"),
		responseOwnerTestIdentity(t, handler, "alpha"),
		requestPayload,
		completed,
	); got != ownerClaimed {
		t.Fatalf("local claim = %v, want claimed", got)
	}
	if _, _, ok := handler.responsesState.getCompleted("resp_shared"); !ok {
		t.Fatal("local snapshot was not retained")
	}
	binding, lookup := handler.responseBindings.lookup("resp_shared")
	if lookup != ownerLookupBound || !binding.LocalSnapshot || binding.Provider != "alpha" {
		t.Fatalf("local owner = %#v lookup=%v", binding, lookup)
	}

	if got := handler.retainNativeResponseOwner(
		"resp_shared",
		responseOwnerTestHeaders("beta"),
		responseOwnerTestIdentity(t, handler, "beta"),
	); got != ownerClaimConflict {
		t.Fatalf("conflicting native claim = %v, want conflict", got)
	}
	if _, _, ok := handler.responsesState.getCompleted("resp_shared"); ok {
		t.Fatal("conflicting native owner left the older local snapshot reachable")
	}
	if _, lookup := handler.responseBindings.lookup("resp_shared"); lookup != ownerLookupConflict {
		t.Fatalf("owner lookup = %v, want conflict", lookup)
	}

	payload := map[string]json.RawMessage{
		"previous_response_id": json.RawMessage(`"resp_shared"`),
		"input":                json.RawMessage(`"continue"`),
	}
	for _, explicitProvider := range []string{"", "alpha", "beta"} {
		req := httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
		if explicitProvider != "" {
			req.Header.Set("X-LunarGate-Provider", explicitProvider)
		}
		if _, _, _, err := handler.resolveResponsesHTTPPayload(req, payload); responseBindingErrorCode(err) != "provider_binding_conflict" {
			t.Fatalf("provider %q resolve error = %v, want provider_binding_conflict", explicitProvider, err)
		}
	}
}

func TestResponseOwnerConflictConcurrentRetentionDiscardsSnapshot(t *testing.T) {
	providerConfigs := map[string]config.ProviderConfig{
		"alpha": {
			Type:         "openai",
			APIKey:       "alpha-secret",
			BaseURL:      "http://alpha.invalid/v1",
			DefaultModel: "gpt-native",
			Capabilities: config.ProviderCapabilities{ResponsesLifecycle: true},
		},
		"beta": {
			Type:         "openai",
			APIKey:       "beta-secret",
			BaseURL:      "http://beta.invalid/v1",
			DefaultModel: "gpt-native",
			Capabilities: config.ProviderCapabilities{ResponsesLifecycle: true},
		},
	}
	_, handler, cache := newNativeLifecycleRouterFromConfigs(t, providerConfigs)
	defer cache.Stop()
	alphaOwner := responseOwnerTestIdentity(t, handler, "alpha")
	betaOwner := responseOwnerTestIdentity(t, handler, "beta")
	requestPayload := map[string]json.RawMessage{"input": json.RawMessage(`"hello"`)}

	for iteration := 0; iteration < 64; iteration++ {
		responseID := fmt.Sprintf("resp_race_%d", iteration)
		start := make(chan struct{})
		var workers sync.WaitGroup
		workers.Add(2)
		go func() {
			defer workers.Done()
			<-start
			handler.retainLocalResponseSnapshot(
				responseID,
				responseOwnerTestHeaders("alpha"),
				alphaOwner,
				requestPayload,
				map[string]interface{}{"id": responseID, "object": "response"},
			)
		}()
		go func() {
			defer workers.Done()
			<-start
			handler.retainNativeResponseOwner(responseID, responseOwnerTestHeaders("beta"), betaOwner)
		}()
		close(start)
		workers.Wait()

		if _, lookup := handler.responseBindings.lookup(responseID); lookup != ownerLookupConflict {
			t.Fatalf("iteration %d lookup = %v, want conflict", iteration, lookup)
		}
		if _, ok := handler.responsesState.get(responseID); ok {
			t.Fatalf("iteration %d left conflicting local continuation state", iteration)
		}
	}
}

func TestResponsesContinuationPinsNativeOwner(t *testing.T) {
	var alphaCalls atomic.Int32
	var betaCalls atomic.Int32
	alpha := newResponseOwnerUpstream(t, &alphaCalls, "resp_alpha_next")
	defer alpha.Close()
	beta := newResponseOwnerUpstream(t, &betaCalls, "resp_beta_next")
	defer beta.Close()

	providerConfigs := map[string]config.ProviderConfig{
		"alpha": {
			Type:         "openai",
			APIKey:       "alpha-secret",
			BaseURL:      alpha.URL + "/v1",
			DefaultModel: "gpt-native",
			Capabilities: config.ProviderCapabilities{ResponsesLifecycle: true},
		},
		"beta": {
			Type:         "openai",
			APIKey:       "beta-secret",
			BaseURL:      beta.URL + "/v1",
			DefaultModel: "gpt-native",
			Capabilities: config.ProviderCapabilities{ResponsesLifecycle: true},
		},
	}
	router, handler, cache := newNativeLifecycleRouterFromConfigs(t, providerConfigs)
	defer cache.Stop()
	if got := handler.responseBindings.claim("resp_alpha", mustResponseBinding(t, handler, "alpha")); got != ownerClaimed {
		t.Fatalf("seed owner = %v, want claimed", got)
	}

	payload := []byte(`{"model":"lunargate/auto","previous_response_id":"resp_alpha","input":"continue","store":false}`)
	response := performLifecycleRequest(t, router, http.MethodPost, "/v1/responses", payload)
	if response.Code != http.StatusOK {
		t.Fatalf("follow-up status = %d, want 200; body=%s", response.Code, response.Body.String())
	}
	if alphaCalls.Load() != 1 || betaCalls.Load() != 0 {
		t.Fatalf("provider calls after implicit follow-up: alpha=%d beta=%d", alphaCalls.Load(), betaCalls.Load())
	}

	mismatchRequest := httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(payload))
	mismatchRequest.Header.Set("Content-Type", "application/json")
	mismatchRequest.Header.Set("X-LunarGate-Provider", "beta")
	mismatch := httptest.NewRecorder()
	router.ServeHTTP(mismatch, mismatchRequest)
	if mismatch.Code != http.StatusBadRequest {
		t.Fatalf("mismatch status = %d, want 400; body=%s", mismatch.Code, mismatch.Body.String())
	}
	assertLifecycleError(t, mismatch.Body.Bytes(), "provider", "invalid_value")
	if alphaCalls.Load() != 1 || betaCalls.Load() != 0 {
		t.Fatalf("provider mismatch reached upstream: alpha=%d beta=%d", alphaCalls.Load(), betaCalls.Load())
	}
}

func TestResponsesContinuationPinsExactNativeTarget(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(map[bool]string{false: "non-stream", true: "stream"}[stream], func(t *testing.T) {
			var seenPath string
			var seenModel string
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				seenPath = r.URL.Path
				var payload map[string]interface{}
				if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
					t.Errorf("decode upstream payload: %v", err)
				}
				seenModel, _ = payload["model"].(string)
				if stream {
					w.Header().Set("Content-Type", "text/event-stream")
					_, _ = io.WriteString(w, "event: response.completed\ndata: {\"type\":\"response.completed\",\"sequence_number\":0,\"response\":{\"id\":\"resp_next\",\"object\":\"response\",\"created_at\":1,\"status\":\"completed\",\"model\":\"gpt-native\",\"output\":[]}}\n\n")
					return
				}
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, `{"id":"resp_next","object":"response","created_at":1,"status":"completed","model":"gpt-native","output":[],"output_text":"ok"}`)
			}))
			defer upstream.Close()

			providerConfigs := map[string]config.ProviderConfig{
				"openai": {
					Type:         "openai",
					APIKey:       "provider-secret",
					BaseURL:      upstream.URL + "/v1",
					DefaultModel: "gpt-native",
					Capabilities: config.ProviderCapabilities{ResponsesLifecycle: true},
				},
			}
			router, handler, cache := newNativeLifecycleRouterFromConfigs(t, providerConfigs)
			defer cache.Stop()
			routingConfig := config.RoutingConfig{
				DefaultStrategy: "weighted",
				Routes: []config.RouteConfig{
					{
						Name:    "wrong-route",
						Match:   config.MatchConfig{Path: "/v1/responses"},
						Targets: []config.TargetConfig{{Provider: "openai", Model: "gpt-native", Weight: 1, UpstreamRequestType: requestTypeResponses}},
					},
					{
						Name:  "owner-route",
						Match: config.MatchConfig{Path: "/v1/responses"},
						Targets: []config.TargetConfig{
							{Provider: "openai", Model: "wrong-model", Weight: 100, UpstreamRequestType: requestTypeResponses},
							{Provider: "openai", Model: "gpt-native", Weight: 100, UpstreamRequestType: requestTypeChatCompletions},
							{Provider: "openai", Model: "gpt-native", Weight: 1, UpstreamRequestType: requestTypeResponses},
						},
					},
				},
			}
			if changed, err := handler.UpdateRuntime(providerConfigs, routingConfig, config.ModelSelectionConfig{}); err != nil {
				t.Fatalf("update runtime: %v", err)
			} else if !changed {
				t.Fatal("routing runtime update reported no change")
			}

			binding := mustResponseBinding(t, handler, "openai")
			binding.Route = "owner-route"
			binding.Model = "openai/gpt-native"
			binding.UpstreamRequestType = requestTypeResponses
			if got := handler.responseBindings.claim("resp_owned_target", binding); got != ownerClaimed {
				t.Fatalf("seed owner = %v, want claimed", got)
			}

			streamField := ""
			if stream {
				streamField = `,"stream":true`
			}
			response := performLifecycleRequest(t, router, http.MethodPost, "/v1/responses", []byte(
				`{"model":"lunargate/auto","previous_response_id":"resp_owned_target","input":"continue","store":false`+streamField+`}`,
			))
			if response.Code != http.StatusOK {
				t.Fatalf("status = %d, want 200; body=%s", response.Code, response.Body.String())
			}
			if seenPath != "/v1/responses" {
				t.Fatalf("upstream path = %q, want native /v1/responses", seenPath)
			}
			if seenModel != "gpt-native" {
				t.Fatalf("upstream model = %q, want owner model", seenModel)
			}
			if response.Header().Get("X-LunarGate-Route") != "owner-route" {
				t.Fatalf("selected route = %q, want owner-route", response.Header().Get("X-LunarGate-Route"))
			}
		})
	}
}

func TestResponseOwnerCollisionAllowsOnlyExplicitNativeLifecycle(t *testing.T) {
	var alphaCalls atomic.Int32
	var betaCalls atomic.Int32
	alpha := newResponseOwnerUpstream(t, &alphaCalls, "resp_collision")
	defer alpha.Close()
	beta := newResponseOwnerUpstream(t, &betaCalls, "resp_collision")
	defer beta.Close()
	providerConfigs := map[string]config.ProviderConfig{
		"alpha": {
			Type:         "openai",
			APIKey:       "alpha-secret",
			BaseURL:      alpha.URL + "/v1",
			DefaultModel: "gpt-native",
			Capabilities: config.ProviderCapabilities{ResponsesLifecycle: true},
		},
		"beta": {
			Type:         "openai",
			APIKey:       "beta-secret",
			BaseURL:      beta.URL + "/v1",
			DefaultModel: "gpt-native",
			Capabilities: config.ProviderCapabilities{ResponsesLifecycle: true},
		},
	}
	router, handler, cache := newNativeLifecycleRouterFromConfigs(t, providerConfigs)
	defer cache.Stop()
	if got := handler.responseBindings.claim("resp_collision", mustResponseBinding(t, handler, "alpha")); got != ownerClaimed {
		t.Fatalf("alpha claim = %v, want claimed", got)
	}
	handler.responsesState.putCompleted(
		"resp_collision",
		map[string]json.RawMessage{"input": json.RawMessage(`"stale"`)},
		map[string]interface{}{"id": "resp_collision", "object": "response"},
	)
	if got := handler.responseBindings.claim("resp_collision", mustResponseBinding(t, handler, "beta")); got != ownerClaimConflict {
		t.Fatalf("beta claim = %v, want conflict", got)
	}

	implicit := performLifecycleRequest(t, router, http.MethodGet, "/v1/responses/resp_collision", nil)
	if implicit.Code != http.StatusBadRequest {
		t.Fatalf("implicit status = %d, want 400; body=%s", implicit.Code, implicit.Body.String())
	}
	assertLifecycleError(t, implicit.Body.Bytes(), "response_id", "provider_binding_conflict")
	if _, _, ok := handler.responsesState.getCompleted("resp_collision"); ok {
		t.Fatal("conflict lifecycle lookup left stale local state reachable")
	}
	if alphaCalls.Load() != 0 || betaCalls.Load() != 0 {
		t.Fatalf("implicit conflict reached upstream: alpha=%d beta=%d", alphaCalls.Load(), betaCalls.Load())
	}

	explicitRequest := httptest.NewRequest(http.MethodGet, "/v1/responses/resp_collision", nil)
	explicitRequest.Header.Set("X-LunarGate-Provider", "beta")
	explicit := httptest.NewRecorder()
	router.ServeHTTP(explicit, explicitRequest)
	if explicit.Code != http.StatusOK {
		t.Fatalf("explicit status = %d, want 200; body=%s", explicit.Code, explicit.Body.String())
	}
	if alphaCalls.Load() != 0 || betaCalls.Load() != 1 {
		t.Fatalf("explicit lifecycle provider calls: alpha=%d beta=%d", alphaCalls.Load(), betaCalls.Load())
	}
}

func TestResponseOwnerUsesExecutionGenerationAcrossReload(t *testing.T) {
	started := make(chan struct{}, 1)
	release := make(chan struct{})
	oldUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		started <- struct{}{}
		<-release
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"resp_generation","object":"response","created_at":1,"status":"completed","model":"gpt-native","output":[],"output_text":"ok"}`)
	}))
	defer oldUpstream.Close()
	var newCalls atomic.Int32
	newUpstream := newResponseOwnerUpstream(t, &newCalls, "resp_new_generation")
	defer newUpstream.Close()

	original := map[string]config.ProviderConfig{
		"native": {
			Type:         "openai",
			APIKey:       "old-secret",
			BaseURL:      oldUpstream.URL + "/v1",
			DefaultModel: "gpt-native",
			Capabilities: config.ProviderCapabilities{ResponsesLifecycle: true},
		},
	}
	router, handler, cache := newNativeLifecycleRouterFromConfigs(t, original)
	defer cache.Stop()

	request := httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewBufferString(`{"model":"native/gpt-native","input":"hello"}`))
	request.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()
	done := make(chan struct{})
	go func() {
		defer close(done)
		router.ServeHTTP(recorder, request)
	}()
	select {
	case <-started:
	case <-time.After(5 * time.Second):
		close(release)
		t.Fatal("timed out waiting for the in-flight upstream request")
	}

	updated := map[string]config.ProviderConfig{
		"native": {
			Type:         "openai",
			APIKey:       "new-secret",
			BaseURL:      newUpstream.URL + "/v1",
			DefaultModel: "gpt-native",
			Capabilities: config.ProviderCapabilities{ResponsesLifecycle: true},
		},
	}
	handler.UpdateProviderConfigs(updated)
	close(release)
	<-done
	if recorder.Code != http.StatusOK {
		t.Fatalf("create status = %d, want 200; body=%s", recorder.Code, recorder.Body.String())
	}
	binding, lookup := handler.responseBindings.lookup("resp_generation")
	if lookup != ownerLookupBound {
		t.Fatalf("owner lookup = %v, want bound", lookup)
	}
	oldFingerprint := conversationAccountFingerprint("openai", oldUpstream.URL+"/v1", "", "old-secret")
	if binding.AccountFingerprint != oldFingerprint {
		t.Fatalf("captured fingerprint = %q, want execution generation", binding.AccountFingerprint)
	}
	currentFingerprint, ok := handler.bindRuntime().responseAccountFingerprint("native")
	if !ok || currentFingerprint == oldFingerprint {
		t.Fatalf("current fingerprint = %q ok=%t, want changed generation", currentFingerprint, ok)
	}

	followUp := performLifecycleRequest(t, router, http.MethodPost, "/v1/responses", []byte(`{"model":"lunargate/auto","previous_response_id":"resp_generation","input":"continue","store":false}`))
	if followUp.Code != http.StatusBadRequest {
		t.Fatalf("follow-up status = %d, want 400; body=%s", followUp.Code, followUp.Body.String())
	}
	assertLifecycleError(t, followUp.Body.Bytes(), "provider", "provider_binding_stale")
	if newCalls.Load() != 0 {
		t.Fatalf("stale continuation reached reloaded provider %d times", newCalls.Load())
	}
}

func TestNativeResponseWithoutLifecycleFallsBackToOwnedLocalSnapshot(t *testing.T) {
	var calls atomic.Int32
	var followUpPayload map[string]json.RawMessage
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		call := calls.Add(1)
		var payload map[string]json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Errorf("decode upstream request: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		responseID := "resp_local_fallback"
		if call > 1 {
			followUpPayload = payload
			responseID = "resp_local_followup"
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"`+responseID+`","object":"response","created_at":1,"status":"completed","model":"gpt-native","output":[],"output_text":"ok"}`)
	}))
	defer upstream.Close()

	providerConfigs := map[string]config.ProviderConfig{
		"native": {
			Type:         "openai",
			APIKey:       "provider-secret",
			BaseURL:      upstream.URL + "/v1",
			DefaultModel: "gpt-native",
		},
	}
	router, handler, cache := newNativeLifecycleRouterFromConfigs(t, providerConfigs)
	defer cache.Stop()

	created := performLifecycleRequest(t, router, http.MethodPost, "/v1/responses", []byte(`{"model":"native/gpt-native","input":"hello"}`))
	if created.Code != http.StatusOK {
		t.Fatalf("create status = %d, want 200; body=%s", created.Code, created.Body.String())
	}
	binding, lookup := handler.responseBindings.lookup("resp_local_fallback")
	if lookup != ownerLookupBound || !binding.LocalSnapshot || binding.Provider != "native" {
		t.Fatalf("fallback binding = %#v lookup=%v", binding, lookup)
	}
	if _, _, ok := handler.responsesState.getCompleted("resp_local_fallback"); !ok {
		t.Fatal("native response without lifecycle support was not retained locally")
	}

	followUp := performLifecycleRequest(t, router, http.MethodPost, "/v1/responses", []byte(`{"model":"lunargate/auto","previous_response_id":"resp_local_fallback","input":"again","store":false}`))
	if followUp.Code != http.StatusOK {
		t.Fatalf("follow-up status = %d, want 200; body=%s", followUp.Code, followUp.Body.String())
	}
	if calls.Load() != 2 {
		t.Fatalf("upstream calls = %d, want 2", calls.Load())
	}
	if _, exists := followUpPayload["previous_response_id"]; exists {
		t.Fatalf("locally resolved follow-up leaked previous_response_id upstream: %#v", followUpPayload)
	}
	if len(followUpPayload["input"]) == 0 {
		t.Fatalf("locally resolved follow-up omitted input: %#v", followUpPayload)
	}
}

func TestResponsesStateRetentionReportsFailure(t *testing.T) {
	store := newResponsesStateStore(0)
	store.maxEntries = 0
	if store.put("resp_disabled", map[string]json.RawMessage{"input": json.RawMessage(`"hello"`)}) {
		t.Fatal("disabled state store reported successful retention")
	}
	if store.putCompleted(
		"resp_disabled",
		map[string]json.RawMessage{"input": json.RawMessage(`"hello"`)},
		map[string]interface{}{"id": "resp_disabled", "object": "response"},
	) {
		t.Fatal("disabled completed store reported successful retention")
	}
	if _, ok := store.get("resp_disabled"); ok {
		t.Fatal("disabled state store retained payload")
	}
}

func TestOrphanedLocalResponseStateFailsClosed(t *testing.T) {
	tests := []struct {
		name       string
		method     string
		path       string
		body       []byte
		wantStatus int
		wantParam  string
		wantCode   string
	}{
		{
			name:       "continuation",
			method:     http.MethodPost,
			path:       "/v1/responses",
			body:       []byte(`{"model":"beta/gpt-native","previous_response_id":"resp_orphan","input":"again","store":false}`),
			wantStatus: http.StatusBadRequest,
			wantParam:  "previous_response_id",
			wantCode:   "previous_response_not_found",
		},
		{name: "retrieve", method: http.MethodGet, path: "/v1/responses/resp_orphan", wantStatus: http.StatusNotFound, wantParam: "response_id", wantCode: "response_not_found"},
		{name: "input items", method: http.MethodGet, path: "/v1/responses/resp_orphan/input_items", wantStatus: http.StatusNotFound, wantParam: "response_id", wantCode: "response_not_found"},
		{name: "delete", method: http.MethodDelete, path: "/v1/responses/resp_orphan", wantStatus: http.StatusNotFound, wantParam: "response_id", wantCode: "response_not_found"},
		{name: "cancel", method: http.MethodPost, path: "/v1/responses/resp_orphan/cancel", wantStatus: http.StatusNotFound, wantParam: "response_id", wantCode: "response_not_found"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var alphaCalls atomic.Int32
			var betaCalls atomic.Int32
			alpha := newResponseOwnerUpstream(t, &alphaCalls, "resp_alpha")
			defer alpha.Close()
			beta := newResponseOwnerUpstream(t, &betaCalls, "resp_beta")
			defer beta.Close()
			router, handler, cache := newNativeLifecycleRouterFromConfigs(t, map[string]config.ProviderConfig{
				"alpha": {
					Type:         "openai",
					APIKey:       "alpha-secret",
					BaseURL:      alpha.URL + "/v1",
					DefaultModel: "gpt-native",
					Capabilities: config.ProviderCapabilities{ResponsesLifecycle: true, ResponseCancellation: true},
				},
				"beta": {
					Type:         "openai",
					APIKey:       "beta-secret",
					BaseURL:      beta.URL + "/v1",
					DefaultModel: "gpt-native",
					Capabilities: config.ProviderCapabilities{ResponsesLifecycle: true, ResponseCancellation: true},
				},
			})
			defer cache.Stop()
			handler.responseBindings.maxEntries = 1
			if got := handler.retainLocalResponseSnapshot(
				"resp_orphan",
				responseOwnerTestHeaders("alpha"),
				responseOwnerTestIdentity(t, handler, "alpha"),
				map[string]json.RawMessage{"input": json.RawMessage(`"private alpha history"`)},
				map[string]interface{}{"id": "resp_orphan", "object": "response", "status": "completed", "output_text": "private alpha output"},
			); got != ownerClaimed {
				t.Fatalf("local owner claim = %v, want claimed", got)
			}
			if got := handler.responseBindings.claim("resp_evictor", mustResponseBinding(t, handler, "beta")); got != ownerClaimed {
				t.Fatalf("evictor claim = %v, want claimed", got)
			}
			if _, lookup := handler.responseBindings.lookup("resp_orphan"); lookup != ownerLookupMissing {
				t.Fatalf("orphan binding lookup = %v, want missing", lookup)
			}
			if _, _, ok := handler.responsesState.getCompleted("resp_orphan"); !ok {
				t.Fatal("test setup did not leave an orphaned local snapshot")
			}

			response := performLifecycleRequest(t, router, test.method, test.path, test.body)
			if response.Code != test.wantStatus {
				t.Fatalf("status = %d, want %d; body=%s", response.Code, test.wantStatus, response.Body.String())
			}
			assertLifecycleError(t, response.Body.Bytes(), test.wantParam, test.wantCode)
			if _, ok := handler.responsesState.get("resp_orphan"); ok {
				t.Fatal("orphaned local state survived request")
			}
			if alphaCalls.Load() != 0 || betaCalls.Load() != 0 {
				t.Fatalf("orphan request reached upstream: alpha=%d beta=%d", alphaCalls.Load(), betaCalls.Load())
			}
		})
	}
}

func newResponseOwnerUpstream(t *testing.T, calls *atomic.Int32, responseID string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		if !strings.HasPrefix(r.URL.Path, "/v1/responses") {
			t.Errorf("upstream path = %q, want Responses API path", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"`+responseID+`","object":"response","created_at":1,"status":"completed","model":"gpt-native","output":[],"output_text":"ok"}`)
	}))
}

func responseOwnerTestHeaders(provider string) http.Header {
	headers := make(http.Header)
	headers.Set("X-LunarGate-Provider", provider)
	headers.Set("X-LunarGate-Route", "responses")
	headers.Set("X-LunarGate-Model", provider+"/gpt-native")
	return headers
}

func responseOwnerTestIdentity(t *testing.T, handler *Handler, provider string) responseExecutionOwner {
	t.Helper()
	fingerprint, ok := handler.responseAccountFingerprint(provider)
	if !ok {
		t.Fatalf("provider %q has no account fingerprint", provider)
	}
	return responseExecutionOwner{
		Provider:            provider,
		Route:               "responses",
		Model:               provider + "/gpt-native",
		UpstreamRequestType: requestTypeResponses,
		AccountFingerprint:  fingerprint,
	}
}

func responseBindingErrorCode(err error) string {
	resolutionErr, _ := err.(*responseBindingResolutionError)
	if resolutionErr == nil {
		return ""
	}
	return resolutionErr.code
}
