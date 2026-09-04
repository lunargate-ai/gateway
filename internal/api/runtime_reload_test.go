package api

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/health"
	"github.com/lunargate-ai/gateway/internal/middleware"
	"github.com/lunargate-ai/gateway/internal/modelselect"
	"github.com/lunargate-ai/gateway/internal/modelstore"
	"github.com/lunargate-ai/gateway/internal/observability"
	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/internal/resilience"
	"github.com/lunargate-ai/gateway/internal/routing"
	"github.com/lunargate-ai/gateway/internal/streaming"
	"github.com/prometheus/client_golang/prometheus"
)

type runtimeReloadUpstreamObservation struct {
	path  string
	auth  string
	model string
}

func TestRuntimeReloadPinsChatAndEmbeddingsToOneGeneration(t *testing.T) {
	tests := []struct {
		name       string
		path       string
		request    func(model string) string
		response   func(label string, model string) string
		assertBody func(*testing.T, string, string)
	}{
		{
			name: "chat completions",
			path: "/v1/chat/completions",
			request: func(model string) string {
				return fmt.Sprintf(`{"model":%q,"messages":[{"role":"user","content":"hello"}]}`, "shared/"+model)
			},
			response: func(label string, model string) string {
				return fmt.Sprintf(`{"id":"chat-%s","object":"chat.completion","created":1,"model":%q,"choices":[{"index":0,"message":{"role":"assistant","content":%q},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`, label, model, label)
			},
			assertBody: func(t *testing.T, body string, label string) {
				t.Helper()
				if !strings.Contains(body, `"content":"`+label+`"`) {
					t.Fatalf("response body = %s, want content %q", body, label)
				}
			},
		},
		{
			name: "embeddings",
			path: "/v1/embeddings",
			request: func(model string) string {
				return fmt.Sprintf(`{"model":%q,"input":"hello"}`, "shared/"+model)
			},
			response: func(label string, model string) string {
				value := 0.1
				if label == "new" {
					value = 0.9
				}
				return fmt.Sprintf(`{"object":"list","data":[{"object":"embedding","embedding":[%g],"index":0}],"model":%q,"usage":{"prompt_tokens":1,"total_tokens":1}}`, value, model)
			},
			assertBody: func(t *testing.T, body string, label string) {
				t.Helper()
				want := `"embedding":[0.1]`
				if label == "new" {
					want = `"embedding":[0.9]`
				}
				if !strings.Contains(body, want) {
					t.Fatalf("response body = %s, want %s", body, want)
				}
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			oldStarted := make(chan runtimeReloadUpstreamObservation, 1)
			releaseOld := make(chan struct{})
			var releaseOldOnce sync.Once
			releaseOldRequest := func() { releaseOldOnce.Do(func() { close(releaseOld) }) }
			defer releaseOldRequest()
			oldUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				observation := observeRuntimeReloadUpstream(r)
				oldStarted <- observation
				<-releaseOld
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, test.response("old", "model-old"))
			}))
			defer oldUpstream.Close()

			newSeen := make(chan runtimeReloadUpstreamObservation, 1)
			newUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				newSeen <- observeRuntimeReloadUpstream(r)
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, test.response("new", "model-new"))
			}))
			defer newUpstream.Close()

			oldProviders := runtimeReloadProviderConfig(oldUpstream.URL, "key-old", "model-old")
			oldRouting := runtimeReloadRoutingConfig(test.path, "route-old", "model-old", "")
			handler, gateway, cache := newRuntimeReloadTestHandler(oldProviders, oldRouting, config.ModelSelectionConfig{}, config.CacheConfig{})
			defer cache.Stop()

			oldRecorder := httptest.NewRecorder()
			oldDone := make(chan struct{})
			go func() {
				defer close(oldDone)
				gateway.ServeHTTP(oldRecorder, runtimeReloadRequest(test.path, test.request("model-old")))
			}()

			oldObservation := receiveRuntimeReloadObservation(t, oldStarted, "old request")
			assertRuntimeReloadObservation(t, oldObservation, test.path, "key-old", "model-old")

			newProviders := runtimeReloadProviderConfig(newUpstream.URL, "key-new", "model-new")
			newRouting := runtimeReloadRoutingConfig(test.path, "route-new", "model-new", "")
			changed, err := handler.UpdateRuntime(newProviders, newRouting, config.ModelSelectionConfig{})
			if err != nil || !changed {
				releaseOldRequest()
				t.Fatalf("runtime update = changed %t, err %v", changed, err)
			}

			newRecorder := httptest.NewRecorder()
			gateway.ServeHTTP(newRecorder, runtimeReloadRequest(test.path, test.request("model-new")))
			newObservation := receiveRuntimeReloadObservation(t, newSeen, "new request")
			assertRuntimeReloadObservation(t, newObservation, test.path, "key-new", "model-new")
			if newRecorder.Code != http.StatusOK {
				releaseOldRequest()
				t.Fatalf("new response status = %d, body=%s", newRecorder.Code, newRecorder.Body.String())
			}
			if got := newRecorder.Header().Get("X-LunarGate-Route"); got != "route-new" {
				releaseOldRequest()
				t.Fatalf("new route header = %q, want route-new", got)
			}
			test.assertBody(t, newRecorder.Body.String(), "new")

			releaseOldRequest()
			select {
			case <-oldDone:
			case <-time.After(5 * time.Second):
				t.Fatal("old request did not finish")
			}
			if oldRecorder.Code != http.StatusOK {
				t.Fatalf("old response status = %d, body=%s", oldRecorder.Code, oldRecorder.Body.String())
			}
			if got := oldRecorder.Header().Get("X-LunarGate-Route"); got != "route-old" {
				t.Fatalf("old route header = %q, want route-old", got)
			}
			test.assertBody(t, oldRecorder.Body.String(), "old")
		})
	}
}

func TestRuntimeReloadProviderNamespacePreventsCachePoisoning(t *testing.T) {
	releaseOld := make(chan struct{})
	var releaseOldOnce sync.Once
	releaseOldRequest := func() { releaseOldOnce.Do(func() { close(releaseOld) }) }
	defer releaseOldRequest()
	oldStarted := make(chan struct{}, 1)
	oldUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		oldStarted <- struct{}{}
		<-releaseOld
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, runtimeReloadChatResponse("old", "shared-model"))
	}))
	defer oldUpstream.Close()

	var newCalls atomic.Int32
	newUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		newCalls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, runtimeReloadChatResponse("new", "shared-model"))
	}))
	defer newUpstream.Close()

	cacheConfig := config.CacheConfig{
		Enabled:       true,
		TTL:           time.Minute,
		MaxSize:       16,
		MaxEntryBytes: 1 << 20,
		MaxBytes:      4 << 20,
	}
	routingConfig := runtimeReloadRoutingConfig("/v1/chat/completions", "chat", "shared-model", "")
	oldProviders := runtimeReloadProviderConfig(oldUpstream.URL, "key-old", "shared-model")
	handler, gateway, cache := newRuntimeReloadTestHandler(oldProviders, routingConfig, config.ModelSelectionConfig{}, cacheConfig)
	defer cache.Stop()

	requestBody := `{"messages":[{"role":"user","content":"same request"}]}`
	oldRecorder := httptest.NewRecorder()
	oldDone := make(chan struct{})
	go func() {
		defer close(oldDone)
		gateway.ServeHTTP(oldRecorder, runtimeReloadRequest("/v1/chat/completions", requestBody))
	}()
	select {
	case <-oldStarted:
	case <-time.After(5 * time.Second):
		releaseOldRequest()
		t.Fatal("old cache-miss request did not reach upstream")
	}

	newProviders := runtimeReloadProviderConfig(newUpstream.URL, "key-new", "shared-model")
	if changed, err := handler.UpdateRuntime(newProviders, routingConfig, config.ModelSelectionConfig{}); err != nil || !changed {
		releaseOldRequest()
		t.Fatalf("runtime update = changed %t, err %v", changed, err)
	}

	newRecorder := httptest.NewRecorder()
	gateway.ServeHTTP(newRecorder, runtimeReloadRequest("/v1/chat/completions", requestBody))
	if newRecorder.Code != http.StatusOK || !strings.Contains(newRecorder.Body.String(), `"content":"new"`) {
		releaseOldRequest()
		t.Fatalf("new response = status %d, body=%s", newRecorder.Code, newRecorder.Body.String())
	}
	if got := newCalls.Load(); got != 1 {
		releaseOldRequest()
		t.Fatalf("new upstream calls = %d, want 1", got)
	}

	releaseOldRequest()
	select {
	case <-oldDone:
	case <-time.After(5 * time.Second):
		t.Fatal("old request did not finish")
	}

	cachedRecorder := httptest.NewRecorder()
	gateway.ServeHTTP(cachedRecorder, runtimeReloadRequest("/v1/chat/completions", requestBody))
	if cachedRecorder.Code != http.StatusOK || cachedRecorder.Header().Get("X-LunarGate-Cache-Status") != "HIT" {
		t.Fatalf("cached response = status %d cache=%q body=%s", cachedRecorder.Code, cachedRecorder.Header().Get("X-LunarGate-Cache-Status"), cachedRecorder.Body.String())
	}
	if !strings.Contains(cachedRecorder.Body.String(), `"content":"new"`) {
		t.Fatalf("old generation poisoned new cache namespace: %s", cachedRecorder.Body.String())
	}
	if got := newCalls.Load(); got != 1 {
		t.Fatalf("new upstream calls after cache hit = %d, want 1", got)
	}
}

func TestRuntimeReloadReusesUnaffectedComponentsAndRefreshesCallbacks(t *testing.T) {
	providersConfig := runtimeReloadProviderConfig("http://provider-old.invalid", "key-old", "model-old")
	routingConfig := runtimeReloadRoutingConfig("/v1/chat/completions", "route-old", "model-old", "")
	selectionConfig := config.ModelSelectionConfig{Enabled: false}
	handler, _, cache := newRuntimeReloadTestHandler(providersConfig, routingConfig, selectionConfig, config.CacheConfig{})
	defer cache.Stop()

	initial := handler.currentRuntimeGeneration()
	if changed, err := handler.UpdateRuntime(providersConfig, routingConfig, selectionConfig); err != nil || changed {
		t.Fatalf("semantic no-op = changed %t, err %v", changed, err)
	}
	if got := handler.currentRuntimeGeneration(); got != initial {
		t.Fatal("semantic no-op replaced runtime generation")
	}

	changedRouting := runtimeReloadRoutingConfig("/v1/chat/completions", "route-new", "model-old", "")
	if changed, err := handler.UpdateRuntime(providersConfig, changedRouting, selectionConfig); err != nil || !changed {
		t.Fatalf("route update = changed %t, err %v", changed, err)
	}
	afterRoute := handler.currentRuntimeGeneration()
	if afterRoute.router == initial.router || afterRoute.registry != initial.registry || afterRoute.store != initial.store || afterRoute.providerClients != initial.providerClients || afterRoute.selector != initial.selector {
		t.Fatal("route-only reload did not replace exactly the routing component")
	}
	if got := handler.RuntimeRouteNames(); !reflect.DeepEqual(got, []string{"route-new"}) {
		t.Fatalf("runtime route callback = %v", got)
	}

	changedSelection := config.ModelSelectionConfig{Enabled: true, OverrideUserModel: true}
	if changed, err := handler.UpdateRuntime(providersConfig, changedRouting, changedSelection); err != nil || !changed {
		t.Fatalf("selection update = changed %t, err %v", changed, err)
	}
	afterSelection := handler.currentRuntimeGeneration()
	if afterSelection.selector == afterRoute.selector || afterSelection.router != afterRoute.router || afterSelection.registry != afterRoute.registry || afterSelection.store != afterRoute.store || afterSelection.providerClients != afterRoute.providerClients {
		t.Fatal("selector-only reload did not replace exactly the selector")
	}

	changedProviders := runtimeReloadProviderConfig("http://provider-new.invalid", "key-new", "model-new")
	handler.UpdateProviderConfigs(changedProviders)
	afterProvider := handler.currentRuntimeGeneration()
	if afterProvider == afterSelection {
		t.Fatal("provider compatibility update did not publish a new generation")
	}
	if afterProvider.registry == afterSelection.registry || afterProvider.store == afterSelection.store || afterProvider.providerClients == afterSelection.providerClients {
		t.Fatal("provider reload reused provider-owned components")
	}
	if afterProvider.router != afterSelection.router || afterProvider.selector != afterSelection.selector {
		t.Fatal("provider-only reload replaced routing or selector")
	}
	if got := handler.RuntimeModelSnapshotIDs(); !reflect.DeepEqual(got, []string{"shared/model-new"}) {
		t.Fatalf("runtime model snapshot callback = %v", got)
	}
	if got := handler.RuntimeModelIDs(context.Background()); !reflect.DeepEqual(got, []string{"shared/model-new"}) {
		t.Fatalf("runtime model callback = %v", got)
	}
	if changed, err := handler.UpdateRuntime(
		map[string]config.ProviderConfig{"invalid": {Type: "unknown"}},
		changedRouting,
		changedSelection,
	); err == nil || changed {
		t.Fatalf("invalid provider update = changed %t, err %v", changed, err)
	}
	if got := handler.currentRuntimeGeneration(); got != afterProvider {
		t.Fatal("failed provider build replaced the active generation")
	}
}

func TestResponsesWebSocketPinsRuntimeAtConnectionEntry(t *testing.T) {
	var oldCalls atomic.Int32
	oldModels := make(chan string, 1)
	oldUpstream := httptest.NewServer(runtimeReloadStreamingUpstream(&oldCalls, oldModels, "old"))
	defer oldUpstream.Close()
	var newCalls atomic.Int32
	newModels := make(chan string, 1)
	newUpstream := httptest.NewServer(runtimeReloadStreamingUpstream(&newCalls, newModels, "new"))
	defer newUpstream.Close()

	oldProviders := runtimeReloadProviderConfig(oldUpstream.URL, "key-old", "model-old")
	oldRouting := runtimeReloadRoutingConfig("/v1/responses", "responses-old", "model-old", requestTypeChatCompletions)
	handler, gateway, cache := newRuntimeReloadTestHandler(oldProviders, oldRouting, config.ModelSelectionConfig{}, config.CacheConfig{})
	defer cache.Stop()
	server := httptest.NewServer(gateway)
	defer server.Close()

	oldConnection := mustDialResponsesWebSocket(t, server.URL)
	defer oldConnection.Close()

	newProviders := runtimeReloadProviderConfig(newUpstream.URL, "key-new", "model-new")
	newRouting := runtimeReloadRoutingConfig("/v1/responses", "responses-new", "model-new", requestTypeChatCompletions)
	if changed, err := handler.UpdateRuntime(newProviders, newRouting, config.ModelSelectionConfig{}); err != nil || !changed {
		t.Fatalf("runtime update = changed %t, err %v", changed, err)
	}

	sendResponsesWebSocketJSON(t, oldConnection, map[string]interface{}{
		"type":  "response.create",
		"input": "old connection",
	})
	oldEvents := readResponsesWebSocketEventsUntilTerminal(t, oldConnection)
	if !hasResponsesWebSocketEventType(oldEvents, "response.completed") {
		t.Fatalf("old connection events = %v", eventTypes(oldEvents))
	}
	if got := receiveRuntimeReloadModel(t, oldModels, "old websocket"); got != "model-old" {
		t.Fatalf("old websocket model = %q", got)
	}
	if oldCalls.Load() != 1 || newCalls.Load() != 0 {
		t.Fatalf("calls after old connection: old=%d new=%d", oldCalls.Load(), newCalls.Load())
	}

	newConnection := mustDialResponsesWebSocket(t, server.URL)
	defer newConnection.Close()
	sendResponsesWebSocketJSON(t, newConnection, map[string]interface{}{
		"type":  "response.create",
		"input": "new connection",
	})
	newEvents := readResponsesWebSocketEventsUntilTerminal(t, newConnection)
	if !hasResponsesWebSocketEventType(newEvents, "response.completed") {
		t.Fatalf("new connection events = %v", eventTypes(newEvents))
	}
	if got := receiveRuntimeReloadModel(t, newModels, "new websocket"); got != "model-new" {
		t.Fatalf("new websocket model = %q", got)
	}
	if oldCalls.Load() != 1 || newCalls.Load() != 1 {
		t.Fatalf("calls after new connection: old=%d new=%d", oldCalls.Load(), newCalls.Load())
	}
}

func newRuntimeReloadTestHandler(
	providerConfig map[string]config.ProviderConfig,
	routingConfig config.RoutingConfig,
	selectionConfig config.ModelSelectionConfig,
	cacheConfig config.CacheConfig,
) (*Handler, http.Handler, *middleware.Cache) {
	registry := providers.NewRegistry(providerConfig)
	router := routing.NewEngine(routingConfig)
	selector := modelselect.NewEngine(selectionConfig)
	cache := middleware.NewCache(cacheConfig)
	handler := NewHandler(
		registry,
		router,
		resilience.NewFallbackExecutor(
			resilience.NewRetrier(config.RetryConfig{Enabled: false}),
			resilience.NewCircuitBreakerManager(),
		),
		cache,
		streaming.NewHandler(),
		observability.NewMetricsWithRegisterer(prometheus.NewRegistry()),
		nil,
		selector,
		modelstore.NewStore(registry, providerConfig),
	)
	return handler, NewRouter(handler, nil, nil, health.NewChecker("test")), cache
}

func runtimeReloadProviderConfig(baseURL string, apiKey string, model string) map[string]config.ProviderConfig {
	return map[string]config.ProviderConfig{
		"shared": {
			Type:         "openai",
			APIKey:       apiKey,
			BaseURL:      strings.TrimRight(baseURL, "/") + "/v1",
			DefaultModel: model,
			Models: config.ProviderModelsConfig{
				Mode:   "static",
				Static: []string{model},
			},
		},
	}
}

func runtimeReloadRoutingConfig(path string, route string, model string, upstreamRequestType string) config.RoutingConfig {
	return config.RoutingConfig{
		DefaultStrategy: "weighted",
		Routes: []config.RouteConfig{{
			Name:  route,
			Match: config.MatchConfig{Path: path},
			Targets: []config.TargetConfig{{
				Provider:            "shared",
				Model:               model,
				Weight:              1,
				UpstreamRequestType: upstreamRequestType,
			}},
		}},
	}
}

func runtimeReloadRequest(path string, body string) *http.Request {
	request := httptest.NewRequest(http.MethodPost, path, bytes.NewBufferString(body))
	request.Header.Set("Content-Type", "application/json")
	return request
}

func observeRuntimeReloadUpstream(request *http.Request) runtimeReloadUpstreamObservation {
	var payload map[string]interface{}
	_ = json.NewDecoder(request.Body).Decode(&payload)
	model, _ := payload["model"].(string)
	return runtimeReloadUpstreamObservation{
		path:  request.URL.Path,
		auth:  request.Header.Get("Authorization"),
		model: model,
	}
}

func receiveRuntimeReloadObservation(t *testing.T, observations <-chan runtimeReloadUpstreamObservation, label string) runtimeReloadUpstreamObservation {
	t.Helper()
	select {
	case observation := <-observations:
		return observation
	case <-time.After(5 * time.Second):
		t.Fatalf("timed out waiting for %s", label)
		return runtimeReloadUpstreamObservation{}
	}
}

func assertRuntimeReloadObservation(t *testing.T, observation runtimeReloadUpstreamObservation, gatewayPath string, apiKey string, model string) {
	t.Helper()
	wantPath := "/v1/chat/completions"
	if gatewayPath == "/v1/embeddings" {
		wantPath = "/v1/embeddings"
	}
	if observation.path != wantPath || observation.auth != "Bearer "+apiKey || observation.model != model {
		t.Fatalf("upstream observation = %#v, want path=%q auth=%q model=%q", observation, wantPath, "Bearer "+apiKey, model)
	}
}

func runtimeReloadChatResponse(label string, model string) string {
	return fmt.Sprintf(`{"id":"chat-%s","object":"chat.completion","created":1,"model":%q,"choices":[{"index":0,"message":{"role":"assistant","content":%q},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`, label, model, label)
}

func runtimeReloadStreamingUpstream(calls *atomic.Int32, models chan<- string, label string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		calls.Add(1)
		observation := observeRuntimeReloadUpstream(request)
		models <- observation.model
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, fmt.Sprintf(
			"data: {\"id\":\"chat-%s\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":%q,\"choices\":[{\"index\":0,\"delta\":{\"content\":%q},\"finish_reason\":null}]}\n\n",
			label,
			observation.model,
			label,
		))
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	})
}

func receiveRuntimeReloadModel(t *testing.T, models <-chan string, label string) string {
	t.Helper()
	select {
	case model := <-models:
		return model
	case <-time.After(5 * time.Second):
		t.Fatalf("timed out waiting for %s", label)
		return ""
	}
}
