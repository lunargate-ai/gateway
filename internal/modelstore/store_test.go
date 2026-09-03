package modelstore

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestAllModelsSnapshotDoesNotFetchColdProvider(t *testing.T) {
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		requests.Add(1)
		writeOpenAIModels(t, w, "remote-only")
	}))
	defer server.Close()

	providerConfigs := fetchProviderConfigs(server.URL, "local-default")
	store := NewStore(providers.NewRegistry(providerConfigs), providerConfigs)

	got := store.AllModelsSnapshot()
	if requests.Load() != 0 {
		t.Fatalf("snapshot made %d provider requests, want zero", requests.Load())
	}
	if !hasModel(got, "custom/local-default") {
		t.Fatalf("snapshot models = %#v, want local default", got)
	}
	if hasModel(got, "custom/remote-only") {
		t.Fatalf("snapshot models = %#v, unexpectedly contain unfetched remote model", got)
	}
}

func TestCanceledFetchDoesNotCacheFallback(t *testing.T) {
	var requests atomic.Int32
	firstRequestStarted := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if requests.Add(1) == 1 {
			close(firstRequestStarted)
			<-r.Context().Done()
			return
		}
		writeOpenAIModels(t, w, "fresh-model")
	}))
	defer server.Close()

	providerConfigs := fetchProviderConfigs(server.URL, "local-default")
	store := NewStore(providers.NewRegistry(providerConfigs), providerConfigs)
	ctx, cancel := context.WithCancel(context.Background())
	firstResult := make(chan []models.ModelInfo, 1)
	go func() { firstResult <- store.AllModels(ctx) }()

	select {
	case <-firstRequestStarted:
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for first provider request")
	}
	cancel()
	select {
	case <-firstResult:
	case <-time.After(time.Second):
		t.Fatal("canceled model fetch did not return")
	}

	got := store.AllModels(context.Background())
	if requests.Load() != 2 {
		t.Fatalf("provider requests = %d, want retry after canceled fetch", requests.Load())
	}
	if !hasModel(got, "custom/fresh-model") {
		t.Fatalf("refreshed models = %#v, want fresh model", got)
	}
	if snapshot := store.AllModelsSnapshot(); !hasModel(snapshot, "custom/fresh-model") {
		t.Fatalf("cached snapshot models = %#v, want fresh model", snapshot)
	}
	if requests.Load() != 2 {
		t.Fatalf("cached snapshot made provider request; total = %d, want 2", requests.Load())
	}
}

func TestOldFetchCannotPoisonCacheAfterConfigReload(t *testing.T) {
	oldRequestStarted := make(chan struct{})
	releaseOldRequest := make(chan struct{})
	released := false
	defer func() {
		if !released {
			close(releaseOldRequest)
		}
	}()
	oldServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		close(oldRequestStarted)
		<-releaseOldRequest
		writeOpenAIModels(t, w, "old-remote")
	}))
	defer oldServer.Close()

	var newRequests atomic.Int32
	newServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		newRequests.Add(1)
		writeOpenAIModels(t, w, "new-remote")
	}))
	defer newServer.Close()

	oldConfigs := fetchProviderConfigs(oldServer.URL, "old-default")
	registry := providers.NewRegistry(oldConfigs)
	store := NewStore(registry, oldConfigs)
	oldResult := make(chan []models.ModelInfo, 1)
	go func() { oldResult <- store.AllModels(context.Background()) }()

	select {
	case <-oldRequestStarted:
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for old provider request")
	}
	newConfigs := fetchProviderConfigs(newServer.URL, "new-default")
	if !registry.UpdateProvidersConfig(newConfigs) {
		t.Fatal("new provider config was rejected")
	}
	store.UpdateProvidersConfig(newConfigs)
	close(releaseOldRequest)
	released = true

	select {
	case result := <-oldResult:
		if hasModel(result, "custom/old-remote") {
			t.Fatalf("stale fetch result leaked after config reload: %#v", result)
		}
	case <-time.After(time.Second):
		t.Fatal("old provider fetch did not return")
	}

	got := store.AllModels(context.Background())
	if newRequests.Load() != 1 {
		t.Fatalf("new provider requests = %d, want 1", newRequests.Load())
	}
	if !hasModel(got, "custom/new-remote") || hasModel(got, "custom/old-remote") {
		t.Fatalf("models after reload = %#v, want only new remote result", got)
	}
}

func fetchProviderConfigs(baseURL, defaultModel string) map[string]config.ProviderConfig {
	return map[string]config.ProviderConfig{
		"custom": {
			Type:         "openai",
			BaseURL:      baseURL,
			DefaultModel: defaultModel,
			Models: config.ProviderModelsConfig{
				Mode:  "fetch",
				Fetch: config.ModelsFetchConfig{TTL: time.Hour},
			},
		},
	}
}

func writeOpenAIModels(t *testing.T, w http.ResponseWriter, ids ...string) {
	t.Helper()
	data := make([]map[string]string, 0, len(ids))
	for _, id := range ids {
		data = append(data, map[string]string{"id": id})
	}
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(map[string]interface{}{"object": "list", "data": data}); err != nil {
		t.Errorf("encode models response: %v", err)
	}
}

func hasModel(items []models.ModelInfo, id string) bool {
	for _, item := range items {
		if item.ID == id {
			return true
		}
	}
	return false
}
