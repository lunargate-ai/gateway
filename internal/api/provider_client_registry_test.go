package api

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/lunargate-ai/gateway/internal/config"
)

func TestProviderClientRegistrySnapshotDoesNotMixReloadGenerations(t *testing.T) {
	const provider = "native"
	first := map[string]config.ProviderConfig{
		provider: {BaseURL: "https://first.example/v1", APIKey: "first-key", Timeout: 11 * time.Millisecond},
	}
	second := map[string]config.ProviderConfig{
		provider: {BaseURL: "https://second.example/v1", APIKey: "second-key", Timeout: 22 * time.Millisecond},
	}
	registry := newProviderClientRegistry(first)

	var invalid atomic.Bool
	var workers sync.WaitGroup
	for worker := 0; worker < 8; worker++ {
		workers.Add(1)
		go func() {
			defer workers.Done()
			for iteration := 0; iteration < 2000; iteration++ {
				clientCfg, providerCfg, ok := registry.Snapshot(provider)
				if !ok {
					invalid.Store(true)
					return
				}
				switch providerCfg.BaseURL {
				case "https://first.example/v1":
					if providerCfg.APIKey != "first-key" || clientCfg.timeout != 11*time.Millisecond {
						invalid.Store(true)
						return
					}
				case "https://second.example/v1":
					if providerCfg.APIKey != "second-key" || clientCfg.timeout != 22*time.Millisecond {
						invalid.Store(true)
						return
					}
				default:
					invalid.Store(true)
					return
				}
			}
		}()
	}
	for iteration := 0; iteration < 2000; iteration++ {
		if iteration%2 == 0 {
			registry.Update(second)
		} else {
			registry.Update(first)
		}
	}
	workers.Wait()
	if invalid.Load() {
		t.Fatal("provider client snapshot mixed HTTP client and credential generations")
	}
}
