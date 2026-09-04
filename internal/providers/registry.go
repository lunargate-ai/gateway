package providers

import (
	"fmt"
	"reflect"
	"strings"
	"sync"
	"time"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/pkg/models"
	"github.com/rs/zerolog/log"
)

type registryEntry struct {
	translator        models.ProviderTranslator
	providerType      string
	capabilities      config.ProviderCapabilities
	circuitBreakerKey string
}

// ProviderSnapshot is an immutable view of one provider registry entry. It is
// intended to be retained for the whole upstream request lifecycle so a
// concurrent configuration reload cannot mix translators, provider types, or
// capabilities from different registry generations.
type ProviderSnapshot struct {
	Translator        models.ProviderTranslator
	ProviderType      string
	Capabilities      config.ProviderCapabilities
	circuitBreakerKey string
}

// CircuitBreakerKey returns an opaque, request-internal identity key for this
// provider generation. It must never be logged or returned to API clients.
func (s ProviderSnapshot) CircuitBreakerKey() string {
	return s.circuitBreakerKey
}

// Registry manages all registered provider translators.
type Registry struct {
	mu        sync.RWMutex
	providers map[string]registryEntry
	configs   map[string]config.ProviderConfig
}

// NewRegistry creates a new provider registry from config.
func NewRegistry(providers map[string]config.ProviderConfig) *Registry {
	ownedConfigs := cloneProviderConfigs(providers)
	r := &Registry{
		providers: make(map[string]registryEntry),
		configs:   ownedConfigs,
	}
	r.providers = buildRegistryEntries(ownedConfigs)

	return r
}

// UpdateProvidersConfig hot-reloads provider translators in-place.
// It preserves the current registry and reports false when the effective input
// is unchanged or the new config would leave the gateway without valid providers.
func (r *Registry) UpdateProvidersConfig(providers map[string]config.ProviderConfig) bool {
	nextConfigs := cloneProviderConfigs(providers)
	r.mu.RLock()
	unchanged := reflect.DeepEqual(r.configs, nextConfigs)
	r.mu.RUnlock()
	if unchanged {
		return false
	}

	next := buildRegistryEntries(nextConfigs)
	if len(next) == 0 {
		log.Error().Msg("provider config reload produced zero valid providers; keeping existing registry")
		return false
	}

	r.mu.Lock()
	if reflect.DeepEqual(r.configs, nextConfigs) {
		r.mu.Unlock()
		return false
	}
	r.providers = next
	r.configs = nextConfigs
	r.mu.Unlock()

	log.Info().Int("providers", len(next)).Msg("provider registry updated")
	return true
}

func cloneProviderConfigs(providers map[string]config.ProviderConfig) map[string]config.ProviderConfig {
	cloned := make(map[string]config.ProviderConfig, len(providers))
	for id, provider := range providers {
		if provider.Temperature != nil {
			value := *provider.Temperature
			provider.Temperature = &value
		}
		if provider.TopP != nil {
			value := *provider.TopP
			provider.TopP = &value
		}
		if provider.TopK != nil {
			value := *provider.TopK
			provider.TopK = &value
		}
		if len(provider.Extra) > 0 {
			extra := provider.Extra
			provider.Extra = make(map[string]string, len(extra))
			for key, value := range extra {
				provider.Extra[key] = value
			}
		} else {
			provider.Extra = nil
		}
		provider.Models.Static = append([]string(nil), provider.Models.Static...)
		provider.Capabilities = cloneProviderCapabilities(provider.Capabilities)
		cloned[id] = provider
	}
	return cloned
}

func buildRegistryEntries(providers map[string]config.ProviderConfig) map[string]registryEntry {
	entries := make(map[string]registryEntry, len(providers))
	for id, cfg := range providers {
		providerType, err := resolveProviderType(id, cfg)
		if err != nil {
			log.Warn().Err(err).Str("provider", id).Msg("invalid provider config, skipping")
			continue
		}

		translator, err := createTranslator(providerType, cfg)
		if err != nil {
			log.Warn().Err(err).Str("provider", id).Str("provider_type", providerType).Msg("failed to create provider translator, skipping")
			continue
		}
		entries[id] = registryEntry{
			translator:        translator,
			providerType:      providerType,
			capabilities:      cloneProviderCapabilities(cfg.Capabilities),
			circuitBreakerKey: providerCircuitBreakerKey(id, providerType, cfg, translator),
		}
		log.Info().
			Str("provider", id).
			Str("provider_type", providerType).
			Str("default_model", translator.DefaultModel()).
			Msg("registered provider")
	}
	return entries
}

func cloneProviderCapabilities(capabilities config.ProviderCapabilities) config.ProviderCapabilities {
	capabilities.HostedTools = append([]string(nil), capabilities.HostedTools...)
	capabilities.ReasoningEffortLevels = append([]string(nil), capabilities.ReasoningEffortLevels...)
	return capabilities
}

func resolveProviderType(providerID string, cfg config.ProviderConfig) (string, error) {
	if t := strings.TrimSpace(cfg.Type); t != "" {
		return strings.ToLower(t), nil
	}

	// If type isn't explicitly set, only allow built-in provider IDs.
	switch providerID {
	case "openai", "anthropic", "ollama":
		return providerID, nil
	default:
		return "", fmt.Errorf("provider type is required for provider_id=%q", providerID)
	}
}

func createTranslator(providerType string, cfg config.ProviderConfig) (models.ProviderTranslator, error) {
	switch providerType {
	case "openai":
		return NewOpenAITranslator(cfg), nil
	case "anthropic":
		return NewAnthropicTranslator(cfg), nil
	case "ollama":
		return NewOllamaTranslator(cfg), nil
	default:
		return nil, fmt.Errorf("unknown provider type: %s", providerType)
	}
}

// Get returns a provider translator by name.
func (r *Registry) Get(name string) (models.ProviderTranslator, bool) {
	snapshot, ok := r.Snapshot(name)
	if !ok {
		return nil, false
	}
	return snapshot.Translator, true
}

// Type returns the resolved provider type for a configured provider ID.
func (r *Registry) Type(name string) (string, bool) {
	snapshot, ok := r.Snapshot(name)
	if !ok {
		return "", false
	}
	return snapshot.ProviderType, true
}

// Capabilities returns the explicitly configured optional API contracts for a
// provider. The returned value does not alias registry-owned slices.
func (r *Registry) Capabilities(name string) (config.ProviderCapabilities, bool) {
	snapshot, ok := r.Snapshot(name)
	if !ok {
		return config.ProviderCapabilities{}, false
	}
	return snapshot.Capabilities, true
}

// Snapshot returns a translator, provider type, and capabilities from one
// registry generation. The returned capabilities do not alias registry-owned
// slices.
func (r *Registry) Snapshot(name string) (ProviderSnapshot, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	entry, ok := r.providers[name]
	if !ok {
		return ProviderSnapshot{}, false
	}
	return ProviderSnapshot{
		Translator:        entry.translator,
		ProviderType:      entry.providerType,
		Capabilities:      cloneProviderCapabilities(entry.capabilities),
		circuitBreakerKey: entry.circuitBreakerKey,
	}, true
}

// List returns all registered provider names.
func (r *Registry) List() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	names := make([]string, 0, len(r.providers))
	for name := range r.providers {
		names = append(names, name)
	}
	return names
}

// ConfigSnapshot returns an owned copy of the provider configuration used to
// build this registry generation.
func (r *Registry) ConfigSnapshot() map[string]config.ProviderConfig {
	if r == nil {
		return map[string]config.ProviderConfig{}
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	return cloneProviderConfigs(r.configs)
}

// AllModels returns all models across all registered providers.
func (r *Registry) AllModels() []models.ModelInfo {
	r.mu.RLock()
	defer r.mu.RUnlock()
	seen := make(map[string]struct{}, 64)
	all := make([]models.ModelInfo, 0, 64)
	for _, entry := range r.providers {
		for _, m := range entry.translator.Models() {
			id := strings.TrimSpace(m.ID)
			if id == "" {
				continue
			}
			if _, ok := seen[id]; ok {
				continue
			}
			seen[id] = struct{}{}
			all = append(all, m)
		}

		if dm := strings.TrimSpace(entry.translator.DefaultModel()); dm != "" {
			if _, ok := seen[dm]; !ok {
				seen[dm] = struct{}{}
				all = append(all, models.ModelInfo{ID: dm, Object: "model", Created: time.Now().Unix(), OwnedBy: entry.translator.Name()})
			}
		}
	}
	return all
}
