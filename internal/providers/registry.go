package providers

import (
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/pkg/models"
	"github.com/rs/zerolog/log"
)

type registryEntry struct {
	translator   models.ProviderTranslator
	providerType string
}

// Registry manages all registered provider translators.
type Registry struct {
	mu        sync.RWMutex
	providers map[string]registryEntry
}

// NewRegistry creates a new provider registry from config.
func NewRegistry(providers map[string]config.ProviderConfig) *Registry {
	r := &Registry{
		providers: make(map[string]registryEntry),
	}

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
		r.providers[id] = registryEntry{translator: translator, providerType: providerType}
		log.Info().
			Str("provider", id).
			Str("provider_type", providerType).
			Str("default_model", translator.DefaultModel()).
			Msg("registered provider")
	}

	return r
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
	r.mu.RLock()
	defer r.mu.RUnlock()
	entry, ok := r.providers[name]
	if !ok {
		return nil, false
	}
	return entry.translator, true
}

// Type returns the resolved provider type for a configured provider ID.
func (r *Registry) Type(name string) (string, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	entry, ok := r.providers[name]
	if !ok {
		return "", false
	}
	return entry.providerType, true
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
