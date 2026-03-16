package modelstore

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/modelid"
	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/pkg/models"
	"github.com/rs/zerolog/log"
)

type cacheEntry struct {
	models    []string
	expiresAt time.Time
}

type Store struct {
	registry *providers.Registry
	client   *http.Client

	cfg atomic.Value

	mu    sync.RWMutex
	cache map[string]cacheEntry
}

func NewStore(reg *providers.Registry, providersCfg map[string]config.ProviderConfig) *Store {
	s := &Store{
		registry: reg,
		client: &http.Client{Timeout: 15 * time.Second},
		cache: make(map[string]cacheEntry),
	}
	s.UpdateProvidersConfig(providersCfg)
	return s
}

func (s *Store) UpdateProvidersConfig(cfg map[string]config.ProviderConfig) {
	copyMap := make(map[string]config.ProviderConfig, len(cfg))
	for k, v := range cfg {
		copyMap[k] = v
	}
	s.cfg.Store(copyMap)

	s.mu.Lock()
	s.cache = make(map[string]cacheEntry)
	s.mu.Unlock()
}

func (s *Store) AllModels(ctx context.Context) []models.ModelInfo {
	cfgAny := s.cfg.Load()
	providersCfg, _ := cfgAny.(map[string]config.ProviderConfig)

	seen := make(map[string]struct{}, 128)
	out := make([]models.ModelInfo, 0, 128)

	providerIDs := s.registry.List()
	sort.Strings(providerIDs)

	for _, providerID := range providerIDs {
		pcfg := providersCfg[providerID]
		ids := s.modelsForProvider(ctx, providerID, pcfg)
		for _, raw := range ids {
			m := strings.TrimSpace(raw)
			if m == "" {
				continue
			}
			canonical := modelid.BuildCanonical(providerID, m)
			if _, ok := seen[canonical]; ok {
				continue
			}
			seen[canonical] = struct{}{}
			out = append(out, models.ModelInfo{ID: canonical, Object: "model", Created: time.Now().Unix(), OwnedBy: providerID})
		}
	}

	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

func (s *Store) modelsForProvider(ctx context.Context, providerID string, pcfg config.ProviderConfig) []string {
	mode := strings.ToLower(strings.TrimSpace(pcfg.Models.Mode))
	if mode == "" {
		mode = "translator"
	}

	switch mode {
	case "static":
		modelsList := make([]string, 0, len(pcfg.Models.Static)+1)
		for _, m := range pcfg.Models.Static {
			mm := strings.TrimSpace(m)
			if mm != "" {
				modelsList = append(modelsList, mm)
			}
		}
		if dm := strings.TrimSpace(pcfg.DefaultModel); dm != "" {
			modelsList = append(modelsList, dm)
		}
		return uniqueStrings(modelsList)

	case "fetch":
		ttl := pcfg.Models.Fetch.TTL
		if ttl <= 0 {
			ttl = 10 * time.Minute
		}

		s.mu.RLock()
		ce, ok := s.cache[providerID]
		s.mu.RUnlock()
		if ok && time.Now().Before(ce.expiresAt) {
			return ce.models
		}

		modelsList, err := s.fetchModels(ctx, providerID, pcfg)
		if err != nil {
			log.Warn().Err(err).Str("provider", providerID).Msg("failed to fetch models")
			modelsList = s.modelsFromTranslator(providerID)
			if len(modelsList) == 0 {
				if dm := strings.TrimSpace(pcfg.DefaultModel); dm != "" {
					modelsList = append(modelsList, dm)
				}
			}
		}
		modelsList = uniqueStrings(modelsList)

		s.mu.Lock()
		s.cache[providerID] = cacheEntry{models: modelsList, expiresAt: time.Now().Add(ttl)}
		s.mu.Unlock()
		return modelsList

	case "translator":
		fallthrough
	default:
		modelsList := s.modelsFromTranslator(providerID)
		if dm := strings.TrimSpace(pcfg.DefaultModel); dm != "" {
			modelsList = append(modelsList, dm)
		}
		return uniqueStrings(modelsList)
	}
}

func (s *Store) modelsFromTranslator(providerID string) []string {
	translator, ok := s.registry.Get(providerID)
	if !ok || translator == nil {
		return nil
	}

	out := make([]string, 0, 16)
	for _, mi := range translator.Models() {
		if id := strings.TrimSpace(mi.ID); id != "" {
			out = append(out, id)
		}
	}
	if dm := strings.TrimSpace(translator.DefaultModel()); dm != "" {
		out = append(out, dm)
	}
	return out
}

type openAIModelsList struct {
	Object string `json:"object"`
	Data   []struct {
		ID string `json:"id"`
	} `json:"data"`
}

type ollamaTagsResponse struct {
	Models []struct {
		Name string `json:"name"`
	} `json:"models"`
}

func (s *Store) fetchModels(ctx context.Context, providerID string, pcfg config.ProviderConfig) ([]string, error) {
	providerType, _ := s.registry.Type(providerID)
	providerType = strings.ToLower(strings.TrimSpace(providerType))

	baseURL := strings.TrimRight(strings.TrimSpace(pcfg.BaseURL), "/")
	if baseURL == "" {
		return nil, fmt.Errorf("provider base_url is empty")
	}

	if providerType == "ollama" {
		url := baseURL + "/api/tags"
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if err != nil {
			return nil, fmt.Errorf("failed to create ollama tags request: %w", err)
		}
		resp, err := s.client.Do(req)
		if err != nil {
			return nil, fmt.Errorf("failed to call ollama tags: %w", err)
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf("ollama tags returned status=%d", resp.StatusCode)
		}
		var tr ollamaTagsResponse
		if err := json.NewDecoder(resp.Body).Decode(&tr); err != nil {
			return nil, fmt.Errorf("failed to decode ollama tags response: %w", err)
		}
		out := make([]string, 0, len(tr.Models))
		for _, m := range tr.Models {
			if ss := strings.TrimSpace(m.Name); ss != "" {
				out = append(out, ss)
			}
		}
		return out, nil
	}

	if providerType == "openai" {
		url := baseURL + "/models"
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if err != nil {
			return nil, fmt.Errorf("failed to create openai models request: %w", err)
		}
		if strings.TrimSpace(pcfg.APIKey) != "" {
			req.Header.Set("Authorization", "Bearer "+strings.TrimSpace(pcfg.APIKey))
		}
		if strings.TrimSpace(pcfg.Organization) != "" {
			req.Header.Set("OpenAI-Organization", strings.TrimSpace(pcfg.Organization))
		}
		resp, err := s.client.Do(req)
		if err != nil {
			return nil, fmt.Errorf("failed to call openai models: %w", err)
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf("openai models returned status=%d", resp.StatusCode)
		}
		var ml openAIModelsList
		if err := json.NewDecoder(resp.Body).Decode(&ml); err != nil {
			return nil, fmt.Errorf("failed to decode openai models response: %w", err)
		}
		out := make([]string, 0, len(ml.Data))
		for _, d := range ml.Data {
			if ss := strings.TrimSpace(d.ID); ss != "" {
				out = append(out, ss)
			}
		}
		return out, nil
	}

	return nil, fmt.Errorf("fetch models not supported for provider_type=%q", providerType)
}

func uniqueStrings(in []string) []string {
	seen := make(map[string]struct{}, len(in))
	out := make([]string, 0, len(in))
	for _, v := range in {
		vv := strings.TrimSpace(v)
		if vv == "" {
			continue
		}
		if _, ok := seen[vv]; ok {
			continue
		}
		seen[vv] = struct{}{}
		out = append(out, vv)
	}
	return out
}
