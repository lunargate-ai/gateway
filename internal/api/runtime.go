package api

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"reflect"
	"strconv"
	"sync"
	"sync/atomic"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/modelselect"
	"github.com/lunargate-ai/gateway/internal/modelstore"
	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/internal/routing"
	"github.com/lunargate-ai/gateway/pkg/models"
	"github.com/rs/zerolog/log"
)

var runtimeCacheNamespaceCounter atomic.Uint64

// runtimeGeneration is immutable after publication. Requests retain its
// component pointers for their complete lifecycle, while a reload constructs
// and publishes a replacement generation with one atomic store.
type runtimeGeneration struct {
	id uint64

	registry        *providers.Registry
	router          *routing.Engine
	selector        *modelselect.Engine
	store           *modelstore.Store
	providerClients *providerClientRegistry

	providerConfig  map[string]config.ProviderConfig
	cacheNamespaces map[string]string
	routingConfig   config.RoutingConfig
	selectionConfig config.ModelSelectionConfig
}

type runtimeController struct {
	current atomic.Pointer[runtimeGeneration]

	mu     sync.Mutex
	nextID uint64
}

func newRuntimeController(
	registry *providers.Registry,
	router *routing.Engine,
	selector *modelselect.Engine,
	store *modelstore.Store,
	providerClients *providerClientRegistry,
) *runtimeController {
	providerConfig := map[string]config.ProviderConfig{}
	if registry != nil {
		providerConfig = registry.ConfigSnapshot()
	}
	routingConfig := cloneRuntimeRoutingConfig(config.RoutingConfig{})
	if router != nil {
		routingConfig = cloneRuntimeRoutingConfig(router.Config())
	}
	selectionConfig := cloneRuntimeSelectionConfig(config.ModelSelectionConfig{})
	if selector != nil {
		selectionConfig = cloneRuntimeSelectionConfig(selector.Config())
	}
	controller := &runtimeController{nextID: 1}
	controller.current.Store(&runtimeGeneration{
		id:              1,
		registry:        registry,
		router:          router,
		selector:        selector,
		store:           store,
		providerClients: providerClients,
		providerConfig:  providerConfig,
		cacheNamespaces: updatedProviderCacheNamespaces(nil, providerConfig, nil),
		routingConfig:   routingConfig,
		selectionConfig: selectionConfig,
	})
	return controller
}

func (c *runtimeController) update(
	providerConfig map[string]config.ProviderConfig,
	routingConfig config.RoutingConfig,
	selectionConfig config.ModelSelectionConfig,
) (bool, error) {
	if c == nil {
		return false, fmt.Errorf("runtime controller is not initialized")
	}

	nextProviders := cloneProviderConfigs(providerConfig)
	nextRouting := cloneRuntimeRoutingConfig(routingConfig)
	nextSelection := cloneRuntimeSelectionConfig(selectionConfig)

	c.mu.Lock()
	defer c.mu.Unlock()
	return c.updateLocked(nextProviders, nextRouting, nextSelection)
}

func (c *runtimeController) updateProviders(providerConfig map[string]config.ProviderConfig) (bool, error) {
	if c == nil {
		return false, fmt.Errorf("runtime controller is not initialized")
	}
	nextProviders := cloneProviderConfigs(providerConfig)
	c.mu.Lock()
	defer c.mu.Unlock()
	current := c.current.Load()
	if current == nil {
		return false, fmt.Errorf("runtime generation is not initialized")
	}
	return c.updateLocked(nextProviders, current.routingConfig, current.selectionConfig)
}

func (c *runtimeController) updateLocked(
	nextProviders map[string]config.ProviderConfig,
	nextRouting config.RoutingConfig,
	nextSelection config.ModelSelectionConfig,
) (bool, error) {
	current := c.current.Load()
	if current == nil {
		return false, fmt.Errorf("runtime generation is not initialized")
	}
	providersChanged := !reflect.DeepEqual(current.providerConfig, nextProviders)
	routingChanged := !reflect.DeepEqual(current.routingConfig, nextRouting)
	selectionChanged := !reflect.DeepEqual(current.selectionConfig, nextSelection)
	if !providersChanged && !routingChanged && !selectionChanged {
		return false, nil
	}

	next := &runtimeGeneration{
		registry:        current.registry,
		router:          current.router,
		selector:        current.selector,
		store:           current.store,
		providerClients: current.providerClients,
		providerConfig:  current.providerConfig,
		cacheNamespaces: current.cacheNamespaces,
		routingConfig:   current.routingConfig,
		selectionConfig: current.selectionConfig,
	}
	if providersChanged {
		next.registry = providers.NewRegistry(nextProviders)
		if len(next.registry.List()) == 0 {
			return false, fmt.Errorf("provider reload produced zero valid providers")
		}
		next.store = modelstore.NewStore(next.registry, nextProviders)
		next.providerClients = newProviderClientRegistry(nextProviders)
		next.providerConfig = nextProviders
		next.cacheNamespaces = updatedProviderCacheNamespaces(
			current.providerConfig,
			nextProviders,
			current.cacheNamespaces,
		)
	}
	if routingChanged {
		next.router = routing.NewEngine(nextRouting)
		next.routingConfig = nextRouting
	}
	if selectionChanged {
		next.selector = modelselect.NewEngine(nextSelection)
		next.selectionConfig = nextSelection
	}

	c.nextID++
	next.id = c.nextID
	c.current.Store(next)
	log.Info().
		Uint64("generation", next.id).
		Bool("providers_changed", providersChanged).
		Bool("routing_changed", routingChanged).
		Bool("model_selection_changed", selectionChanged).
		Msg("runtime generation swapped")
	return true, nil
}

// UpdateRuntime builds a coherent runtime generation and publishes it in one
// atomic operation. A failed build leaves the current generation untouched.
func (h *Handler) UpdateRuntime(
	providerConfig map[string]config.ProviderConfig,
	routingConfig config.RoutingConfig,
	selectionConfig config.ModelSelectionConfig,
) (bool, error) {
	owner := h.runtimeOwner()
	if owner == nil || owner.runtime == nil {
		return false, fmt.Errorf("handler runtime is not initialized")
	}
	return owner.runtime.update(providerConfig, routingConfig, selectionConfig)
}

func (h *Handler) runtimeOwner() *Handler {
	if h == nil {
		return nil
	}
	if h.runtimeRoot != nil {
		return h.runtimeRoot
	}
	return h
}

func (h *Handler) currentRuntimeGeneration() *runtimeGeneration {
	if h == nil {
		return nil
	}
	if h.boundRuntime != nil {
		return h.boundRuntime
	}
	owner := h.runtimeOwner()
	if owner == nil || owner.runtime == nil {
		return nil
	}
	return owner.runtime.current.Load()
}

// bindRuntime returns a lightweight request-scoped Handler. It deliberately
// copies only pointers and immutable generation members; mutex-owning state is
// retained on the root Handler.
func (h *Handler) bindRuntime() *Handler {
	if h == nil || h.boundRuntime != nil {
		return h
	}
	generation := h.currentRuntimeGeneration()
	if generation == nil {
		return h
	}
	owner := h.runtimeOwner()
	return &Handler{
		registry:               generation.registry,
		router:                 generation.router,
		fallback:               owner.fallback,
		cache:                  owner.cache,
		streamer:               owner.streamer,
		metrics:                owner.metrics,
		collector:              owner.collector,
		selector:               generation.selector,
		store:                  generation.store,
		providerClients:        generation.providerClients,
		responsesState:         owner.responsesState,
		responseBindings:       owner.responseBindings,
		chatCompletionBindings: owner.chatCompletionBindings,
		conversationBindings:   owner.conversationBindings,
		conversationsState:     owner.conversationsState,
		runtime:                owner.runtime,
		boundRuntime:           generation,
		runtimeRoot:            owner,
	}
}

type runtimeHTTPHandler func(*Handler, http.ResponseWriter, *http.Request)

func (h *Handler) withRuntime(next runtimeHTTPHandler) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		next(h.bindRuntime(), w, r)
	}
}

func (h *Handler) responsesWebSocketRegistryRef() *responsesWebSocketRegistry {
	owner := h.runtimeOwner()
	if owner == nil {
		return nil
	}
	return &owner.responsesWebSockets
}

// RuntimeProviderNames returns provider aliases from one current generation.
func (h *Handler) RuntimeProviderNames() []string {
	generation := h.currentRuntimeGeneration()
	if generation == nil || generation.registry == nil {
		return nil
	}
	return generation.registry.List()
}

// RuntimeRouteNames returns route names from one current generation.
func (h *Handler) RuntimeRouteNames() []string {
	generation := h.currentRuntimeGeneration()
	if generation == nil || generation.router == nil {
		return nil
	}
	return generation.router.RouteNames()
}

// RuntimeModelSnapshotIDs returns the non-blocking model inventory from one
// current generation.
func (h *Handler) RuntimeModelSnapshotIDs() []string {
	generation := h.currentRuntimeGeneration()
	if generation == nil || generation.store == nil {
		return nil
	}
	return modelIDsFromInventory(generation.store.AllModelsSnapshot())
}

// RuntimeModelIDs returns the complete model inventory from one current
// generation, including any provider fetches performed by that generation.
func (h *Handler) RuntimeModelIDs(ctx context.Context) []string {
	generation := h.currentRuntimeGeneration()
	if generation == nil || generation.store == nil {
		return nil
	}
	return modelIDsFromInventory(generation.store.AllModels(ctx))
}

func modelIDsFromInventory(inventory []models.ModelInfo) []string {
	ids := make([]string, 0, len(inventory))
	for _, item := range inventory {
		if item.ID != "" {
			ids = append(ids, item.ID)
		}
	}
	return ids
}

func updatedProviderCacheNamespaces(
	currentConfig map[string]config.ProviderConfig,
	nextConfig map[string]config.ProviderConfig,
	currentNamespaces map[string]string,
) map[string]string {
	namespaces := make(map[string]string, len(nextConfig))
	for provider, nextProvider := range nextConfig {
		if currentProvider, ok := currentConfig[provider]; ok && reflect.DeepEqual(currentProvider, nextProvider) {
			if namespace := currentNamespaces[provider]; namespace != "" {
				namespaces[provider] = namespace
				continue
			}
		}
		namespaces[provider] = strconv.FormatUint(runtimeCacheNamespaceCounter.Add(1), 36)
	}
	return namespaces
}

// runtimeCacheKey keeps entries from different provider configurations in
// separate namespaces. The namespace is an opaque process-local epoch rather
// than a credential-derived fingerprint, and only the combined digest reaches
// cache diagnostics.
func (h *Handler) runtimeCacheKey(baseKey string, provider string) string {
	if baseKey == "" {
		return ""
	}
	generation := h.currentRuntimeGeneration()
	if generation == nil {
		return baseKey
	}
	namespace := generation.cacheNamespaces[provider]
	if namespace == "" {
		return baseKey
	}
	digest := sha256.Sum256([]byte(namespace + "\x00" + baseKey))
	return hex.EncodeToString(digest[:16])
}

func cloneRuntimeRoutingConfig(cfg config.RoutingConfig) config.RoutingConfig {
	cloned := cfg
	cloned.Routes = make([]config.RouteConfig, len(cfg.Routes))
	for i, route := range cfg.Routes {
		cloned.Routes[i] = route
		cloned.Routes[i].Targets = append([]config.TargetConfig(nil), route.Targets...)
		cloned.Routes[i].Fallback = append([]config.TargetConfig(nil), route.Fallback...)
		if len(route.Match.Headers) > 0 {
			cloned.Routes[i].Match.Headers = cloneStringMap(route.Match.Headers)
		} else {
			cloned.Routes[i].Match.Headers = nil
		}
	}
	return cloned
}

func cloneRuntimeSelectionConfig(cfg config.ModelSelectionConfig) config.ModelSelectionConfig {
	cloned := cfg
	cloneInt := func(value *int) *int {
		if value == nil {
			return nil
		}
		copy := *value
		return &copy
	}
	cloned.Complexity.Simple.MaxUserChars = cloneInt(cfg.Complexity.Simple.MaxUserChars)
	cloned.Complexity.Simple.MinUserChars = cloneInt(cfg.Complexity.Simple.MinUserChars)
	cloned.Complexity.Simple.MaxMessages = cloneInt(cfg.Complexity.Simple.MaxMessages)
	cloned.Complexity.Simple.MinMessages = cloneInt(cfg.Complexity.Simple.MinMessages)
	cloned.Complexity.Simple.AnyOf = append([]string(nil), cfg.Complexity.Simple.AnyOf...)
	cloned.Complexity.Complex.MaxUserChars = cloneInt(cfg.Complexity.Complex.MaxUserChars)
	cloned.Complexity.Complex.MinUserChars = cloneInt(cfg.Complexity.Complex.MinUserChars)
	cloned.Complexity.Complex.MaxMessages = cloneInt(cfg.Complexity.Complex.MaxMessages)
	cloned.Complexity.Complex.MinMessages = cloneInt(cfg.Complexity.Complex.MinMessages)
	cloned.Complexity.Complex.AnyOf = append([]string(nil), cfg.Complexity.Complex.AnyOf...)
	cloned.Skills = make([]config.ModelSelectionSkillRule, len(cfg.Skills))
	for i, skill := range cfg.Skills {
		cloned.Skills[i] = skill
		cloned.Skills[i].RegexAny = append([]string(nil), skill.RegexAny...)
	}
	return cloned
}
