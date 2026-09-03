package routing

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"sort"
	"strings"
	"sync/atomic"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/rs/zerolog/log"
)

// Target represents a resolved routing target (provider + model).
type Target struct {
	Provider            string
	Model               string
	Weight              int
	UpstreamRequestType string
	circuitBreakerKey   string
}

// WithCircuitBreakerKey returns a copy carrying the opaque internal identity
// used for circuit-breaker lookup. The key is deliberately unexported so it is
// never serialized with route metadata.
func (t Target) WithCircuitBreakerKey(key string) Target {
	if strings.TrimSpace(key) == "" {
		t.circuitBreakerKey = ""
	} else {
		t.circuitBreakerKey = key
	}
	return t
}

// CircuitBreakerKey returns the internal breaker lookup key, falling back to
// the provider alias for targets created outside the request handler.
// Callers must never log or expose the returned value.
func (t Target) CircuitBreakerKey() string {
	if strings.TrimSpace(t.circuitBreakerKey) != "" {
		return t.circuitBreakerKey
	}
	return t.Provider
}

// GoString keeps the internal breaker identity out of diagnostic formatting.
func (t Target) GoString() string {
	return fmt.Sprintf(
		"routing.Target{Provider:%q, Model:%q, Weight:%d, UpstreamRequestType:%q}",
		t.Provider,
		t.Model,
		t.Weight,
		t.UpstreamRequestType,
	)
}

type pinnedTargetContextKey struct{}

type pinnedProviderProtocolContextKey struct{}

type pinnedTarget struct {
	route               string
	provider            string
	model               string
	upstreamRequestType string
}

type pinnedProviderProtocol struct {
	provider            string
	upstreamRequestType string
}

// WithPinnedTarget constrains one stateful follow-up to the exact route and
// upstream protocol that created its owner binding. The constraint lives in
// the trusted request context rather than a caller-controlled HTTP header.
func WithPinnedTarget(ctx context.Context, route string, target Target) context.Context {
	return context.WithValue(ctx, pinnedTargetContextKey{}, pinnedTarget{
		route:               strings.TrimSpace(route),
		provider:            strings.TrimSpace(target.Provider),
		model:               strings.TrimSpace(target.Model),
		upstreamRequestType: canonicalTargetUpstreamRequestType(target.UpstreamRequestType),
	})
}

// WithPinnedProviderProtocol constrains one stateful request to its owner
// provider and upstream protocol while preserving any route and model
// constraints supplied by the request. This is used by bindings that retain
// provider ownership but deliberately do not retain a route or model.
func WithPinnedProviderProtocol(ctx context.Context, target Target) context.Context {
	return context.WithValue(ctx, pinnedProviderProtocolContextKey{}, pinnedProviderProtocol{
		provider:            strings.TrimSpace(target.Provider),
		upstreamRequestType: canonicalTargetUpstreamRequestType(target.UpstreamRequestType),
	})
}

// ResolvedRoute contains the matched route and selected target.
type ResolvedRoute struct {
	RouteName string
	Target    Target
	Fallbacks []Target
	Index     int
}

// ErrNoRouteMatched reports that no configured route matched the request.
var ErrNoRouteMatched = errors.New("no route matched")

type RequestedTargetUnavailableError struct {
	Provider string
	Model    string
}

func (e *RequestedTargetUnavailableError) Error() string {
	if e == nil {
		return "requested target is not available"
	}
	if strings.TrimSpace(e.Model) != "" {
		return fmt.Sprintf("requested model %q is not available", e.Model)
	}
	return fmt.Sprintf("requested provider %q is not available", e.Provider)
}

// Engine handles route matching and target selection.
type Engine struct {
	config  atomic.Value // stores *config.RoutingConfig
	counter atomic.Uint64
}

// NewEngine creates a new routing engine.
func NewEngine(cfg config.RoutingConfig) *Engine {
	e := &Engine{}
	owned := cloneRoutingConfig(cfg)
	e.config.Store(&owned)
	return e
}

// UpdateConfig hot-reloads the routing configuration.
func (e *Engine) UpdateConfig(cfg config.RoutingConfig) {
	owned := cloneRoutingConfig(cfg)
	e.config.Store(&owned)
	log.Info().Msg("routing config updated")
}

// Config returns an owned copy of the active routing configuration.
func (e *Engine) Config() config.RoutingConfig {
	if e == nil {
		return config.RoutingConfig{}
	}
	cfg, _ := e.config.Load().(*config.RoutingConfig)
	if cfg == nil {
		return config.RoutingConfig{}
	}
	return cloneRoutingConfig(*cfg)
}

func cloneRoutingConfig(cfg config.RoutingConfig) config.RoutingConfig {
	cloned := cfg
	cloned.Routes = make([]config.RouteConfig, len(cfg.Routes))
	for i, route := range cfg.Routes {
		cloned.Routes[i] = route
		cloned.Routes[i].Targets = append([]config.TargetConfig(nil), route.Targets...)
		cloned.Routes[i].Fallback = append([]config.TargetConfig(nil), route.Fallback...)
		if len(route.Match.Headers) > 0 {
			cloned.Routes[i].Match.Headers = make(map[string]string, len(route.Match.Headers))
			for key, value := range route.Match.Headers {
				cloned.Routes[i].Match.Headers[key] = value
			}
		} else {
			cloned.Routes[i].Match.Headers = nil
		}
	}
	return cloned
}

func (e *Engine) RouteNames() []string {
	cfg := e.config.Load().(*config.RoutingConfig)
	out := make([]string, 0, len(cfg.Routes))
	for _, route := range cfg.Routes {
		name := strings.TrimSpace(route.Name)
		if name == "" {
			continue
		}
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

// FirstMatchingRouteTargetAvailable reports whether a route matched and, when
// one did, whether that first route contains a target for the trusted provider
// and upstream protocol. Model selection is deliberately ignored so callers
// can distinguish an unavailable protocol from a separate model-selection
// error. The check is read-only and does not advance load-balancer state.
func (e *Engine) FirstMatchingRouteTargetAvailable(
	path string,
	headers map[string]string,
	provider string,
	upstreamRequestType string,
) (matched bool, available bool) {
	cfg := e.config.Load().(*config.RoutingConfig)
	requestedRoute := strings.TrimSpace(headers["x-lunargate-route"])
	for _, route := range cfg.Routes {
		if requestedRoute != "" && strings.TrimSpace(route.Name) != requestedRoute {
			continue
		}
		if !matchRoute(route, path, headers) {
			continue
		}
		if len(route.Targets) == 0 {
			continue
		}
		targets, _ := filterTargets(route.Targets, provider, "", upstreamRequestType)
		return true, len(targets) > 0
	}
	return false, false
}

// Resolve finds the best route and target for the given request context.
func (e *Engine) Resolve(ctx context.Context, path string, headers map[string]string) (*ResolvedRoute, error) {
	cfg := e.config.Load().(*config.RoutingConfig)
	requestedProvider := strings.TrimSpace(headers["x-lunargate-provider"])
	requestedModel := modelNameFromHeader(strings.TrimSpace(headers["x-lunargate-model"]))
	requestedRoute := strings.TrimSpace(headers["x-lunargate-route"])
	requestedUpstreamRequestType := ""
	if constraint, ok := ctx.Value(pinnedTargetContextKey{}).(pinnedTarget); ok {
		requestedProvider = constraint.provider
		requestedModel = modelNameFromHeader(constraint.model)
		requestedRoute = constraint.route
		requestedUpstreamRequestType = constraint.upstreamRequestType
	}
	if constraint, ok := ctx.Value(pinnedProviderProtocolContextKey{}).(pinnedProviderProtocol); ok {
		requestedProvider = constraint.provider
		requestedUpstreamRequestType = constraint.upstreamRequestType
	}

	for _, route := range cfg.Routes {
		if requestedRoute != "" && strings.TrimSpace(route.Name) != requestedRoute {
			continue
		}
		if matchRoute(route, path, headers) {
			if len(route.Targets) == 0 {
				continue
			}
			selectedTargets, indexMap := filterTargets(route.Targets, requestedProvider, requestedModel, requestedUpstreamRequestType)
			if len(selectedTargets) == 0 {
				return nil, &RequestedTargetUnavailableError{Provider: requestedProvider, Model: requestedModel}
			}

			target, idx := e.selectTarget(cfg.DefaultStrategy, selectedTargets)
			if indexMap != nil && idx >= 0 && idx < len(indexMap) {
				idx = indexMap[idx]
			}

			fallbackConfigs, _ := filterTargets(route.Fallback, requestedProvider, requestedModel, requestedUpstreamRequestType)
			var fallbacks []Target
			for _, fb := range fallbackConfigs {
				fallbacks = append(fallbacks, Target{
					Provider:            fb.Provider,
					Model:               fb.Model,
					Weight:              fb.Weight,
					UpstreamRequestType: fb.UpstreamRequestType,
				})
			}

			return &ResolvedRoute{
				RouteName: route.Name,
				Target:    target,
				Fallbacks: fallbacks,
				Index:     idx,
			}, nil
		}
	}
	return nil, fmt.Errorf("%w for path=%s", ErrNoRouteMatched, path)
}

func modelNameFromHeader(modelHeader string) string {
	m := strings.TrimSpace(modelHeader)
	if m == "" {
		return ""
	}
	idx := strings.IndexByte(m, '/')
	if idx <= 0 || idx >= len(m)-1 {
		return m
	}
	return strings.TrimSpace(m[idx+1:])
}

func filterTargets(targets []config.TargetConfig, provider string, model string, upstreamRequestType string) ([]config.TargetConfig, []int) {
	p := strings.TrimSpace(provider)
	m := strings.TrimSpace(model)
	u := canonicalTargetUpstreamRequestType(upstreamRequestType)
	if p == "" && m == "" && strings.TrimSpace(upstreamRequestType) == "" {
		return targets, nil
	}
	filtered := make([]config.TargetConfig, 0, len(targets))
	idxMap := make([]int, 0, len(targets))
	for i := range targets {
		t := targets[i]
		if p != "" && strings.TrimSpace(t.Provider) != p {
			continue
		}
		if m != "" {
			if strings.TrimSpace(t.Model) != "" && strings.TrimSpace(t.Model) != m {
				continue
			}
		}
		if strings.TrimSpace(upstreamRequestType) != "" && canonicalTargetUpstreamRequestType(t.UpstreamRequestType) != u {
			continue
		}
		filtered = append(filtered, t)
		idxMap = append(idxMap, i)
	}
	return filtered, idxMap
}

func canonicalTargetUpstreamRequestType(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	if value == "" {
		return "chat_completions"
	}
	return value
}

func matchRoute(route config.RouteConfig, path string, headers map[string]string) bool {
	// Match path
	if route.Match.Path != "" && route.Match.Path != "*" {
		if !strings.HasPrefix(path, route.Match.Path) {
			return false
		}
	}

	// Match headers
	for key, val := range route.Match.Headers {
		if headers[key] != val {
			return false
		}
	}

	return true
}

func (e *Engine) selectTarget(strategy string, targets []config.TargetConfig) (Target, int) {
	switch strategy {
	case "weighted":
		return e.weightedSelect(targets)
	case "round-robin":
		return e.roundRobinSelect(targets)
	case "random":
		return e.randomSelect(targets)
	default:
		return e.weightedSelect(targets)
	}
}

func (e *Engine) weightedSelect(targets []config.TargetConfig) (Target, int) {
	totalWeight := 0
	for _, t := range targets {
		w := t.Weight
		if w <= 0 {
			w = 1
		}
		totalWeight += w
	}

	r := rand.Intn(totalWeight)
	for i, t := range targets {
		w := t.Weight
		if w <= 0 {
			w = 1
		}
		r -= w
		if r < 0 {
			return Target{Provider: t.Provider, Model: t.Model, Weight: t.Weight, UpstreamRequestType: t.UpstreamRequestType}, i
		}
	}

	// Fallback to first
	t := targets[0]
	return Target{Provider: t.Provider, Model: t.Model, Weight: t.Weight, UpstreamRequestType: t.UpstreamRequestType}, 0
}

func (e *Engine) roundRobinSelect(targets []config.TargetConfig) (Target, int) {
	idx := int(e.counter.Add(1)-1) % len(targets)
	t := targets[idx]
	return Target{Provider: t.Provider, Model: t.Model, Weight: t.Weight, UpstreamRequestType: t.UpstreamRequestType}, idx
}

func (e *Engine) randomSelect(targets []config.TargetConfig) (Target, int) {
	idx := rand.Intn(len(targets))
	t := targets[idx]
	return Target{Provider: t.Provider, Model: t.Model, Weight: t.Weight, UpstreamRequestType: t.UpstreamRequestType}, idx
}
