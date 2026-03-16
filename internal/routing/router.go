package routing

import (
	"context"
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
	Provider string
	Model    string
	Weight   int
}

// ResolvedRoute contains the matched route and selected target.
type ResolvedRoute struct {
	RouteName string
	Target    Target
	Fallbacks []Target
	Index     int
}

// Engine handles route matching and target selection.
type Engine struct {
	config  atomic.Value // stores *config.RoutingConfig
	counter atomic.Uint64
}

// NewEngine creates a new routing engine.
func NewEngine(cfg config.RoutingConfig) *Engine {
	e := &Engine{}
	e.config.Store(&cfg)
	return e
}

// UpdateConfig hot-reloads the routing configuration.
func (e *Engine) UpdateConfig(cfg config.RoutingConfig) {
	e.config.Store(&cfg)
	log.Info().Msg("routing config updated")
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

// Resolve finds the best route and target for the given request context.
func (e *Engine) Resolve(ctx context.Context, path string, headers map[string]string) (*ResolvedRoute, error) {
	cfg := e.config.Load().(*config.RoutingConfig)
	requestedProvider := strings.TrimSpace(headers["x-lunargate-provider"])
	requestedModel := modelNameFromHeader(strings.TrimSpace(headers["x-lunargate-model"]))
	requestedRoute := strings.TrimSpace(headers["x-lunargate-route"])

	for _, route := range cfg.Routes {
		if requestedRoute != "" && strings.TrimSpace(route.Name) != requestedRoute {
			continue
		}
		if matchRoute(route, path, headers) {
			if len(route.Targets) == 0 {
				continue
			}
			selectedTargets, indexMap := filterTargets(route.Targets, requestedProvider, requestedModel)
			if len(selectedTargets) == 0 {
				selectedTargets = route.Targets
				indexMap = nil
			}

			target, idx := e.selectTarget(cfg.DefaultStrategy, selectedTargets)
			if indexMap != nil && idx >= 0 && idx < len(indexMap) {
				idx = indexMap[idx]
			}

			var fallbacks []Target
			for _, fb := range route.Fallback {
				fallbacks = append(fallbacks, Target{
					Provider: fb.Provider,
					Model:    fb.Model,
					Weight:   fb.Weight,
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

	return nil, fmt.Errorf("no route matched for path=%s", path)
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

func filterTargets(targets []config.TargetConfig, provider string, model string) ([]config.TargetConfig, []int) {
	p := strings.TrimSpace(provider)
	m := strings.TrimSpace(model)
	if p == "" && m == "" {
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
		filtered = append(filtered, t)
		idxMap = append(idxMap, i)
	}
	return filtered, idxMap
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
			return Target{Provider: t.Provider, Model: t.Model, Weight: t.Weight}, i
		}
	}

	// Fallback to first
	t := targets[0]
	return Target{Provider: t.Provider, Model: t.Model, Weight: t.Weight}, 0
}

func (e *Engine) roundRobinSelect(targets []config.TargetConfig) (Target, int) {
	idx := int(e.counter.Add(1)-1) % len(targets)
	t := targets[idx]
	return Target{Provider: t.Provider, Model: t.Model, Weight: t.Weight}, idx
}

func (e *Engine) randomSelect(targets []config.TargetConfig) (Target, int) {
	idx := rand.Intn(len(targets))
	t := targets[idx]
	return Target{Provider: t.Provider, Model: t.Model, Weight: t.Weight}, idx
}
