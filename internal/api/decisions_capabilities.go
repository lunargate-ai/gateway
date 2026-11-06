package api

import (
	"sort"
	"strings"

	"github.com/lunargate-ai/gateway/internal/modelid"
	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/pkg/models"
)

// RuntimeDecisionCapabilities uses one immutable runtime generation and never
// does provider I/O. A route is eligible if its filtered primary pool is nonempty.
func (h *Handler) RuntimeDecisionCapabilities() models.DecisionCapabilities {
	result := models.DecisionCapabilities{}
	generation := h.currentRuntimeGeneration()
	if generation == nil || generation.registry == nil || generation.router == nil {
		return result
	}
	configs := generation.registry.ConfigSnapshot()
	for id, cfg := range configs {
		typ, ok := generation.registry.Type(id)
		if !ok {
			continue
		}
		for _, model := range cfg.Capabilities.DecisionModels {
			if providers.DecisionsModelEnabled(typ, cfg.Capabilities, model) {
				result.Models = append(result.Models, modelid.BuildCanonical(id, model))
			}
		}
		if typ == "typesafe" {
			if generation.store != nil {
				for _, model := range generation.store.AllModelsSnapshot() {
					p, _, _ := modelid.SplitCanonical(model.ID)
					if p == id {
						result.OnlyModels = append(result.OnlyModels, model.ID)
					}
				}
			}
			for _, model := range cfg.Capabilities.DecisionModels {
				result.OnlyModels = append(result.OnlyModels, modelid.BuildCanonical(id, model))
			}
		}
	}
	for _, route := range generation.router.Config().Routes {
		onlyDecisions := len(route.Targets) > 0
		for _, target := range route.Targets {
			typ, ok := generation.registry.Type(target.Provider)
			if !ok || typ != "typesafe" {
				onlyDecisions = false
				break
			}
		}
		if onlyDecisions {
			result.OnlyRoutes = append(result.OnlyRoutes, route.Name)
		}
		path := route.Match.Path
		if path != "" && path != "*" && !strings.HasPrefix("/v1/decisions", path) {
			continue
		}
		for _, target := range route.Targets {
			snapshot, ok := generation.registry.Snapshot(target.Provider)
			if !ok {
				continue
			}
			model := strings.TrimSpace(target.Model)
			if model == "" {
				model = snapshot.Translator.DefaultModel()
			}
			if providers.DecisionsModelEnabled(snapshot.ProviderType, snapshot.Capabilities, model) {
				result.Routes = append(result.Routes, route.Name)
				break
			}
		}
	}
	result.Models = uniqueDecisionStrings(result.Models)
	result.Routes = uniqueDecisionStrings(result.Routes)
	result.OnlyModels = uniqueDecisionStrings(result.OnlyModels)
	result.OnlyRoutes = uniqueDecisionStrings(result.OnlyRoutes)
	return result
}

func uniqueDecisionStrings(values []string) []string {
	seen := map[string]bool{}
	result := make([]string, 0, len(values))
	for _, v := range values {
		if v != "" && !seen[v] {
			seen[v] = true
			result = append(result, v)
		}
	}
	sort.Strings(result)
	return result
}
