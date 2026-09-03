package routing

import (
	"context"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
)

func TestResolvePinsFallbacksToExplicitProviderAndModel(t *testing.T) {
	engine := NewEngine(config.RoutingConfig{
		DefaultStrategy: "round-robin",
		Routes: []config.RouteConfig{{
			Name:  "default",
			Match: config.MatchConfig{Path: "*"},
			Targets: []config.TargetConfig{
				{Provider: "ollama", Model: "qwen3.5"},
				{Provider: "openai", Model: "gpt-5.4"},
			},
			Fallback: []config.TargetConfig{
				{Provider: "openai", Model: "gpt-5.4"},
				{Provider: "ollama", Model: "other-local"},
				{Provider: "ollama", Model: "qwen3.5"},
			},
		}},
	})

	resolved, err := engine.Resolve(context.Background(), "/v1/chat/completions", map[string]string{
		"x-lunargate-provider": "ollama",
		"x-lunargate-model":    "ollama/qwen3.5",
	})
	if err != nil {
		t.Fatalf("Resolve returned error: %v", err)
	}
	if resolved.Target.Provider != "ollama" || resolved.Target.Model != "qwen3.5" {
		t.Fatalf("primary target = %#v", resolved.Target)
	}
	if len(resolved.Fallbacks) != 1 {
		t.Fatalf("fallbacks = %#v, want one pinned fallback", resolved.Fallbacks)
	}
	if got := resolved.Fallbacks[0]; got.Provider != "ollama" || got.Model != "qwen3.5" {
		t.Fatalf("fallback = %#v", got)
	}
}

func TestResolveKeepsConfiguredFallbacksWithoutExplicitPin(t *testing.T) {
	engine := NewEngine(config.RoutingConfig{
		DefaultStrategy: "round-robin",
		Routes: []config.RouteConfig{{
			Name:     "default",
			Match:    config.MatchConfig{Path: "*"},
			Targets:  []config.TargetConfig{{Provider: "ollama", Model: "qwen3.5"}},
			Fallback: []config.TargetConfig{{Provider: "openai", Model: "gpt-5.4"}},
		}},
	})

	resolved, err := engine.Resolve(context.Background(), "/v1/chat/completions", map[string]string{})
	if err != nil {
		t.Fatalf("Resolve returned error: %v", err)
	}
	if len(resolved.Fallbacks) != 1 || resolved.Fallbacks[0].Provider != "openai" {
		t.Fatalf("fallbacks = %#v", resolved.Fallbacks)
	}
}

func TestResolveRejectsUnavailableExplicitModel(t *testing.T) {
	engine := NewEngine(config.RoutingConfig{
		DefaultStrategy: "round-robin",
		Routes: []config.RouteConfig{{
			Name:    "default",
			Match:   config.MatchConfig{Path: "*"},
			Targets: []config.TargetConfig{{Provider: "openai", Model: "gpt-5.4"}},
		}},
	})

	_, err := engine.Resolve(context.Background(), "/v1/chat/completions", map[string]string{
		"x-lunargate-model": "gpt-4o",
	})
	unavailable, ok := err.(*RequestedTargetUnavailableError)
	if !ok {
		t.Fatalf("error = %T %v, want RequestedTargetUnavailableError", err, err)
	}
	if unavailable.Model != "gpt-4o" {
		t.Fatalf("unavailable model = %q", unavailable.Model)
	}
}

func TestResolveDoesNotEscapeFirstMatchingRouteForExplicitModel(t *testing.T) {
	engine := NewEngine(config.RoutingConfig{
		DefaultStrategy: "round-robin",
		Routes: []config.RouteConfig{
			{
				Name:    "private",
				Match:   config.MatchConfig{Path: "*", Headers: map[string]string{"x-team": "private"}},
				Targets: []config.TargetConfig{{Provider: "ollama", Model: "qwen3.5"}},
			},
			{
				Name:    "cloud-catch-all",
				Match:   config.MatchConfig{Path: "*"},
				Targets: []config.TargetConfig{{Provider: "openai", Model: "gpt-5.4"}},
			},
		},
	})

	_, err := engine.Resolve(context.Background(), "/v1/chat/completions", map[string]string{
		"x-team":            "private",
		"x-lunargate-model": "gpt-5.4",
	})
	if _, ok := err.(*RequestedTargetUnavailableError); !ok {
		t.Fatalf("error = %T %v, want first-route RequestedTargetUnavailableError", err, err)
	}
}
