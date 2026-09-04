package routing

import (
	"context"
	"errors"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
)

func TestResolveClassifiesNoRouteMatched(t *testing.T) {
	engine := NewEngine(config.RoutingConfig{
		Routes: []config.RouteConfig{{
			Name:    "chat-only",
			Match:   config.MatchConfig{Path: "/v1/chat/completions"},
			Targets: []config.TargetConfig{{Provider: "openai", Model: "gpt-5.4"}},
		}},
	})

	_, err := engine.Resolve(context.Background(), "/v1/responses", nil)
	if !errors.Is(err, ErrNoRouteMatched) {
		t.Fatalf("error = %T %v, want ErrNoRouteMatched", err, err)
	}
	var unavailable *RequestedTargetUnavailableError
	if errors.As(err, &unavailable) {
		t.Fatalf("error = %T %v, do not want RequestedTargetUnavailableError", err, err)
	}
}

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
	if errors.Is(err, ErrNoRouteMatched) {
		t.Fatalf("error = %T %v, do not want ErrNoRouteMatched", err, err)
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

func TestResolvePinnedProviderProtocolPreservesRequestedRouteAndModel(t *testing.T) {
	engine := NewEngine(config.RoutingConfig{
		DefaultStrategy: "round-robin",
		Routes: []config.RouteConfig{
			{
				Name:    "other-route",
				Match:   config.MatchConfig{Path: "/v1/responses"},
				Targets: []config.TargetConfig{{Provider: "native", Model: "gpt-native", UpstreamRequestType: "responses"}},
			},
			{
				Name:  "requested-route",
				Match: config.MatchConfig{Path: "/v1/responses"},
				Targets: []config.TargetConfig{
					{Provider: "native", Model: "gpt-native", UpstreamRequestType: "chat_completions"},
					{Provider: "native", Model: "gpt-other", UpstreamRequestType: "responses"},
					{Provider: "native", Model: "gpt-native", UpstreamRequestType: "responses"},
					{Provider: "other", Model: "gpt-native", UpstreamRequestType: "responses"},
				},
			},
		},
	})

	ctx := WithPinnedProviderProtocol(context.Background(), Target{
		Provider:            "native",
		UpstreamRequestType: "responses",
	})
	resolved, err := engine.Resolve(ctx, "/v1/responses", map[string]string{
		"x-lunargate-provider": "other",
		"x-lunargate-model":    "native/gpt-native",
		"x-lunargate-route":    "requested-route",
	})
	if err != nil {
		t.Fatalf("Resolve returned error: %v", err)
	}
	if resolved.RouteName != "requested-route" {
		t.Fatalf("route = %q, want requested-route", resolved.RouteName)
	}
	if got := resolved.Target; got.Provider != "native" || got.Model != "gpt-native" || got.UpstreamRequestType != "responses" {
		t.Fatalf("target = %#v, want native/gpt-native over Responses", got)
	}
}

func TestFirstMatchingRouteTargetAvailableIsReadOnlyAndIgnoresModel(t *testing.T) {
	engine := NewEngine(config.RoutingConfig{
		DefaultStrategy: "round-robin",
		Routes: []config.RouteConfig{
			{
				Name:    "skipped",
				Match:   config.MatchConfig{Path: "/v1/responses"},
				Targets: []config.TargetConfig{{Provider: "native", Model: "gpt-native", UpstreamRequestType: "responses"}},
			},
			{
				Name:  "requested",
				Match: config.MatchConfig{Path: "/v1/responses", Headers: map[string]string{"x-scope": "stateful"}},
				Targets: []config.TargetConfig{
					{Provider: "native", Model: "gpt-native", UpstreamRequestType: "chat_completions"},
					{Provider: "native", Model: "gpt-other", UpstreamRequestType: "responses"},
					{Provider: "native", Model: "gpt-native", UpstreamRequestType: "responses"},
				},
			},
		},
	})
	headers := map[string]string{
		"x-lunargate-route": "requested",
		"x-lunargate-model": "native/not-configured",
		"x-scope":           "stateful",
	}

	matched, available := engine.FirstMatchingRouteTargetAvailable(
		"/v1/responses",
		headers,
		"native",
		"responses",
	)
	if !matched || !available {
		t.Fatalf("availability = matched:%t available:%t, want true/true", matched, available)
	}

	delete(headers, "x-lunargate-model")
	resolved, err := engine.Resolve(
		WithPinnedProviderProtocol(context.Background(), Target{Provider: "native", UpstreamRequestType: "responses"}),
		"/v1/responses",
		headers,
	)
	if err != nil {
		t.Fatalf("Resolve returned error: %v", err)
	}
	if resolved.Target.Model != "gpt-other" {
		t.Fatalf("target model = %q, want first eligible model after a non-selecting availability check", resolved.Target.Model)
	}
}

func TestFirstMatchingRouteTargetAvailableDoesNotEscapeMatchedRoute(t *testing.T) {
	engine := NewEngine(config.RoutingConfig{
		Routes: []config.RouteConfig{
			{
				Name:    "first",
				Match:   config.MatchConfig{Path: "/v1/responses"},
				Targets: []config.TargetConfig{{Provider: "native", Model: "gpt-native", UpstreamRequestType: "chat_completions"}},
			},
			{
				Name:    "later",
				Match:   config.MatchConfig{Path: "/v1/responses"},
				Targets: []config.TargetConfig{{Provider: "native", Model: "gpt-native", UpstreamRequestType: "responses"}},
			},
		},
	})

	matched, available := engine.FirstMatchingRouteTargetAvailable(
		"/v1/responses",
		map[string]string{},
		"native",
		"responses",
	)
	if !matched || available {
		t.Fatalf("availability = matched:%t available:%t, want true/false", matched, available)
	}
}

func TestFirstMatchingRouteTargetAvailableSkipsEmptyRoutesLikeResolve(t *testing.T) {
	engine := NewEngine(config.RoutingConfig{
		Routes: []config.RouteConfig{
			{
				Name:  "empty",
				Match: config.MatchConfig{Path: "/v1/responses"},
			},
			{
				Name:    "selectable",
				Match:   config.MatchConfig{Path: "/v1/responses"},
				Targets: []config.TargetConfig{{Provider: "native", Model: "gpt-native", UpstreamRequestType: "responses"}},
			},
		},
	})

	matched, available := engine.FirstMatchingRouteTargetAvailable(
		"/v1/responses",
		map[string]string{},
		"native",
		"responses",
	)
	if !matched || !available {
		t.Fatalf("availability = matched:%t available:%t, want true/true", matched, available)
	}
}
