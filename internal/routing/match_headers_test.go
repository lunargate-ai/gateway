package routing

import (
	"reflect"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
)

func TestMatchHeaderNamesReturnsNormalizedUniqueSnapshot(t *testing.T) {
	engine := NewEngine(config.RoutingConfig{Routes: []config.RouteConfig{
		{Match: config.MatchConfig{Headers: map[string]string{
			" X-Tenant ": "alpha",
			"x-team":     "private",
		}}},
		{Match: config.MatchConfig{Headers: map[string]string{
			"x-tenant": "beta",
		}}},
	}})

	if got, want := engine.MatchHeaderNames(), []string{"x-team", "x-tenant"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("MatchHeaderNames() = %#v, want %#v", got, want)
	}

	engine.UpdateConfig(config.RoutingConfig{Routes: []config.RouteConfig{
		{Match: config.MatchConfig{Headers: map[string]string{"X-Environment": "staging"}}},
	}})
	if got, want := engine.MatchHeaderNames(), []string{"x-environment"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("MatchHeaderNames() after reload = %#v, want %#v", got, want)
	}
}

func TestMatchHeaderNamesHandlesNilEngine(t *testing.T) {
	var engine *Engine
	if got := engine.MatchHeaderNames(); got != nil {
		t.Fatalf("nil engine MatchHeaderNames() = %#v, want nil", got)
	}
}
