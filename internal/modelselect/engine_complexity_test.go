package modelselect

import (
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestEnrichHeadersUsesConfiguredLegacyComplexityRules(t *testing.T) {
	minimumChars := 1
	engine := NewEngine(config.ModelSelectionConfig{
		Enabled: true,
		OutputHeaders: config.ModelSelectionOutputHeaders{
			Complexity: "x-lunargate-complexity",
			Score:      "x-lunargate-complexity-score",
		},
		ComplexityTiers: config.ModelSelectionComplexityTiersConfig{
			Tier01Max: 1,
			Tier23Max: 3,
			Tier45Max: 5,
		},
		Complexity: config.ModelSelectionComplexityRules{
			Complex: config.ModelSelectionComplexityRule{MinUserChars: &minimumChars},
		},
	})
	headers := map[string]string{}

	complexity, _ := engine.EnrichHeaders(&models.UnifiedRequest{
		Messages: []models.Message{{Role: "user", Content: "hello"}},
	}, headers)

	if complexity != "complex" {
		t.Fatalf("complexity = %q, want legacy rule result %q", complexity, "complex")
	}
	if headers["x-lunargate-complexity"] != "complex" {
		t.Fatalf("complexity header = %q, want %q", headers["x-lunargate-complexity"], "complex")
	}
	if headers["x-lunargate-complexity-score"] != "0" {
		t.Fatalf("score header = %q, want score to remain available", headers["x-lunargate-complexity-score"])
	}
}

func TestEnrichHeadersUsesScoredTiersWithoutLegacyRules(t *testing.T) {
	engine := NewEngine(config.ModelSelectionConfig{
		Enabled: true,
		OutputHeaders: config.ModelSelectionOutputHeaders{
			Complexity: "x-lunargate-complexity",
		},
		ComplexityTiers: config.ModelSelectionComplexityTiersConfig{
			Tier01Max: 1,
			Tier23Max: 3,
			Tier45Max: 5,
		},
	})

	complexity, _ := engine.EnrichHeaders(&models.UnifiedRequest{
		Messages: []models.Message{{Role: "user", Content: "hello"}},
	}, nil)

	if complexity != "0-1" {
		t.Fatalf("complexity = %q, want scored tier %q", complexity, "0-1")
	}
}

func TestEnrichHeadersTreatsJSONSchemaAsJSONRequirement(t *testing.T) {
	engine := NewEngine(config.ModelSelectionConfig{
		Enabled: true,
		Complexity: config.ModelSelectionComplexityRules{
			Complex: config.ModelSelectionComplexityRule{AnyOf: []string{"requires_json"}},
		},
	})

	complexity, _ := engine.EnrichHeaders(&models.UnifiedRequest{
		Messages:       []models.Message{{Role: "user", Content: "hello"}},
		ResponseFormat: &models.ResponseFormat{Type: "JSON_SCHEMA"},
	}, nil)

	if complexity != "complex" {
		t.Fatalf("complexity = %q, want JSON schema to match requires_json", complexity)
	}
}
