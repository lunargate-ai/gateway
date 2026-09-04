package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestValidateModelSelectionConfigCanonicalizesValidRules(t *testing.T) {
	cfg := validRuntimeConfig()
	cfg.ModelSelect = validModelSelectionConfig()
	cfg.ModelSelect.OutputHeaders.Complexity = " X-Complexity "
	cfg.ModelSelect.Complexity.Simple.AnyOf = []string{" HAS_TOOLS ", "requires_JSON"}
	cfg.ModelSelect.Skills[0].Name = " coding "
	cfg.ModelSelect.Skills[0].RegexAny[0] = "  (?i)code  "

	if err := validateConfig(cfg); err != nil {
		t.Fatalf("validateConfig returned error: %v", err)
	}
	if got, want := cfg.ModelSelect.OutputHeaders.Complexity, "x-complexity"; got != want {
		t.Fatalf("complexity output header = %q, want %q", got, want)
	}
	if got, want := cfg.ModelSelect.Complexity.Simple.AnyOf, []string{"has_tools", "requires_json"}; !equalStrings(got, want) {
		t.Fatalf("simple any_of = %#v, want %#v", got, want)
	}
	if got, want := cfg.ModelSelect.Skills[0].Name, "coding"; got != want {
		t.Fatalf("skill name = %q, want %q", got, want)
	}
	if got, want := cfg.ModelSelect.Skills[0].RegexAny[0], "(?i)code"; got != want {
		t.Fatalf("skill regex = %q, want %q", got, want)
	}
}

func TestValidateModelSelectionConfigRejectsInvalidRules(t *testing.T) {
	negative := -1
	zero := 0
	one := 1
	two := 2
	tests := []struct {
		name    string
		mutate  func(*ModelSelectionConfig)
		wantErr string
	}{
		{
			name: "invalid skill regex",
			mutate: func(cfg *ModelSelectionConfig) {
				cfg.Skills[0].RegexAny[0] = "("
			},
			wantErr: "model_selection.skills[0].regex_any[0]",
		},
		{
			name: "empty skill name",
			mutate: func(cfg *ModelSelectionConfig) {
				cfg.Skills[0].Name = " "
			},
			wantErr: "model_selection.skills[0].name",
		},
		{
			name: "empty skill patterns",
			mutate: func(cfg *ModelSelectionConfig) {
				cfg.Skills[0].RegexAny = nil
			},
			wantErr: "model_selection.skills[0].regex_any",
		},
		{
			name: "duplicate skill name",
			mutate: func(cfg *ModelSelectionConfig) {
				cfg.Skills = append(cfg.Skills, ModelSelectionSkillRule{Name: "CODING", RegexAny: []string{"test"}})
			},
			wantErr: "model_selection.skills[1].name",
		},
		{
			name: "reserved provider output header",
			mutate: func(cfg *ModelSelectionConfig) {
				cfg.OutputHeaders.Skill = "X-LunarGate-Provider"
			},
			wantErr: "reserved routing header",
		},
		{
			name: "duplicate output header",
			mutate: func(cfg *ModelSelectionConfig) {
				cfg.OutputHeaders.Score = " X-LUNARGATE-COMPLEXITY "
			},
			wantErr: "model_selection.output_headers.score duplicates",
		},
		{
			name: "non-positive first tier",
			mutate: func(cfg *ModelSelectionConfig) {
				cfg.ComplexityTiers.Tier01Max = 0
			},
			wantErr: "model_selection.complexity_tiers.tier_01_max",
		},
		{
			name: "equal middle tier",
			mutate: func(cfg *ModelSelectionConfig) {
				cfg.ComplexityTiers.Tier23Max = cfg.ComplexityTiers.Tier01Max
			},
			wantErr: "model_selection.complexity_tiers.tier_23_max",
		},
		{
			name: "descending final tier",
			mutate: func(cfg *ModelSelectionConfig) {
				cfg.ComplexityTiers.Tier45Max = cfg.ComplexityTiers.Tier23Max - 1
			},
			wantErr: "model_selection.complexity_tiers.tier_45_max",
		},
		{
			name: "negative character bound",
			mutate: func(cfg *ModelSelectionConfig) {
				cfg.Complexity.Simple.MaxUserChars = &negative
			},
			wantErr: "model_selection.complexity.simple.max_user_chars",
		},
		{
			name: "inverted character range",
			mutate: func(cfg *ModelSelectionConfig) {
				cfg.Complexity.Simple.MinUserChars = &two
				cfg.Complexity.Simple.MaxUserChars = &one
			},
			wantErr: "min_user_chars must not exceed max_user_chars",
		},
		{
			name: "inverted message range",
			mutate: func(cfg *ModelSelectionConfig) {
				cfg.Complexity.Complex.MinMessages = &two
				cfg.Complexity.Complex.MaxMessages = &one
			},
			wantErr: "min_messages must not exceed max_messages",
		},
		{
			name: "unknown any-of condition",
			mutate: func(cfg *ModelSelectionConfig) {
				cfg.Complexity.Simple.AnyOf = []string{"has_images"}
			},
			wantErr: "model_selection.complexity.simple.any_of[0]",
		},
		{
			name: "duplicate any-of condition",
			mutate: func(cfg *ModelSelectionConfig) {
				cfg.Complexity.Simple.AnyOf = []string{"has_tools", " HAS_TOOLS "}
			},
			wantErr: "model_selection.complexity.simple.any_of[1] is duplicated",
		},
		{
			name: "contradictory tools conditions",
			mutate: func(cfg *ModelSelectionConfig) {
				cfg.Complexity.Simple.AnyOf = []string{"has_tools"}
				cfg.Complexity.Simple.RequireNoTools = true
			},
			wantErr: "require_no_tools",
		},
		{
			name: "contradictory JSON conditions",
			mutate: func(cfg *ModelSelectionConfig) {
				cfg.Complexity.Simple.AnyOf = []string{"requires_json"}
				cfg.Complexity.Simple.RequireNoJSON = true
			},
			wantErr: "require_no_json",
		},
		{
			name: "negative message bound",
			mutate: func(cfg *ModelSelectionConfig) {
				cfg.Complexity.Simple.MinMessages = &negative
				cfg.Complexity.Simple.MaxMessages = &zero
			},
			wantErr: "model_selection.complexity.simple.min_messages",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cfg := validRuntimeConfig()
			cfg.ModelSelect = validModelSelectionConfig()
			test.mutate(&cfg.ModelSelect)

			err := validateConfig(cfg)
			if err == nil {
				t.Fatal("validateConfig returned nil error")
			}
			if !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("validateConfig error = %q, want substring %q", err, test.wantErr)
			}
		})
	}
}

func TestValidateModelSelectionConfigAcceptsDisabledDraftRules(t *testing.T) {
	cfg := validRuntimeConfig()
	cfg.ModelSelect = ModelSelectionConfig{
		Enabled: false,
		Skills:  []ModelSelectionSkillRule{{RegexAny: []string{"("}}},
	}

	if err := validateConfig(cfg); err != nil {
		t.Fatalf("validateConfig returned error for disabled draft rules: %v", err)
	}
}

func TestNewManagerRejectsInvalidModelSelectionRegex(t *testing.T) {
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	configBody := `providers:
  openai:
    api_key: test-key
routing:
  routes:
    - name: default
      targets:
        - provider: openai
          weight: 1
model_selection:
  enabled: true
  skills:
    - name: coding
      regex_any: ["("]
`
	if err := os.WriteFile(configPath, []byte(configBody), 0o600); err != nil {
		t.Fatalf("write config: %v", err)
	}

	_, err := NewManager(configPath)
	if err == nil {
		t.Fatal("NewManager returned nil error for invalid model selection regex")
	}
	if !strings.Contains(err.Error(), "model_selection.skills[0].regex_any[0]") {
		t.Fatalf("NewManager error = %q", err)
	}
}

func validModelSelectionConfig() ModelSelectionConfig {
	return ModelSelectionConfig{
		Enabled: true,
		OutputHeaders: ModelSelectionOutputHeaders{
			Complexity: "x-lunargate-complexity",
			Score:      "x-lunargate-complexity-score",
			Skill:      "x-lunargate-skill",
		},
		ComplexityTiers: ModelSelectionComplexityTiersConfig{
			Tier01Max: 1,
			Tier23Max: 3,
			Tier45Max: 5,
		},
		Skills: []ModelSelectionSkillRule{{
			Name:     "coding",
			RegexAny: []string{"(?i)code"},
		}},
	}
}

func equalStrings(left, right []string) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}
