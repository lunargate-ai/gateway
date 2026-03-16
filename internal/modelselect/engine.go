package modelselect

import (
	"regexp"
	"strconv"
	"strings"
	"sync/atomic"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/pkg/models"
	"github.com/rs/zerolog/log"
)

type compiledSkillRule struct {
	name  string
	rexes []*regexp.Regexp
}

type compiledConfig struct {
	cfg    config.ModelSelectionConfig
	skills []compiledSkillRule
}

type Engine struct {
	current atomic.Value
}

func NewEngine(cfg config.ModelSelectionConfig) *Engine {
	e := &Engine{}
	e.UpdateConfig(cfg)
	return e
}

func (e *Engine) UpdateConfig(cfg config.ModelSelectionConfig) {
	cc := &compiledConfig{cfg: cfg}

	if cfg.Enabled {
		cc.skills = make([]compiledSkillRule, 0, len(cfg.Skills))
		for _, sr := range cfg.Skills {
			name := strings.TrimSpace(sr.Name)
			if name == "" {
				continue
			}
			compiled := compiledSkillRule{name: name}
			for _, pat := range sr.RegexAny {
				p := strings.TrimSpace(pat)
				if p == "" {
					continue
				}
				r, err := regexp.Compile(p)
				if err != nil {
					log.Warn().Err(err).Str("skill", name).Msg("invalid skill regex, skipping")
					continue
				}
				compiled.rexes = append(compiled.rexes, r)
			}
			if len(compiled.rexes) == 0 {
				continue
			}
			cc.skills = append(cc.skills, compiled)
		}
	}

	e.current.Store(cc)
}

func (e *Engine) Config() config.ModelSelectionConfig {
	v, ok := e.current.Load().(*compiledConfig)
	if !ok || v == nil {
		return config.ModelSelectionConfig{}
	}
	return v.cfg
}

func (e *Engine) Enabled() bool {
	return e.Config().Enabled
}

func (e *Engine) EnrichHeaders(req *models.UnifiedRequest, headers map[string]string) (complexity string, skill string) {
	v, ok := e.current.Load().(*compiledConfig)
	if !ok || v == nil {
		return "", ""
	}
	cfg := v.cfg
	if !cfg.Enabled {
		return "", ""
	}

	userChars := 0
	userText := make([]string, 0, 4)
	msgCount := 0
	if req != nil {
		msgCount = len(req.Messages)
		for i := range req.Messages {
			m := req.Messages[i]
			if m.Role != "user" {
				continue
			}
			s := m.ContentString()
			if s == "" {
				continue
			}
			userChars += len(s)
			userText = append(userText, s)
		}
	}

	hasTools := req != nil && (len(req.Tools) > 0 || req.ToolChoice != nil)
	requiresJSON := false
	if req != nil && req.ResponseFormat != nil {
		t := strings.TrimSpace(req.ResponseFormat.Type)
		requiresJSON = t == "json" || t == "json_object"
	}

	score := classifyComplexityScore(cfg.ComplexityScoring, userChars, userText, hasTools)
	complexity = classifyComplexityTier(cfg.ComplexityTiers, score)
	if complexity == "" {
		complexity = classifyComplexity(cfg.Complexity, userChars, msgCount, hasTools, requiresJSON)
	}
	skill = classifySkill(v.skills, userText)

	if strings.TrimSpace(skill) == "" {
		skill = "default"
	}

	if headers != nil {
		if k := strings.TrimSpace(cfg.OutputHeaders.Complexity); k != "" && strings.TrimSpace(complexity) != "" {
			headers[strings.ToLower(k)] = complexity
		}
		if k := strings.TrimSpace(cfg.OutputHeaders.Score); k != "" {
			headers[strings.ToLower(k)] = strconv.Itoa(score)
		}
		if k := strings.TrimSpace(cfg.OutputHeaders.Skill); k != "" && strings.TrimSpace(skill) != "" {
			headers[strings.ToLower(k)] = skill
		}
		// Add tools requirement header for routing
		if hasTools {
			headers["x-lunargate-requires-tools"] = "true"
		}
	}

	return complexity, skill
}

func classifyComplexityScore(cfg config.ModelSelectionComplexityScoringConfig, userChars int, userText []string, hasTools bool) int {
	score := 0
	threshold := cfg.InputTokensThreshold
	if threshold <= 0 {
		threshold = 2000
	}
	approxTokens := userChars / 4
	if approxTokens > threshold {
		score += clampWeight(cfg.WeightInputTokens, 2)
	}

	combined := strings.ToLower(strings.Join(userText, "\n"))
	if combined != "" {
		if detectContainsCode(combined) {
			score += clampWeight(cfg.WeightContainsCode, 2)
		}
		if detectMathReasoning(combined) {
			score += clampWeight(cfg.WeightMathReasoning, 2)
		}
		if detectAnalysisSynthesis(combined) {
			score += clampWeight(cfg.WeightAnalysisSynthesis, 1)
		}
		if detectSafetySensitive(combined) {
			score += clampWeight(cfg.WeightSafetySensitive, 2)
		}
	}

	if hasTools {
		score += clampWeight(cfg.WeightTools, 2)
	}

	if score < 0 {
		return 0
	}
	return score
}

func classifyComplexityTier(cfg config.ModelSelectionComplexityTiersConfig, score int) string {
	max01 := cfg.Tier01Max
	max23 := cfg.Tier23Max
	max45 := cfg.Tier45Max
	if max01 <= 0 {
		max01 = 1
	}
	if max23 <= 0 {
		max23 = 3
	}
	if max45 <= 0 {
		max45 = 5
	}
	if score <= max01 {
		return "0-1"
	}
	if score <= max23 {
		return "2-3"
	}
	if score <= max45 {
		return "4-5"
	}
	return "6+"
}

func clampWeight(w int, def int) int {
	if w == 0 {
		return def
	}
	if w < 0 {
		return 0
	}
	return w
}

func detectContainsCode(s string) bool {
	if strings.Contains(s, "```") {
		return true
	}
	for _, kw := range []string{"func ", "package ", "import ", "struct ", "interface ", "class ", "def ", "const ", "var ", "let ", "return ", "#include", "public ", "private ", "select ", "from ", "where ", "dockerfile", "kubernetes", "kubectl", "helm ", "terraform"} {
		if strings.Contains(s, kw) {
			return true
		}
	}
	if strings.Contains(s, "{\n") || strings.Contains(s, ";\n") {
		return true
	}
	return false
}

func detectMathReasoning(s string) bool {
	for _, kw := range []string{"prove", "derive", "calculate", "solve", "equation", "theorem", "integral", "derivative", "matrix", "vector", "probability", "statistics", "algorithm", "complexity", "o(", "big o"} {
		if strings.Contains(s, kw) {
			return true
		}
	}
	for _, sym := range []string{"=", "!=", ">=", "<=", "\u2211", "\u222b", "\u2202"} {
		if strings.Contains(s, sym) {
			return true
		}
	}
	return false
}

func detectAnalysisSynthesis(s string) bool {
	for _, kw := range []string{"analy", "compare", "trade-off", "tradeoff", "pros and cons", "evaluate", "synthes", "summar", "plan", "design", "architecture", "recommend"} {
		if strings.Contains(s, kw) {
			return true
		}
	}
	return false
}

func detectSafetySensitive(s string) bool {
	for _, kw := range []string{"exploit", "vulnerability", "attack", "phishing", "malware", "ransomware", "ddos", "sql injection", "xss", "csrf", "weapon", "bomb", "poison", "suicide", "self-harm", "drugs", "fraud", "steal"} {
		if strings.Contains(s, kw) {
			return true
		}
	}
	return false
}

func classifySkill(rules []compiledSkillRule, userText []string) string {
	if len(rules) == 0 || len(userText) == 0 {
		return ""
	}
	for _, rule := range rules {
		for _, txt := range userText {
			for _, rx := range rule.rexes {
				if rx.MatchString(txt) {
					return rule.name
				}
			}
		}
	}
	return ""
}

func classifyComplexity(
	rules config.ModelSelectionComplexityRules,
	userChars int,
	msgCount int,
	hasTools bool,
	requiresJSON bool,
) string {
	// If no rules are configured, use sane defaults.
	isEmpty := rules.Simple.MaxUserChars == nil && rules.Simple.MinUserChars == nil && rules.Simple.MaxMessages == nil && rules.Simple.MinMessages == nil && len(rules.Simple.AnyOf) == 0 && !rules.Simple.RequireNoTools && !rules.Simple.RequireNoJSON &&
		rules.Complex.MaxUserChars == nil && rules.Complex.MinUserChars == nil && rules.Complex.MaxMessages == nil && rules.Complex.MinMessages == nil && len(rules.Complex.AnyOf) == 0 && !rules.Complex.RequireNoTools && !rules.Complex.RequireNoJSON

	if isEmpty {
		if userChars <= 800 && msgCount <= 6 && !hasTools && !requiresJSON {
			return "simple"
		}
		return "complex"
	}

	if matchesComplexityRule(rules.Complex, userChars, msgCount, hasTools, requiresJSON) {
		return "complex"
	}
	if matchesComplexityRule(rules.Simple, userChars, msgCount, hasTools, requiresJSON) {
		return "simple"
	}

	return "simple"
}

func matchesComplexityRule(
	rule config.ModelSelectionComplexityRule,
	userChars int,
	msgCount int,
	hasTools bool,
	requiresJSON bool,
) bool {
	if rule.MaxUserChars != nil && userChars > *rule.MaxUserChars {
		return false
	}
	if rule.MinUserChars != nil && userChars < *rule.MinUserChars {
		return false
	}
	if rule.MaxMessages != nil && msgCount > *rule.MaxMessages {
		return false
	}
	if rule.MinMessages != nil && msgCount < *rule.MinMessages {
		return false
	}
	if rule.RequireNoTools && hasTools {
		return false
	}
	if rule.RequireNoJSON && requiresJSON {
		return false
	}
	if len(rule.AnyOf) > 0 {
		ok := false
		for _, cond := range rule.AnyOf {
			switch strings.ToLower(strings.TrimSpace(cond)) {
			case "has_tools":
				if hasTools {
					ok = true
				}
			case "requires_json":
				if requiresJSON {
					ok = true
				}
			}
		}
		if !ok {
			return false
		}
	}
	return true
}
