package config

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/fsnotify/fsnotify"
	"github.com/rs/zerolog/log"
	"github.com/spf13/viper"
)

const defaultDataSharingBackendURL = "https://api.lunargate.ai/v1"

// Config holds the entire gateway configuration.
type Config struct {
	Server        ServerConfig              `mapstructure:"server"`
	Providers     map[string]ProviderConfig `mapstructure:"providers"`
	Routing       RoutingConfig             `mapstructure:"routing"`
	ModelSelect   ModelSelectionConfig      `mapstructure:"model_selection"`
	RateLimit     RateLimitConfig           `mapstructure:"rate_limiting"`
	Cache         CacheConfig               `mapstructure:"caching"`
	Retry         RetryConfig               `mapstructure:"retry"`
	Logging       LoggingConfig             `mapstructure:"logging"`
	Security      SecurityConfig            `mapstructure:"security"`
	DataSharing   DataSharingConfig         `mapstructure:"data_sharing"`
}

type ServerConfig struct {
	Host         string        `mapstructure:"host"`
	Port         int           `mapstructure:"port"`
	ReadTimeout  time.Duration `mapstructure:"read_timeout"`
	WriteTimeout time.Duration `mapstructure:"write_timeout"`
	IdleTimeout  time.Duration `mapstructure:"idle_timeout"`
}

func (s ServerConfig) Address() string {
	return fmt.Sprintf("%s:%d", s.Host, s.Port)
}

type ProviderConfig struct {
	Type         string               `mapstructure:"type"`
	APIKey       string               `mapstructure:"api_key"`
	BaseURL      string               `mapstructure:"base_url"`
	DefaultModel string               `mapstructure:"default_model"`
	Organization string               `mapstructure:"organization"`
	APIVersion   string               `mapstructure:"api_version"`
	Extra        map[string]string    `mapstructure:"extra"`
	Models       ProviderModelsConfig `mapstructure:"models"`
}

type ProviderModelsConfig struct {
	Mode   string            `mapstructure:"mode"`
	Static []string          `mapstructure:"static"`
	Fetch  ModelsFetchConfig `mapstructure:"fetch"`
}

type ModelsFetchConfig struct {
	TTL time.Duration `mapstructure:"ttl"`
}

type ModelSelectionConfig struct {
	Enabled           bool                                  `mapstructure:"enabled"`
	OverrideUserModel bool                                  `mapstructure:"override_user_model"`
	OutputHeaders     ModelSelectionOutputHeaders           `mapstructure:"output_headers"`
	ComplexityScoring ModelSelectionComplexityScoringConfig `mapstructure:"complexity_scoring"`
	ComplexityTiers   ModelSelectionComplexityTiersConfig   `mapstructure:"complexity_tiers"`
	Complexity        ModelSelectionComplexityRules         `mapstructure:"complexity"`
	Skills            []ModelSelectionSkillRule             `mapstructure:"skills"`
}

type ModelSelectionOutputHeaders struct {
	Complexity string `mapstructure:"complexity"`
	Score      string `mapstructure:"score"`
	Skill      string `mapstructure:"skill"`
}

type ModelSelectionComplexityScoringConfig struct {
	InputTokensThreshold    int `mapstructure:"input_tokens_threshold"`
	WeightInputTokens       int `mapstructure:"weight_input_tokens"`
	WeightContainsCode      int `mapstructure:"weight_contains_code"`
	WeightMathReasoning     int `mapstructure:"weight_math_reasoning"`
	WeightAnalysisSynthesis int `mapstructure:"weight_analysis_synthesis"`
	WeightSafetySensitive   int `mapstructure:"weight_safety_sensitive"`
	WeightTools             int `mapstructure:"weight_tools"`
}

type ModelSelectionComplexityTiersConfig struct {
	Tier01Max int `mapstructure:"tier_01_max"`
	Tier23Max int `mapstructure:"tier_23_max"`
	Tier45Max int `mapstructure:"tier_45_max"`
}

type ModelSelectionComplexityRules struct {
	Simple  ModelSelectionComplexityRule `mapstructure:"simple"`
	Complex ModelSelectionComplexityRule `mapstructure:"complex"`
}

type ModelSelectionComplexityRule struct {
	MaxUserChars   *int     `mapstructure:"max_user_chars"`
	MinUserChars   *int     `mapstructure:"min_user_chars"`
	MaxMessages    *int     `mapstructure:"max_messages"`
	MinMessages    *int     `mapstructure:"min_messages"`
	AnyOf          []string `mapstructure:"any_of"`
	RequireNoTools bool     `mapstructure:"require_no_tools"`
	RequireNoJSON  bool     `mapstructure:"require_no_json"`
}

type ModelSelectionSkillRule struct {
	Name     string   `mapstructure:"name"`
	RegexAny []string `mapstructure:"regex_any"`
}

type RoutingConfig struct {
	DefaultStrategy string        `mapstructure:"default_strategy"`
	Routes          []RouteConfig `mapstructure:"routes"`
}

type RouteConfig struct {
	Name     string         `mapstructure:"name"`
	Match    MatchConfig    `mapstructure:"match"`
	Targets  []TargetConfig `mapstructure:"targets"`
	Fallback []TargetConfig `mapstructure:"fallback"`
}

type MatchConfig struct {
	Path       string            `mapstructure:"path"`
	Headers    map[string]string `mapstructure:"headers"`
	Conditions []ConditionConfig `mapstructure:"conditions"`
}

type ConditionConfig struct {
	Field    string `mapstructure:"field"`
	Operator string `mapstructure:"operator"`
	Value    string `mapstructure:"value"`
}

type TargetConfig struct {
	Provider string `mapstructure:"provider"`
	Model    string `mapstructure:"model"`
	Weight   int    `mapstructure:"weight"`
}

type RateLimitConfig struct {
	Enabled           bool `mapstructure:"enabled"`
	RequestsPerMinute int  `mapstructure:"requests_per_minute"`
	BurstSize         int  `mapstructure:"burst_size"`
}

type CacheConfig struct {
	Enabled bool          `mapstructure:"enabled"`
	TTL     time.Duration `mapstructure:"ttl"`
	MaxSize int           `mapstructure:"max_size"`
}

type RetryConfig struct {
	Enabled         bool          `mapstructure:"enabled"`
	MaxAttempts     int           `mapstructure:"max_attempts"`
	InitialDelay    time.Duration `mapstructure:"initial_delay"`
	MaxDelay        time.Duration `mapstructure:"max_delay"`
	Multiplier      float64       `mapstructure:"multiplier"`
	JitterFactor    float64       `mapstructure:"jitter_factor"`
	RetryableErrors []int         `mapstructure:"retryable_errors"`
}

type LoggingConfig struct {
	Level  string `mapstructure:"level"`
	Format string `mapstructure:"format"`
}

type SecurityConfig struct {
	APIKeysEnabled bool     `mapstructure:"api_keys_enabled"`
	APIKeys        []string `mapstructure:"api_keys"`
}

// DataSharingConfig controls what request/response data the gateway forwards
// to the SaaS backend. When disabled (default), ONLY metrics are sent (zero data leakage).
// When enabled, prompts and/or responses can be forwarded for log inspection in the dashboard.
type DataSharingConfig struct {
	Enabled        bool   `mapstructure:"enabled"`
	SharePrompts   bool   `mapstructure:"share_prompts"`
	ShareResponses bool   `mapstructure:"share_responses"`
	BackendURL     string `mapstructure:"backend_url"`
	GatewayID      string `mapstructure:"gateway_id"`
	APIKey         string `mapstructure:"api_key"`
	GatewayLat     string `mapstructure:"gateway_lat"`
	GatewayLon     string `mapstructure:"gateway_lon"`
	RemoteControl  bool   `mapstructure:"remote_control"`
}

// Manager handles config loading, validation, and hot-reloading.
type Manager struct {
	path     string
	current  atomic.Value // stores *Config
	onChange []func(*Config)
	v        *viper.Viper
}

// NewManager creates a new config manager and loads the initial config.
func NewManager(path string) (*Manager, error) {
	m := &Manager{
		path: path,
		v:    viper.New(),
	}

	if err := loadDotEnv(path); err != nil {
		return nil, fmt.Errorf("failed to load .env: %w", err)
	}

	m.setDefaults()
	m.v.SetConfigFile(path)

	// Enable env var expansion
	m.v.AutomaticEnv()
	m.v.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))

	if err := m.load(); err != nil {
		return nil, fmt.Errorf("failed to load config: %w", err)
	}

	return m, nil
}

func (m *Manager) setDefaults() {
	m.v.SetDefault("server.host", "0.0.0.0")
	m.v.SetDefault("server.port", 8080)
	m.v.SetDefault("server.read_timeout", "30s")
	m.v.SetDefault("server.write_timeout", "0s")
	m.v.SetDefault("server.idle_timeout", "60s")

	m.v.SetDefault("routing.default_strategy", "round-robin")

	m.v.SetDefault("rate_limiting.enabled", false)
	m.v.SetDefault("rate_limiting.requests_per_minute", 60)
	m.v.SetDefault("rate_limiting.burst_size", 10)

	m.v.SetDefault("caching.enabled", false)
	m.v.SetDefault("caching.ttl", "1h")
	m.v.SetDefault("caching.max_size", 1000)

	m.v.SetDefault("retry.enabled", true)
	m.v.SetDefault("retry.max_attempts", 3)
	m.v.SetDefault("retry.initial_delay", "1s")
	m.v.SetDefault("retry.max_delay", "30s")
	m.v.SetDefault("retry.multiplier", 2.0)
	m.v.SetDefault("retry.jitter_factor", 0.2)
	m.v.SetDefault("retry.retryable_errors", []int{429, 500, 502, 503, 504})

	m.v.SetDefault("logging.level", "info")
	m.v.SetDefault("logging.format", "console")

	m.v.SetDefault("model_selection.enabled", false)
	m.v.SetDefault("model_selection.override_user_model", false)
	m.v.SetDefault("model_selection.output_headers.complexity", "x-lunargate-complexity")
	m.v.SetDefault("model_selection.output_headers.score", "x-lunargate-complexity-score")
	m.v.SetDefault("model_selection.output_headers.skill", "x-lunargate-skill")
	m.v.SetDefault("model_selection.complexity_scoring.input_tokens_threshold", 2000)
	m.v.SetDefault("model_selection.complexity_scoring.weight_input_tokens", 2)
	m.v.SetDefault("model_selection.complexity_scoring.weight_contains_code", 2)
	m.v.SetDefault("model_selection.complexity_scoring.weight_math_reasoning", 2)
	m.v.SetDefault("model_selection.complexity_scoring.weight_analysis_synthesis", 1)
	m.v.SetDefault("model_selection.complexity_scoring.weight_safety_sensitive", 2)
	m.v.SetDefault("model_selection.complexity_scoring.weight_tools", 2)
	m.v.SetDefault("model_selection.complexity_tiers.tier_01_max", 1)
	m.v.SetDefault("model_selection.complexity_tiers.tier_23_max", 3)
	m.v.SetDefault("model_selection.complexity_tiers.tier_45_max", 5)

	m.v.SetDefault("data_sharing.enabled", false)
	m.v.SetDefault("data_sharing.share_prompts", false)
	m.v.SetDefault("data_sharing.share_responses", false)
	m.v.SetDefault("data_sharing.backend_url", defaultDataSharingBackendURL)
	m.v.SetDefault("data_sharing.gateway_id", "")
	m.v.SetDefault("data_sharing.api_key", "")
	m.v.SetDefault("data_sharing.remote_control", false)
}

func (m *Manager) load() error {
	if err := m.v.ReadInConfig(); err != nil {
		return fmt.Errorf("failed to read config file: %w", err)
	}

	cfg := &Config{}
	if err := m.v.Unmarshal(cfg); err != nil {
		return fmt.Errorf("failed to unmarshal config: %w", err)
	}

	// Expand environment variables in provider API keys
	for name, p := range cfg.Providers {
		p.APIKey = expandEnv(p.APIKey)
		p.BaseURL = expandEnv(p.BaseURL)
		p.Organization = expandEnv(p.Organization)
		cfg.Providers[name] = p
	}

	cfg.DataSharing.BackendURL = expandEnv(cfg.DataSharing.BackendURL)
	cfg.DataSharing.BackendURL = strings.TrimRight(strings.TrimSpace(cfg.DataSharing.BackendURL), "/")
	if strings.HasSuffix(cfg.DataSharing.BackendURL, "/collector") {
		cfg.DataSharing.BackendURL = strings.TrimSuffix(cfg.DataSharing.BackendURL, "/collector")
	}
	if strings.TrimSpace(cfg.DataSharing.BackendURL) == "" {
		cfg.DataSharing.BackendURL = defaultDataSharingBackendURL
	}
	cfg.DataSharing.GatewayID = expandEnv(cfg.DataSharing.GatewayID)
	cfg.DataSharing.APIKey = expandEnv(cfg.DataSharing.APIKey)

	m.current.Store(cfg)
	return nil
}

// expandEnv replaces ${VAR} patterns with environment variable values.
func expandEnv(s string) string {
	if strings.Contains(s, "${") {
		return os.ExpandEnv(s)
	}
	return s
}

func loadDotEnv(configPath string) error {
	for _, candidate := range dotEnvCandidates(configPath) {
		if err := loadDotEnvFile(candidate); err != nil {
			return err
		}
	}
	return nil
}

func dotEnvCandidates(configPath string) []string {
	seen := map[string]struct{}{}
	candidates := make([]string, 0, 2)

	add := func(path string) {
		if path == "" {
			return
		}
		clean := filepath.Clean(path)
		if _, ok := seen[clean]; ok {
			return
		}
		seen[clean] = struct{}{}
		candidates = append(candidates, clean)
	}

	add(filepath.Join(filepath.Dir(configPath), ".env"))

	if wd, err := os.Getwd(); err == nil {
		add(filepath.Join(wd, ".env"))
	}

	return candidates
}

func loadDotEnvFile(path string) error {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for lineNo := 1; scanner.Scan(); lineNo++ {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if strings.HasPrefix(line, "export ") {
			line = strings.TrimSpace(strings.TrimPrefix(line, "export "))
		}

		key, value, ok := strings.Cut(line, "=")
		if !ok {
			return fmt.Errorf("invalid .env entry in %s at line %d", path, lineNo)
		}

		key = strings.TrimSpace(key)
		if key == "" {
			return fmt.Errorf("empty .env key in %s at line %d", path, lineNo)
		}
		if _, exists := os.LookupEnv(key); exists {
			continue
		}

		value = strings.TrimSpace(value)
		if len(value) >= 2 {
			switch {
			case value[0] == '"' && value[len(value)-1] == '"':
				unquoted, err := strconv.Unquote(value)
				if err != nil {
					return fmt.Errorf("invalid quoted .env value for %s in %s at line %d: %w", key, path, lineNo, err)
				}
				value = unquoted
			case value[0] == '\'' && value[len(value)-1] == '\'':
				value = value[1 : len(value)-1]
			}
		}

		if err := os.Setenv(key, os.ExpandEnv(value)); err != nil {
			return fmt.Errorf("set env %s from %s: %w", key, path, err)
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("read %s: %w", path, err)
	}

	return nil
}

// Get returns the current config (lock-free read).
func (m *Manager) Get() *Config {
	return m.current.Load().(*Config)
}

// OnChange registers a callback for config changes.
func (m *Manager) OnChange(fn func(*Config)) {
	m.onChange = append(m.onChange, fn)
}

// WatchChanges starts watching the config file for changes and hot-reloads.
func (m *Manager) WatchChanges() {
	m.v.OnConfigChange(func(e fsnotify.Event) {
		log.Info().Str("file", e.Name).Msg("config file changed, reloading")

		if err := m.load(); err != nil {
			log.Error().Err(err).Msg("failed to reload config")
			return
		}

		newCfg := m.Get()
		for _, fn := range m.onChange {
			fn(newCfg)
		}

		log.Info().Msg("config reloaded successfully")
	})
	m.v.WatchConfig()
}
