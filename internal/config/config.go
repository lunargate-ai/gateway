package config

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
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
	Server      ServerConfig              `mapstructure:"server"`
	Providers   map[string]ProviderConfig `mapstructure:"providers"`
	Routing     RoutingConfig             `mapstructure:"routing"`
	ModelSelect ModelSelectionConfig      `mapstructure:"model_selection"`
	RateLimit   RateLimitConfig           `mapstructure:"rate_limiting"`
	Cache       CacheConfig               `mapstructure:"caching"`
	Retry       RetryConfig               `mapstructure:"retry"`
	Logging     LoggingConfig             `mapstructure:"logging"`
	Security    SecurityConfig            `mapstructure:"security"`
	DataSharing DataSharingConfig         `mapstructure:"data_sharing"`
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
	Type                   string               `mapstructure:"type"`
	APIKey                 string               `mapstructure:"api_key"`
	BaseURL                string               `mapstructure:"base_url"`
	DefaultModel           string               `mapstructure:"default_model"`
	Temperature            *float64             `mapstructure:"temperature"`
	TopP                   *float64             `mapstructure:"top_p"`
	TopK                   *int                 `mapstructure:"top_k"`
	Organization           string               `mapstructure:"organization"`
	APIVersion             string               `mapstructure:"api_version"`
	Timeout                time.Duration        `mapstructure:"timeout"`
	TimeoutMode            string               `mapstructure:"timeout_mode"`
	CompatibilityProfile   string               `mapstructure:"compatibility_profile"`
	NormalizeDeveloperRole bool                 `mapstructure:"normalize_developer_role"`
	Extra                  map[string]string    `mapstructure:"extra"`
	Models                 ProviderModelsConfig `mapstructure:"models"`
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
	Path    string            `mapstructure:"path"`
	Headers map[string]string `mapstructure:"headers"`
}

type TargetConfig struct {
	Provider            string `mapstructure:"provider"`
	Model               string `mapstructure:"model"`
	Weight              int    `mapstructure:"weight"`
	UpstreamRequestType string `mapstructure:"upstream_request_type"`
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
	Enabled  bool               `mapstructure:"enabled"`
	Provider string             `mapstructure:"provider"`
	APIKey   APIKeyAuthConfig   `mapstructure:"api_key"`
	External ExternalAuthConfig `mapstructure:"external"`

	// Deprecated compatibility fields. Prefer `security.enabled`,
	// `security.provider`, and `security.api_key.*`.
	APIKeysEnabled bool     `mapstructure:"api_keys_enabled"`
	APIKeys        []string `mapstructure:"api_keys"`
}

type APIKeyAuthConfig struct {
	Header       string             `mapstructure:"header"`
	Prefix       string             `mapstructure:"prefix"`
	AllowXAPIKey bool               `mapstructure:"allow_x_api_key"`
	Keys         []APIKeyCredential `mapstructure:"keys"`
}

type APIKeyCredential struct {
	Name  string `mapstructure:"name"`
	Value string `mapstructure:"value"`
}

type ExternalAuthConfig struct {
	Type             string        `mapstructure:"type"`
	JWKSURL          string        `mapstructure:"jwks_url"`
	IntrospectionURL string        `mapstructure:"introspection_url"`
	Issuer           string        `mapstructure:"issuer"`
	Audience         []string      `mapstructure:"audience"`
	Timeout          time.Duration `mapstructure:"timeout"`
}

// DataSharingConfig controls what request/response data the gateway forwards
// to the SaaS backend. When disabled (default), ONLY metrics are sent (zero data leakage).
// When enabled, prompts and/or responses can be forwarded for log inspection in the dashboard.
type DataSharingConfig struct {
	Enabled        bool   `mapstructure:"enabled"`
	SharePrompts   bool   `mapstructure:"share_prompts"`
	ShareResponses bool   `mapstructure:"share_responses"`
	BackendURL     string `mapstructure:"backend_url"`
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

	m.v.SetDefault("security.enabled", false)
	m.v.SetDefault("security.provider", "none")
	m.v.SetDefault("security.api_key.header", "Authorization")
	m.v.SetDefault("security.api_key.prefix", "Bearer")
	m.v.SetDefault("security.api_key.allow_x_api_key", true)
	m.v.SetDefault("security.external.timeout", "5s")

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

	expandConfigEnv(cfg)
	normalizeSecurityConfig(cfg)
	if err := validateConfig(cfg); err != nil {
		return fmt.Errorf("invalid config: %w", err)
	}

	cfg.DataSharing.BackendURL = expandEnv(cfg.DataSharing.BackendURL)
	cfg.DataSharing.BackendURL = strings.TrimRight(strings.TrimSpace(cfg.DataSharing.BackendURL), "/")
	if strings.HasSuffix(cfg.DataSharing.BackendURL, "/collector") {
		cfg.DataSharing.BackendURL = strings.TrimSuffix(cfg.DataSharing.BackendURL, "/collector")
	}
	if strings.TrimSpace(cfg.DataSharing.BackendURL) == "" {
		cfg.DataSharing.BackendURL = defaultDataSharingBackendURL
	}
	cfg.DataSharing.APIKey = expandEnv(cfg.DataSharing.APIKey)

	m.current.Store(cfg)
	return nil
}

func normalizeSecurityConfig(cfg *Config) {
	if cfg == nil {
		return
	}

	securityCfg := &cfg.Security

	if securityCfg.APIKeysEnabled {
		securityCfg.Enabled = true
	}

	securityCfg.Provider = strings.ToLower(strings.TrimSpace(securityCfg.Provider))
	if securityCfg.Provider == "" {
		switch {
		case len(securityCfg.APIKey.Keys) > 0 || len(securityCfg.APIKeys) > 0:
			securityCfg.Provider = "api_key"
		default:
			securityCfg.Provider = "none"
		}
	}
	if securityCfg.Provider == "none" && (securityCfg.APIKeysEnabled || len(securityCfg.APIKeys) > 0) {
		securityCfg.Provider = "api_key"
	}

	if len(securityCfg.APIKey.Keys) == 0 && len(securityCfg.APIKeys) > 0 {
		securityCfg.APIKey.Keys = make([]APIKeyCredential, 0, len(securityCfg.APIKeys))
		for idx, value := range securityCfg.APIKeys {
			securityCfg.APIKey.Keys = append(securityCfg.APIKey.Keys, APIKeyCredential{
				Name:  fmt.Sprintf("legacy-key-%d", idx+1),
				Value: value,
			})
		}
	}

	if strings.TrimSpace(securityCfg.APIKey.Header) == "" {
		securityCfg.APIKey.Header = "Authorization"
	}
	if strings.EqualFold(strings.TrimSpace(securityCfg.APIKey.Header), "Authorization") &&
		strings.TrimSpace(securityCfg.APIKey.Prefix) == "" {
		securityCfg.APIKey.Prefix = "Bearer"
	}

	for idx := range securityCfg.APIKey.Keys {
		key := &securityCfg.APIKey.Keys[idx]
		key.Name = strings.TrimSpace(key.Name)
		key.Value = strings.TrimSpace(key.Value)
		if key.Name == "" {
			key.Name = fmt.Sprintf("key-%d", idx+1)
		}
	}

	if securityCfg.Provider == "api_key" && len(securityCfg.APIKey.Keys) > 0 {
		securityCfg.Enabled = true
	}
}

func validateConfig(cfg *Config) error {
	if cfg == nil {
		return nil
	}

	securityCfg := cfg.Security
	provider := strings.ToLower(strings.TrimSpace(securityCfg.Provider))
	if provider == "" {
		provider = "none"
	}

	if !securityCfg.Enabled && provider != "api_key" {
		return nil
	}

	switch provider {
	case "none":
		return nil
	case "api_key":
		if len(securityCfg.APIKey.Keys) == 0 {
			return fmt.Errorf("security.api_key.keys must contain at least one key when security.provider is api_key")
		}
		for idx, key := range securityCfg.APIKey.Keys {
			if strings.TrimSpace(key.Value) == "" {
				return fmt.Errorf("security.api_key.keys[%d].value must not be empty", idx)
			}
		}
		return nil
	case "external":
		if !securityCfg.Enabled {
			return nil
		}
		return fmt.Errorf("security.provider=external is reserved for future inbound auth integrations and is not implemented yet")
	default:
		return fmt.Errorf("unsupported security.provider %q", securityCfg.Provider)
	}
}

// expandEnv replaces ${VAR} patterns with environment variable values.
func expandEnv(s string) string {
	if strings.Contains(s, "${") {
		return os.ExpandEnv(s)
	}
	return s
}

func expandConfigEnv(cfg *Config) {
	if cfg == nil {
		return
	}

	reflect.ValueOf(cfg).Elem().Set(expandEnvValue(reflect.ValueOf(*cfg)))
}

func expandEnvValue(v reflect.Value) reflect.Value {
	if !v.IsValid() {
		return v
	}

	switch v.Kind() {
	case reflect.Pointer:
		if v.IsNil() {
			return v
		}
		out := reflect.New(v.Type().Elem())
		out.Elem().Set(expandEnvValue(v.Elem()))
		return out
	case reflect.Struct:
		out := reflect.New(v.Type()).Elem()
		out.Set(v)
		for i := 0; i < out.NumField(); i++ {
			field := out.Field(i)
			if field.CanSet() {
				field.Set(expandEnvValue(field))
			}
		}
		return out
	case reflect.Slice:
		if v.IsNil() {
			return v
		}
		out := reflect.MakeSlice(v.Type(), v.Len(), v.Len())
		for i := 0; i < v.Len(); i++ {
			out.Index(i).Set(expandEnvValue(v.Index(i)))
		}
		return out
	case reflect.Array:
		out := reflect.New(v.Type()).Elem()
		for i := 0; i < v.Len(); i++ {
			out.Index(i).Set(expandEnvValue(v.Index(i)))
		}
		return out
	case reflect.Map:
		if v.IsNil() {
			return v
		}
		out := reflect.MakeMapWithSize(v.Type(), v.Len())
		iter := v.MapRange()
		for iter.Next() {
			out.SetMapIndex(iter.Key(), expandEnvValue(iter.Value()))
		}
		return out
	case reflect.String:
		return reflect.ValueOf(expandEnv(v.String())).Convert(v.Type())
	default:
		return v
	}
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
