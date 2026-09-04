package config

import (
	"bufio"
	"bytes"
	"fmt"
	"math"
	"net"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/fsnotify/fsnotify"
	"github.com/lunargate-ai/gateway/internal/safeurl"
	"github.com/rs/zerolog/log"
	"github.com/spf13/viper"
	"gopkg.in/yaml.v3"
)

const (
	defaultBackendURL         = "https://api.lunargate.ai/v1"
	defaultUpdateCheckURL     = "https://get.lunargate.ai/latest"
	defaultUpdateCheckPeriod  = 24 * time.Hour
	defaultUpdateCheckTimeout = 3 * time.Second
	// DefaultModelsFetchTTL is the effective cache duration for fetch-mode discovery.
	DefaultModelsFetchTTL = 10 * time.Minute
	// MaxRetryAttempts bounds cumulative provider timeouts and retry load per request.
	MaxRetryAttempts = 10
)

// Config holds the entire gateway configuration.
type Config struct {
	General     GeneralConfig             `mapstructure:"general"`
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
	UpdateCheck UpdateCheckConfig         `mapstructure:"update_check"`
}

type GeneralConfig struct {
	APIKey     string `mapstructure:"api_key"`
	BackendURL string `mapstructure:"backend_url"`
}

type UpdateCheckConfig struct {
	Enabled  bool          `mapstructure:"enabled"`
	Endpoint string        `mapstructure:"endpoint"`
	Interval time.Duration `mapstructure:"interval"`
	Timeout  time.Duration `mapstructure:"timeout"`
}

type ServerConfig struct {
	Host         string        `mapstructure:"host"`
	Port         int           `mapstructure:"port"`
	ReadTimeout  time.Duration `mapstructure:"read_timeout"`
	WriteTimeout time.Duration `mapstructure:"write_timeout"`
	IdleTimeout  time.Duration `mapstructure:"idle_timeout"`
}

func (s ServerConfig) Address() string {
	host := strings.TrimSpace(s.Host)
	if normalized, err := normalizeServerHost(host); err == nil {
		host = normalized
	}
	return net.JoinHostPort(host, strconv.Itoa(s.Port))
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
	ExtractReasoningTags   bool                 `mapstructure:"extract_reasoning_tags"`
	Extra                  map[string]string    `mapstructure:"extra"`
	Models                 ProviderModelsConfig `mapstructure:"models"`
	Capabilities           ProviderCapabilities `mapstructure:"capabilities"`
}

// ProviderCapabilities declares optional API contracts that must never be
// inferred from a provider's type or URL. Zero values are deliberately safe.
type ProviderCapabilities struct {
	ChatCompletionsLifecycle bool `mapstructure:"chat_completions_lifecycle"`
	ResponsesLifecycle       bool `mapstructure:"responses_lifecycle"`
	Conversations            bool `mapstructure:"conversations"`
	BackgroundResponses      bool `mapstructure:"background_responses"`
	ResponseCancellation     bool `mapstructure:"response_cancellation"`
	ResponseCompaction       bool `mapstructure:"response_compaction"`
	ResponseInputTokens      bool `mapstructure:"response_input_tokens"`
	EmbeddingsBase64         bool `mapstructure:"embeddings_base64"`
	StructuredOutputs        bool `mapstructure:"structured_outputs"`
	ReasoningEffort          bool `mapstructure:"reasoning_effort"`
	// ReasoningEffortLevels narrows model-dependent levels. An empty list
	// enables only the common low, medium, and high levels.
	ReasoningEffortLevels []string `mapstructure:"reasoning_effort_levels"`
	// AdaptiveThinking permits thinking.type=adaptive in translated requests.
	AdaptiveThinking bool     `mapstructure:"adaptive_thinking"`
	HostedTools      []string `mapstructure:"hosted_tools"`
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
	// Byte limits account for the cache key and retained serialized response
	// buffers. They intentionally exclude fixed per-entry Go object overhead.
	MaxEntryBytes int `mapstructure:"max_entry_bytes"`
	MaxBytes      int `mapstructure:"max_bytes"`
}

const (
	DefaultCacheMaxEntryBytes = 16 << 20
	DefaultCacheMaxBytes      = 64 << 20
)

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

// DataSharingConfig controls all gateway communication with the SaaS backend.
// Enabled is the master switch for collection and remote control. Prompt and
// response forwarding remain independently opt-in when the master switch is on.
type DataSharingConfig struct {
	Enabled        bool   `mapstructure:"enabled"`
	SharePrompts   bool   `mapstructure:"share_prompts"`
	ShareResponses bool   `mapstructure:"share_responses"`
	APIKey         string `mapstructure:"api_key"`
	// BackendURL is a deprecated compatibility alias for general.backend_url.
	BackendURL    string `mapstructure:"backend_url"`
	GatewayLat    string `mapstructure:"gateway_lat"`
	GatewayLon    string `mapstructure:"gateway_lon"`
	RemoteControl bool   `mapstructure:"remote_control"`
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
	m.v.SetDefault("general.api_key", "")

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
	m.v.SetDefault("caching.max_entry_bytes", DefaultCacheMaxEntryBytes)
	m.v.SetDefault("caching.max_bytes", DefaultCacheMaxBytes)

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
	m.v.SetDefault("data_sharing.api_key", "")
	m.v.SetDefault("data_sharing.remote_control", false)

	m.v.SetDefault("update_check.enabled", true)
	m.v.SetDefault("update_check.endpoint", defaultUpdateCheckURL)
	m.v.SetDefault("update_check.interval", defaultUpdateCheckPeriod)
	m.v.SetDefault("update_check.timeout", defaultUpdateCheckTimeout)
}

func (m *Manager) load() error {
	rawConfig, err := os.ReadFile(m.path)
	if err != nil {
		return fmt.Errorf("failed to read config file: %w", err)
	}
	if isYAMLConfigPath(m.path) {
		if err := validateRoutingHeaderKeyDuplicates(rawConfig); err != nil {
			return fmt.Errorf("invalid config: %w", err)
		}
	}
	if err := m.v.ReadConfig(bytes.NewReader(rawConfig)); err != nil {
		return fmt.Errorf("failed to read config file: %w", err)
	}

	cfg := &Config{}
	if err := m.v.UnmarshalExact(cfg); err != nil {
		return fmt.Errorf("failed to unmarshal config: %w", err)
	}

	expandConfigEnv(cfg)
	normalizeProviderCapabilities(cfg)
	normalizeSecurityConfig(cfg)
	resolveGatewayAPIKey(cfg)
	cfg.UpdateCheck.Endpoint = strings.TrimSpace(cfg.UpdateCheck.Endpoint)
	if cfg.UpdateCheck.Endpoint == "" {
		cfg.UpdateCheck.Endpoint = defaultUpdateCheckURL
	}
	if err := validateConfig(cfg); err != nil {
		return fmt.Errorf("invalid config: %w", err)
	}

	if cfg.UpdateCheck.Interval <= 0 {
		cfg.UpdateCheck.Interval = defaultUpdateCheckPeriod
	}
	if cfg.UpdateCheck.Timeout <= 0 {
		cfg.UpdateCheck.Timeout = defaultUpdateCheckTimeout
	}

	m.current.Store(cfg)
	return nil
}

func isYAMLConfigPath(path string) bool {
	extension := strings.ToLower(filepath.Ext(path))
	return extension == ".yaml" || extension == ".yml"
}

func validateRoutingHeaderKeyDuplicates(rawConfig []byte) error {
	var document yaml.Node
	if err := yaml.Unmarshal(rawConfig, &document); err != nil {
		return err
	}
	if len(document.Content) == 0 {
		return nil
	}

	routing := yamlMappingValue(document.Content[0], "routing")
	routes := yamlMappingValue(routing, "routes")
	if routes == nil || routes.Kind != yaml.SequenceNode {
		return nil
	}

	for routeIndex, route := range routes.Content {
		match := yamlMappingValue(route, "match")
		headers := yamlMappingValue(match, "headers")
		if headers == nil || headers.Kind != yaml.MappingNode {
			continue
		}

		seen := make(map[string]struct{}, len(headers.Content)/2)
		for index := 0; index+1 < len(headers.Content); index += 2 {
			keyNode := dereferenceYAMLNode(headers.Content[index])
			if keyNode == nil || keyNode.Kind != yaml.ScalarNode {
				continue
			}
			name := strings.ToLower(strings.TrimSpace(keyNode.Value))
			if _, exists := seen[name]; exists {
				return fmt.Errorf("routing.routes[%d].match.headers contains duplicate header %q after normalization", routeIndex, name)
			}
			seen[name] = struct{}{}
		}
	}
	return nil
}

func yamlMappingValue(node *yaml.Node, key string) *yaml.Node {
	node = dereferenceYAMLNode(node)
	if node == nil || node.Kind != yaml.MappingNode {
		return nil
	}
	for index := 0; index+1 < len(node.Content); index += 2 {
		keyNode := dereferenceYAMLNode(node.Content[index])
		if keyNode != nil && keyNode.Kind == yaml.ScalarNode && strings.EqualFold(strings.TrimSpace(keyNode.Value), key) {
			return dereferenceYAMLNode(node.Content[index+1])
		}
	}
	return nil
}

func dereferenceYAMLNode(node *yaml.Node) *yaml.Node {
	for node != nil && node.Kind == yaml.AliasNode {
		node = node.Alias
	}
	return node
}

func normalizeProviderCapabilities(cfg *Config) {
	if cfg == nil {
		return
	}
	for providerID, providerCfg := range cfg.Providers {
		if len(providerCfg.Capabilities.HostedTools) > 0 {
			providerCfg.Capabilities.HostedTools = normalizeCapabilityValues(providerCfg.Capabilities.HostedTools)
		}
		if len(providerCfg.Capabilities.ReasoningEffortLevels) > 0 {
			providerCfg.Capabilities.ReasoningEffortLevels = normalizeCapabilityValues(providerCfg.Capabilities.ReasoningEffortLevels)
		}
		cfg.Providers[providerID] = providerCfg
	}
}

func normalizeCapabilityValues(values []string) []string {
	if len(values) == 0 {
		return nil
	}
	seen := make(map[string]struct{}, len(values))
	normalized := make([]string, 0, len(values))
	for _, raw := range values {
		value := strings.ToLower(strings.TrimSpace(raw))
		if value == "" {
			continue
		}
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		normalized = append(normalized, value)
	}
	return normalized
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

func resolveGatewayAPIKey(cfg *Config) {
	if cfg == nil {
		return
	}

	generalKey := strings.TrimSpace(expandEnv(cfg.General.APIKey))
	legacyDataSharingKey := strings.TrimSpace(expandEnv(cfg.DataSharing.APIKey))

	switch {
	case generalKey != "" && legacyDataSharingKey != "" && generalKey != legacyDataSharingKey:
		log.Warn().Msg("both general.api_key and deprecated data_sharing.api_key are set; using general.api_key")
	case generalKey == "" && legacyDataSharingKey != "":
		generalKey = legacyDataSharingKey
		log.Warn().Msg("data_sharing.api_key is deprecated; move this value to general.api_key")
	}

	cfg.General.APIKey = generalKey
	cfg.DataSharing.APIKey = generalKey
}

func validateConfig(cfg *Config) error {
	if cfg == nil {
		return nil
	}
	normalizeBackendConfig(cfg)
	if err := normalizeRoutingConfig(&cfg.Routing); err != nil {
		return err
	}
	if err := validateServerConfig(cfg.Server); err != nil {
		return err
	}
	providerTypes, err := validateProviderConfigs(cfg.Providers)
	if err != nil {
		return err
	}
	if err := validateRoutingConfig(cfg.Routing, providerTypes); err != nil {
		return err
	}
	if err := validateModelSelectionConfig(&cfg.ModelSelect); err != nil {
		return err
	}
	if err := validateRateLimitConfig(cfg.RateLimit); err != nil {
		return err
	}
	if err := validateCacheConfig(cfg.Cache); err != nil {
		return err
	}
	if err := validateLoggingConfig(&cfg.Logging); err != nil {
		return err
	}
	if err := validateRetryConfig(cfg.Retry); err != nil {
		return err
	}
	if err := validateUpdateCheckConfig(cfg.UpdateCheck); err != nil {
		return err
	}
	if err := validateBackendConfig(cfg); err != nil {
		return err
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
		if securityCfg.Enabled {
			return fmt.Errorf("security.provider must not be none when security.enabled is true")
		}
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

func normalizeBackendConfig(cfg *Config) {
	if cfg == nil {
		return
	}
	backendURL := strings.TrimSpace(expandEnv(cfg.General.BackendURL))
	if backendURL == "" {
		backendURL = strings.TrimSpace(expandEnv(cfg.DataSharing.BackendURL))
	}
	if backendURL == "" {
		backendURL = defaultBackendURL
	}
	cfg.General.BackendURL = backendURL
}

func validateBackendConfig(cfg *Config) error {
	parsed, err := safeurl.ParseHTTPBaseURL(cfg.General.BackendURL)
	if err != nil {
		return fmt.Errorf("general.backend_url must be an absolute HTTP or HTTPS URL")
	}
	if err := normalizeBackendURLPath(parsed); err != nil {
		return fmt.Errorf("general.backend_url must be an absolute HTTP or HTTPS URL")
	}
	cfg.General.BackendURL = parsed.String()

	if cfg.DataSharing.RemoteControl && !cfg.DataSharing.Enabled {
		return fmt.Errorf("data_sharing.enabled must be true when data_sharing.remote_control is enabled")
	}
	if cfg.DataSharing.Enabled && strings.TrimSpace(cfg.General.APIKey) == "" {
		return fmt.Errorf("general.api_key must not be empty when data_sharing.enabled is true")
	}
	return nil
}

func normalizeBackendURLPath(parsed *url.URL) error {
	escapedPath := strings.TrimRight(parsed.EscapedPath(), "/")
	if strings.HasSuffix(escapedPath, "/collector") {
		escapedPath = strings.TrimSuffix(escapedPath, "/collector")
	}
	decodedPath, err := url.PathUnescape(escapedPath)
	if err != nil {
		return err
	}
	parsed.Path = decodedPath
	parsed.RawPath = escapedPath
	return nil
}

func normalizeRoutingConfig(cfg *RoutingConfig) error {
	if cfg == nil {
		return nil
	}

	cfg.DefaultStrategy = strings.ToLower(strings.TrimSpace(cfg.DefaultStrategy))
	if cfg.DefaultStrategy == "" {
		cfg.DefaultStrategy = "round-robin"
	}

	for routeIndex := range cfg.Routes {
		headers := cfg.Routes[routeIndex].Match.Headers
		if len(headers) == 0 {
			continue
		}

		normalizedHeaders := make(map[string]string, len(headers))
		for rawName, value := range headers {
			name := strings.ToLower(strings.TrimSpace(rawName))
			if name == "" {
				return fmt.Errorf("routing.routes[%d].match.headers must not contain an empty header name", routeIndex)
			}
			if _, exists := normalizedHeaders[name]; exists {
				return fmt.Errorf("routing.routes[%d].match.headers contains duplicate header %q after normalization", routeIndex, name)
			}
			normalizedHeaders[name] = value
		}
		cfg.Routes[routeIndex].Match.Headers = normalizedHeaders
	}
	return nil
}

func validateServerConfig(cfg ServerConfig) error {
	if _, err := normalizeServerHost(cfg.Host); err != nil {
		return fmt.Errorf("server.host must be empty, a valid IP address, or a valid DNS hostname")
	}
	if cfg.Port < 1 || cfg.Port > 65535 {
		return fmt.Errorf("server.port must be between 1 and 65535")
	}
	timeouts := []struct {
		name  string
		value time.Duration
	}{
		{name: "read_timeout", value: cfg.ReadTimeout},
		{name: "write_timeout", value: cfg.WriteTimeout},
		{name: "idle_timeout", value: cfg.IdleTimeout},
	}
	for _, timeout := range timeouts {
		if timeout.value < 0 {
			return fmt.Errorf("server.%s must not be negative", timeout.name)
		}
	}
	return nil
}

func normalizeServerHost(raw string) (string, error) {
	host := strings.TrimSpace(raw)
	if host == "" {
		return "", nil
	}

	if strings.ContainsAny(host, "[]") {
		if len(host) < 3 || host[0] != '[' || host[len(host)-1] != ']' {
			return "", fmt.Errorf("malformed bracketed host")
		}
		host = host[1 : len(host)-1]
		if strings.ContainsAny(host, "[]") || !validIPv6Host(host) {
			return "", fmt.Errorf("brackets are only valid around an IPv6 address")
		}
		return host, nil
	}

	if net.ParseIP(host) != nil || validIPv6Host(host) || validDNSHostname(host) {
		return host, nil
	}
	return "", fmt.Errorf("invalid host")
}

func validIPv6Host(host string) bool {
	address, zone, zoned := strings.Cut(host, "%")
	if !strings.Contains(address, ":") || net.ParseIP(address) == nil {
		return false
	}
	if !zoned {
		return true
	}
	return zone != "" && !strings.ContainsAny(zone, "%[]/\\\t\r\n ")
}

func validDNSHostname(host string) bool {
	trimmed := strings.TrimSuffix(host, ".")
	if trimmed == "" || len(trimmed) > 253 {
		return false
	}
	if strings.Contains(trimmed, ":") {
		return false
	}

	numeric := true
	for _, char := range trimmed {
		if (char < '0' || char > '9') && char != '.' {
			numeric = false
			break
		}
	}
	if numeric && strings.Contains(trimmed, ".") {
		return false
	}

	for _, label := range strings.Split(trimmed, ".") {
		if len(label) == 0 || len(label) > 63 || label[0] == '-' || label[len(label)-1] == '-' {
			return false
		}
		for _, char := range label {
			if (char >= 'a' && char <= 'z') || (char >= 'A' && char <= 'Z') ||
				(char >= '0' && char <= '9') || char == '-' {
				continue
			}
			return false
		}
	}
	return true
}

func validateProviderConfigs(providers map[string]ProviderConfig) (map[string]string, error) {
	providerIDs := make([]string, 0, len(providers))
	for providerID := range providers {
		providerIDs = append(providerIDs, providerID)
	}
	sort.Strings(providerIDs)

	resolvedTypes := make(map[string]string, len(providers))
	for _, providerID := range providerIDs {
		if strings.TrimSpace(providerID) == "" {
			return nil, fmt.Errorf("providers must not contain an empty provider ID")
		}
		providerCfg := providers[providerID]
		providerType := strings.ToLower(strings.TrimSpace(providerCfg.Type))
		if providerType == "" {
			switch providerID {
			case "openai", "anthropic", "ollama":
				providerType = providerID
			default:
				return nil, fmt.Errorf("providers[%q].type is required for a custom provider", providerID)
			}
		}
		switch providerType {
		case "openai", "anthropic", "ollama":
		default:
			return nil, fmt.Errorf("providers[%q].type %q is not supported", providerID, providerCfg.Type)
		}
		if strings.TrimSpace(providerCfg.BaseURL) != "" {
			if _, err := safeurl.ParseHTTPBaseURL(providerCfg.BaseURL); err != nil {
				return nil, fmt.Errorf("providers[%q].base_url must be an absolute HTTP or HTTPS URL", providerID)
			}
		}
		if providerCfg.Timeout < 0 {
			return nil, fmt.Errorf("providers[%q].timeout must not be negative", providerID)
		}
		timeoutMode := strings.ToLower(strings.TrimSpace(providerCfg.TimeoutMode))
		switch timeoutMode {
		case "", "ttft", "total", "last_byte":
		default:
			return nil, fmt.Errorf("providers[%q].timeout_mode must be ttft, total, or last_byte", providerID)
		}
		modelMode := strings.ToLower(strings.TrimSpace(providerCfg.Models.Mode))
		if modelMode == "" {
			modelMode = "translator"
		}
		switch modelMode {
		case "translator", "static", "fetch":
		default:
			return nil, fmt.Errorf("providers[%q].models.mode must be translator, static, or fetch", providerID)
		}
		if providerCfg.Models.Fetch.TTL < 0 {
			return nil, fmt.Errorf("providers[%q].models.fetch.ttl must not be negative", providerID)
		}
		if modelMode == "fetch" {
			switch providerType {
			case "openai", "ollama":
			default:
				return nil, fmt.Errorf("providers[%q].models.mode=fetch is not supported for provider type %q", providerID, providerType)
			}
			if providerCfg.Models.Fetch.TTL == 0 {
				providerCfg.Models.Fetch.TTL = DefaultModelsFetchTTL
			}
		}
		providerCfg.Models.Mode = modelMode
		providers[providerID] = providerCfg
		resolvedTypes[providerID] = providerType
	}
	return resolvedTypes, nil
}

func validateRoutingConfig(cfg RoutingConfig, providerTypes map[string]string) error {
	switch cfg.DefaultStrategy {
	case "weighted", "round-robin", "random":
	default:
		return fmt.Errorf("routing.default_strategy must be weighted, round-robin, or random")
	}
	if len(cfg.Routes) == 0 {
		return fmt.Errorf("routing.routes must contain at least one route")
	}
	routeNames := make(map[string]struct{}, len(cfg.Routes))
	for routeIndex, route := range cfg.Routes {
		routeName := strings.TrimSpace(route.Name)
		if routeName == "" {
			return fmt.Errorf("routing.routes[%d].name must not be empty", routeIndex)
		}
		if _, exists := routeNames[routeName]; exists {
			return fmt.Errorf("routing.routes[%d].name %q is duplicated", routeIndex, routeName)
		}
		routeNames[routeName] = struct{}{}
		if len(route.Targets) == 0 {
			return fmt.Errorf("routing.routes[%d].targets must contain at least one target", routeIndex)
		}

		groups := []struct {
			name    string
			targets []TargetConfig
		}{
			{name: "targets", targets: route.Targets},
			{name: "fallback", targets: route.Fallback},
		}
		for _, group := range groups {
			totalWeight := 0
			for targetIndex, target := range group.targets {
				providerField := fmt.Sprintf("routing.routes[%d].%s[%d].provider", routeIndex, group.name, targetIndex)
				if strings.TrimSpace(target.Provider) == "" {
					return fmt.Errorf("%s must not be empty", providerField)
				}
				providerType, ok := providerTypes[target.Provider]
				if !ok {
					return fmt.Errorf("%s references unknown or invalid provider %q", providerField, target.Provider)
				}
				weightField := fmt.Sprintf("routing.routes[%d].%s[%d].weight", routeIndex, group.name, targetIndex)
				if target.Weight <= 0 {
					return fmt.Errorf("%s must be greater than zero", weightField)
				}
				if target.Weight > math.MaxInt-totalWeight {
					return fmt.Errorf("routing.routes[%d].%s weights exceed the supported total", routeIndex, group.name)
				}
				totalWeight += target.Weight

				requestType := strings.ToLower(strings.TrimSpace(target.UpstreamRequestType))
				if requestType == "" || requestType == "chat_completions" {
					continue
				}
				field := fmt.Sprintf("routing.routes[%d].%s[%d].upstream_request_type", routeIndex, group.name, targetIndex)
				if requestType != "responses" {
					return fmt.Errorf("%s must be chat_completions or responses", field)
				}
				if providerType != "openai" {
					return fmt.Errorf("%s requires an openai provider, got %q", field, providerType)
				}
			}
		}
	}
	return nil
}

func validateModelSelectionConfig(cfg *ModelSelectionConfig) error {
	if cfg == nil || !cfg.Enabled {
		return nil
	}

	reservedHeaders := map[string]struct{}{
		"x-lunargate-model":          {},
		"x-lunargate-provider":       {},
		"x-lunargate-request-type":   {},
		"x-lunargate-requires-tools": {},
		"x-lunargate-route":          {},
	}
	outputHeaders := []struct {
		field string
		value *string
	}{
		{field: "complexity", value: &cfg.OutputHeaders.Complexity},
		{field: "score", value: &cfg.OutputHeaders.Score},
		{field: "skill", value: &cfg.OutputHeaders.Skill},
	}
	seenHeaders := make(map[string]string, len(outputHeaders))
	for _, outputHeader := range outputHeaders {
		name := strings.ToLower(strings.TrimSpace(*outputHeader.value))
		*outputHeader.value = name
		if name == "" {
			continue
		}
		field := "model_selection.output_headers." + outputHeader.field
		if _, reserved := reservedHeaders[name]; reserved {
			return fmt.Errorf("%s must not use reserved routing header %q", field, name)
		}
		if previousField, exists := seenHeaders[name]; exists {
			return fmt.Errorf("%s duplicates model_selection.output_headers.%s", field, previousField)
		}
		seenHeaders[name] = outputHeader.field
	}

	tiers := cfg.ComplexityTiers
	if tiers.Tier01Max <= 0 {
		return fmt.Errorf("model_selection.complexity_tiers.tier_01_max must be greater than zero when enabled")
	}
	if tiers.Tier23Max <= tiers.Tier01Max {
		return fmt.Errorf("model_selection.complexity_tiers.tier_23_max must be greater than tier_01_max")
	}
	if tiers.Tier45Max <= tiers.Tier23Max {
		return fmt.Errorf("model_selection.complexity_tiers.tier_45_max must be greater than tier_23_max")
	}

	if err := validateModelSelectionComplexityRule("simple", &cfg.Complexity.Simple); err != nil {
		return err
	}
	if err := validateModelSelectionComplexityRule("complex", &cfg.Complexity.Complex); err != nil {
		return err
	}

	seenSkills := make(map[string]struct{}, len(cfg.Skills))
	for skillIndex := range cfg.Skills {
		skill := &cfg.Skills[skillIndex]
		skill.Name = strings.TrimSpace(skill.Name)
		nameField := fmt.Sprintf("model_selection.skills[%d].name", skillIndex)
		if skill.Name == "" {
			return fmt.Errorf("%s must not be empty", nameField)
		}
		canonicalName := strings.ToLower(skill.Name)
		if _, exists := seenSkills[canonicalName]; exists {
			return fmt.Errorf("%s is duplicated", nameField)
		}
		seenSkills[canonicalName] = struct{}{}
		if len(skill.RegexAny) == 0 {
			return fmt.Errorf("model_selection.skills[%d].regex_any must contain at least one regular expression", skillIndex)
		}
		for regexIndex := range skill.RegexAny {
			pattern := strings.TrimSpace(skill.RegexAny[regexIndex])
			field := fmt.Sprintf("model_selection.skills[%d].regex_any[%d]", skillIndex, regexIndex)
			if pattern == "" {
				return fmt.Errorf("%s must not be empty", field)
			}
			if _, err := regexp.Compile(pattern); err != nil {
				return fmt.Errorf("%s must be a valid regular expression", field)
			}
			skill.RegexAny[regexIndex] = pattern
		}
	}
	return nil
}

func validateModelSelectionComplexityRule(name string, rule *ModelSelectionComplexityRule) error {
	if rule == nil {
		return nil
	}
	prefix := "model_selection.complexity." + name
	bounds := []struct {
		field string
		value *int
	}{
		{field: "max_user_chars", value: rule.MaxUserChars},
		{field: "min_user_chars", value: rule.MinUserChars},
		{field: "max_messages", value: rule.MaxMessages},
		{field: "min_messages", value: rule.MinMessages},
	}
	for _, bound := range bounds {
		if bound.value != nil && *bound.value < 0 {
			return fmt.Errorf("%s.%s must not be negative", prefix, bound.field)
		}
	}
	if rule.MinUserChars != nil && rule.MaxUserChars != nil && *rule.MinUserChars > *rule.MaxUserChars {
		return fmt.Errorf("%s.min_user_chars must not exceed max_user_chars", prefix)
	}
	if rule.MinMessages != nil && rule.MaxMessages != nil && *rule.MinMessages > *rule.MaxMessages {
		return fmt.Errorf("%s.min_messages must not exceed max_messages", prefix)
	}

	seenConditions := make(map[string]struct{}, len(rule.AnyOf))
	for conditionIndex := range rule.AnyOf {
		condition := strings.ToLower(strings.TrimSpace(rule.AnyOf[conditionIndex]))
		field := fmt.Sprintf("%s.any_of[%d]", prefix, conditionIndex)
		switch condition {
		case "has_tools":
			if rule.RequireNoTools {
				return fmt.Errorf("%s conflicts with %s.require_no_tools", field, prefix)
			}
		case "requires_json":
			if rule.RequireNoJSON {
				return fmt.Errorf("%s conflicts with %s.require_no_json", field, prefix)
			}
		default:
			return fmt.Errorf("%s must be has_tools or requires_json", field)
		}
		if _, exists := seenConditions[condition]; exists {
			return fmt.Errorf("%s is duplicated", field)
		}
		seenConditions[condition] = struct{}{}
		rule.AnyOf[conditionIndex] = condition
	}
	return nil
}

func validateRateLimitConfig(cfg RateLimitConfig) error {
	if !cfg.Enabled {
		return nil
	}
	if cfg.RequestsPerMinute <= 0 {
		return fmt.Errorf("rate_limiting.requests_per_minute must be greater than zero when enabled")
	}
	if cfg.BurstSize < 0 {
		return fmt.Errorf("rate_limiting.burst_size must not be negative when enabled")
	}
	return nil
}

func validateCacheConfig(cfg CacheConfig) error {
	if !cfg.Enabled {
		return nil
	}
	if cfg.TTL <= 0 {
		return fmt.Errorf("caching.ttl must be greater than zero when enabled")
	}
	if cfg.MaxSize <= 0 {
		return fmt.Errorf("caching.max_size must be greater than zero when enabled")
	}
	if cfg.MaxEntryBytes <= 0 {
		return fmt.Errorf("caching.max_entry_bytes must be greater than zero when enabled")
	}
	if cfg.MaxBytes <= 0 {
		return fmt.Errorf("caching.max_bytes must be greater than zero when enabled")
	}
	if cfg.MaxEntryBytes > cfg.MaxBytes {
		return fmt.Errorf("caching.max_entry_bytes must not exceed caching.max_bytes when enabled")
	}
	return nil
}

func validateLoggingConfig(cfg *LoggingConfig) error {
	if cfg == nil {
		return nil
	}

	level := strings.ToLower(strings.TrimSpace(cfg.Level))
	if level == "" {
		level = "info"
	}
	switch level {
	case "trace", "debug", "info", "warn", "error", "fatal", "panic", "disabled":
	default:
		return fmt.Errorf("logging.level must be one of trace, debug, info, warn, error, fatal, panic, or disabled")
	}

	format := strings.ToLower(strings.TrimSpace(cfg.Format))
	if format == "" {
		format = "console"
	}
	switch format {
	case "console", "json":
	default:
		return fmt.Errorf("logging.format must be console or json")
	}

	cfg.Level = level
	cfg.Format = format
	return nil
}

func validateRetryConfig(cfg RetryConfig) error {
	if !cfg.Enabled {
		return nil
	}
	if cfg.MaxAttempts < 1 {
		return fmt.Errorf("retry.max_attempts must be at least 1 when enabled")
	}
	if cfg.MaxAttempts > MaxRetryAttempts {
		return fmt.Errorf("retry.max_attempts must not exceed %d when enabled", MaxRetryAttempts)
	}
	if cfg.InitialDelay < 0 {
		return fmt.Errorf("retry.initial_delay must not be negative when enabled")
	}
	if cfg.MaxDelay < 0 {
		return fmt.Errorf("retry.max_delay must not be negative when enabled")
	}
	if cfg.MaxDelay < cfg.InitialDelay {
		return fmt.Errorf("retry.max_delay must be greater than or equal to retry.initial_delay")
	}
	if math.IsNaN(cfg.Multiplier) || math.IsInf(cfg.Multiplier, 0) || cfg.Multiplier < 1 {
		return fmt.Errorf("retry.multiplier must be a finite value greater than or equal to 1")
	}
	if math.IsNaN(cfg.JitterFactor) || math.IsInf(cfg.JitterFactor, 0) || cfg.JitterFactor < 0 || cfg.JitterFactor > 1 {
		return fmt.Errorf("retry.jitter_factor must be a finite value between 0 and 1")
	}
	for index, status := range cfg.RetryableErrors {
		if status < 400 || status > 599 {
			return fmt.Errorf("retry.retryable_errors[%d] must be between 400 and 599", index)
		}
	}
	return nil
}

func validateUpdateCheckConfig(cfg UpdateCheckConfig) error {
	if !cfg.Enabled {
		return nil
	}
	endpoint := strings.TrimSpace(cfg.Endpoint)
	parsed, err := url.Parse(endpoint)
	if err != nil || !parsed.IsAbs() || parsed.Host == "" {
		return fmt.Errorf("update_check.endpoint must be an absolute HTTP or HTTPS URL when enabled")
	}
	switch strings.ToLower(parsed.Scheme) {
	case "http", "https":
	default:
		return fmt.Errorf("update_check.endpoint must be an absolute HTTP or HTTPS URL when enabled")
	}
	if cfg.Interval <= 0 {
		return fmt.Errorf("update_check.interval must be greater than zero when enabled")
	}
	if cfg.Timeout <= 0 {
		return fmt.Errorf("update_check.timeout must be greater than zero when enabled")
	}
	return nil
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
