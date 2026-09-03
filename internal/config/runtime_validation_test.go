package config

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestValidateConfigAcceptsSupportedRuntimeConfiguration(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*Config)
	}{
		{name: "minimum built-in provider"},
		{
			name: "round-robin routing strategy",
			mutate: func(cfg *Config) {
				cfg.Routing.DefaultStrategy = " ROUND-ROBIN "
			},
		},
		{
			name: "weighted routing strategy",
			mutate: func(cfg *Config) {
				cfg.Routing.DefaultStrategy = "weighted"
			},
		},
		{
			name: "random routing strategy",
			mutate: func(cfg *Config) {
				cfg.Routing.DefaultStrategy = "random"
			},
		},
		{
			name: "custom OpenAI-compatible provider without API key",
			mutate: func(cfg *Config) {
				cfg.Providers = map[string]ProviderConfig{"abacus": {Type: " OpenAI "}}
				cfg.Routing.Routes[0].Targets[0].Provider = "abacus"
			},
		},
		{
			name: "provider base URL with transport credentials",
			mutate: func(cfg *Config) {
				cfg.Providers["openai"] = ProviderConfig{
					BaseURL: "HTTPS://url-user:url-password@example.test/v1?api_key=query-secret#unused-fragment",
				}
			},
		},
		{
			name: "server boundary values",
			mutate: func(cfg *Config) {
				cfg.Server.Port = 1
				cfg.Server.ReadTimeout = 0
				cfg.Server.WriteTimeout = 0
				cfg.Server.IdleTimeout = 0
			},
		},
		{
			name: "maximum server port",
			mutate: func(cfg *Config) {
				cfg.Server.Port = 65535
			},
		},
		{
			name: "provider timeout modes",
			mutate: func(cfg *Config) {
				cfg.Providers = map[string]ProviderConfig{
					"openai":    {TimeoutMode: " TTFT "},
					"total":     {Type: "openai", TimeoutMode: "total"},
					"last-byte": {Type: "anthropic", TimeoutMode: "LAST_BYTE"},
				}
				cfg.Routing.Routes[0].Targets = []TargetConfig{{Provider: "openai", Weight: 1}}
				cfg.Routing.Routes[0].Fallback = []TargetConfig{
					{Provider: "total", Weight: 1},
					{Provider: "last-byte", Weight: 1},
				}
			},
		},
		{
			name: "maximum route weight total",
			mutate: func(cfg *Config) {
				cfg.Routing.Routes[0].Targets = []TargetConfig{
					{Provider: "openai", Weight: math.MaxInt - 1},
					{Provider: "openai", Weight: 1},
				}
			},
		},
		{
			name: "enabled optional sections at lower bounds",
			mutate: func(cfg *Config) {
				cfg.RateLimit = RateLimitConfig{Enabled: true, RequestsPerMinute: 1, BurstSize: 0}
				cfg.Cache = CacheConfig{
					Enabled:       true,
					TTL:           time.Nanosecond,
					MaxSize:       1,
					MaxEntryBytes: 1,
					MaxBytes:      1,
				}
				cfg.Retry = validRetryConfig()
				cfg.UpdateCheck = validUpdateCheckConfig()
			},
		},
		{
			name: "maximum retry attempts",
			mutate: func(cfg *Config) {
				cfg.Retry = validRetryConfig()
				cfg.Retry.MaxAttempts = MaxRetryAttempts
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cfg := validRuntimeConfig()
			if test.mutate != nil {
				test.mutate(cfg)
			}
			if err := validateConfig(cfg); err != nil {
				t.Fatalf("validateConfig returned error: %v", err)
			}
		})
	}
}

func TestValidateConfigRejectsBrokenRuntimeConfiguration(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(*Config)
		wantErr string
	}{
		{
			name: "server port zero",
			mutate: func(cfg *Config) {
				cfg.Server.Port = 0
			},
			wantErr: "server.port",
		},
		{
			name: "server port above maximum",
			mutate: func(cfg *Config) {
				cfg.Server.Port = 65536
			},
			wantErr: "server.port",
		},
		{
			name: "negative server read timeout",
			mutate: func(cfg *Config) {
				cfg.Server.ReadTimeout = -time.Nanosecond
			},
			wantErr: "server.read_timeout",
		},
		{
			name: "negative server write timeout",
			mutate: func(cfg *Config) {
				cfg.Server.WriteTimeout = -time.Nanosecond
			},
			wantErr: "server.write_timeout",
		},
		{
			name: "negative server idle timeout",
			mutate: func(cfg *Config) {
				cfg.Server.IdleTimeout = -time.Nanosecond
			},
			wantErr: "server.idle_timeout",
		},
		{
			name: "empty provider ID",
			mutate: func(cfg *Config) {
				cfg.Providers = map[string]ProviderConfig{" ": {Type: "openai"}}
			},
			wantErr: "empty provider ID",
		},
		{
			name: "custom provider without type",
			mutate: func(cfg *Config) {
				cfg.Providers = map[string]ProviderConfig{"custom": {}}
			},
			wantErr: `providers["custom"].type`,
		},
		{
			name: "unsupported provider type",
			mutate: func(cfg *Config) {
				cfg.Providers["openai"] = ProviderConfig{Type: "gemini"}
			},
			wantErr: `providers["openai"].type`,
		},
		{
			name: "enabled security without provider",
			mutate: func(cfg *Config) {
				cfg.Security = SecurityConfig{Enabled: true, Provider: "none"}
			},
			wantErr: "security.provider must not be none",
		},
		{
			name: "invalid provider base URL",
			mutate: func(cfg *Config) {
				cfg.Providers["openai"] = ProviderConfig{BaseURL: "relative/provider"}
			},
			wantErr: `providers["openai"].base_url`,
		},
		{
			name: "negative provider timeout",
			mutate: func(cfg *Config) {
				cfg.Providers["openai"] = ProviderConfig{Timeout: -time.Nanosecond}
			},
			wantErr: `providers["openai"].timeout`,
		},
		{
			name: "unsupported provider timeout mode",
			mutate: func(cfg *Config) {
				cfg.Providers["openai"] = ProviderConfig{TimeoutMode: "first_byte"}
			},
			wantErr: `providers["openai"].timeout_mode`,
		},
		{
			name: "unknown routing strategy",
			mutate: func(cfg *Config) {
				cfg.Routing.DefaultStrategy = "least-connections"
			},
			wantErr: "routing.default_strategy",
		},
		{
			name: "no routes",
			mutate: func(cfg *Config) {
				cfg.Routing.Routes = nil
			},
			wantErr: "routing.routes",
		},
		{
			name: "empty route name",
			mutate: func(cfg *Config) {
				cfg.Routing.Routes[0].Name = " "
			},
			wantErr: "routing.routes[0].name",
		},
		{
			name: "duplicate route name",
			mutate: func(cfg *Config) {
				duplicate := cfg.Routing.Routes[0]
				duplicate.Name = " default "
				cfg.Routing.Routes = append(cfg.Routing.Routes, duplicate)
			},
			wantErr: "duplicated",
		},
		{
			name: "route without primary target",
			mutate: func(cfg *Config) {
				cfg.Routing.Routes[0].Targets = nil
			},
			wantErr: "routing.routes[0].targets",
		},
		{
			name: "empty primary provider",
			mutate: func(cfg *Config) {
				cfg.Routing.Routes[0].Targets[0].Provider = " "
			},
			wantErr: "routing.routes[0].targets[0].provider",
		},
		{
			name: "unknown primary provider",
			mutate: func(cfg *Config) {
				cfg.Routing.Routes[0].Targets[0].Provider = "missing"
			},
			wantErr: "unknown or invalid provider",
		},
		{
			name: "unknown fallback provider",
			mutate: func(cfg *Config) {
				cfg.Routing.Routes[0].Fallback = []TargetConfig{{Provider: "missing", Weight: 1}}
			},
			wantErr: "routing.routes[0].fallback[0].provider",
		},
		{
			name: "zero primary target weight",
			mutate: func(cfg *Config) {
				cfg.Routing.Routes[0].Targets[0].Weight = 0
			},
			wantErr: "routing.routes[0].targets[0].weight",
		},
		{
			name: "negative primary target weight",
			mutate: func(cfg *Config) {
				cfg.Routing.Routes[0].Targets[0].Weight = -1
			},
			wantErr: "routing.routes[0].targets[0].weight",
		},
		{
			name: "zero fallback target weight",
			mutate: func(cfg *Config) {
				cfg.Routing.Routes[0].Fallback = []TargetConfig{{Provider: "openai"}}
			},
			wantErr: "routing.routes[0].fallback[0].weight",
		},
		{
			name: "negative fallback target weight",
			mutate: func(cfg *Config) {
				cfg.Routing.Routes[0].Fallback = []TargetConfig{{Provider: "openai", Weight: -1}}
			},
			wantErr: "routing.routes[0].fallback[0].weight",
		},
		{
			name: "primary target weights overflow",
			mutate: func(cfg *Config) {
				cfg.Routing.Routes[0].Targets = []TargetConfig{
					{Provider: "openai", Weight: math.MaxInt},
					{Provider: "openai", Weight: 1},
				}
			},
			wantErr: "routing.routes[0].targets weights",
		},
		{
			name: "fallback target weights overflow",
			mutate: func(cfg *Config) {
				cfg.Routing.Routes[0].Fallback = []TargetConfig{
					{Provider: "openai", Weight: math.MaxInt},
					{Provider: "openai", Weight: 1},
				}
			},
			wantErr: "routing.routes[0].fallback weights",
		},
		{
			name: "enabled rate limit with zero RPM",
			mutate: func(cfg *Config) {
				cfg.RateLimit = RateLimitConfig{Enabled: true, BurstSize: 1}
			},
			wantErr: "rate_limiting.requests_per_minute",
		},
		{
			name: "enabled rate limit with negative burst",
			mutate: func(cfg *Config) {
				cfg.RateLimit = RateLimitConfig{Enabled: true, RequestsPerMinute: 1, BurstSize: -1}
			},
			wantErr: "rate_limiting.burst_size",
		},
		{
			name: "enabled cache with zero TTL",
			mutate: func(cfg *Config) {
				cfg.Cache = CacheConfig{Enabled: true, MaxSize: 1, MaxEntryBytes: 1, MaxBytes: 1}
			},
			wantErr: "caching.ttl",
		},
		{
			name: "enabled cache with zero max size",
			mutate: func(cfg *Config) {
				cfg.Cache = CacheConfig{Enabled: true, TTL: time.Second, MaxEntryBytes: 1, MaxBytes: 1}
			},
			wantErr: "caching.max_size",
		},
		{
			name: "enabled cache with zero max entry bytes",
			mutate: func(cfg *Config) {
				cfg.Cache = CacheConfig{Enabled: true, TTL: time.Second, MaxSize: 1, MaxBytes: 1}
			},
			wantErr: "caching.max_entry_bytes",
		},
		{
			name: "enabled cache with zero max bytes",
			mutate: func(cfg *Config) {
				cfg.Cache = CacheConfig{Enabled: true, TTL: time.Second, MaxSize: 1, MaxEntryBytes: 1}
			},
			wantErr: "caching.max_bytes",
		},
		{
			name: "cache entry limit exceeds total limit",
			mutate: func(cfg *Config) {
				cfg.Cache = CacheConfig{Enabled: true, TTL: time.Second, MaxSize: 1, MaxEntryBytes: 2, MaxBytes: 1}
			},
			wantErr: "caching.max_entry_bytes",
		},
		{
			name: "enabled retry with zero attempts",
			mutate: func(cfg *Config) {
				cfg.Retry = validRetryConfig()
				cfg.Retry.MaxAttempts = 0
			},
			wantErr: "retry.max_attempts",
		},
		{
			name: "enabled retry above attempt limit",
			mutate: func(cfg *Config) {
				cfg.Retry = validRetryConfig()
				cfg.Retry.MaxAttempts = MaxRetryAttempts + 1
			},
			wantErr: "retry.max_attempts",
		},
		{
			name: "enabled retry with negative initial delay",
			mutate: func(cfg *Config) {
				cfg.Retry = validRetryConfig()
				cfg.Retry.InitialDelay = -time.Nanosecond
			},
			wantErr: "retry.initial_delay",
		},
		{
			name: "enabled retry with negative max delay",
			mutate: func(cfg *Config) {
				cfg.Retry = validRetryConfig()
				cfg.Retry.MaxDelay = -time.Nanosecond
			},
			wantErr: "retry.max_delay",
		},
		{
			name: "enabled retry with descending delays",
			mutate: func(cfg *Config) {
				cfg.Retry = validRetryConfig()
				cfg.Retry.InitialDelay = time.Second
				cfg.Retry.MaxDelay = time.Millisecond
			},
			wantErr: "retry.max_delay",
		},
		{
			name: "enabled retry with multiplier below one",
			mutate: func(cfg *Config) {
				cfg.Retry = validRetryConfig()
				cfg.Retry.Multiplier = 0.5
			},
			wantErr: "retry.multiplier",
		},
		{
			name: "enabled retry with non-finite multiplier",
			mutate: func(cfg *Config) {
				cfg.Retry = validRetryConfig()
				cfg.Retry.Multiplier = math.Inf(1)
			},
			wantErr: "retry.multiplier",
		},
		{
			name: "enabled retry with negative jitter",
			mutate: func(cfg *Config) {
				cfg.Retry = validRetryConfig()
				cfg.Retry.JitterFactor = -0.1
			},
			wantErr: "retry.jitter_factor",
		},
		{
			name: "enabled retry with jitter above one",
			mutate: func(cfg *Config) {
				cfg.Retry = validRetryConfig()
				cfg.Retry.JitterFactor = 1.1
			},
			wantErr: "retry.jitter_factor",
		},
		{
			name: "enabled retry with non-finite jitter",
			mutate: func(cfg *Config) {
				cfg.Retry = validRetryConfig()
				cfg.Retry.JitterFactor = math.NaN()
			},
			wantErr: "retry.jitter_factor",
		},
		{
			name: "enabled retry with status below HTTP error range",
			mutate: func(cfg *Config) {
				cfg.Retry = validRetryConfig()
				cfg.Retry.RetryableErrors = []int{399}
			},
			wantErr: "retry.retryable_errors[0]",
		},
		{
			name: "enabled retry with status above HTTP range",
			mutate: func(cfg *Config) {
				cfg.Retry = validRetryConfig()
				cfg.Retry.RetryableErrors = []int{600}
			},
			wantErr: "retry.retryable_errors[0]",
		},
		{
			name: "enabled update check with relative endpoint",
			mutate: func(cfg *Config) {
				cfg.UpdateCheck = validUpdateCheckConfig()
				cfg.UpdateCheck.Endpoint = "/latest"
			},
			wantErr: "update_check.endpoint",
		},
		{
			name: "enabled update check with non-HTTP endpoint",
			mutate: func(cfg *Config) {
				cfg.UpdateCheck = validUpdateCheckConfig()
				cfg.UpdateCheck.Endpoint = "ftp://updates.example/latest"
			},
			wantErr: "update_check.endpoint",
		},
		{
			name: "enabled update check without host",
			mutate: func(cfg *Config) {
				cfg.UpdateCheck = validUpdateCheckConfig()
				cfg.UpdateCheck.Endpoint = "https:/latest"
			},
			wantErr: "update_check.endpoint",
		},
		{
			name: "enabled update check with zero interval",
			mutate: func(cfg *Config) {
				cfg.UpdateCheck = validUpdateCheckConfig()
				cfg.UpdateCheck.Interval = 0
			},
			wantErr: "update_check.interval",
		},
		{
			name: "enabled update check with zero timeout",
			mutate: func(cfg *Config) {
				cfg.UpdateCheck = validUpdateCheckConfig()
				cfg.UpdateCheck.Timeout = 0
			},
			wantErr: "update_check.timeout",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cfg := validRuntimeConfig()
			test.mutate(cfg)
			err := validateConfig(cfg)
			if err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("validateConfig error = %v, want substring %q", err, test.wantErr)
			}
		})
	}
}

func TestValidateConfigProviderBaseURLErrorDoesNotEchoValue(t *testing.T) {
	const secret = "base-url-validation-secret"
	cfg := validRuntimeConfig()
	cfg.Providers["openai"] = ProviderConfig{
		BaseURL: "https://example.test/%zz?api_key=" + secret,
	}

	err := validateConfig(cfg)
	if err == nil || !strings.Contains(err.Error(), `providers["openai"].base_url`) {
		t.Fatalf("validateConfig error = %v, want provider base_url error", err)
	}
	if strings.Contains(err.Error(), secret) {
		t.Fatalf("validation error leaked base URL: %v", err)
	}
}

func TestValidateConfigIgnoresDisabledSectionZeroValues(t *testing.T) {
	cfg := validRuntimeConfig()
	cfg.RateLimit = RateLimitConfig{}
	cfg.Cache = CacheConfig{}
	cfg.Retry = RetryConfig{}
	cfg.UpdateCheck = UpdateCheckConfig{}

	if err := validateConfig(cfg); err != nil {
		t.Fatalf("validateConfig returned error for disabled zero values: %v", err)
	}
}

func TestNewManagerDefaultsEmptyExpandedUpdateEndpoint(t *testing.T) {
	const endpointEnv = "LUNARGATE_TEST_EMPTY_UPDATE_ENDPOINT"
	if err := os.Unsetenv(endpointEnv); err != nil {
		t.Fatalf("unset endpoint environment variable: %v", err)
	}
	t.Cleanup(func() { _ = os.Unsetenv(endpointEnv) })

	configPath := filepath.Join(t.TempDir(), "config.yaml")
	configBody := `providers:
  openai:
    type: openai
routing:
  routes:
    - name: default
      targets:
        - provider: openai
          weight: 100
update_check:
  enabled: true
  endpoint: "${LUNARGATE_TEST_EMPTY_UPDATE_ENDPOINT}"
`
	if err := os.WriteFile(configPath, []byte(configBody), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	manager, err := NewManager(configPath)
	if err != nil {
		t.Fatalf("NewManager returned error: %v", err)
	}
	if got := manager.Get().UpdateCheck.Endpoint; got != defaultUpdateCheckURL {
		t.Fatalf("update_check.endpoint = %q, want %q", got, defaultUpdateCheckURL)
	}
}

func TestNewManagerDefaultsCacheByteLimits(t *testing.T) {
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	configBody := `providers:
  openai:
    type: openai
routing:
  routes:
    - name: default
      targets:
        - provider: openai
          weight: 100
caching:
  enabled: true
  ttl: 1m
  max_size: 8
`
	if err := os.WriteFile(configPath, []byte(configBody), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	manager, err := NewManager(configPath)
	if err != nil {
		t.Fatalf("NewManager returned error: %v", err)
	}
	cache := manager.Get().Cache
	if cache.MaxEntryBytes != DefaultCacheMaxEntryBytes {
		t.Fatalf("cache max_entry_bytes = %d, want %d", cache.MaxEntryBytes, DefaultCacheMaxEntryBytes)
	}
	if cache.MaxBytes != DefaultCacheMaxBytes {
		t.Fatalf("cache max_bytes = %d, want %d", cache.MaxBytes, DefaultCacheMaxBytes)
	}
}

func validRuntimeConfig() *Config {
	return &Config{
		Server: ServerConfig{Port: 8080},
		Providers: map[string]ProviderConfig{
			"openai": {},
		},
		Routing: RoutingConfig{Routes: []RouteConfig{{
			Name:    "default",
			Targets: []TargetConfig{{Provider: "openai", Weight: 1}},
		}}},
	}
}

func validRetryConfig() RetryConfig {
	return RetryConfig{
		Enabled:         true,
		MaxAttempts:     1,
		InitialDelay:    0,
		MaxDelay:        0,
		Multiplier:      1,
		JitterFactor:    0,
		RetryableErrors: []int{400, 599},
	}
}

func validUpdateCheckConfig() UpdateCheckConfig {
	return UpdateCheckConfig{
		Enabled:  true,
		Endpoint: "https://updates.example/latest",
		Interval: time.Hour,
		Timeout:  time.Second,
	}
}
