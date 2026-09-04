package updatecheck

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/safeurl"
	"github.com/rs/zerolog/log"
	"golang.org/x/mod/semver"
)

const (
	initialDelay    = 30 * time.Second
	maxResponseSize = 4 << 10
)

var ErrDisabled = errors.New("automatic update checks are disabled")

type checkRequest struct {
	Version string `json:"version"`
	Arch    string `json:"arch"`
}

type CheckResult struct {
	Version string `json:"version"`
}

type Checker struct {
	mu      sync.RWMutex
	config  config.UpdateCheckConfig
	version string
	arch    string
	client  *http.Client
	wake    chan struct{}
}

func NewChecker(cfg config.UpdateCheckConfig, version string) *Checker {
	return newChecker(cfg, version, runtime.GOARCH, &http.Client{
		CheckRedirect: func(_ *http.Request, _ []*http.Request) error {
			return http.ErrUseLastResponse
		},
	})
}

func newChecker(cfg config.UpdateCheckConfig, version, arch string, client *http.Client) *Checker {
	return &Checker{
		config:  cfg,
		version: strings.TrimPrefix(strings.TrimSpace(version), "v"),
		arch:    arch,
		client:  client,
		wake:    make(chan struct{}, 1),
	}
}

func (c *Checker) Start(ctx context.Context) {
	go c.run(ctx)
}

func (c *Checker) UpdateConfig(cfg config.UpdateCheckConfig) {
	c.mu.Lock()
	c.config = cfg
	c.mu.Unlock()

	select {
	case c.wake <- struct{}{}:
	default:
	}
}

func (c *Checker) Check(ctx context.Context) (CheckResult, error) {
	cfg := c.snapshot()
	if !cfg.Enabled {
		return CheckResult{}, ErrDisabled
	}

	body, err := json.Marshal(checkRequest{Version: c.version, Arch: c.arch})
	if err != nil {
		return CheckResult{}, fmt.Errorf("encode update check: %w", err)
	}

	requestCtx, cancel := context.WithTimeout(ctx, cfg.Timeout)
	defer cancel()

	req, err := http.NewRequestWithContext(requestCtx, http.MethodPost, cfg.Endpoint, bytes.NewReader(body))
	if err != nil {
		return CheckResult{}, fmt.Errorf("create update check: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	// Suppress Go's default User-Agent. The endpoint receives only version and architecture.
	req.Header.Set("User-Agent", "")

	resp, err := c.client.Do(req)
	if err != nil {
		err = safeurl.RedactTransportError(err, req.URL)
		return CheckResult{}, fmt.Errorf("request latest version: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, maxResponseSize))
		return CheckResult{}, fmt.Errorf("latest version endpoint returned %s", resp.Status)
	}

	responseBody, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseSize+1))
	if err != nil {
		return CheckResult{}, fmt.Errorf("read latest version: %w", err)
	}
	if len(responseBody) > maxResponseSize {
		return CheckResult{}, fmt.Errorf("decode latest version: response exceeds %d bytes", maxResponseSize)
	}

	var result CheckResult
	decoder := json.NewDecoder(bytes.NewReader(responseBody))
	if err := decoder.Decode(&result); err != nil {
		return CheckResult{}, fmt.Errorf("decode latest version: %w", err)
	}
	var extra json.RawMessage
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		if err == nil {
			return CheckResult{}, errors.New("decode latest version: multiple JSON documents")
		}
		return CheckResult{}, fmt.Errorf("decode latest version: %w", err)
	}
	result.Version = strings.TrimPrefix(strings.TrimSpace(result.Version), "v")
	if !semver.IsValid("v" + result.Version) {
		return CheckResult{}, fmt.Errorf("latest version endpoint returned invalid version %q", result.Version)
	}

	return result, nil
}

func IsUpdateAvailable(current, latest string) bool {
	currentVersion := "v" + strings.TrimPrefix(strings.TrimSpace(current), "v")
	latestVersion := "v" + strings.TrimPrefix(strings.TrimSpace(latest), "v")
	if !semver.IsValid(currentVersion) || !semver.IsValid(latestVersion) {
		return false
	}
	return semver.Compare(latestVersion, currentVersion) > 0
}

func (c *Checker) run(ctx context.Context) {
	if !wait(ctx, c.wake, initialDelay) {
		return
	}

	for {
		cfg := c.snapshot()
		if cfg.Enabled {
			result, err := c.Check(ctx)
			switch {
			case err != nil && !errors.Is(err, context.Canceled):
				log.Debug().Err(err).Msg("automatic update check failed")
			case err == nil && IsUpdateAvailable(c.version, result.Version):
				log.Warn().
					Str("current_version", c.version).
					Str("latest_version", result.Version).
					Msg("a newer LunarGate version is available")
			}
		}

		interval := cfg.Interval
		if interval <= 0 {
			interval = 24 * time.Hour
		}
		if !wait(ctx, c.wake, interval) {
			return
		}
	}
}

func (c *Checker) snapshot() config.UpdateCheckConfig {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.config
}

func wait(ctx context.Context, wake <-chan struct{}, duration time.Duration) bool {
	timer := time.NewTimer(duration)
	defer timer.Stop()

	select {
	case <-ctx.Done():
		return false
	case <-wake:
		return true
	case <-timer.C:
		return true
	}
}
