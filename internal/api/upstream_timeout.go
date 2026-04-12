package api

import (
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/lunargate-ai/gateway/internal/config"
)

const defaultUpstreamTimeout = 120 * time.Second

const (
	upstreamTimeoutModeTTFT  = "ttft"
	upstreamTimeoutModeTotal = "total"
)

var errUpstreamTTFTTimeout = errors.New("upstream timed out waiting for first token")
var errUpstreamTotalTimeout = errors.New("upstream timed out before full response completed")

type providerClientConfig struct {
	client  *http.Client
	timeout time.Duration
	mode    string
}

func buildProviderClients(providerConfigs map[string]config.ProviderConfig) map[string]providerClientConfig {
	clients := make(map[string]providerClientConfig, len(providerConfigs))
	for providerID, providerCfg := range providerConfigs {
		timeout := providerCfg.Timeout
		if timeout <= 0 {
			timeout = defaultUpstreamTimeout
		}
		mode := normalizeUpstreamTimeoutMode(providerCfg.TimeoutMode)
		clients[providerID] = providerClientConfig{
			client:  newProviderHTTPClient(timeout),
			timeout: timeout,
			mode:    mode,
		}
	}
	return clients
}

func normalizeUpstreamTimeoutMode(mode string) string {
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case "", upstreamTimeoutModeTTFT:
		return upstreamTimeoutModeTTFT
	case "last_byte", upstreamTimeoutModeTotal:
		return upstreamTimeoutModeTotal
	default:
		return upstreamTimeoutModeTTFT
	}
}

func wrapBodyWithTTFTTimeout(body io.ReadCloser, timeout time.Duration) io.ReadCloser {
	if body == nil || timeout <= 0 {
		return body
	}

	wrapped := &timeoutBody{
		body:       body,
		timeoutErr: errUpstreamTTFTTimeout,
		stopOnData: true,
	}
	wrapped.timer = time.AfterFunc(timeout, func() {
		if wrapped.state.CompareAndSwap(timeoutStateWaiting, timeoutStateTimedOut) {
			_ = wrapped.body.Close()
		}
	})
	return wrapped
}

func wrapBodyWithTotalTimeout(body io.ReadCloser, timeout time.Duration) io.ReadCloser {
	if body == nil || timeout <= 0 {
		return body
	}

	wrapped := &timeoutBody{
		body:       body,
		timeoutErr: errUpstreamTotalTimeout,
		stopOnData: false,
	}
	wrapped.timer = time.AfterFunc(timeout, func() {
		if wrapped.state.CompareAndSwap(timeoutStateWaiting, timeoutStateTimedOut) {
			_ = wrapped.body.Close()
		}
	})
	return wrapped
}

func isUpstreamTTFTTimeout(err error) bool {
	return errors.Is(err, errUpstreamTTFTTimeout)
}

func isUpstreamTotalTimeout(err error) bool {
	return errors.Is(err, errUpstreamTotalTimeout)
}

func isHTTPTimeoutError(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return true
	}
	var netErr net.Error
	return errors.As(err, &netErr) && netErr.Timeout()
}

type timeoutBody struct {
	body       io.ReadCloser
	timer      *time.Timer
	timeoutErr error
	stopOnData bool
	state      atomic.Int32
}

const (
	timeoutStateWaiting int32 = iota
	timeoutStateDone
	timeoutStateTimedOut
)

func (b *timeoutBody) Read(p []byte) (int, error) {
	n, err := b.body.Read(p)

	if b.stopOnData && n > 0 {
		if b.state.CompareAndSwap(timeoutStateWaiting, timeoutStateDone) && b.timer != nil {
			b.timer.Stop()
		}
	}

	if err != nil {
		if b.state.CompareAndSwap(timeoutStateWaiting, timeoutStateDone) && b.timer != nil {
			b.timer.Stop()
		}
	}

	if b.state.Load() == timeoutStateTimedOut {
		return 0, b.timeoutErr
	}

	return n, err
}

func (b *timeoutBody) Close() error {
	if b.state.CompareAndSwap(timeoutStateWaiting, timeoutStateDone) && b.timer != nil {
		b.timer.Stop()
	}
	return b.body.Close()
}

type providerClientRegistry struct {
	mu      sync.RWMutex
	clients map[string]providerClientConfig
	configs map[string]config.ProviderConfig
}

func newProviderClientRegistry(providerConfigs map[string]config.ProviderConfig) *providerClientRegistry {
	return &providerClientRegistry{
		clients: buildProviderClients(providerConfigs),
		configs: cloneProviderConfigs(providerConfigs),
	}
}

func (r *providerClientRegistry) Get(providerID string) (providerClientConfig, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	cfg, ok := r.clients[providerID]
	return cfg, ok
}

func (r *providerClientRegistry) Config(providerID string) (config.ProviderConfig, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	cfg, ok := r.configs[providerID]
	return cfg, ok
}

func (r *providerClientRegistry) Update(providerConfigs map[string]config.ProviderConfig) {
	clients := buildProviderClients(providerConfigs)
	r.mu.Lock()
	r.clients = clients
	r.configs = cloneProviderConfigs(providerConfigs)
	r.mu.Unlock()
}

func cloneProviderConfigs(providerConfigs map[string]config.ProviderConfig) map[string]config.ProviderConfig {
	if len(providerConfigs) == 0 {
		return map[string]config.ProviderConfig{}
	}
	out := make(map[string]config.ProviderConfig, len(providerConfigs))
	for providerID, providerCfg := range providerConfigs {
		out[providerID] = providerCfg
	}
	return out
}
