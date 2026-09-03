package api

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/safeurl"
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
			client:  newProviderHTTPClient(),
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

func doProviderRequest(
	request *http.Request,
	clientCfg providerClientConfig,
	provider string,
	failureAction string,
) (*http.Response, error) {
	if request == nil {
		return nil, fmt.Errorf("%s %s: request is nil", failureAction, provider)
	}
	if clientCfg.client == nil {
		return nil, fmt.Errorf("%s %s: HTTP client is nil", failureAction, provider)
	}

	timeout := clientCfg.timeout
	if timeout <= 0 {
		timeout = defaultUpstreamTimeout
	}
	mode := normalizeUpstreamTimeoutMode(clientCfg.mode)
	parent := request.Context()
	attempt := newUpstreamTimeoutAttempt(parent, timeout, mode)
	attemptRequest := request.WithContext(attempt.ctx)

	response, err := clientCfg.client.Do(attemptRequest)
	if err != nil {
		timeoutErr := attempt.finish()
		if response != nil && response.Body != nil {
			_ = response.Body.Close()
		}
		err = safeurl.RedactTransportError(err, attemptRequest.URL)
		if timeoutErr != nil {
			return nil, upstreamTimeoutError(mode, provider)
		}
		// A parent deadline belongs to the client request, not the provider.
		// Preserve it so retry and fallback remain terminal for cancellation.
		if parent.Err() == nil && isHTTPTimeoutError(err) {
			return nil, upstreamTimeoutError(mode, provider)
		}
		return nil, fmt.Errorf("%s %s: %w", failureAction, provider, err)
	}
	if response.Request == nil {
		response.Request = attemptRequest
	}
	if response.Body == nil {
		_ = attempt.finish()
		return response, nil
	}
	wrappedBody := &upstreamTimeoutBody{
		body:    response.Body,
		attempt: attempt,
		mode:    mode,
	}
	if attachErr := attempt.attachBody(wrappedBody.closeUnderlying); attachErr != nil {
		_ = wrappedBody.closeUnderlying()
		if errors.Is(attachErr, errUpstreamTTFTTimeout) || errors.Is(attachErr, errUpstreamTotalTimeout) {
			return nil, upstreamTimeoutError(mode, provider)
		}
		return nil, fmt.Errorf("%s %s: %w", failureAction, provider, attachErr)
	}
	if parentErr := parent.Err(); parentErr != nil {
		_ = attempt.finish()
		_ = wrappedBody.closeUnderlying()
		return nil, fmt.Errorf("%s %s: %w", failureAction, provider, parentErr)
	}
	if timeoutErr := attempt.timeoutError(); timeoutErr != nil {
		_ = wrappedBody.closeUnderlying()
		return nil, upstreamTimeoutError(mode, provider)
	}
	response.Body = wrappedBody
	return response, nil
}

func upstreamTimeoutError(mode string, provider string) error {
	if mode == upstreamTimeoutModeTotal {
		return fmt.Errorf("%w: provider %s", errUpstreamTotalTimeout, provider)
	}
	return fmt.Errorf("%w: provider %s", errUpstreamTTFTTimeout, provider)
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

type upstreamTimeoutAttempt struct {
	ctx        context.Context
	cancel     context.CancelCauseFunc
	timeoutErr error
	timer      *time.Timer
	closeBody  func() error
	mu         sync.Mutex
	state      upstreamTimeoutState
}

type upstreamTimeoutState uint8

const (
	upstreamTimeoutActive upstreamTimeoutState = iota
	upstreamTimeoutStopped
	upstreamTimeoutFinished
	upstreamTimeoutFired
)

func newUpstreamTimeoutAttempt(parent context.Context, timeout time.Duration, mode string) *upstreamTimeoutAttempt {
	ctx, cancel := context.WithCancelCause(parent)
	attempt := &upstreamTimeoutAttempt{
		ctx:        ctx,
		cancel:     cancel,
		timeoutErr: errUpstreamTTFTTimeout,
	}
	if mode == upstreamTimeoutModeTotal {
		attempt.timeoutErr = errUpstreamTotalTimeout
	}
	attempt.timer = time.AfterFunc(timeout, attempt.fire)
	return attempt
}

func (a *upstreamTimeoutAttempt) fire() {
	var closeBody func() error
	a.mu.Lock()
	if a.state != upstreamTimeoutActive {
		a.mu.Unlock()
		return
	}
	a.cancel(a.timeoutErr)
	if errors.Is(context.Cause(a.ctx), a.timeoutErr) {
		a.state = upstreamTimeoutFired
		closeBody = a.closeBody
	} else {
		// The parent cancellation won the race. Do not relabel it as a provider
		// timeout even though this timer callback was also ready to run.
		a.state = upstreamTimeoutFinished
	}
	a.mu.Unlock()
	if closeBody != nil {
		_ = closeBody()
	}
}

func (a *upstreamTimeoutAttempt) attachBody(closeBody func() error) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	switch a.state {
	case upstreamTimeoutActive, upstreamTimeoutStopped:
		a.closeBody = closeBody
		return nil
	case upstreamTimeoutFired:
		return a.timeoutErr
	default:
		return context.Cause(a.ctx)
	}
}

func (a *upstreamTimeoutAttempt) stopDeadline() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	switch a.state {
	case upstreamTimeoutActive:
		a.state = upstreamTimeoutStopped
		if a.timer != nil {
			a.timer.Stop()
		}
		return nil
	case upstreamTimeoutFired:
		return a.timeoutErr
	default:
		return nil
	}
}

func (a *upstreamTimeoutAttempt) finish() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.state == upstreamTimeoutFired {
		return a.timeoutErr
	}
	if a.state == upstreamTimeoutFinished {
		return nil
	}
	a.state = upstreamTimeoutFinished
	if a.timer != nil {
		a.timer.Stop()
	}
	a.cancel(nil)
	return nil
}

func (a *upstreamTimeoutAttempt) timeoutError() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.state == upstreamTimeoutFired {
		return a.timeoutErr
	}
	return nil
}

type upstreamTimeoutBody struct {
	body      io.ReadCloser
	attempt   *upstreamTimeoutAttempt
	mode      string
	closeOnce sync.Once
	closeErr  error
}

func (b *upstreamTimeoutBody) Read(p []byte) (int, error) {
	n, err := b.body.Read(p)

	var timeoutErr error
	if b.mode == upstreamTimeoutModeTTFT && n > 0 {
		timeoutErr = b.attempt.stopDeadline()
	}
	if err != nil {
		if finishErr := b.attempt.finish(); timeoutErr == nil {
			timeoutErr = finishErr
		}
	}
	if timeoutErr == nil {
		timeoutErr = b.attempt.timeoutError()
	}
	if timeoutErr != nil {
		return 0, timeoutErr
	}
	return n, err
}

func (b *upstreamTimeoutBody) Close() error {
	timeoutErr := b.attempt.finish()
	err := b.closeUnderlying()
	if timeoutErr != nil {
		return timeoutErr
	}
	return err
}

func (b *upstreamTimeoutBody) closeUnderlying() error {
	b.closeOnce.Do(func() {
		b.closeErr = b.body.Close()
	})
	return b.closeErr
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

func (r *providerClientRegistry) Snapshot(providerID string) (providerClientConfig, config.ProviderConfig, bool) {
	if r == nil {
		return providerClientConfig{}, config.ProviderConfig{}, false
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	clientCfg, clientOK := r.clients[providerID]
	providerCfg, configOK := r.configs[providerID]
	return clientCfg, providerCfg, clientOK && configOK
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
		if providerCfg.Temperature != nil {
			value := *providerCfg.Temperature
			providerCfg.Temperature = &value
		}
		if providerCfg.TopP != nil {
			value := *providerCfg.TopP
			providerCfg.TopP = &value
		}
		if providerCfg.TopK != nil {
			value := *providerCfg.TopK
			providerCfg.TopK = &value
		}
		if len(providerCfg.Extra) > 0 {
			providerCfg.Extra = cloneStringMap(providerCfg.Extra)
		} else {
			providerCfg.Extra = nil
		}
		providerCfg.Models.Static = append([]string(nil), providerCfg.Models.Static...)
		providerCfg.Capabilities.HostedTools = append([]string(nil), providerCfg.Capabilities.HostedTools...)
		providerCfg.Capabilities.ReasoningEffortLevels = append([]string(nil), providerCfg.Capabilities.ReasoningEffortLevels...)
		out[providerID] = providerCfg
	}
	return out
}

func cloneStringMap(values map[string]string) map[string]string {
	cloned := make(map[string]string, len(values))
	for key, value := range values {
		cloned[key] = value
	}
	return cloned
}
