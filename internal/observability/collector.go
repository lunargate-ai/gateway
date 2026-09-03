package observability

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/rs/zerolog/log"
)

const (
	defaultCollectorQueueCapacity   = 1000
	defaultCollectorMaxPayloadBytes = 16 << 20
	defaultCollectorMaxQueueBytes   = 64 << 20
)

type Event struct {
	Type string      `json:"type"`
	Data interface{} `json:"data"`
}

type CollectorRequest struct {
	Version        string    `json:"version"`
	GatewayID      string    `json:"gateway_id,omitempty"`
	GatewayVersion string    `json:"gateway_version,omitempty"`
	Timestamp      time.Time `json:"timestamp"`
	Events         []Event   `json:"events"`
}

type MetricEventData struct {
	RequestID               string            `json:"request_id"`
	Timestamp               time.Time         `json:"timestamp"`
	RequestType             string            `json:"request_type,omitempty"`
	UpstreamRequestType     string            `json:"upstream_request_type,omitempty"`
	DurationMS              int64             `json:"duration_ms"`
	GatewayPreUpstreamMS    *int64            `json:"gateway_pre_upstream_ms,omitempty"`
	TtftMS                  *int64            `json:"ttft_ms,omitempty"`
	TtltMS                  *int64            `json:"ttlt_ms,omitempty"`
	Provider                string            `json:"provider"`
	Model                   string            `json:"model"`
	User                    *string           `json:"user,omitempty"`
	SessionID               *string           `json:"session_id,omitempty"`
	TokensInput             int               `json:"tokens_input"`
	TokensOutput            int               `json:"tokens_output"`
	TokensInputCached       int               `json:"tokens_input_cached,omitempty"`
	TokensInputCacheWrite   int               `json:"tokens_input_cache_write,omitempty"`
	TokensInputCacheWrite5m int               `json:"tokens_input_cache_write_5m,omitempty"`
	TokensInputCacheWrite1h int               `json:"tokens_input_cache_write_1h,omitempty"`
	CostUSD                 float64           `json:"cost_usd"`
	StatusCode              int               `json:"status_code"`
	ErrorCode               *string           `json:"error_code,omitempty"`
	CacheHit                bool              `json:"cache_hit"`
	CacheKey                *string           `json:"cache_key,omitempty"`
	RouteUsed               *string           `json:"route_used,omitempty"`
	TargetIndex             *int              `json:"target_index,omitempty"`
	FallbackUsed            bool              `json:"fallback_used"`
	RetryCount              int               `json:"retry_count"`
	CircuitBreakerState     *string           `json:"circuit_breaker_state,omitempty"`
	Tags                    map[string]string `json:"tags,omitempty"`
}

// MetricErrorClass reduces failures to a finite, non-content-bearing label.
// Provider error types and messages are untrusted and may contain request
// content or credentials, so they belong only in explicitly enabled request
// logs, never in metrics-only events.
func MetricErrorClass(statusCode int, failed bool) *string {
	if !failed && statusCode < http.StatusBadRequest {
		return nil
	}
	class := "request_error"
	switch statusCode {
	case 499:
		class = "client_cancelled"
	case http.StatusBadRequest, http.StatusMethodNotAllowed, http.StatusUnprocessableEntity:
		class = "invalid_request"
	case http.StatusUnauthorized:
		class = "authentication"
	case http.StatusForbidden:
		class = "permission"
	case http.StatusNotFound:
		class = "not_found"
	case http.StatusConflict:
		class = "conflict"
	case http.StatusRequestEntityTooLarge:
		class = "request_too_large"
	case http.StatusRequestTimeout, http.StatusGatewayTimeout:
		class = "timeout"
	case http.StatusTooManyRequests:
		class = "rate_limited"
	default:
		if statusCode >= http.StatusInternalServerError || statusCode < http.StatusBadRequest {
			class = "upstream_error"
		}
	}
	return &class
}

type TraceEventData struct {
	RequestID string            `json:"request_id"`
	Timestamp time.Time         `json:"timestamp"`
	Phase     string            `json:"phase"`
	Tags      map[string]string `json:"tags,omitempty"`
}

type RequestLogEventData struct {
	RequestID           string            `json:"request_id"`
	Timestamp           time.Time         `json:"timestamp"`
	GatewayID           string            `json:"gateway_id,omitempty"`
	RequestType         string            `json:"request_type,omitempty"`
	UpstreamRequestType string            `json:"upstream_request_type,omitempty"`
	User                *string           `json:"user,omitempty"`
	SessionID           *string           `json:"session_id,omitempty"`
	Provider            string            `json:"provider"`
	Model               string            `json:"model"`
	StatusCode          int               `json:"status_code"`
	DurationMS          int64             `json:"duration_ms"`
	RouteUsed           *string           `json:"route_used,omitempty"`
	CacheHit            bool              `json:"cache_hit"`
	FallbackUsed        bool              `json:"fallback_used"`
	RetryCount          int               `json:"retry_count"`
	ErrorCode           *string           `json:"error_code,omitempty"`
	ErrorMessage        *string           `json:"error_message,omitempty"`
	Tags                map[string]string `json:"tags,omitempty"`
	Request             interface{}       `json:"request,omitempty"`
	Response            interface{}       `json:"response,omitempty"`
}

type collectorItem struct {
	requestID     string
	payload       []byte
	identity      collectorIdentity
	accountedSize int64
}

type collectorEnqueueStatus uint8

const (
	collectorEnqueueAccepted collectorEnqueueStatus = iota
	collectorEnqueueStopped
	collectorEnqueuePayloadTooLarge
	collectorEnqueueBudgetExceeded
	collectorEnqueueQueueFull
)

type collectorIdentity struct {
	backendURL string
	apiKey     string
}

type collectorRuntimeConfig struct {
	enabled        bool
	backendURL     string
	apiKey         string
	gatewayLat     string
	gatewayLon     string
	sharePrompts   bool
	shareResponses bool
}

type CollectorClient struct {
	gatewayVersion string

	httpClient      *http.Client
	queue           chan collectorItem
	ctx             context.Context
	cancel          context.CancelFunc
	wg              sync.WaitGroup
	stopOnce        sync.Once
	mu              sync.RWMutex
	cfg             collectorRuntimeConfig
	queueMu         sync.Mutex
	queueBytes      int64
	maxPayloadBytes int64
	maxQueueBytes   int64
	lastLogKey      string
	lastLogAt       time.Time
}

func NewCollectorClient(general config.GeneralConfig, cfg config.DataSharingConfig, gatewayVersion string) *CollectorClient {
	ctx, cancel := context.WithCancel(context.Background())
	c := &CollectorClient{
		gatewayVersion: gatewayVersion,
		httpClient: &http.Client{
			Timeout: 10 * time.Second,
			CheckRedirect: func(_ *http.Request, _ []*http.Request) error {
				return http.ErrUseLastResponse
			},
		},
		queue:           make(chan collectorItem, defaultCollectorQueueCapacity),
		ctx:             ctx,
		cancel:          cancel,
		cfg:             normalizeCollectorConfig(general, cfg),
		maxPayloadBytes: defaultCollectorMaxPayloadBytes,
		maxQueueBytes:   defaultCollectorMaxQueueBytes,
	}

	c.wg.Add(1)
	go c.worker()
	return c
}

func normalizeCollectorConfig(general config.GeneralConfig, cfg config.DataSharingConfig) collectorRuntimeConfig {
	backendURL := strings.TrimSpace(general.BackendURL)
	apiKey := strings.TrimSpace(general.APIKey)

	return collectorRuntimeConfig{
		enabled:        cfg.Enabled && backendURL != "" && apiKey != "",
		backendURL:     backendURL,
		apiKey:         apiKey,
		gatewayLat:     strings.TrimSpace(cfg.GatewayLat),
		gatewayLon:     strings.TrimSpace(cfg.GatewayLon),
		sharePrompts:   cfg.SharePrompts,
		shareResponses: cfg.ShareResponses,
	}
}

func (c *CollectorClient) snapshot() collectorRuntimeConfig {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.cfg
}

// UpdateConfig hot-reloads collector behavior without restarting the process.
func (c *CollectorClient) UpdateConfig(general config.GeneralConfig, cfg config.DataSharingConfig) {
	if c == nil {
		return
	}
	c.mu.Lock()
	c.cfg = normalizeCollectorConfig(general, cfg)
	c.mu.Unlock()
	log.Info().Bool("enabled", c.Enabled()).Msg("collector config updated")
}

func (c *CollectorClient) GatewayLat() string {
	if c == nil {
		return ""
	}
	return c.snapshot().gatewayLat
}

func (c *CollectorClient) GatewayLon() string {
	if c == nil {
		return ""
	}
	return c.snapshot().gatewayLon
}

func (c *CollectorClient) Enabled() bool {
	return c != nil && c.snapshot().enabled
}

func (c *CollectorClient) SharePrompts() bool {
	if c == nil {
		return false
	}
	cfg := c.snapshot()
	return cfg.enabled && cfg.sharePrompts
}

func (c *CollectorClient) ShareResponses() bool {
	if c == nil {
		return false
	}
	cfg := c.snapshot()
	return cfg.enabled && cfg.shareResponses
}

func (c *CollectorClient) Enqueue(ctx context.Context, requestID string, events []Event) {
	if c == nil {
		return
	}
	if len(events) == 0 {
		return
	}
	cfg := c.snapshot()
	if !cfg.enabled {
		return
	}

	req := CollectorRequest{
		Version:        "1.0",
		GatewayVersion: c.gatewayVersion,
		Timestamp:      time.Now().UTC(),
		Events:         events,
	}

	b, err := json.Marshal(req)
	if err != nil {
		log.Error().Err(err).Str("request_id", requestID).Msg("failed to marshal collector payload")
		return
	}

	item := collectorItem{
		requestID: requestID,
		payload:   b,
		identity: collectorIdentity{
			backendURL: cfg.backendURL,
			apiKey:     cfg.apiKey,
		},
	}
	status, queuedBytes := c.tryEnqueue(item)
	switch status {
	case collectorEnqueueAccepted, collectorEnqueueStopped:
		return
	case collectorEnqueuePayloadTooLarge:
		log.Warn().
			Str("request_id", requestID).
			Int("payload_bytes", len(b)).
			Int64("max_payload_bytes", c.payloadByteLimit()).
			Msg("collector payload too large, dropping event")
	case collectorEnqueueBudgetExceeded:
		log.Warn().
			Str("request_id", requestID).
			Int("payload_bytes", len(b)).
			Int64("queued_bytes", queuedBytes).
			Int64("max_queue_bytes", c.queueByteLimit()).
			Msg("collector queue byte limit reached, dropping event")
	case collectorEnqueueQueueFull:
		log.Warn().Str("request_id", requestID).Msg("collector queue full, dropping event")
	}
}

func (c *CollectorClient) tryEnqueue(item collectorItem) (collectorEnqueueStatus, int64) {
	payloadBytes := int64(len(item.payload))
	if payloadBytes > c.payloadByteLimit() {
		return collectorEnqueuePayloadTooLarge, c.queuedPayloadBytes()
	}

	c.queueMu.Lock()
	defer c.queueMu.Unlock()

	if c.ctx != nil && c.ctx.Err() != nil {
		return collectorEnqueueStopped, c.queueBytes
	}
	queueLimit := c.queueByteLimit()
	if payloadBytes > queueLimit || c.queueBytes > queueLimit-payloadBytes {
		return collectorEnqueueBudgetExceeded, c.queueBytes
	}

	item.accountedSize = payloadBytes
	select {
	case c.queue <- item:
		c.queueBytes += payloadBytes
		return collectorEnqueueAccepted, c.queueBytes
	default:
		return collectorEnqueueQueueFull, c.queueBytes
	}
}

func (c *CollectorClient) payloadByteLimit() int64 {
	if c.maxPayloadBytes > 0 {
		return c.maxPayloadBytes
	}
	return defaultCollectorMaxPayloadBytes
}

func (c *CollectorClient) queueByteLimit() int64 {
	if c.maxQueueBytes > 0 {
		return c.maxQueueBytes
	}
	return defaultCollectorMaxQueueBytes
}

func (c *CollectorClient) queuedPayloadBytes() int64 {
	c.queueMu.Lock()
	defer c.queueMu.Unlock()
	return c.queueBytes
}

func (c *CollectorClient) releaseQueuedPayload(item collectorItem) {
	if item.accountedSize <= 0 {
		return
	}
	c.queueMu.Lock()
	c.queueBytes -= item.accountedSize
	if c.queueBytes < 0 {
		c.queueBytes = 0
	}
	c.queueMu.Unlock()
}

func (c *CollectorClient) drainQueue() {
	c.queueMu.Lock()
	defer c.queueMu.Unlock()

	for {
		select {
		case item := <-c.queue:
			c.queueBytes -= item.accountedSize
		default:
			if c.queueBytes < 0 {
				c.queueBytes = 0
			}
			return
		}
	}
}

// Stop shuts down the background collector worker.
func (c *CollectorClient) Stop() {
	if c == nil {
		return
	}
	c.stopOnce.Do(func() {
		c.cancel()
		c.wg.Wait()
		c.drainQueue()
	})
}

func (c *CollectorClient) worker() {
	defer c.wg.Done()
	for {
		select {
		case <-c.ctx.Done():
			return
		case item := <-c.queue:
			c.releaseQueuedPayload(item)
			c.sendWithRetry(c.ctx, item)
		}
	}
}

func (c *CollectorClient) sendWithRetry(ctx context.Context, item collectorItem) {
	var lastErr error
	for attempt := 0; attempt < 3; attempt++ {
		if ctx.Err() != nil {
			return
		}
		if err := c.send(ctx, item); err == nil {
			return
		} else {
			lastErr = err
			if !isRetryableSendError(err) {
				break
			}
			timer := time.NewTimer(time.Duration(attempt+1) * 500 * time.Millisecond)
			select {
			case <-ctx.Done():
				timer.Stop()
				return
			case <-timer.C:
			}
		}
	}

	if lastErr != nil {
		c.logSendError(item.requestID, lastErr)
	}
}

func (c *CollectorClient) send(ctx context.Context, item collectorItem) error {
	cfg := c.snapshot()
	if !cfg.enabled {
		return nil
	}
	if cfg.backendURL != item.identity.backendURL || cfg.apiKey != item.identity.apiKey {
		log.Debug().Str("request_id", item.requestID).Msg("collector target changed, dropping queued payload")
		return nil
	}

	collectorURL, err := url.JoinPath(item.identity.backendURL, "collector")
	if err != nil {
		return err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", collectorURL, bytes.NewReader(item.payload))
	if err != nil {
		return err
	}

	req.Header.Set("Authorization", "Bearer "+item.identity.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		io.Copy(io.Discard, resp.Body)
		return nil
	}

	return &httpStatusError{
		statusCode: resp.StatusCode,
		detail:     readResponseSnippet(resp.Body),
	}
}

type httpStatusError struct {
	statusCode int
	detail     string
}

func (e *httpStatusError) Error() string {
	if e.detail == "" {
		return "unexpected status code: " + strconv.Itoa(e.statusCode) + " (" + http.StatusText(e.statusCode) + ")"
	}
	return "unexpected status code: " + strconv.Itoa(e.statusCode) + " (" + http.StatusText(e.statusCode) + "): " + e.detail
}

func readResponseSnippet(r io.Reader) string {
	if r == nil {
		return ""
	}
	body, err := io.ReadAll(io.LimitReader(r, 2048))
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(body))
}

func isRetryableSendError(err error) bool {
	var statusErr *httpStatusError
	if !errors.As(err, &statusErr) {
		return true
	}
	return statusErr.statusCode == http.StatusTooManyRequests || statusErr.statusCode >= 500
}

func (c *CollectorClient) logSendError(requestID string, err error) {
	if err == nil {
		return
	}

	key := err.Error()
	now := time.Now()
	if key == c.lastLogKey && now.Sub(c.lastLogAt) < 30*time.Second {
		return
	}
	c.lastLogKey = key
	c.lastLogAt = now

	var statusErr *httpStatusError
	if errors.As(err, &statusErr) && (statusErr.statusCode == http.StatusUnauthorized || statusErr.statusCode == http.StatusForbidden) {
		event := log.Warn().
			Str("request_id", requestID).
			Int("status_code", statusErr.statusCode).
			Str("status_text", http.StatusText(statusErr.statusCode))
		if statusErr.detail != "" {
			event = event.Str("detail", statusErr.detail)
		}
		event.Msg("collector authentication rejected by lunargate.ai; go to app.lunargate.ai and check general.api_key")
		return
	}

	log.Warn().Err(err).Str("request_id", requestID).Msg("failed to send collector payload after retries")
}
