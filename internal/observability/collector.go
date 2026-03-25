package observability

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/rs/zerolog/log"
)

type Event struct {
	Type string      `json:"type"`
	Data interface{} `json:"data"`
}

type CollectorRequest struct {
	Version        string    `json:"version"`
	GatewayID      string    `json:"gateway_id"`
	GatewayVersion string    `json:"gateway_version,omitempty"`
	Timestamp      time.Time `json:"timestamp"`
	Events         []Event   `json:"events"`
}

type MetricEventData struct {
	RequestID           string            `json:"request_id"`
	Timestamp           time.Time         `json:"timestamp"`
	RequestType         string            `json:"request_type,omitempty"`
	DurationMS          int64             `json:"duration_ms"`
	GatewayPreUpstreamMS *int64           `json:"gateway_pre_upstream_ms,omitempty"`
	TtftMS              *int64            `json:"ttft_ms,omitempty"`
	TtltMS              *int64            `json:"ttlt_ms,omitempty"`
	Provider            string            `json:"provider"`
	Model               string            `json:"model"`
	User                *string           `json:"user,omitempty"`
	SessionID           *string           `json:"session_id,omitempty"`
	TokensInput         int               `json:"tokens_input"`
	TokensOutput        int               `json:"tokens_output"`
	CostUSD             float64           `json:"cost_usd"`
	StatusCode          int               `json:"status_code"`
	ErrorCode           *string           `json:"error_code,omitempty"`
	ErrorMessage        *string           `json:"error_message,omitempty"`
	CacheHit            bool              `json:"cache_hit"`
	CacheKey            *string           `json:"cache_key,omitempty"`
	RouteUsed           *string           `json:"route_used,omitempty"`
	TargetIndex         *int              `json:"target_index,omitempty"`
	FallbackUsed        bool              `json:"fallback_used"`
	RetryCount          int               `json:"retry_count"`
	CircuitBreakerState *string           `json:"circuit_breaker_state,omitempty"`
	Tags                map[string]string `json:"tags,omitempty"`
}

type TraceEventData struct {
	RequestID  string            `json:"request_id"`
	Timestamp  time.Time         `json:"timestamp"`
	Phase      string            `json:"phase"`
	Tags       map[string]string `json:"tags,omitempty"`
}

type RequestLogEventData struct {
	RequestID    string            `json:"request_id"`
	Timestamp    time.Time         `json:"timestamp"`
	GatewayID    string            `json:"gateway_id"`
	RequestType  string            `json:"request_type,omitempty"`
	User         *string           `json:"user,omitempty"`
	SessionID    *string           `json:"session_id,omitempty"`
	Provider     string            `json:"provider"`
	Model        string            `json:"model"`
	StatusCode   int               `json:"status_code"`
	DurationMS   int64             `json:"duration_ms"`
	RouteUsed    *string           `json:"route_used,omitempty"`
	CacheHit     bool              `json:"cache_hit"`
	FallbackUsed bool              `json:"fallback_used"`
	RetryCount   int               `json:"retry_count"`
	ErrorCode    *string           `json:"error_code,omitempty"`
	ErrorMessage *string           `json:"error_message,omitempty"`
	Tags         map[string]string `json:"tags,omitempty"`
	Request      interface{}       `json:"request,omitempty"`
	Response     interface{}       `json:"response,omitempty"`
}

type collectorItem struct {
	requestID string
	payload   []byte
}

type CollectorClient struct {
	backendURL     string
	gatewayID      string
	gatewayVersion string
	apiKey         string
	gatewayLat     string
	gatewayLon     string

	sharePrompts   bool
	shareResponses bool

	httpClient *http.Client
	queue      chan collectorItem
}

func NewCollectorClient(cfg config.DataSharingConfig, gatewayVersion string) *CollectorClient {
	if !cfg.Enabled {
		return nil
	}
	if strings.TrimSpace(cfg.BackendURL) == "" || strings.TrimSpace(cfg.GatewayID) == "" || strings.TrimSpace(cfg.APIKey) == "" {
		return nil
	}

	c := &CollectorClient{
		backendURL:     strings.TrimSpace(cfg.BackendURL),
		gatewayID:      strings.TrimSpace(cfg.GatewayID),
		gatewayVersion: gatewayVersion,
		apiKey:         strings.TrimSpace(cfg.APIKey),
		gatewayLat:     strings.TrimSpace(cfg.GatewayLat),
		gatewayLon:     strings.TrimSpace(cfg.GatewayLon),
		sharePrompts:   cfg.SharePrompts,
		shareResponses: cfg.ShareResponses,
		httpClient: &http.Client{
			Timeout: 10 * time.Second,
		},
		queue: make(chan collectorItem, 1000),
	}

	go c.worker()
	return c
}

func (c *CollectorClient) GatewayLat() string {
	if c == nil {
		return ""
	}
	return c.gatewayLat
}

func (c *CollectorClient) GatewayLon() string {
	if c == nil {
		return ""
	}
	return c.gatewayLon
}

func (c *CollectorClient) Enabled() bool {
	return c != nil
}

func (c *CollectorClient) SharePrompts() bool {
	return c != nil && c.sharePrompts
}

func (c *CollectorClient) ShareResponses() bool {
	return c != nil && c.shareResponses
}

func (c *CollectorClient) GatewayID() string {
	if c == nil {
		return ""
	}
	return c.gatewayID
}

func (c *CollectorClient) Enqueue(ctx context.Context, requestID string, events []Event) {
	if c == nil {
		return
	}
	if len(events) == 0 {
		return
	}

	req := CollectorRequest{
		Version:        "1.0",
		GatewayID:      c.gatewayID,
		GatewayVersion: c.gatewayVersion,
		Timestamp:      time.Now().UTC(),
		Events:         events,
	}

	b, err := json.Marshal(req)
	if err != nil {
		log.Error().Err(err).Str("request_id", requestID).Msg("failed to marshal collector payload")
		return
	}

	item := collectorItem{requestID: requestID, payload: b}
	select {
	case c.queue <- item:
	default:
		log.Warn().Str("request_id", requestID).Msg("collector queue full, dropping event")
	}
}

func (c *CollectorClient) worker() {
	for item := range c.queue {
		c.sendWithRetry(item)
	}
}

func (c *CollectorClient) sendWithRetry(item collectorItem) {
	var lastErr error
	for attempt := 0; attempt < 3; attempt++ {
		if err := c.send(item.payload); err == nil {
			return
		} else {
			lastErr = err
			time.Sleep(time.Duration(attempt+1) * 500 * time.Millisecond)
		}
	}

	if lastErr != nil {
		log.Warn().Err(lastErr).Str("request_id", item.requestID).Msg("failed to send collector payload after retries")
	}
}

func (c *CollectorClient) send(payload []byte) error {
	collectorURL, err := url.JoinPath(c.backendURL, "collector")
	if err != nil {
		return err
	}

	req, err := http.NewRequest("POST", collectorURL, bytes.NewReader(payload))
	if err != nil {
		return err
	}

	req.Header.Set("Authorization", "Bearer "+c.apiKey)
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

	io.Copy(io.Discard, resp.Body)
	return &httpStatusError{statusCode: resp.StatusCode}
}

type httpStatusError struct {
	statusCode int
}

func (e *httpStatusError) Error() string {
	return "unexpected status code: " + strconv.Itoa(e.statusCode) + " (" + http.StatusText(e.statusCode) + ")"
}
