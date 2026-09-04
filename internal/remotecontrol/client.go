package remotecontrol

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/safeurl"
	"github.com/rs/zerolog/log"
)

type ModelListFunc func(context.Context) []string
type ModelSnapshotFunc func() []string

const (
	defaultHeartbeatInterval     = 20 * time.Second
	defaultModelRefreshTimeout   = 8 * time.Second
	defaultSandboxConcurrency    = 8
	defaultSandboxResponseLimit  = 16 << 20
	defaultWebSocketReadLimit    = 16 << 20
	defaultWebSocketWriteTimeout = 10 * time.Second
	initialReconnectBackoff      = time.Second
	maximumReconnectBackoff      = 15 * time.Second
)

type Client struct {
	backendURL            string
	apiKey                string
	instanceID            string
	version               string
	localBaseURL          string
	localAuthHeader       string
	localAuthValue        string
	routeNames            func() []string
	modelSnapshotIDs      ModelSnapshotFunc
	modelIDs              ModelListFunc
	httpClient            *http.Client
	refreshCh             chan struct{}
	heartbeatInterval     time.Duration
	modelRefreshTimeout   time.Duration
	sandboxCommandLimit   int
	sandboxCommandOnce    sync.Once
	sandboxCommandSlots   chan struct{}
	sandboxResponseLimit  int64
	websocketReadLimit    int64
	websocketWriteTimeout time.Duration
	lastLogKey            string
	lastLogAt             time.Time
}

type helloMessage struct {
	Type       string   `json:"type"`
	InstanceID string   `json:"instance_id"`
	Version    string   `json:"version,omitempty"`
	Routes     []string `json:"routes,omitempty"`
	Models     []string `json:"models,omitempty"`
}

type heartbeatMessage struct {
	Type string `json:"type"`
}

type sandboxExecuteMessage struct {
	Type        string                 `json:"type"`
	CommandID   string                 `json:"command_id"`
	Target      sandboxTarget          `json:"target"`
	RequestType string                 `json:"request_type"`
	Request     map[string]interface{} `json:"request"`
}

type sandboxTarget struct {
	Mode  string `json:"mode"`
	Value string `json:"value"`
}

type sandboxResponseMessage struct {
	Type       string            `json:"type"`
	CommandID  string            `json:"command_id"`
	InstanceID string            `json:"instance_id,omitempty"`
	OK         bool              `json:"ok"`
	StatusCode int               `json:"status_code"`
	RequestID  string            `json:"request_id,omitempty"`
	Provider   string            `json:"provider,omitempty"`
	Model      string            `json:"model,omitempty"`
	RouteUsed  string            `json:"route_used,omitempty"`
	Headers    map[string]string `json:"headers,omitempty"`
	Body       interface{}       `json:"body,omitempty"`
	Error      string            `json:"error,omitempty"`
}

func NewClient(
	general config.GeneralConfig,
	dataSharing config.DataSharingConfig,
	security config.SecurityConfig,
	version string,
	localBaseURL string,
	routeNames func() []string,
	modelSnapshotIDs ModelSnapshotFunc,
	modelIDs ModelListFunc,
) *Client {
	if !dataSharing.Enabled {
		return nil
	}
	if !dataSharing.RemoteControl {
		return nil
	}
	apiKey := strings.TrimSpace(general.APIKey)
	if apiKey == "" {
		log.Warn().Msg("remote control disabled: general.api_key missing in config")
		return nil
	}
	backendURL := strings.TrimSpace(general.BackendURL)
	if backendURL == "" {
		log.Warn().Msg("remote control disabled: general.backend_url missing in config")
		return nil
	}
	localBaseURL = strings.TrimRight(strings.TrimSpace(localBaseURL), "/")
	if !isLoopbackBaseURL(localBaseURL) {
		log.Warn().Msg("remote control disabled: local sandbox endpoint is not loopback")
		return nil
	}
	instanceID, err := localInstanceID()
	if err != nil {
		log.Error().Err(err).Msg("remote control disabled: failed to create instance identity")
		return nil
	}
	localAuthHeader, localAuthValue := loopbackCredential(security)
	return &Client{
		backendURL:       backendURL,
		apiKey:           apiKey,
		instanceID:       instanceID,
		version:          version,
		localBaseURL:     localBaseURL,
		localAuthHeader:  localAuthHeader,
		localAuthValue:   localAuthValue,
		routeNames:       routeNames,
		modelSnapshotIDs: modelSnapshotIDs,
		modelIDs:         modelIDs,
		httpClient: &http.Client{
			Timeout: 5 * time.Minute,
			CheckRedirect: func(_ *http.Request, _ []*http.Request) error {
				return http.ErrUseLastResponse
			},
		},
		refreshCh:             make(chan struct{}, 1),
		heartbeatInterval:     defaultHeartbeatInterval,
		modelRefreshTimeout:   defaultModelRefreshTimeout,
		sandboxCommandLimit:   defaultSandboxConcurrency,
		websocketReadLimit:    defaultWebSocketReadLimit,
		websocketWriteTimeout: defaultWebSocketWriteTimeout,
	}
}

func isLoopbackBaseURL(raw string) bool {
	parsed, err := url.Parse(raw)
	if err != nil || parsed == nil || parsed.Opaque != "" || parsed.Hostname() == "" {
		return false
	}
	switch strings.ToLower(parsed.Scheme) {
	case "http", "https":
	default:
		return false
	}
	ip := net.ParseIP(strings.TrimSpace(parsed.Hostname()))
	return ip != nil && ip.IsLoopback()
}

func loopbackCredential(security config.SecurityConfig) (string, string) {
	if !security.Enabled || !strings.EqualFold(strings.TrimSpace(security.Provider), "api_key") || len(security.APIKey.Keys) == 0 {
		return "", ""
	}

	header := strings.TrimSpace(security.APIKey.Header)
	if header == "" {
		header = "Authorization"
	}
	credential := strings.TrimSpace(security.APIKey.Keys[0].Value)
	if credential == "" {
		return "", ""
	}
	if prefix := strings.TrimSpace(security.APIKey.Prefix); prefix != "" {
		credential = prefix + " " + credential
	}
	return header, credential
}

func (c *Client) Start(ctx context.Context) {
	if c == nil {
		return
	}
	go c.run(ctx)
}

func (c *Client) RefreshHello() {
	if c == nil {
		return
	}
	select {
	case c.refreshCh <- struct{}{}:
	default:
	}
}

func (c *Client) InstanceID() string {
	if c == nil {
		return ""
	}
	return c.instanceID
}

func (c *Client) run(ctx context.Context) {
	backoff := newReconnectBackoff()
	for {
		if ctx.Err() != nil {
			return
		}
		healthy := make(chan struct{}, 1)
		if err := c.connectAndServeWithHealth(ctx, healthy); err != nil && ctx.Err() == nil {
			c.logConnectionIssue(err)
		}
		wasHealthy := false
		select {
		case <-healthy:
			wasHealthy = true
		default:
		}
		reconnectDelay := backoff.nextDelay(wasHealthy)
		timer := time.NewTimer(reconnectDelay)
		select {
		case <-ctx.Done():
			timer.Stop()
			return
		case <-timer.C:
		}
	}
}

func (c *Client) connectAndServe(ctx context.Context) error {
	return c.connectAndServeWithHealth(ctx, nil)
}

func (c *Client) connectAndServeWithHealth(ctx context.Context, healthy chan<- struct{}) error {
	wsURL, err := c.websocketURL()
	if err != nil {
		return err
	}
	requestURL, err := url.Parse(wsURL)
	if err != nil {
		return fmt.Errorf("invalid remote control WebSocket URL")
	}
	headers := http.Header{}
	headers.Set("Authorization", "Bearer "+strings.TrimSpace(c.apiKey))
	conn, resp, err := websocket.DefaultDialer.DialContext(ctx, wsURL, headers)
	if err != nil {
		return classifyDialError(err, resp, requestURL)
	}
	defer conn.Close()
	readLimit := c.websocketReadLimit
	if readLimit <= 0 {
		readLimit = defaultWebSocketReadLimit
	}
	conn.SetReadLimit(readLimit)

	closeCh := make(chan struct{})
	go closeConnectionOnContext(ctx, conn, closeCh)
	defer close(closeCh)

	writeJSON := c.boundedWebSocketWriter(conn)

	if err := writeJSON(c.buildSnapshotHello()); err != nil {
		return err
	}

	connectionCtx, cancelConnection := context.WithCancel(ctx)
	defer cancelConnection()
	errCh := make(chan error, 1)
	asyncResponses := make(chan sandboxResponseMessage, c.sandboxConcurrency())
	go c.sandboxResponseLoop(connectionCtx, writeJSON, asyncResponses, errCh)
	go c.readLoop(connectionCtx, conn, writeJSON, asyncResponses, errCh, healthy)

	heartbeatInterval := c.heartbeatInterval
	if heartbeatInterval <= 0 {
		heartbeatInterval = defaultHeartbeatInterval
	}
	heartbeatTicker := time.NewTicker(heartbeatInterval)
	defer heartbeatTicker.Stop()

	modelRefreshCtx, cancelModelRefresh := context.WithCancel(ctx)
	var modelRefreshWG sync.WaitGroup
	defer func() {
		cancelModelRefresh()
		modelRefreshWG.Wait()
	}()

	type modelRefreshResult struct {
		models []string
		ok     bool
	}
	modelRefreshResults := make(chan modelRefreshResult, 1)
	modelRefreshActive := false
	modelRefreshPending := false
	startModelRefresh := func() {
		if modelRefreshActive || c.modelIDs == nil {
			return
		}
		modelRefreshActive = true
		modelRefreshWG.Add(1)
		go func() {
			defer modelRefreshWG.Done()
			timeout := c.modelRefreshTimeout
			if timeout <= 0 {
				timeout = defaultModelRefreshTimeout
			}
			refreshCtx, cancel := context.WithTimeout(modelRefreshCtx, timeout)
			models := c.modelIDs(refreshCtx)
			completed := refreshCtx.Err() == nil
			cancel()
			result := modelRefreshResult{models: models, ok: completed}
			select {
			case modelRefreshResults <- result:
			case <-modelRefreshCtx.Done():
			}
		}()
	}
	startModelRefresh()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-heartbeatTicker.C:
			if err := writeJSON(heartbeatMessage{Type: "heartbeat"}); err != nil {
				return err
			}
		case <-c.refreshCh:
			if err := writeJSON(c.buildSnapshotHello()); err != nil {
				return err
			}
			if modelRefreshActive {
				modelRefreshPending = true
			} else {
				startModelRefresh()
			}
		case result := <-modelRefreshResults:
			modelRefreshActive = false
			if result.ok {
				if err := writeJSON(c.buildHello(result.models)); err != nil {
					return err
				}
			}
			if modelRefreshPending {
				modelRefreshPending = false
				startModelRefresh()
			}
		case err := <-errCh:
			return err
		}
	}
}

type reconnectBackoff struct {
	current time.Duration
}

func newReconnectBackoff() reconnectBackoff {
	return reconnectBackoff{current: initialReconnectBackoff}
}

func (b *reconnectBackoff) nextDelay(healthy bool) time.Duration {
	if healthy || b.current <= 0 {
		b.current = initialReconnectBackoff
	}

	delay := b.current
	b.current *= 2
	if b.current > maximumReconnectBackoff {
		b.current = maximumReconnectBackoff
	}
	return delay
}

type websocketJSONWriter interface {
	SetWriteDeadline(time.Time) error
	WriteJSON(interface{}) error
}

func (c *Client) boundedWebSocketWriter(conn websocketJSONWriter) func(interface{}) error {
	var writeMu sync.Mutex
	return func(value interface{}) error {
		writeMu.Lock()
		defer writeMu.Unlock()

		timeout := c.websocketWriteTimeout
		if timeout <= 0 {
			timeout = defaultWebSocketWriteTimeout
		}
		if err := conn.SetWriteDeadline(time.Now().Add(timeout)); err != nil {
			return err
		}
		return conn.WriteJSON(value)
	}
}

func closeConnectionOnContext(ctx context.Context, connection io.Closer, stop <-chan struct{}) {
	select {
	case <-ctx.Done():
		_ = connection.Close()
	case <-stop:
	}
}

func (c *Client) readLoop(
	ctx context.Context,
	conn *websocket.Conn,
	writeJSON func(interface{}) error,
	asyncResponses chan<- sandboxResponseMessage,
	errCh chan<- error,
	healthy chan<- struct{},
) {
	for {
		_, payload, err := conn.ReadMessage()
		if err != nil {
			select {
			case errCh <- err:
			default:
			}
			return
		}
		var msg sandboxExecuteMessage
		if err := decodeSingleJSONDocument(payload, &msg); err != nil {
			if strings.TrimSpace(msg.Type) != "sandbox.execute" {
				select {
				case errCh <- fmt.Errorf("failed to decode remote control message: %w", err):
				default:
				}
				return
			}
			if !queueSandboxResponse(asyncResponses, errCh, c.sandboxErrorResponse(msg.CommandID, fmt.Errorf("failed to decode sandbox command: %w", err))) {
				return
			}
			continue
		}
		msgType := strings.TrimSpace(msg.Type)
		switch msgType {
		case "heartbeat":
			signalConnectionHealthy(healthy)
			continue
		case "sandbox.execute":
			signalConnectionHealthy(healthy)
			slots := c.sandboxSlots()
			select {
			case slots <- struct{}{}:
				go func(command sandboxExecuteMessage) {
					defer func() { <-slots }()
					c.handleSandboxCommand(ctx, writeJSON, command)
				}(msg)
			default:
				if !queueSandboxResponse(asyncResponses, errCh, c.sandboxOverloadResponse(msg.CommandID)) {
					return
				}
			}
		}
	}
}

func signalConnectionHealthy(healthy chan<- struct{}) {
	if healthy == nil {
		return
	}
	select {
	case healthy <- struct{}{}:
	default:
	}
}

func queueSandboxResponse(responses chan<- sandboxResponseMessage, errCh chan<- error, response sandboxResponseMessage) bool {
	select {
	case responses <- response:
		return true
	default:
		select {
		case errCh <- fmt.Errorf("sandbox response queue is full"):
		default:
		}
		return false
	}
}

func (c *Client) sandboxConcurrency() int {
	if c.sandboxCommandLimit > 0 {
		return c.sandboxCommandLimit
	}
	return defaultSandboxConcurrency
}

func (c *Client) sandboxSlots() chan struct{} {
	c.sandboxCommandOnce.Do(func() {
		if c.sandboxCommandSlots == nil {
			c.sandboxCommandSlots = make(chan struct{}, c.sandboxConcurrency())
		}
	})
	return c.sandboxCommandSlots
}

func (c *Client) sandboxOverloadResponse(commandID string) sandboxResponseMessage {
	return sandboxResponseMessage{
		Type:       "sandbox.response",
		CommandID:  commandID,
		InstanceID: c.instanceID,
		OK:         false,
		StatusCode: http.StatusTooManyRequests,
		Error:      "sandbox command concurrency limit reached",
		Headers:    map[string]string{},
	}
}

func (c *Client) sandboxResponseLoop(
	ctx context.Context,
	writeJSON func(interface{}) error,
	responses <-chan sandboxResponseMessage,
	errCh chan<- error,
) {
	for {
		select {
		case <-ctx.Done():
			return
		case response := <-responses:
			if err := writeJSON(response); err != nil {
				select {
				case errCh <- err:
				default:
				}
				return
			}
		}
	}
}

func (c *Client) handleSandboxCommand(ctx context.Context, writeJSON func(interface{}) error, msg sandboxExecuteMessage) {
	resp := sandboxResponseMessage{
		Type:       "sandbox.response",
		CommandID:  msg.CommandID,
		InstanceID: c.instanceID,
		Headers:    map[string]string{},
	}

	statusCode, headers, body, err := c.executeSandbox(ctx, msg)
	resp.StatusCode = statusCode
	resp.Headers = headers
	resp.Body = body
	if requestID := strings.TrimSpace(headers["X-LunarGate-Request-ID"]); requestID != "" {
		resp.RequestID = requestID
	}
	if provider := strings.TrimSpace(headers["X-LunarGate-Provider"]); provider != "" {
		resp.Provider = provider
	}
	if model := strings.TrimSpace(headers["X-LunarGate-Model"]); model != "" {
		resp.Model = model
	}
	if route := strings.TrimSpace(headers["X-LunarGate-Route"]); route != "" {
		resp.RouteUsed = route
	}
	if err != nil {
		resp.OK = false
		resp.Error = err.Error()
		if resp.StatusCode == 0 {
			resp.StatusCode = http.StatusBadGateway
		}
	} else {
		resp.OK = statusCode < 400
	}
	if writeErr := writeJSON(resp); writeErr != nil {
		log.Warn().Err(writeErr).Str("command_id", msg.CommandID).Msg("failed to send sandbox response")
	}
}

func (c *Client) sandboxErrorResponse(commandID string, err error) sandboxResponseMessage {
	return sandboxResponseMessage{
		Type:       "sandbox.response",
		CommandID:  commandID,
		InstanceID: c.instanceID,
		OK:         false,
		StatusCode: http.StatusBadRequest,
		Error:      err.Error(),
		Headers:    map[string]string{},
	}
}

func (c *Client) executeSandbox(ctx context.Context, msg sandboxExecuteMessage) (int, map[string]string, interface{}, error) {
	endpointPath, err := sandboxEndpointPath(msg.RequestType)
	if err != nil {
		return http.StatusBadRequest, map[string]string{}, nil, err
	}

	requestPayload := make(map[string]interface{}, len(msg.Request))
	for key, value := range msg.Request {
		requestPayload[key] = value
	}
	if msg.Target.Mode == "route" {
		requestPayload["model"] = "lunargate/auto"
	}
	bodyBytes, err := json.Marshal(requestPayload)
	if err != nil {
		return 0, map[string]string{}, nil, fmt.Errorf("failed to encode request body: %w", err)
	}
	endpoint := c.localBaseURL + endpointPath
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(bodyBytes))
	if err != nil {
		return 0, map[string]string{}, nil, fmt.Errorf("failed to create local sandbox request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-LunarGate-No-Cache", "true")
	if c.localAuthHeader != "" && c.localAuthValue != "" {
		req.Header.Set(c.localAuthHeader, c.localAuthValue)
	}
	if msg.Target.Mode == "route" {
		req.Header.Set("X-LunarGate-Route", strings.TrimSpace(msg.Target.Value))
	} else if msg.Target.Mode == "model" {
		req.Header.Set("X-LunarGate-Model", strings.TrimSpace(msg.Target.Value))
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return 0, map[string]string{}, nil, fmt.Errorf("failed to execute local sandbox request: %w", err)
	}
	defer resp.Body.Close()
	responseLimit := c.sandboxResponseLimit
	if responseLimit <= 0 {
		responseLimit = defaultSandboxResponseLimit
	}
	respBytes, err := readSandboxResponseBody(resp.Body, responseLimit)
	if err != nil {
		return resp.StatusCode, collectHeaders(resp.Header), nil, fmt.Errorf("failed to read sandbox response: %w", err)
	}
	parsedBody := parseBody(respBytes)
	return resp.StatusCode, collectHeaders(resp.Header), parsedBody, nil
}

func readSandboxResponseBody(body io.Reader, limit int64) ([]byte, error) {
	payload, err := io.ReadAll(io.LimitReader(body, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(payload)) > limit {
		return nil, fmt.Errorf("sandbox response exceeds %d byte limit", limit)
	}
	return payload, nil
}

func sandboxEndpointPath(requestType string) (string, error) {
	normalized := strings.ToLower(strings.TrimSpace(requestType))
	switch normalized {
	case "", "chat", "chat_completions":
		return "/v1/chat/completions", nil
	case "responses":
		return "/v1/responses", nil
	case "embeddings":
		return "/v1/embeddings", nil
	default:
		return "", fmt.Errorf(
			"unsupported sandbox request_type %q: expected chat_completions, responses, or embeddings",
			requestType,
		)
	}
}

func (c *Client) buildSnapshotHello() helloMessage {
	models := []string{}
	if c.modelSnapshotIDs != nil {
		models = c.modelSnapshotIDs()
	}
	return c.buildHello(models)
}

func (c *Client) buildHello(models []string) helloMessage {
	routes := []string{}
	if c.routeNames != nil {
		routes = c.routeNames()
	}
	return helloMessage{
		Type:       "hello",
		InstanceID: c.instanceID,
		Version:    c.version,
		Routes:     routes,
		Models:     models,
	}
}

func (c *Client) websocketURL() (string, error) {
	base, err := safeurl.ParseHTTPBaseURL(c.backendURL)
	if err != nil {
		return "", fmt.Errorf("invalid general.backend_url: %w", err)
	}
	base = base.JoinPath("remote-control", "ws", "gateway")
	base.Fragment = ""
	base.RawFragment = ""
	switch base.Scheme {
	case "https":
		base.Scheme = "wss"
	case "http":
		base.Scheme = "ws"
	default:
		return "", fmt.Errorf("invalid general.backend_url")
	}
	return base.String(), nil
}

func (c *Client) logConnectionIssue(err error) {
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

	var statusErr *dialStatusError
	if errors.As(err, &statusErr) && (statusErr.statusCode == http.StatusUnauthorized || statusErr.statusCode == http.StatusForbidden) {
		log.Warn().
			Int("status_code", statusErr.statusCode).
			Str("status_text", http.StatusText(statusErr.statusCode)).
			Msg("remote control authentication rejected by lunargate.ai; go to app.lunargate.ai and check general.api_key")
		return
	}

	log.Warn().Err(err).Msg("remote control connection closed")
}

func classifyDialError(err error, resp *http.Response, requestURL *url.URL) error {
	redactedErr := newWebSocketDialError(err, requestURL)
	if resp == nil {
		return redactedErr
	}
	if resp.Body != nil {
		defer resp.Body.Close()
	}

	if resp.StatusCode == 0 {
		return redactedErr
	}
	return &dialStatusError{statusCode: resp.StatusCode}
}

type dialStatusError struct {
	statusCode int
}

func (e *dialStatusError) Error() string {
	return fmt.Sprintf("websocket handshake failed with status %d (%s)", e.statusCode, http.StatusText(e.statusCode))
}

type webSocketDialError struct {
	cause    error
	endpoint string
}

func newWebSocketDialError(err error, requestURL *url.URL) error {
	if err == nil {
		return nil
	}
	return &webSocketDialError{
		cause:    safeurl.RedactTransportError(err, requestURL),
		endpoint: redactedWebSocketEndpoint(requestURL),
	}
}

func (e *webSocketDialError) Error() string {
	if e.endpoint == "" {
		return "remote control WebSocket dial failed"
	}
	return "remote control WebSocket dial failed for " + e.endpoint
}

func (e *webSocketDialError) Unwrap() error {
	return e.cause
}

func redactedWebSocketEndpoint(endpoint *url.URL) string {
	if endpoint == nil {
		return ""
	}
	redacted := *endpoint
	redacted.User = nil
	redacted.RawQuery = ""
	redacted.ForceQuery = false
	redacted.Fragment = ""
	redacted.RawFragment = ""
	return redacted.String()
}

func collectHeaders(h http.Header) map[string]string {
	out := map[string]string{}
	for _, key := range []string{
		"X-LunarGate-Request-ID",
		"X-LunarGate-Provider",
		"X-LunarGate-Model",
		"X-LunarGate-Route",
		"Content-Type",
	} {
		if value := strings.TrimSpace(h.Get(key)); value != "" {
			out[key] = value
		}
	}
	return out
}

func parseBody(body []byte) interface{} {
	trimmed := bytes.TrimSpace(body)
	if len(trimmed) == 0 {
		return map[string]interface{}{}
	}
	var decoded interface{}
	if err := decodeSingleJSONDocument(trimmed, &decoded); err == nil {
		return decoded
	}
	return string(trimmed)
}

func decodeSingleJSONDocument(payload []byte, target interface{}) error {
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.UseNumber()
	if err := decoder.Decode(target); err != nil {
		return err
	}

	var trailing interface{}
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		if err == nil {
			return fmt.Errorf("multiple JSON documents are not allowed")
		}
		return fmt.Errorf("invalid data after JSON document: %w", err)
	}
	return nil
}

func localInstanceID() (string, error) {
	return instanceIDFromReader(rand.Reader)
}

func instanceIDFromReader(reader io.Reader) (string, error) {
	const entropyBytes = 16

	buf := make([]byte, entropyBytes)
	if _, err := io.ReadFull(reader, buf); err != nil {
		return "", fmt.Errorf("failed to read instance identity entropy: %w", err)
	}
	return hex.EncodeToString(buf), nil
}
