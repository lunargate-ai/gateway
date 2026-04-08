package remotecontrol

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/rs/zerolog/log"
)

type ModelListFunc func(context.Context) []string

type Client struct {
	dataSharing  config.DataSharingConfig
	instanceID   string
	version      string
	localBaseURL string
	routeNames   func() []string
	modelIDs     ModelListFunc
	httpClient   *http.Client
	refreshCh    chan struct{}
	lastLogKey   string
	lastLogAt    time.Time
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
	Type      string                 `json:"type"`
	CommandID string                 `json:"command_id"`
	Target    sandboxTarget          `json:"target"`
	Request   map[string]interface{} `json:"request"`
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
	dataSharing config.DataSharingConfig,
	version string,
	localBaseURL string,
	routeNames func() []string,
	modelIDs ModelListFunc,
) *Client {
	if !dataSharing.RemoteControl {
		return nil
	}
	if strings.TrimSpace(dataSharing.APIKey) == "" {
		log.Warn().Msg("remote control disabled: api_key missing in data_sharing config")
		return nil
	}
	return &Client{
		dataSharing:  dataSharing,
		instanceID:   localInstanceID(),
		version:      version,
		localBaseURL: strings.TrimRight(strings.TrimSpace(localBaseURL), "/"),
		routeNames:   routeNames,
		modelIDs:     modelIDs,
		httpClient:   &http.Client{Timeout: 5 * time.Minute},
		refreshCh:    make(chan struct{}, 1),
	}
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
	backoff := time.Second
	for {
		if ctx.Err() != nil {
			return
		}
		if err := c.connectAndServe(ctx); err != nil && ctx.Err() == nil {
			c.logConnectionIssue(err)
		}
		select {
		case <-ctx.Done():
			return
		case <-time.After(backoff):
		}
		if backoff < 15*time.Second {
			backoff *= 2
		}
	}
}

func (c *Client) connectAndServe(ctx context.Context) error {
	wsURL, err := c.websocketURL()
	if err != nil {
		return err
	}
	headers := http.Header{}
	headers.Set("Authorization", "Bearer "+strings.TrimSpace(c.dataSharing.APIKey))
	conn, resp, err := websocket.DefaultDialer.DialContext(ctx, wsURL, headers)
	if err != nil {
		return classifyDialError(err, resp)
	}
	defer conn.Close()

	var writeMu sync.Mutex
	writeJSON := func(v interface{}) error {
		writeMu.Lock()
		defer writeMu.Unlock()
		return conn.WriteJSON(v)
	}

	if err := writeJSON(c.buildHello(ctx)); err != nil {
		return err
	}

	errCh := make(chan error, 1)
	go c.readLoop(ctx, conn, writeJSON, errCh)

	closeCh := make(chan struct{})
	go func() {
		select {
		case <-ctx.Done():
			writeMu.Lock()
			_ = conn.Close()
			writeMu.Unlock()
		case <-closeCh:
		}
	}()
	defer close(closeCh)

	heartbeatTicker := time.NewTicker(20 * time.Second)
	defer heartbeatTicker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-heartbeatTicker.C:
			if err := writeJSON(heartbeatMessage{Type: "heartbeat"}); err != nil {
				return err
			}
		case <-c.refreshCh:
			if err := writeJSON(c.buildHello(ctx)); err != nil {
				return err
			}
		case err := <-errCh:
			return err
		}
	}
}

func (c *Client) readLoop(
	ctx context.Context,
	conn *websocket.Conn,
	writeJSON func(interface{}) error,
	errCh chan<- error,
) {
	for {
		var payload map[string]interface{}
		if err := conn.ReadJSON(&payload); err != nil {
			select {
			case errCh <- err:
			default:
			}
			return
		}
		msgType := strings.TrimSpace(fmt.Sprint(payload["type"]))
		switch msgType {
		case "heartbeat":
			continue
		case "sandbox.execute":
			var msg sandboxExecuteMessage
			b, err := json.Marshal(payload)
			if err != nil {
				go c.sendSandboxError(writeJSON, "", fmt.Errorf("failed to marshal sandbox command: %w", err))
				continue
			}
			if err := json.Unmarshal(b, &msg); err != nil {
				go c.sendSandboxError(writeJSON, "", fmt.Errorf("failed to decode sandbox command: %w", err))
				continue
			}
			go c.handleSandboxCommand(ctx, writeJSON, msg)
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

func (c *Client) sendSandboxError(writeJSON func(interface{}) error, commandID string, err error) {
	_ = writeJSON(sandboxResponseMessage{
		Type:       "sandbox.response",
		CommandID:  commandID,
		InstanceID: c.instanceID,
		OK:         false,
		StatusCode: http.StatusBadRequest,
		Error:      err.Error(),
		Headers:    map[string]string{},
	})
}

func (c *Client) executeSandbox(ctx context.Context, msg sandboxExecuteMessage) (int, map[string]string, interface{}, error) {
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
	endpoint := c.localBaseURL + "/v1/chat/completions"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(bodyBytes))
	if err != nil {
		return 0, map[string]string{}, nil, fmt.Errorf("failed to create local sandbox request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-LunarGate-No-Cache", "true")
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
	respBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return resp.StatusCode, collectHeaders(resp.Header), nil, fmt.Errorf("failed to read sandbox response: %w", err)
	}
	parsedBody := parseBody(respBytes)
	return resp.StatusCode, collectHeaders(resp.Header), parsedBody, nil
}

func (c *Client) buildHello(ctx context.Context) helloMessage {
	models := []string{}
	if c.modelIDs != nil {
		models = c.modelIDs(ctx)
	}
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
	base, err := url.Parse(strings.TrimSpace(c.dataSharing.BackendURL))
	if err != nil {
		return "", fmt.Errorf("invalid data_sharing.backend_url: %w", err)
	}
	switch base.Scheme {
	case "https":
		base.Scheme = "wss"
	case "http":
		base.Scheme = "ws"
	default:
		return "", fmt.Errorf("unsupported data_sharing.backend_url scheme: %s", base.Scheme)
	}
	base.Path = strings.TrimRight(base.Path, "/") + "/remote-control/ws/gateway"
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
		event := log.Warn().
			Int("status_code", statusErr.statusCode).
			Str("status_text", http.StatusText(statusErr.statusCode))
		if statusErr.detail != "" {
			event = event.Str("detail", statusErr.detail)
		}
		event.Msg("remote control authentication rejected by lunargate.ai; go to app.lunargate.ai and check data_sharing.api_key")
		return
	}

	log.Warn().Err(err).Msg("remote control connection closed")
}

func classifyDialError(err error, resp *http.Response) error {
	if resp == nil {
		return err
	}
	defer resp.Body.Close()

	detail := readResponseSnippet(resp.Body)
	if resp.StatusCode == 0 {
		return err
	}
	return &dialStatusError{
		statusCode: resp.StatusCode,
		detail:     detail,
	}
}

type dialStatusError struct {
	statusCode int
	detail     string
}

func (e *dialStatusError) Error() string {
	if e.detail == "" {
		return fmt.Sprintf("websocket handshake failed with status %d (%s)", e.statusCode, http.StatusText(e.statusCode))
	}
	return fmt.Sprintf("websocket handshake failed with status %d (%s): %s", e.statusCode, http.StatusText(e.statusCode), e.detail)
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
	if err := json.Unmarshal(trimmed, &decoded); err == nil {
		return decoded
	}
	return string(trimmed)
}

func localInstanceID() string {
	const alphabet = "abcdefghijklmnopqrstuvwxyz"
	const size = 4
	buf := make([]byte, size)
	if _, err := rand.Read(buf); err != nil {
		return "zzzz"
	}
	out := make([]byte, size)
	for i, b := range buf {
		out[i] = alphabet[int(b)%len(alphabet)]
	}
	return string(out)
}
