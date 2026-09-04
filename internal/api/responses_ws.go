package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/websocket"
	"github.com/lunargate-ai/gateway/internal/middleware"
	"github.com/lunargate-ai/gateway/internal/security"
	"github.com/lunargate-ai/gateway/internal/streaming"
	"github.com/lunargate-ai/gateway/pkg/models"
	"github.com/rs/zerolog/log"
)

const (
	responsesWebSocketWriteTimeout = 10 * time.Second
	responsesWebSocketPongTimeout  = 60 * time.Second
	responsesWebSocketPingInterval = 30 * time.Second
	// A complete SSE record may be followed by a CRLF blank-line delimiter,
	// which is transport framing and is not counted in MaxStreamRecordBytes.
	responsesWebSocketSSEBufferLimit  = streaming.MaxStreamRecordBytes + 2
	responsesWebSocketMaxCachedStates = 1
	responsesWebSocketMaxCachedBytes  = 16 << 20
)

var responsesWebSocketUpgrader = websocket.Upgrader{
	ReadBufferSize:  4096,
	WriteBufferSize: 4096,
	CheckOrigin:     checkResponsesWebSocketOrigin,
}

func checkResponsesWebSocketOrigin(r *http.Request) bool {
	if r == nil {
		return false
	}

	origin := strings.TrimSpace(r.Header.Get("Origin"))
	if origin == "" {
		// Non-browser clients generally omit Origin and authenticate using the
		// regular API credentials enforced by the route middleware.
		return true
	}

	parsed, err := url.Parse(origin)
	if err != nil || parsed.User != nil || parsed.Path != "" || parsed.RawQuery != "" || parsed.Fragment != "" {
		return false
	}

	scheme := strings.ToLower(strings.TrimSpace(parsed.Scheme))
	switch scheme {
	case "http", "https", "ws", "wss":
	default:
		return false
	}

	originHost, ok := canonicalWebSocketOriginHost(parsed.Host, scheme)
	if !ok {
		return false
	}
	requestHost, ok := canonicalWebSocketOriginHost(r.Host, scheme)
	if !ok {
		return false
	}

	return originHost == requestHost
}

func canonicalWebSocketOriginHost(rawHost string, scheme string) (string, bool) {
	parsed, err := url.Parse("//" + strings.TrimSpace(rawHost))
	if err != nil || parsed.User != nil || parsed.Hostname() == "" {
		return "", false
	}

	hostname := strings.ToLower(strings.TrimSuffix(parsed.Hostname(), "."))
	port := parsed.Port()
	if port == "" {
		switch scheme {
		case "http", "ws":
			port = "80"
		case "https", "wss":
			port = "443"
		default:
			return "", false
		}
	}

	return net.JoinHostPort(hostname, port), true
}

type responsesWebSocketSession struct {
	conn                *websocket.Conn
	sessionID           string
	cachedStates        map[string]*responsesWebSocketCachedState
	cachedStateBytes    int
	maxCachedStateBytes int
}

type responsesWebSocketMessagePolicy func(*http.Request) (*http.Request, *responsesWebSocketEventError)

type responsesWebSocketProxy struct {
	session            *responsesWebSocketSession
	headers            http.Header
	statusCode         int
	buffer             bytes.Buffer
	errorBodyTruncated bool
	done               bool
	terminalSeen       bool
	responseID         string
	completedResponse  map[string]interface{}
	terminalResponse   map[string]interface{}
	terminalError      *responsesWebSocketEventError
	cacheBasePayload   map[string]json.RawMessage
	stateCached        bool
	syntheticFailure   bool
	lastSequence       int64
}

type responsesWebSocketCachedState struct {
	responseID string
	payload    map[string]json.RawMessage
	sizeBytes  int
}

type responsesWebSocketCreateRequest struct {
	payload            map[string]json.RawMessage
	previousResponseID string
	generate           bool
}

type responsesWebSocketEventError struct {
	status  int
	errType string
	code    string
	param   string
	message string
}

func (e *responsesWebSocketEventError) Error() string {
	if e == nil {
		return ""
	}
	return strings.TrimSpace(e.message)
}

func (h *Handler) ResponsesWebSocket(w http.ResponseWriter, r *http.Request) {
	h.bindRuntime().responsesWebSocket(w, r, nil)
}

func (h *Handler) responsesWebSocketHandler(policy responsesWebSocketMessagePolicy) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		h.bindRuntime().responsesWebSocket(w, r, policy)
	}
}

func (h *Handler) responsesWebSocket(w http.ResponseWriter, r *http.Request, policy responsesWebSocketMessagePolicy) {
	connectionCtx, cancelConnection := context.WithCancel(r.Context())
	webSockets := h.responsesWebSocketRegistryRef()
	registration, ok := webSockets.register(cancelConnection)
	if !ok {
		cancelConnection()
		writeError(w, http.StatusServiceUnavailable, errResponsesWebSocketShuttingDown.Error(), "server_error")
		return
	}
	defer webSockets.unregister(registration)
	r = r.Clone(connectionCtx)

	conn, err := responsesWebSocketUpgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Warn().Err(err).Msg("responses websocket upgrade failed")
		return
	}
	if !webSockets.attach(registration, conn) {
		return
	}
	var stopHeartbeat func()
	defer func() {
		_ = conn.Close()
		if stopHeartbeat != nil {
			stopHeartbeat()
		}
	}()

	conn.SetReadLimit(maxRequestBodyBytes)
	if err := conn.SetReadDeadline(time.Now().Add(responsesWebSocketPongTimeout)); err != nil {
		log.Warn().Err(err).Msg("responses websocket read deadline setup failed")
		return
	}
	conn.SetPongHandler(func(_ string) error {
		return conn.SetReadDeadline(time.Now().Add(responsesWebSocketPongTimeout))
	})
	stopHeartbeat = startResponsesWebSocketHeartbeat(conn)

	sessionID := strings.TrimSpace(r.Header.Get("x-lunargate-sessionid"))
	if sessionID == "" {
		sessionID = "wsresp_" + uuid.NewString()
	}
	session := &responsesWebSocketSession{
		conn:                conn,
		sessionID:           sessionID,
		cachedStates:        make(map[string]*responsesWebSocketCachedState),
		maxCachedStateBytes: responsesWebSocketMaxCachedBytes,
	}

	for {
		msgType, payload, err := conn.ReadMessage()
		if err != nil {
			if isBenignResponsesWebSocketClose(err) {
				return
			}
			log.Warn().Err(err).Msg("responses websocket read failed")
			return
		}
		if msgType != websocket.TextMessage && msgType != websocket.BinaryMessage {
			continue
		}
		if err := conn.SetReadDeadline(time.Now().Add(responsesWebSocketPongTimeout)); err != nil {
			log.Warn().Err(err).Msg("responses websocket read deadline refresh failed")
			return
		}

		messageReq := r
		if policy != nil {
			var policyErr *responsesWebSocketEventError
			messageReq, policyErr = policy(r.Clone(r.Context()))
			if policyErr != nil {
				if err := session.writeErrorEvent(policyErr); err != nil {
					log.Warn().Err(err).Msg("responses websocket policy error write failed")
					return
				}
				continue
			}
		}

		if err := session.handleCreate(h, messageReq, payload); err != nil {
			log.Warn().Err(err).Msg("responses websocket request failed")
			_ = session.writeErrorEvent(responsesWebSocketEventErrorFromError(err))
		}
	}
}

func newResponsesWebSocketMessagePolicy(
	authManager *security.Manager,
	rateLimiter *middleware.RateLimiter,
) responsesWebSocketMessagePolicy {
	return func(r *http.Request) (*http.Request, *responsesWebSocketEventError) {
		recorder := newCapturedResponseWriter()
		var accepted *http.Request
		guarded := http.Handler(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
			accepted = req
			w.WriteHeader(http.StatusNoContent)
		}))
		if rateLimiter != nil {
			guarded = rateLimiter.Middleware(guarded)
		}
		if authManager != nil {
			guarded = authManager.Middleware(guarded)
		}

		guarded.ServeHTTP(recorder, r)
		if accepted != nil {
			return accepted, nil
		}

		status := recorder.statusCode
		if status == 0 {
			status = http.StatusInternalServerError
		}
		return nil, parseResponsesHTTPError(status, recorder.body.Bytes())
	}
}

func startResponsesWebSocketHeartbeat(conn *websocket.Conn) func() {
	done := make(chan struct{})
	stopped := make(chan struct{})
	go func() {
		defer close(stopped)
		ticker := time.NewTicker(responsesWebSocketPingInterval)
		defer ticker.Stop()
		for {
			select {
			case <-done:
				return
			case <-ticker.C:
				if err := conn.WriteControl(
					websocket.PingMessage,
					nil,
					time.Now().Add(responsesWebSocketWriteTimeout),
				); err != nil {
					return
				}
			}
		}
	}()

	return func() {
		close(done)
		<-stopped
	}
}

func isBenignResponsesWebSocketClose(err error) bool {
	if err == nil {
		return true
	}
	var closeErr *websocket.CloseError
	if errors.As(err, &closeErr) {
		return closeErr.Code == websocket.CloseNormalClosure ||
			closeErr.Code == websocket.CloseGoingAway ||
			closeErr.Code == websocket.CloseNoStatusReceived ||
			// Browser/CLI clients may disconnect TCP without sending a close frame.
			// Gorilla surfaces this as 1006 + unexpected EOF, which is benign here.
			closeErr.Code == websocket.CloseAbnormalClosure
	}
	var netErr net.Error
	if errors.As(err, &netErr) && netErr.Timeout() {
		return true
	}
	return strings.Contains(strings.ToLower(err.Error()), "unexpected eof")
}

func (s *responsesWebSocketSession) handleCreate(h *Handler, baseReq *http.Request, rawPayload []byte) error {
	createReq, err := parseResponsesWebSocketCreateRequest(rawPayload)
	if err != nil {
		_ = s.writeErrorEvent(responsesWebSocketEventErrorFromError(err))
		return nil
	}

	resolvedPayload, err := s.resolveCreatePayload(createReq)
	if err != nil {
		_ = s.writeErrorEvent(responsesWebSocketEventErrorFromError(err))
		return nil
	}

	model := parseJSONStringRaw(resolvedPayload["model"])
	if !createReq.generate {
		if model == "" {
			_ = s.writeErrorEvent(&responsesWebSocketEventError{
				status:  http.StatusBadRequest,
				errType: "invalid_request_error",
				param:   "model",
				message: "model is required",
			})
			return nil
		}
		responseID := "resp_ws_" + uuid.NewString()
		if cacheErr := s.cacheState(responseID, resolvedPayload); cacheErr != nil {
			_ = s.writeErrorEvent(cacheErr)
			return nil
		}
		return s.writeWarmupResponse(responseID, model, resolvedPayload)
	}

	body, err := marshalResponsesWebSocketHTTPBody(resolvedPayload)
	if err != nil {
		return err
	}

	req := makeResponsesWebSocketHTTPRequest(baseReq, body, s.sessionID)
	proxy := newResponsesWebSocketProxy(s)
	proxy.cacheBasePayload = resolvedPayload
	h.Responses(proxy, req)
	if err := proxy.finalize(); err != nil {
		if createReq.previousResponseID != "" {
			s.evictState(createReq.previousResponseID)
		}
		return err
	}
	if proxy.terminalError != nil {
		if createReq.previousResponseID != "" {
			s.evictState(createReq.previousResponseID)
		}
		return nil
	}
	return nil
}

func parseResponsesWebSocketCreateRequest(rawPayload []byte) (*responsesWebSocketCreateRequest, error) {
	var envelope map[string]json.RawMessage
	if err := decodeJSONStrict(bytes.NewReader(rawPayload), &envelope); err != nil {
		return nil, &responsesWebSocketEventError{
			status:  http.StatusBadRequest,
			errType: "invalid_request_error",
			message: "invalid websocket JSON payload",
		}
	}
	if len(envelope) == 0 {
		return nil, &responsesWebSocketEventError{
			status:  http.StatusBadRequest,
			errType: "invalid_request_error",
			message: "empty websocket payload",
		}
	}

	eventType := parseJSONStringRaw(envelope["type"])
	if eventType == "" {
		return nil, &responsesWebSocketEventError{
			status:  http.StatusBadRequest,
			errType: "invalid_request_error",
			message: "websocket payload requires a string type",
		}
	}
	if eventType != "response.create" {
		return nil, &responsesWebSocketEventError{
			status:  http.StatusBadRequest,
			errType: "invalid_request_error",
			message: fmt.Sprintf("unsupported websocket event type %q", eventType),
		}
	}

	payload := make(map[string]json.RawMessage)
	if nestedRaw, ok := envelope["response"]; ok && len(nestedRaw) > 0 {
		var nested map[string]json.RawMessage
		if err := decodeJSONStrict(bytes.NewReader(nestedRaw), &nested); err != nil {
			return nil, &responsesWebSocketEventError{
				status:  http.StatusBadRequest,
				errType: "invalid_request_error",
				message: "response must be a JSON object",
			}
		}
		for key, value := range nested {
			payload[key] = value
		}
	}
	for key, value := range envelope {
		if key == "type" || key == "response" {
			continue
		}
		payload[key] = value
	}

	previousResponseID, _, previousResponseErr := optionalOpaqueResourceID(
		payload["previous_response_id"],
		"previous_response_id",
	)
	if previousResponseErr != nil {
		return nil, &responsesWebSocketEventError{
			status:  http.StatusBadRequest,
			errType: "invalid_request_error",
			param:   "previous_response_id",
			code:    "invalid_value",
			message: previousResponseErr.Error(),
		}
	}

	generate := true
	if rawGenerate, ok := payload["generate"]; ok && len(rawGenerate) > 0 {
		if err := json.Unmarshal(rawGenerate, &generate); err != nil {
			return nil, &responsesWebSocketEventError{
				status:  http.StatusBadRequest,
				errType: "invalid_request_error",
				param:   "generate",
				message: "generate must be a boolean",
			}
		}
	}

	delete(payload, "previous_response_id")
	delete(payload, "generate")

	return &responsesWebSocketCreateRequest{
		payload:            payload,
		previousResponseID: previousResponseID,
		generate:           generate,
	}, nil
}

func makeResponsesWebSocketHTTPRequest(baseReq *http.Request, body []byte, sessionID string) *http.Request {
	req := baseReq.Clone(baseReq.Context())
	req.Method = http.MethodPost
	req.URL.Path = "/v1/responses"
	req.URL.RawPath = ""
	req.RequestURI = "/v1/responses"
	req.Body = io.NopCloser(bytes.NewReader(body))
	req.ContentLength = int64(len(body))
	req.Header = baseReq.Header.Clone()
	// One WebSocket can carry many response.create messages. Reusing the
	// handshake idempotency key for every create would collapse distinct calls.
	req.Header.Del("Idempotency-Key")
	req.Header.Set("Content-Type", "application/json")
	if strings.TrimSpace(req.Header.Get("x-lunargate-sessionid")) == "" && strings.TrimSpace(sessionID) != "" {
		req.Header.Set("x-lunargate-sessionid", strings.TrimSpace(sessionID))
	}
	return req
}

func newResponsesWebSocketProxy(session *responsesWebSocketSession) *responsesWebSocketProxy {
	return &responsesWebSocketProxy{
		session:      session,
		headers:      make(http.Header),
		lastSequence: -1,
	}
}

func (p *responsesWebSocketProxy) Header() http.Header {
	return p.headers
}

func (p *responsesWebSocketProxy) WriteHeader(statusCode int) {
	p.statusCode = statusCode
}

func (p *responsesWebSocketProxy) markNativeResponsesSyntheticFailure() {
	p.syntheticFailure = true
}

func (p *responsesWebSocketProxy) Write(b []byte) (int, error) {
	written := len(b)
	if p.statusCode >= 400 {
		p.appendHTTPErrorBody(b)
		return written, nil
	}

	for len(b) > 0 {
		available := responsesWebSocketSSEBufferLimit - p.buffer.Len()
		if available <= 0 {
			p.buffer = bytes.Buffer{}
			return written, streaming.ErrStreamRecordTooLarge
		}
		chunkSize := len(b)
		if chunkSize > available {
			chunkSize = available
		}
		_, _ = p.buffer.Write(b[:chunkSize])
		b = b[chunkSize:]

		if err := p.drainCompleteSSEFrames(); err != nil {
			p.buffer = bytes.Buffer{}
			return written, err
		}
	}
	return written, nil
}

func (p *responsesWebSocketProxy) appendHTTPErrorBody(b []byte) {
	if p.errorBodyTruncated || len(b) == 0 {
		return
	}
	available := upstreamErrorBodyLimit - p.buffer.Len()
	if available <= 0 {
		p.errorBodyTruncated = true
		return
	}
	if len(b) > available {
		_, _ = p.buffer.Write(b[:available])
		p.errorBodyTruncated = true
		return
	}
	_, _ = p.buffer.Write(b)
}

func (p *responsesWebSocketProxy) drainCompleteSSEFrames() error {
	remaining := p.buffer.Bytes()
	consumed := false
	for len(remaining) > 0 {
		frame, next, ok, err := nextResponsesSSEFrame(remaining)
		if err != nil {
			return err
		}
		if !ok {
			break
		}
		consumed = true
		if _, err := p.processSSEFrame(frame); err != nil {
			return err
		}
		remaining = next
	}
	if consumed {
		tail := append([]byte(nil), remaining...)
		p.buffer = bytes.Buffer{}
		_, _ = p.buffer.Write(tail)
	}
	return nil
}

// FlushError satisfies http.ResponseController. Each complete SSE frame is
// forwarded immediately by Write, so the WebSocket adapter has no buffered
// transport state to flush.
func (p *responsesWebSocketProxy) FlushError() error {
	return nil
}

func (p *responsesWebSocketProxy) finalize() error {
	if p.statusCode >= 400 {
		if p.errorBodyTruncated {
			p.terminalError = &responsesWebSocketEventError{
				status:  http.StatusBadGateway,
				errType: "provider_error",
				code:    "upstream_response_too_large",
				message: "upstream error response exceeds the 1 MiB limit",
			}
		} else {
			p.terminalError = parseResponsesHTTPError(p.statusCode, p.buffer.Bytes())
		}
		p.buffer = bytes.Buffer{}
		return p.session.writeErrorEvent(p.terminalError)
	}

	if p.buffer.Len() > 0 {
		p.buffer = bytes.Buffer{}
		return p.writeIncompleteStreamError()
	}
	if !p.terminalSeen {
		return p.writeIncompleteStreamError()
	}
	return nil
}

func (p *responsesWebSocketProxy) writeIncompleteStreamError() error {
	p.done = true
	p.terminalError = &responsesWebSocketEventError{
		status:  http.StatusBadGateway,
		errType: "provider_error",
		code:    "upstream_stream_incomplete",
		message: "response stream ended before a terminal event",
	}
	return p.session.writeErrorEventAfter(p.terminalError, p.lastSequence)
}

func (p *responsesWebSocketProxy) processSSEFrame(frame []byte) (bool, error) {
	payload, ok := responsesSSEData(frame)
	if !ok || len(bytes.TrimSpace(payload)) == 0 {
		return false, nil
	}
	if string(bytes.TrimSpace(payload)) == "[DONE]" {
		p.done = true
		return false, nil
	}
	if p.terminalSeen {
		// A native Responses stream has exactly one authoritative terminal.
		// Consume every later application event while still allowing [DONE]
		// above to close the underlying transport normally.
		return false, nil
	}
	if err := p.sendEvent(payload); err != nil {
		return true, err
	}
	return true, nil
}

// nextResponsesSSEFrame extracts one complete SSE event while accepting both
// LF and CRLF line endings. The returned slices borrow data and remain valid
// only until the source buffer is reused.
func nextResponsesSSEFrame(data []byte) (frame []byte, remaining []byte, ok bool, err error) {
	lineStart := 0
	for lineStart < len(data) {
		relativeEnd := bytes.IndexByte(data[lineStart:], '\n')
		if relativeEnd < 0 {
			if len(data) > streaming.MaxStreamRecordBytes {
				return nil, nil, false, streaming.ErrStreamRecordTooLarge
			}
			return nil, nil, false, nil
		}
		lineEnd := lineStart + relativeEnd + 1
		line := data[lineStart:lineEnd]
		if bytes.Equal(line, []byte{'\n'}) || bytes.Equal(line, []byte{'\r', '\n'}) {
			if lineStart > streaming.MaxStreamRecordBytes {
				return nil, nil, false, streaming.ErrStreamRecordTooLarge
			}
			return data[:lineStart], data[lineEnd:], true, nil
		}
		if lineEnd > streaming.MaxStreamRecordBytes {
			return nil, nil, false, streaming.ErrStreamRecordTooLarge
		}
		lineStart = lineEnd
	}
	return nil, nil, false, nil
}

// responsesSSEData follows the SSE data-field rules: comments and other
// fields are ignored, one optional leading space is removed, and multiple
// data lines are joined with a newline before the JSON event is decoded.
func responsesSSEData(frame []byte) ([]byte, bool) {
	dataLines := make([][]byte, 0, 1)
	for _, rawLine := range bytes.Split(frame, []byte{'\n'}) {
		line := bytes.TrimSuffix(rawLine, []byte{'\r'})
		if len(line) == 0 || line[0] == ':' {
			continue
		}
		field, value, found := bytes.Cut(line, []byte{':'})
		if !found || !bytes.Equal(field, []byte("data")) {
			continue
		}
		if len(value) > 0 && value[0] == ' ' {
			value = value[1:]
		}
		dataLines = append(dataLines, append([]byte(nil), value...))
	}
	if len(dataLines) == 0 {
		return nil, false
	}
	return bytes.Join(dataLines, []byte{'\n'}), true
}

func (p *responsesWebSocketProxy) sendEvent(payload []byte) error {
	responseID, identityErr := validateResponsesEventIdentity(payload, "", p.responseID)
	if identityErr != nil {
		p.done = true
		p.terminalSeen = true
		p.terminalError = &responsesWebSocketEventError{
			status:  http.StatusBadGateway,
			errType: "provider_error",
			code:    "invalid_response_id",
			message: "response stream returned an invalid or inconsistent response identifier",
		}
		return p.session.writeErrorEventAfter(p.terminalError, p.lastSequence)
	}
	sequence, sequenceErr := validateNativeResponsesEventSequence(payload, p.lastSequence)
	if sequenceErr != nil {
		p.done = true
		p.terminalSeen = true
		p.terminalError = &responsesWebSocketEventError{
			status:  http.StatusBadGateway,
			errType: "provider_error",
			code:    "invalid_sequence_number",
			message: "response stream returned an invalid or inconsistent sequence_number",
		}
		return p.session.writeErrorEventAfter(p.terminalError, p.lastSequence)
	}
	p.responseID = responseID
	p.captureEventState(payload)
	cachedTerminal := false
	if p.terminalSeen && p.terminalError == nil && !p.syntheticFailure && !p.stateCached && p.cacheBasePayload != nil {
		terminalResponse := p.terminalResponse
		if terminalResponse == nil {
			terminalResponse = p.completedResponse
		}
		if terminalResponse != nil && p.responseID != "" {
			nextState := withCompletedResponseHistory(p.cacheBasePayload, terminalResponse)
			if cacheErr := p.session.cacheState(p.responseID, nextState); cacheErr != nil {
				p.terminalError = cacheErr
				p.terminalResponse = nil
				p.completedResponse = nil
				return p.session.writeErrorEventAfter(cacheErr, p.lastSequence)
			}
			p.stateCached = true
			cachedTerminal = true
		}
	}
	if err := p.session.writeMessage(websocket.TextMessage, payload); err != nil {
		if cachedTerminal {
			p.session.evictState(p.responseID)
			p.stateCached = false
		}
		return err
	}
	p.lastSequence = sequence
	return nil
}

func (s *responsesWebSocketSession) writeMessage(messageType int, payload []byte) error {
	if s == nil || s.conn == nil {
		return errors.New("websocket session is closed")
	}
	if err := s.conn.SetWriteDeadline(time.Now().Add(responsesWebSocketWriteTimeout)); err != nil {
		return fmt.Errorf("failed to set websocket write deadline: %w", err)
	}
	if err := s.conn.WriteMessage(messageType, payload); err != nil {
		return fmt.Errorf("failed to write websocket message: %w", err)
	}
	return nil
}

func (s *responsesWebSocketSession) writeErrorEvent(eventErr *responsesWebSocketEventError) error {
	return s.writeErrorEventAfter(eventErr, -1)
}

func (s *responsesWebSocketSession) writeErrorEventAfter(eventErr *responsesWebSocketEventError, previousSequence int64) error {
	if eventErr == nil {
		eventErr = &responsesWebSocketEventError{
			status:  http.StatusBadGateway,
			errType: "provider_error",
			message: "failed to process websocket request",
		}
	}
	if previousSequence < -1 {
		previousSequence = -1
	}
	message := nonEmptyOrDefault(strings.TrimSpace(eventErr.message), "failed to process websocket request")
	code := strings.TrimSpace(eventErr.code)
	param := strings.TrimSpace(eventErr.param)
	payload := map[string]interface{}{
		"type":            "error",
		"code":            nil,
		"message":         message,
		"param":           nil,
		"sequence_number": previousSequence + 1,
		"status": func() int {
			if eventErr.status > 0 {
				return eventErr.status
			}
			return http.StatusBadGateway
		}(),
		"error": map[string]interface{}{
			"type":    nonEmptyOrDefault(strings.TrimSpace(eventErr.errType), "provider_error"),
			"message": message,
		},
	}
	if code != "" {
		payload["code"] = code
		payload["error"].(map[string]interface{})["code"] = code
	}
	if param != "" {
		payload["param"] = param
		payload["error"].(map[string]interface{})["param"] = param
	}
	b, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	return s.writeMessage(websocket.TextMessage, b)
}

func parseResponsesHTTPError(status int, body []byte) *responsesWebSocketEventError {
	errResp := &responsesWebSocketEventError{
		status:  status,
		errType: "provider_error",
		message: fmt.Sprintf("upstream request failed with status %d", status),
	}

	var parsed models.ErrorResponse
	if err := json.Unmarshal(body, &parsed); err == nil {
		if strings.TrimSpace(parsed.Error.Type) != "" {
			errResp.errType = strings.TrimSpace(parsed.Error.Type)
		}
		if strings.TrimSpace(parsed.Error.Message) != "" {
			errResp.message = strings.TrimSpace(parsed.Error.Message)
		}
		if parsed.Error.Code != nil && strings.TrimSpace(*parsed.Error.Code) != "" {
			errResp.code = strings.TrimSpace(*parsed.Error.Code)
		}
		if parsed.Error.Param != nil && strings.TrimSpace(*parsed.Error.Param) != "" {
			errResp.param = strings.TrimSpace(*parsed.Error.Param)
		}
	}
	return errResp
}

func parseJSONStringRaw(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var value string
	if err := json.Unmarshal(raw, &value); err != nil {
		return ""
	}
	return strings.TrimSpace(value)
}

func (s *responsesWebSocketSession) resolveCreatePayload(createReq *responsesWebSocketCreateRequest) (map[string]json.RawMessage, error) {
	if createReq == nil {
		return nil, &responsesWebSocketEventError{
			status:  http.StatusBadRequest,
			errType: "invalid_request_error",
			message: "empty websocket payload",
		}
	}

	if createReq.previousResponseID == "" {
		return normalizeResponsesWebSocketPayload(createReq.payload)
	}
	if !validOpaqueResourceID(createReq.previousResponseID) {
		return nil, &responsesWebSocketEventError{
			status:  http.StatusBadRequest,
			errType: "invalid_request_error",
			code:    "invalid_value",
			param:   "previous_response_id",
			message: "previous_response_id must be a non-empty identifier without surrounding whitespace",
		}
	}

	state, ok := s.cachedStates[createReq.previousResponseID]
	if !ok {
		return nil, &responsesWebSocketEventError{
			status:  http.StatusBadRequest,
			errType: "invalid_request_error",
			code:    "previous_response_not_found",
			param:   "previous_response_id",
			message: fmt.Sprintf("Previous response with id '%s' not found.", createReq.previousResponseID),
		}
	}
	merged, err := mergeResponsesWebSocketPayloads(state.payload, createReq.payload)
	if err != nil {
		return nil, err
	}
	if _, ok := responsesWebSocketCachedStateSize(
		createReq.previousResponseID,
		merged,
		s.cachedStateLimit(),
	); !ok {
		s.clearCachedStates()
		return nil, responsesWebSocketStateTooLargeError()
	}
	return merged, nil
}

func (s *responsesWebSocketSession) cacheState(responseID string, payload map[string]json.RawMessage) *responsesWebSocketEventError {
	if len(payload) == 0 {
		return nil
	}
	if !validOpaqueResourceID(responseID) {
		return &responsesWebSocketEventError{
			status:  http.StatusBadGateway,
			errType: "provider_error",
			code:    "invalid_response_id",
			param:   "response_id",
			message: "upstream response id must be a non-empty identifier without surrounding whitespace",
		}
	}
	stateSize, ok := responsesWebSocketCachedStateSize(responseID, payload, s.cachedStateLimit())
	if !ok {
		s.clearCachedStates()
		return responsesWebSocketStateTooLargeError()
	}
	if s.cachedStates == nil {
		s.cachedStates = make(map[string]*responsesWebSocketCachedState, responsesWebSocketMaxCachedStates)
	}
	if len(s.cachedStates) >= responsesWebSocketMaxCachedStates {
		s.clearCachedStates()
	}
	s.cachedStates[responseID] = &responsesWebSocketCachedState{
		responseID: responseID,
		payload:    cloneResponsesRawMap(payload),
		sizeBytes:  stateSize,
	}
	s.cachedStateBytes = stateSize
	return nil
}

func (s *responsesWebSocketSession) evictState(responseID string) {
	if s == nil || !validOpaqueResourceID(responseID) {
		return
	}
	if state := s.cachedStates[responseID]; state != nil {
		s.cachedStateBytes -= state.sizeBytes
		if s.cachedStateBytes < 0 {
			s.cachedStateBytes = 0
		}
	}
	delete(s.cachedStates, responseID)
	if len(s.cachedStates) == 0 {
		s.cachedStateBytes = 0
	}
}

func (s *responsesWebSocketSession) clearCachedStates() {
	if s == nil {
		return
	}
	s.cachedStates = make(map[string]*responsesWebSocketCachedState, responsesWebSocketMaxCachedStates)
	s.cachedStateBytes = 0
}

func (s *responsesWebSocketSession) cachedStateLimit() int {
	if s != nil && s.maxCachedStateBytes > 0 {
		return s.maxCachedStateBytes
	}
	return responsesWebSocketMaxCachedBytes
}

func responsesWebSocketCachedStateSize(id string, payload map[string]json.RawMessage, limit int) (int, bool) {
	if limit <= 0 || len(id) > limit-len(id) {
		return 0, false
	}
	size := 2 * len(id) // map key plus cached responseID
	for key, value := range payload {
		componentSize := len(key) + len(value)
		if componentSize > limit-size {
			return 0, false
		}
		size += componentSize
	}
	return size, true
}

func responsesWebSocketStateTooLargeError() *responsesWebSocketEventError {
	return &responsesWebSocketEventError{
		status:  http.StatusRequestEntityTooLarge,
		errType: "invalid_request_error",
		code:    "state_too_large",
		param:   "previous_response_id",
		message: "websocket continuation state exceeds the 16 MiB limit",
	}
}

func (s *responsesWebSocketSession) writeWarmupResponse(
	responseID string,
	model string,
	requestPayload map[string]json.RawMessage,
) error {
	createdAt := time.Now().Unix()
	created := map[string]interface{}{
		"type":            "response.created",
		"sequence_number": 0,
		"response": completeSyntheticResponsesEnvelope(map[string]interface{}{
			"id":         responseID,
			"object":     "response",
			"created_at": createdAt,
			"status":     "in_progress",
			"model":      model,
			"output":     []interface{}{},
		}, requestPayload, true),
	}
	completed := map[string]interface{}{
		"type":            "response.completed",
		"sequence_number": 1,
		"response": completeSyntheticResponsesEnvelope(map[string]interface{}{
			"id":         responseID,
			"object":     "response",
			"created_at": createdAt,
			"status":     "completed",
			"model":      model,
			"output":     []interface{}{},
		}, requestPayload, true),
	}
	for _, event := range []map[string]interface{}{created, completed} {
		b, err := json.Marshal(event)
		if err != nil {
			return err
		}
		if err := s.writeMessage(websocket.TextMessage, b); err != nil {
			return err
		}
	}
	return nil
}

func normalizeResponsesWebSocketPayload(payload map[string]json.RawMessage) (map[string]json.RawMessage, error) {
	if len(payload) == 0 {
		return map[string]json.RawMessage{}, nil
	}

	normalized := cloneResponsesRawMap(payload)
	if rawInput, ok := normalized["input"]; ok && len(rawInput) > 0 {
		items, err := responsesInputRawToItems(rawInput)
		if err != nil {
			return nil, &responsesWebSocketEventError{
				status:  http.StatusBadRequest,
				errType: "invalid_request_error",
				param:   "input",
				message: err.Error(),
			}
		}
		encoded, err := json.Marshal(items)
		if err != nil {
			return nil, err
		}
		normalized["input"] = json.RawMessage(encoded)
	}
	delete(normalized, "previous_response_id")
	delete(normalized, "generate")
	delete(normalized, "stream")
	return normalized, nil
}

func mergeResponsesWebSocketPayloads(base map[string]json.RawMessage, delta map[string]json.RawMessage) (map[string]json.RawMessage, error) {
	merged, err := normalizeResponsesWebSocketPayload(base)
	if err != nil {
		return nil, err
	}
	deltaNormalized, err := normalizeResponsesWebSocketPayload(delta)
	if err != nil {
		return nil, err
	}

	baseItems, err := responsesInputRawToItems(merged["input"])
	if err != nil {
		return nil, err
	}
	deltaItems, err := responsesInputRawToItems(deltaNormalized["input"])
	if err != nil {
		return nil, err
	}

	for key, value := range deltaNormalized {
		if key == "input" {
			continue
		}
		merged[key] = cloneResponsesRawMessage(value)
	}

	if len(baseItems) > 0 || len(deltaItems) > 0 {
		combined := make([]interface{}, 0, len(baseItems)+len(deltaItems))
		combined = append(combined, baseItems...)
		combined = append(combined, deltaItems...)
		encoded, err := json.Marshal(combined)
		if err != nil {
			return nil, err
		}
		merged["input"] = json.RawMessage(encoded)
	}

	return merged, nil
}

func marshalResponsesWebSocketHTTPBody(payload map[string]json.RawMessage) ([]byte, error) {
	bodyPayload := cloneResponsesRawMap(payload)
	bodyPayload["stream"] = json.RawMessage("true")
	return json.Marshal(bodyPayload)
}

func withCompletedResponseHistory(payload map[string]json.RawMessage, completedResponse map[string]interface{}) map[string]json.RawMessage {
	next := cloneResponsesRawMap(payload)
	requestItems, err := responsesInputRawToItems(next["input"])
	if err != nil {
		return next
	}

	outputItems := responsesCompletedResponseToInputItems(completedResponse)
	if len(requestItems) == 0 && len(outputItems) == 0 {
		return next
	}

	combined := make([]interface{}, 0, len(requestItems)+len(outputItems))
	combined = append(combined, requestItems...)
	combined = append(combined, outputItems...)
	if encoded, err := json.Marshal(combined); err == nil {
		next["input"] = json.RawMessage(encoded)
	}
	return next
}

func responsesCompletedResponseToInputItems(response map[string]interface{}) []interface{} {
	if response == nil {
		return nil
	}
	rawOutput, _ := response["output"].([]interface{})
	if len(rawOutput) == 0 {
		return nil
	}

	items := make([]interface{}, 0, len(rawOutput))
	for _, item := range rawOutput {
		items = append(items, cloneResponsesContinuationValue(item))
	}
	return items
}

// Manual Responses history must replay every output item verbatim. Output item
// kinds are additive (reasoning, computer use, hosted tools, and future kinds),
// so rebuilding only currently known shapes silently removes model context.
func cloneResponsesContinuationValue(value interface{}) interface{} {
	switch typed := value.(type) {
	case map[string]interface{}:
		cloned := make(map[string]interface{}, len(typed))
		for key, nested := range typed {
			cloned[key] = cloneResponsesContinuationValue(nested)
		}
		return cloned
	case []interface{}:
		cloned := make([]interface{}, len(typed))
		for index, nested := range typed {
			cloned[index] = cloneResponsesContinuationValue(nested)
		}
		return cloned
	case json.RawMessage:
		return cloneResponsesRawMessage(typed)
	default:
		return value
	}
}

func responsesInputRawToItems(raw json.RawMessage) ([]interface{}, error) {
	if len(raw) == 0 {
		return nil, nil
	}

	var decoded interface{}
	if err := decodeJSONStrict(bytes.NewReader(raw), &decoded); err != nil {
		return nil, fmt.Errorf("unsupported input format")
	}
	return responsesInputValueToItems(decoded)
}

func responsesInputValueToItems(value interface{}) ([]interface{}, error) {
	switch typed := value.(type) {
	case nil:
		return nil, nil
	case string:
		if strings.TrimSpace(typed) == "" {
			return nil, nil
		}
		return []interface{}{map[string]interface{}{
			"type": "message",
			"role": "user",
			"content": []interface{}{
				map[string]interface{}{
					"type": "input_text",
					"text": typed,
				},
			},
		}}, nil
	case []interface{}:
		return cloneResponsesInterfaceSlice(typed), nil
	default:
		return nil, fmt.Errorf("unsupported input format")
	}
}

func (p *responsesWebSocketProxy) captureEventState(payload []byte) {
	if responseID, err := validateResponsesEventIdentity(payload, "", p.responseID); err == nil {
		p.responseID = responseID
	}

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(payload, &raw); err != nil {
		return
	}
	eventType := parseJSONStringRaw(raw["type"])
	if eventType == "response.failed" || eventType == "response.incomplete" {
		p.done = true
		p.terminalSeen = true
		var response map[string]interface{}
		if err := decodeJSONStrict(bytes.NewReader(raw["response"]), &response); err != nil || response == nil {
			p.terminalError = &responsesWebSocketEventError{
				status:  http.StatusBadGateway,
				errType: "provider_error",
				code:    "invalid_terminal_response",
				message: "response stream returned an invalid terminal response",
			}
			return
		}
		p.terminalResponse = response
		return
	}
	if eventType == "response.cancelled" || eventType == "response.canceled" || eventType == "error" {
		p.done = true
		p.terminalSeen = true
		p.terminalError = &responsesWebSocketEventError{
			status:  http.StatusBadGateway,
			errType: "provider_error",
			code:    "upstream_stream_error",
			message: "response stream did not complete successfully",
		}
		return
	}
	if eventType != "response.completed" {
		return
	}
	p.done = true
	p.terminalSeen = true
	var response map[string]interface{}
	if err := decodeJSONStrict(bytes.NewReader(raw["response"]), &response); err != nil {
		return
	}
	p.completedResponse = response
}

func cloneResponsesRawMap(src map[string]json.RawMessage) map[string]json.RawMessage {
	if len(src) == 0 {
		return map[string]json.RawMessage{}
	}
	dst := make(map[string]json.RawMessage, len(src))
	for key, value := range src {
		dst[key] = cloneResponsesRawMessage(value)
	}
	return dst
}

func cloneResponsesRawMessage(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return nil
	}
	out := make(json.RawMessage, len(raw))
	copy(out, raw)
	return out
}

func cloneResponsesInterfaceSlice(src []interface{}) []interface{} {
	if len(src) == 0 {
		return nil
	}
	b, err := json.Marshal(src)
	if err != nil {
		out := make([]interface{}, 0, len(src))
		out = append(out, src...)
		return out
	}
	var out []interface{}
	if err := decodeJSONStrict(bytes.NewReader(b), &out); err != nil {
		out = make([]interface{}, 0, len(src))
		out = append(out, src...)
	}
	return out
}

func responsesWebSocketEventErrorFromError(err error) *responsesWebSocketEventError {
	if err == nil {
		return nil
	}
	var eventErr *responsesWebSocketEventError
	if errors.As(err, &eventErr) {
		return eventErr
	}
	return &responsesWebSocketEventError{
		status:  http.StatusBadGateway,
		errType: "provider_error",
		message: strings.TrimSpace(err.Error()),
	}
}

func nonEmptyOrDefault(value string, fallback string) string {
	if strings.TrimSpace(value) == "" {
		return fallback
	}
	return strings.TrimSpace(value)
}
