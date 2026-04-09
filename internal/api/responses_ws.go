package api

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/websocket"
	"github.com/lunargate-ai/gateway/pkg/models"
	"github.com/rs/zerolog/log"
)

var responsesWebSocketUpgrader = websocket.Upgrader{
	ReadBufferSize:  4096,
	WriteBufferSize: 4096,
	CheckOrigin: func(_ *http.Request) bool {
		return true
	},
}

type responsesWebSocketSession struct {
	conn         *websocket.Conn
	sessionID    string
	cachedStates map[string]*responsesWebSocketCachedState
}

type responsesWebSocketProxy struct {
	session           *responsesWebSocketSession
	headers           http.Header
	statusCode        int
	buffer            bytes.Buffer
	done              bool
	responseID        string
	completedResponse map[string]interface{}
	terminalError     *responsesWebSocketEventError
}

type responsesWebSocketCachedState struct {
	responseID string
	payload    map[string]json.RawMessage
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
	conn, err := responsesWebSocketUpgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Warn().Err(err).Msg("responses websocket upgrade failed")
		return
	}
	defer conn.Close()

	conn.SetReadLimit(maxRequestBodyBytes)
	sessionID := strings.TrimSpace(r.Header.Get("x-lunargate-sessionid"))
	if sessionID == "" {
		sessionID = "wsresp_" + uuid.NewString()
	}
	session := &responsesWebSocketSession{
		conn:         conn,
		sessionID:    sessionID,
		cachedStates: make(map[string]*responsesWebSocketCachedState),
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

		if err := session.handleCreate(h, r, payload); err != nil {
			log.Warn().Err(err).Msg("responses websocket request failed")
			_ = session.writeErrorEvent(responsesWebSocketEventErrorFromError(err))
		}
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
		s.cacheState(responseID, resolvedPayload)
		return s.writeWarmupResponse(responseID, model)
	}

	body, err := marshalResponsesWebSocketHTTPBody(resolvedPayload)
	if err != nil {
		return err
	}

	req := makeResponsesWebSocketHTTPRequest(baseReq, body, s.sessionID)
	proxy := newResponsesWebSocketProxy(s)
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

	if proxy.responseID != "" {
		s.cacheState(proxy.responseID, withCompletedResponseHistory(resolvedPayload, proxy.completedResponse))
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

	previousResponseID := parseJSONStringRaw(payload["previous_response_id"])
	if rawPrevious, ok := payload["previous_response_id"]; ok && len(rawPrevious) > 0 && previousResponseID == "" {
		return nil, &responsesWebSocketEventError{
			status:  http.StatusBadRequest,
			errType: "invalid_request_error",
			param:   "previous_response_id",
			message: "previous_response_id must be a string",
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
	req.Header.Set("Content-Type", "application/json")
	if strings.TrimSpace(req.Header.Get("x-lunargate-sessionid")) == "" && strings.TrimSpace(sessionID) != "" {
		req.Header.Set("x-lunargate-sessionid", strings.TrimSpace(sessionID))
	}
	return req
}

func newResponsesWebSocketProxy(session *responsesWebSocketSession) *responsesWebSocketProxy {
	return &responsesWebSocketProxy{
		session: session,
		headers: make(http.Header),
	}
}

func (p *responsesWebSocketProxy) Header() http.Header {
	return p.headers
}

func (p *responsesWebSocketProxy) WriteHeader(statusCode int) {
	p.statusCode = statusCode
}

func (p *responsesWebSocketProxy) Write(b []byte) (int, error) {
	if p.statusCode >= 400 {
		_, _ = p.buffer.Write(b)
		return len(b), nil
	}

	_, _ = p.buffer.Write(b)
	for {
		all := p.buffer.String()
		idx := strings.Index(all, "\n\n")
		if idx < 0 {
			break
		}
		frame := all[:idx]
		remaining := all[idx+2:]
		p.buffer.Reset()
		_, _ = p.buffer.WriteString(remaining)
		if _, err := p.processSSEFrame(frame); err != nil {
			return len(b), err
		}
	}
	return len(b), nil
}

func (p *responsesWebSocketProxy) finalize() error {
	if p.statusCode >= 400 {
		p.terminalError = parseResponsesHTTPError(p.statusCode, p.buffer.Bytes())
		return p.session.writeErrorEvent(p.terminalError)
	}

	if p.buffer.Len() > 0 {
		emitted, err := p.processSSEFrame(p.buffer.String())
		if err != nil {
			return err
		}
		if !emitted && !p.done {
			raw := bytes.TrimSpace(p.buffer.Bytes())
			if len(raw) > 0 {
				if err := p.sendEvent(raw); err != nil {
					return err
				}
			}
		}
		p.buffer.Reset()
	}
	return nil
}

func (p *responsesWebSocketProxy) processSSEFrame(frame string) (bool, error) {
	emitted := false
	lines := strings.Split(frame, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "" {
			continue
		}
		if payload == "[DONE]" {
			p.done = true
			continue
		}
		emitted = true
		if err := p.sendEvent([]byte(payload)); err != nil {
			return emitted, err
		}
	}
	return emitted, nil
}

func (p *responsesWebSocketProxy) sendEvent(payload []byte) error {
	p.captureEventState(payload)
	return p.session.conn.WriteMessage(websocket.TextMessage, payload)
}

func (s *responsesWebSocketSession) writeErrorEvent(eventErr *responsesWebSocketEventError) error {
	if eventErr == nil {
		eventErr = &responsesWebSocketEventError{
			status:  http.StatusBadGateway,
			errType: "provider_error",
			message: "failed to process websocket request",
		}
	}
	payload := map[string]interface{}{
		"type": "error",
		"status": func() int {
			if eventErr.status > 0 {
				return eventErr.status
			}
			return http.StatusBadGateway
		}(),
		"error": map[string]interface{}{
			"type":    nonEmptyOrDefault(strings.TrimSpace(eventErr.errType), "provider_error"),
			"message": nonEmptyOrDefault(strings.TrimSpace(eventErr.message), "failed to process websocket request"),
		},
	}
	if code := strings.TrimSpace(eventErr.code); code != "" {
		payload["error"].(map[string]interface{})["code"] = code
	}
	if param := strings.TrimSpace(eventErr.param); param != "" {
		payload["error"].(map[string]interface{})["param"] = param
	}
	b, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	return s.conn.WriteMessage(websocket.TextMessage, b)
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

func extractResponsesEventResponseID(payload []byte) string {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(payload, &raw); err != nil {
		return ""
	}
	if responseID := parseJSONStringRaw(raw["response_id"]); responseID != "" {
		return responseID
	}

	responseRaw, ok := raw["response"]
	if !ok || len(responseRaw) == 0 {
		return ""
	}
	var responseObj map[string]json.RawMessage
	if err := json.Unmarshal(responseRaw, &responseObj); err != nil {
		return ""
	}
	return parseJSONStringRaw(responseObj["id"])
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

	if strings.TrimSpace(createReq.previousResponseID) == "" {
		return normalizeResponsesWebSocketPayload(createReq.payload)
	}

	state, ok := s.cachedStates[strings.TrimSpace(createReq.previousResponseID)]
	if !ok {
		return nil, &responsesWebSocketEventError{
			status:  http.StatusBadRequest,
			errType: "invalid_request_error",
			code:    "previous_response_not_found",
			param:   "previous_response_id",
			message: fmt.Sprintf("Previous response with id '%s' not found.", strings.TrimSpace(createReq.previousResponseID)),
		}
	}
	return mergeResponsesWebSocketPayloads(state.payload, createReq.payload)
}

func (s *responsesWebSocketSession) cacheState(responseID string, payload map[string]json.RawMessage) {
	id := strings.TrimSpace(responseID)
	if id == "" || len(payload) == 0 {
		return
	}
	s.cachedStates = map[string]*responsesWebSocketCachedState{
		id: {
			responseID: id,
			payload:    cloneResponsesRawMap(payload),
		},
	}
}

func (s *responsesWebSocketSession) evictState(responseID string) {
	if s == nil {
		return
	}
	delete(s.cachedStates, strings.TrimSpace(responseID))
}

func (s *responsesWebSocketSession) writeWarmupResponse(responseID string, model string) error {
	createdAt := time.Now().Unix()
	created := map[string]interface{}{
		"type": "response.created",
		"response": map[string]interface{}{
			"id":         responseID,
			"object":     "response",
			"created_at": createdAt,
			"status":     "in_progress",
			"model":      model,
			"output":     []interface{}{},
		},
	}
	completed := map[string]interface{}{
		"type": "response.completed",
		"response": map[string]interface{}{
			"id":          responseID,
			"object":      "response",
			"created_at":  createdAt,
			"status":      "completed",
			"model":       model,
			"output":      []interface{}{},
			"output_text": "",
		},
	}
	for _, event := range []map[string]interface{}{created, completed} {
		b, err := json.Marshal(event)
		if err != nil {
			return err
		}
		if err := s.conn.WriteMessage(websocket.TextMessage, b); err != nil {
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
	for _, rawItem := range rawOutput {
		item, _ := rawItem.(map[string]interface{})
		if item == nil {
			continue
		}

		switch strings.TrimSpace(responsesValueString(item["type"])) {
		case "message":
			role := strings.TrimSpace(responsesValueString(item["role"]))
			if role == "" {
				role = "assistant"
			}
			content, ok := item["content"].([]interface{})
			if !ok {
				continue
			}
			items = append(items, map[string]interface{}{
				"type":    "message",
				"role":    role,
				"content": cloneResponsesInterfaceSlice(content),
			})
		case "function_call":
			callID := strings.TrimSpace(responsesValueString(item["call_id"]))
			if callID == "" {
				callID = strings.TrimSpace(responsesValueString(item["id"]))
			}
			name := strings.TrimSpace(responsesValueString(item["name"]))
			if callID == "" || name == "" {
				continue
			}
			items = append(items, map[string]interface{}{
				"type":      "function_call",
				"id":        callID,
				"call_id":   callID,
				"name":      name,
				"arguments": responsesValueString(item["arguments"]),
			})
		}
	}
	return items
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
	responseID := extractResponsesEventResponseID(payload)
	if responseID != "" {
		p.responseID = responseID
	}

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(payload, &raw); err != nil {
		return
	}
	if parseJSONStringRaw(raw["type"]) != "response.completed" {
		return
	}
	var response map[string]interface{}
	if err := json.Unmarshal(raw["response"], &response); err != nil {
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
	if err := json.Unmarshal(b, &out); err != nil {
		out = make([]interface{}, 0, len(src))
		out = append(out, src...)
	}
	return out
}

func responsesValueString(value interface{}) string {
	switch typed := value.(type) {
	case string:
		return typed
	default:
		return fmt.Sprintf("%v", value)
	}
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
