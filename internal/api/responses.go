package api

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"

	"github.com/lunargate-ai/gateway/pkg/models"
)

type preservedUnifiedRequestContextKey struct{}

type preservedUnifiedRequest struct {
	rawJSON           json.RawMessage
	sourceRequestType string
}

func withPreservedUnifiedRequest(ctx context.Context, req *models.UnifiedRequest) context.Context {
	if req == nil {
		return ctx
	}
	return context.WithValue(ctx, preservedUnifiedRequestContextKey{}, preservedUnifiedRequest{
		rawJSON:           append(json.RawMessage(nil), req.RawJSON...),
		sourceRequestType: strings.TrimSpace(req.SourceRequestType),
	})
}

func preservedUnifiedRequestFromContext(ctx context.Context) (preservedUnifiedRequest, bool) {
	if ctx == nil {
		return preservedUnifiedRequest{}, false
	}
	preserved, ok := ctx.Value(preservedUnifiedRequestContextKey{}).(preservedUnifiedRequest)
	if !ok || len(bytes.TrimSpace(preserved.rawJSON)) == 0 {
		return preservedUnifiedRequest{}, false
	}
	return preserved, true
}

func responsesRequestToRawMap(req *models.ResponsesRequest) (map[string]json.RawMessage, error) {
	body := []byte(req.RawJSON)
	if len(bytes.TrimSpace(body)) == 0 {
		var err error
		body, err = json.Marshal(req)
		if err != nil {
			return nil, err
		}
	}
	var payload map[string]json.RawMessage
	if err := decodeJSONStrict(bytes.NewReader(body), &payload); err != nil {
		return nil, err
	}
	return payload, nil
}

func responsesRawMapToRequest(payload map[string]json.RawMessage) (*models.ResponsesRequest, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	var req models.ResponsesRequest
	if err := decodeJSONStrict(bytes.NewReader(body), &req); err != nil {
		return nil, err
	}
	req.RawJSON = append(json.RawMessage(nil), body...)
	return &req, nil
}

func responsesResponseToMap(resp *models.ResponsesResponse) (map[string]interface{}, error) {
	body, err := json.Marshal(resp)
	if err != nil {
		return nil, err
	}
	var payload map[string]interface{}
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, err
	}
	return payload, nil
}

func (h *Handler) resolveResponsesHTTPPayload(payload map[string]json.RawMessage) (map[string]json.RawMessage, error) {
	previousResponseID := parseJSONStringRaw(payload["previous_response_id"])
	if previousResponseID == "" {
		return cloneResponsesRawMap(payload), nil
	}
	if h == nil || h.responsesState == nil {
		return cloneResponsesRawMap(payload), nil
	}
	basePayload, ok := h.responsesState.get(previousResponseID)
	if !ok {
		// A native Responses target may own this ID even when it is not in the
		// gateway's bounded local continuation cache. Compatibility validation
		// rejects it later if the selected target requires local translation.
		return cloneResponsesRawMap(payload), nil
	}
	return mergeResponsesWebSocketPayloads(basePayload, payload)
}

func parseResponsesRequest(w http.ResponseWriter, r *http.Request) (*models.ResponsesRequest, bool) {
	limitRequestBody(w, r)
	defer r.Body.Close()

	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeRequestReadError(w, err)
		return nil, false
	}
	var req models.ResponsesRequest
	if err := decodeJSONStrict(bytes.NewReader(body), &req); err != nil {
		writeRequestDecodeError(w, err)
		return nil, false
	}
	req.RawJSON = append(json.RawMessage(nil), body...)
	return &req, true
}

func copyHeaders(dst http.Header, src http.Header) {
	blocked := map[string]struct{}{
		"connection":          {},
		"content-encoding":    {},
		"content-length":      {},
		"content-md5":         {},
		"digest":              {},
		"etag":                {},
		"keep-alive":          {},
		"last-modified":       {},
		"proxy-authenticate":  {},
		"proxy-authorization": {},
		"set-cookie":          {},
		"te":                  {},
		"trailer":             {},
		"transfer-encoding":   {},
		"upgrade":             {},
	}
	for _, value := range src.Values("Connection") {
		for _, token := range strings.Split(value, ",") {
			if token = strings.ToLower(strings.TrimSpace(token)); token != "" {
				blocked[token] = struct{}{}
			}
		}
	}
	for key, values := range src {
		if _, unsafe := blocked[strings.ToLower(strings.TrimSpace(key))]; unsafe {
			continue
		}
		if _, exists := dst[key]; exists {
			continue
		}
		copied := make([]string, 0, len(values))
		for _, value := range values {
			copied = append(copied, value)
		}
		dst[key] = copied
	}
}

func makeResponsesChatRequest(r *http.Request, unifiedReq *models.UnifiedRequest) (*http.Request, error) {
	body, err := json.Marshal(unifiedReq)
	if err != nil {
		return nil, err
	}

	originalPath := strings.TrimSpace(r.URL.Path)
	if originalPath == "" {
		originalPath = "/v1/responses"
	}
	chatReq := r.Clone(withPreservedUnifiedRequest(r.Context(), unifiedReq))
	chatReq.URL.Path = "/v1/chat/completions"
	chatReq.RequestURI = "/v1/chat/completions"
	chatReq.Body = io.NopCloser(bytes.NewReader(body))
	chatReq.ContentLength = int64(len(body))
	chatReq.Header = r.Header.Clone()
	chatReq.Header.Set("Content-Type", "application/json")
	chatReq.Header.Set("X-LunarGate-Request-Type", "responses")
	chatReq.Header.Set("X-LunarGate-Original-Path", originalPath)
	// Creating a Response is stateful and must always yield a fresh response ID.
	chatReq.Header.Set("X-LunarGate-No-Cache", "true")
	chatReq.Header.Set("X-LunarGate-No-Retry", "true")
	chatReq.Header.Set("X-LunarGate-No-Fallback", "true")
	return chatReq, nil
}

func (h *Handler) handleResponsesStream(w http.ResponseWriter, chatReq *http.Request, requestPayload map[string]json.RawMessage, store bool) {
	proxy := newResponsesStreamProxy(w)
	h.ChatCompletions(proxy, chatReq)
	if err := proxy.finalize(); err != nil {
		writeError(w, http.StatusBadGateway, "failed to stream responses event payload", "provider_error")
		return
	}
	if !store || h == nil || h.responsesState == nil || proxy.responseID == "" || proxy.completedResponse == nil {
		return
	}
	h.responsesState.putCompleted(proxy.responseID, requestPayload, proxy.completedResponse)
}

func (h *Handler) handleResponsesNonStream(w http.ResponseWriter, chatReq *http.Request, requestPayload map[string]json.RawMessage, store bool) {
	status, headers, unifiedResp, errorBody, err := h.executeChatCompletionsUnified(chatReq)
	copyHeaders(w.Header(), headers)
	if err != nil {
		writeError(w, http.StatusBadGateway, "failed to parse provider response", "provider_error")
		return
	}
	if status >= 400 {
		w.WriteHeader(status)
		_, _ = w.Write(errorBody)
		return
	}

	resp := models.UnifiedResponseToResponses(unifiedResp)
	if store && h != nil && h.responsesState != nil && resp != nil {
		if completedResponse, err := responsesResponseToMap(resp); err == nil {
			h.responsesState.putCompleted(resp.ID, requestPayload, completedResponse)
		}
	}
	writeJSON(w, status, resp)
}

func (h *Handler) Responses(w http.ResponseWriter, r *http.Request) {
	responsesReq, ok := parseResponsesRequest(w, r)
	if !ok {
		return
	}

	requestPayload, err := responsesRequestToRawMap(responsesReq)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to prepare request", "internal_error")
		return
	}
	resolvedPayload, err := h.resolveResponsesHTTPPayload(requestPayload)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), "invalid_request_error")
		return
	}
	resolvedReq, err := responsesRawMapToRequest(resolvedPayload)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to prepare request", "internal_error")
		return
	}
	resolvedReq.Stream = responsesReq.Stream

	unifiedReq, err := models.ResponsesToUnifiedRequest(resolvedReq)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), "invalid_request_error")
		return
	}
	if err := models.NormalizeUnifiedRequest(unifiedReq); err != nil {
		// Keep unified normalization as a safety net for request invariants
		// shared across all chat-completions flows.
		writeError(w, http.StatusBadRequest, "invalid tool/function calling payload", "invalid_request_error")
		return
	}

	chatReq, err := makeResponsesChatRequest(r, unifiedReq)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to prepare request", "internal_error")
		return
	}

	if unifiedReq.Stream {
		h.handleResponsesStream(w, chatReq, resolvedPayload, responsesReq.Store == nil || *responsesReq.Store)
		return
	}
	h.handleResponsesNonStream(w, chatReq, resolvedPayload, responsesReq.Store == nil || *responsesReq.Store)
}
