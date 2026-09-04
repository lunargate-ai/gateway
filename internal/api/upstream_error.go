package api

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/lunargate-ai/gateway/internal/resilience"
	"github.com/lunargate-ai/gateway/pkg/models"
	"github.com/rs/zerolog/log"
)

const upstreamErrorBodyLimit = 1 << 20

// upstreamHTTPError is a complete client-facing snapshot of one upstream HTTP
// failure. It never includes an earlier retry or fallback response.
type upstreamHTTPError struct {
	status  int
	headers http.Header
	body    []byte
	message string
	errType string
}

func (e *upstreamHTTPError) Error() string {
	if e == nil {
		return "upstream request failed"
	}
	return e.message
}

func upstreamHTTPErrorFromRetry(err error, providerType string) (*upstreamHTTPError, bool) {
	var statusErr *resilience.RetryableStatusError
	if !errors.As(err, &statusErr) || statusErr == nil || statusErr.StatusCode <= 0 {
		return nil, false
	}
	return newUpstreamHTTPError(
		statusErr.StatusCode,
		statusErr.Headers,
		statusErr.Body,
		statusErr.Truncated,
		providerType,
	), true
}

func readUpstreamHTTPError(resp *http.Response, providerType string) *upstreamHTTPError {
	if resp == nil {
		return newUpstreamHTTPError(http.StatusBadGateway, nil, nil, true, providerType)
	}

	headers := resp.Header.Clone()
	body := []byte(nil)
	truncated := false
	if resp.Body != nil {
		var readErr error
		body, readErr = io.ReadAll(io.LimitReader(resp.Body, upstreamErrorBodyLimit+1))
		_ = resp.Body.Close()
		if len(body) > upstreamErrorBodyLimit {
			body = append([]byte(nil), body[:upstreamErrorBodyLimit]...)
			truncated = true
		}
		if readErr != nil {
			truncated = true
		}
	}
	return newUpstreamHTTPError(resp.StatusCode, headers, body, truncated, providerType)
}

func newUpstreamHTTPError(status int, headers http.Header, body []byte, truncated bool, providerType string) *upstreamHTTPError {
	if status < 100 || status > 599 {
		status = http.StatusBadGateway
	}

	detail := extractUpstreamErrorDetail(status, body)
	result := &upstreamHTTPError{
		status:  status,
		headers: headers.Clone(),
		message: detail.Message,
		errType: detail.Type,
	}

	if !truncated && strings.EqualFold(strings.TrimSpace(providerType), "openai") && isOpenAIErrorEnvelope(body) {
		result.body = append([]byte(nil), body...)
		return result
	}

	encoded, err := json.Marshal(models.ErrorResponse{Error: detail})
	if err != nil {
		// models.ErrorResponse contains only strings and cannot fail today. Keep a
		// defensive valid envelope in case the model changes in the future.
		encoded = []byte(`{"error":{"message":"upstream request failed","type":"upstream_error","param":null,"code":null}}`)
	}
	result.body = encoded
	return result
}

func isOpenAIErrorEnvelope(body []byte) bool {
	document := bytes.TrimSpace(body)
	if len(document) == 0 || !json.Valid(document) {
		return false
	}
	var payload map[string]json.RawMessage
	if err := json.Unmarshal(document, &payload); err != nil || payload == nil {
		return false
	}
	var errorObject map[string]json.RawMessage
	if err := json.Unmarshal(payload["error"], &errorObject); err != nil || errorObject == nil {
		return false
	}
	return true
}

func extractUpstreamErrorDetail(status int, body []byte) models.ErrorDetail {
	detail := models.ErrorDetail{
		Message: fmt.Sprintf("upstream request failed with status %d", status),
		Type:    upstreamErrorTypeForStatus(status),
	}

	var payload map[string]json.RawMessage
	if err := json.Unmarshal(bytes.TrimSpace(body), &payload); err != nil || payload == nil {
		return detail
	}

	var nested map[string]json.RawMessage
	if err := json.Unmarshal(payload["error"], &nested); err == nil && nested != nil {
		if value := rawErrorString(nested["message"]); value != "" {
			detail.Message = value
		}
		if value := rawErrorString(nested["type"]); value != "" {
			detail.Type = value
		}
		if value, ok := rawOptionalErrorString(nested["param"]); ok {
			detail.Param = &value
		}
		if value, ok := rawOptionalErrorString(nested["code"]); ok {
			detail.Code = &value
		}
		return detail
	}

	for _, candidate := range []json.RawMessage{payload["error"], payload["message"], payload["detail"]} {
		if value := rawErrorString(candidate); value != "" {
			detail.Message = value
			break
		}
	}
	if value, ok := firstOptionalErrorString(payload, "param"); ok {
		detail.Param = &value
	}
	if value, ok := firstOptionalErrorString(payload, "code", "errorType", "error_type", "errorCode", "error_code"); ok {
		detail.Code = &value
	}
	return detail
}

func upstreamErrorTypeForStatus(status int) string {
	switch status {
	case http.StatusBadRequest, http.StatusNotFound, http.StatusUnprocessableEntity:
		return "invalid_request_error"
	case http.StatusUnauthorized:
		return "authentication_error"
	case http.StatusForbidden:
		return "permission_error"
	case http.StatusConflict:
		return "conflict_error"
	case http.StatusTooManyRequests:
		return "rate_limit_error"
	default:
		if status >= http.StatusInternalServerError {
			return "server_error"
		}
		return "upstream_error"
	}
}

func firstOptionalErrorString(payload map[string]json.RawMessage, keys ...string) (string, bool) {
	for _, key := range keys {
		if value, ok := rawOptionalErrorString(payload[key]); ok {
			return value, true
		}
	}
	return "", false
}

func rawOptionalErrorString(raw json.RawMessage) (string, bool) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return "", false
	}
	if value := rawErrorString(trimmed); value != "" {
		return value, true
	}
	return string(trimmed), true
}

func rawErrorString(raw json.RawMessage) string {
	var value string
	if err := json.Unmarshal(raw, &value); err != nil {
		return ""
	}
	return strings.TrimSpace(value)
}

func (e *upstreamHTTPError) write(w http.ResponseWriter) {
	if e == nil {
		writeError(w, http.StatusBadGateway, "upstream request failed", "upstream_error")
		return
	}
	copyHeaders(w.Header(), e.headers)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(e.status)
	if _, err := w.Write(e.body); err != nil {
		log.Error().Err(err).Msg("failed to write upstream error response")
	}
}
