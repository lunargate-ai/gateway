package api

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"

	"github.com/go-chi/chi/v5"
	"github.com/rs/zerolog/log"
)

const (
	defaultResponseInputItemsLimit = 20
	maxResponseInputItemsLimit     = 100
)

type responseDeleted struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Deleted bool   `json:"deleted"`
}

type responseInputItemList struct {
	Object  string            `json:"object"`
	Data    []json.RawMessage `json:"data"`
	FirstID *string           `json:"first_id"`
	LastID  *string           `json:"last_id"`
	HasMore bool              `json:"has_more"`
}

// RetrieveResponse returns a locally emulated response or proxies the request
// to the native provider that owns the response ID.
func (h *Handler) RetrieveResponse(w http.ResponseWriter, r *http.Request) {
	responseID := strings.TrimSpace(chi.URLParam(r, "response_id"))
	if binding, ok, err := h.boundResponseBinding(r, responseID, responseNativeLifecycle); err != nil {
		writeResponseBindingResolutionError(w, err)
		return
	} else if ok {
		h.proxyResponseLifecycleRequest(w, r, binding, http.MethodGet, "responses/"+url.PathEscape(responseID), nil)
		return
	}

	if h != nil && h.responsesState != nil {
		if response, _, ok := h.responsesState.getCompleted(responseID); ok {
			if param := unsupportedLocalResponseRetrieveParam(r); param != "" {
				code := "unsupported_feature"
				writeErrorDetail(
					w,
					http.StatusBadRequest,
					fmt.Sprintf("%s is not supported for locally stored responses", param),
					"invalid_request_error",
					&param,
					&code,
				)
				return
			}
			writeJSON(w, http.StatusOK, response)
			return
		}
	}

	if binding, ok, err := h.explicitResponseBinding(r, responseNativeLifecycle); err != nil {
		writeResponseBindingResolutionError(w, err)
		return
	} else if ok {
		h.proxyResponseLifecycleRequest(w, r, binding, http.MethodGet, "responses/"+url.PathEscape(responseID), nil)
		return
	}
	writeResponseNotFound(w, responseID)
}

func unsupportedLocalResponseRetrieveParam(r *http.Request) string {
	if r == nil || r.URL == nil {
		return ""
	}
	query := r.URL.Query()
	for _, param := range []string{"include", "include_obfuscation", "starting_after"} {
		for key := range query {
			if strings.TrimSuffix(strings.TrimSpace(key), "[]") == param {
				return param
			}
		}
	}
	return ""
}

// DeleteResponse deletes either locally emulated state or the bound native
// resource. Native deletion is attempted exactly once and the owner binding is
// released only after the upstream confirms success.
func (h *Handler) DeleteResponse(w http.ResponseWriter, r *http.Request) {
	responseID := strings.TrimSpace(chi.URLParam(r, "response_id"))
	if binding, ok, err := h.boundResponseBinding(r, responseID, responseNativeLifecycle); err != nil {
		writeResponseBindingResolutionError(w, err)
		return
	} else if ok {
		response, requestErr := h.makeResponseLifecycleRequest(r, binding, http.MethodDelete, "responses/"+url.PathEscape(responseID), nil)
		if requestErr != nil {
			writeNativeResponseTransportError(w, binding, requestErr)
			return
		}
		if response.StatusCode >= http.StatusOK && response.StatusCode < http.StatusMultipleChoices {
			h.responseBindings.delete(responseID)
		}
		h.proxyNativeResponse(w, r, binding, response)
		return
	}

	if h != nil && h.responsesState != nil && h.responsesState.delete(responseID) {
		writeJSON(w, http.StatusOK, responseDeleted{
			ID:      responseID,
			Object:  "response",
			Deleted: true,
		})
		return
	}

	if binding, ok, err := h.explicitResponseBinding(r, responseNativeLifecycle); err != nil {
		writeResponseBindingResolutionError(w, err)
		return
	} else if ok {
		response, requestErr := h.makeResponseLifecycleRequest(r, binding, http.MethodDelete, "responses/"+url.PathEscape(responseID), nil)
		if requestErr != nil {
			writeNativeResponseTransportError(w, binding, requestErr)
			return
		}
		h.proxyNativeResponse(w, r, binding, response)
		return
	}
	writeResponseNotFound(w, responseID)
}

// CancelResponse proxies native cancellation exactly once. Locally emulated
// responses cannot be cancelled because they are already terminal snapshots.
func (h *Handler) CancelResponse(w http.ResponseWriter, r *http.Request) {
	responseID := strings.TrimSpace(chi.URLParam(r, "response_id"))
	body, ok := readResponseOperationBody(w, r)
	if !ok {
		return
	}
	if binding, bound, err := h.boundResponseBinding(r, responseID, responseNativeCancellation); err != nil {
		writeResponseBindingResolutionError(w, err)
		return
	} else if bound {
		h.proxyResponseLifecycleRequest(w, r, binding, http.MethodPost, "responses/"+url.PathEscape(responseID)+"/cancel", body)
		return
	}

	if h != nil && h.responsesState != nil {
		if _, _, local := h.responsesState.getCompleted(responseID); local {
			param := "response_id"
			code := "unsupported_feature"
			writeErrorDetail(w, http.StatusBadRequest, "cancellation is not supported for locally stored responses", "invalid_request_error", &param, &code)
			return
		}
	}

	if binding, explicit, err := h.explicitResponseBinding(r, responseNativeCancellation); err != nil {
		writeResponseBindingResolutionError(w, err)
		return
	} else if explicit {
		h.proxyResponseLifecycleRequest(w, r, binding, http.MethodPost, "responses/"+url.PathEscape(responseID)+"/cancel", body)
		return
	}
	writeResponseNotFound(w, responseID)
}

// ListResponseInputItems lists the locally retained input items used to create
// a stored response. Results follow the Responses API cursor envelope.
func (h *Handler) ListResponseInputItems(w http.ResponseWriter, r *http.Request) {
	responseID := strings.TrimSpace(chi.URLParam(r, "response_id"))
	if binding, ok, err := h.boundResponseBinding(r, responseID, responseNativeLifecycle); err != nil {
		writeResponseBindingResolutionError(w, err)
		return
	} else if ok {
		h.proxyResponseLifecycleRequest(w, r, binding, http.MethodGet, "responses/"+url.PathEscape(responseID)+"/input_items", nil)
		return
	}

	if h != nil && h.responsesState != nil {
		if _, inputItems, ok := h.responsesState.getCompleted(responseID); ok {
			if responseInputItemsIncludeRequested(r) {
				param := "include"
				code := "unsupported_feature"
				writeErrorDetail(
					w,
					http.StatusBadRequest,
					"include is not supported for locally stored response input items",
					"invalid_request_error",
					&param,
					&code,
				)
				return
			}

			page, param, err := paginateResponseInputItems(
				inputItems,
				strings.TrimSpace(r.URL.Query().Get("after")),
				strings.TrimSpace(r.URL.Query().Get("order")),
				strings.TrimSpace(r.URL.Query().Get("limit")),
			)
			if err != nil {
				code := "invalid_value"
				writeErrorDetail(w, http.StatusBadRequest, err.Error(), "invalid_request_error", &param, &code)
				return
			}

			writeJSON(w, http.StatusOK, page)
			return
		}
	}

	if binding, ok, err := h.explicitResponseBinding(r, responseNativeLifecycle); err != nil {
		writeResponseBindingResolutionError(w, err)
		return
	} else if ok {
		h.proxyResponseLifecycleRequest(w, r, binding, http.MethodGet, "responses/"+url.PathEscape(responseID)+"/input_items", nil)
		return
	}
	writeResponseNotFound(w, responseID)
}

func (h *Handler) proxyResponseLifecycleRequest(
	w http.ResponseWriter,
	r *http.Request,
	binding responseBinding,
	method string,
	path string,
	body []byte,
) {
	response, err := h.makeResponseLifecycleRequest(r, binding, method, path, body)
	if err != nil {
		writeNativeResponseTransportError(w, binding, err)
		return
	}
	h.proxyNativeResponse(w, r, binding, response)
}

func writeNativeResponseTransportError(w http.ResponseWriter, binding responseBinding, err error) {
	log.Error().Err(err).Str("provider", strings.TrimSpace(binding.Provider)).Msg("native response provider request failed")
	writeError(w, http.StatusBadGateway, "upstream response provider request failed", "provider_error")
}

func (h *Handler) makeResponseLifecycleRequest(
	r *http.Request,
	binding responseBinding,
	method string,
	path string,
	body []byte,
) (*http.Response, error) {
	rawQuery := ""
	if r != nil && r.URL != nil {
		rawQuery = r.URL.RawQuery
	}
	return h.nativeResponseRequest(r.Context(), method, binding, path, rawQuery, body, r.Header)
}

func readResponseOperationBody(w http.ResponseWriter, r *http.Request) ([]byte, bool) {
	if r == nil || r.Body == nil {
		return nil, true
	}
	limitRequestBody(w, r)
	defer r.Body.Close()
	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeRequestReadError(w, err)
		return nil, false
	}
	return body, true
}

func writeResponseBindingResolutionError(w http.ResponseWriter, err error) {
	resolutionErr, ok := err.(*responseBindingResolutionError)
	if !ok || resolutionErr == nil {
		writeError(w, http.StatusBadRequest, err.Error(), "invalid_request_error")
		return
	}
	param := resolutionErr.param
	code := resolutionErr.code
	writeErrorDetail(w, http.StatusBadRequest, resolutionErr.message, "invalid_request_error", &param, &code)
}

func responseInputItemsIncludeRequested(r *http.Request) bool {
	if r == nil || r.URL == nil {
		return false
	}
	for key := range r.URL.Query() {
		normalized := strings.TrimSuffix(strings.TrimSpace(key), "[]")
		if normalized == "include" {
			return true
		}
	}
	return false
}

func paginateResponseInputItems(
	inputItems []json.RawMessage,
	after string,
	orderValue string,
	limitValue string,
) (responseInputItemList, string, error) {
	limit := defaultResponseInputItemsLimit
	if limitValue != "" {
		parsed, err := strconv.Atoi(limitValue)
		if err != nil || parsed < 1 || parsed > maxResponseInputItemsLimit {
			return responseInputItemList{}, "limit", fmt.Errorf(
				"limit must be an integer between 1 and %d",
				maxResponseInputItemsLimit,
			)
		}
		limit = parsed
	}

	order := strings.ToLower(orderValue)
	if order == "" {
		order = "desc"
	}
	if order != "asc" && order != "desc" {
		return responseInputItemList{}, "order", fmt.Errorf("order must be 'asc' or 'desc'")
	}

	ordered := cloneResponsesRawMessages(inputItems)
	if order == "desc" {
		for left, right := 0, len(ordered)-1; left < right; left, right = left+1, right-1 {
			ordered[left], ordered[right] = ordered[right], ordered[left]
		}
	}

	start := 0
	if after != "" {
		found := false
		for index, item := range ordered {
			if responseInputItemID(item) == after {
				start = index + 1
				found = true
				break
			}
		}
		if !found {
			return responseInputItemList{}, "after", fmt.Errorf("item with id %q was not found", after)
		}
	}

	end := start + limit
	if end > len(ordered) {
		end = len(ordered)
	}
	data := make([]json.RawMessage, end-start)
	copy(data, ordered[start:end])

	page := responseInputItemList{
		Object:  "list",
		Data:    data,
		HasMore: end < len(ordered),
	}
	if len(data) > 0 {
		firstID := responseInputItemID(data[0])
		lastID := responseInputItemID(data[len(data)-1])
		page.FirstID = &firstID
		page.LastID = &lastID
	}
	return page, "", nil
}

func responseInputItemID(item json.RawMessage) string {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(item, &fields); err != nil {
		return ""
	}
	return parseJSONStringRaw(fields["id"])
}

func writeResponseNotFound(w http.ResponseWriter, responseID string) {
	param := "response_id"
	code := "response_not_found"
	writeErrorDetail(
		w,
		http.StatusNotFound,
		fmt.Sprintf("Response with id '%s' not found.", responseID),
		"invalid_request_error",
		&param,
		&code,
	)
}
