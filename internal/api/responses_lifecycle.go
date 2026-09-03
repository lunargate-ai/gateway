package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"

	"github.com/go-chi/chi/v5"
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

// RetrieveResponse returns a response retained by LunarGate's bounded local
// Responses state. Native provider lifecycle proxying is intentionally not
// implied by this endpoint.
func (h *Handler) RetrieveResponse(w http.ResponseWriter, r *http.Request) {
	responseID := strings.TrimSpace(chi.URLParam(r, "response_id"))
	if h == nil || h.responsesState == nil {
		writeResponseNotFound(w, responseID)
		return
	}

	response, _, ok := h.responsesState.getCompleted(responseID)
	if !ok {
		writeResponseNotFound(w, responseID)
		return
	}
	writeJSON(w, http.StatusOK, response)
}

// DeleteResponse deletes a response retained by the local bounded state.
func (h *Handler) DeleteResponse(w http.ResponseWriter, r *http.Request) {
	responseID := strings.TrimSpace(chi.URLParam(r, "response_id"))
	if h == nil || h.responsesState == nil || !h.responsesState.delete(responseID) {
		writeResponseNotFound(w, responseID)
		return
	}

	writeJSON(w, http.StatusOK, responseDeleted{
		ID:      responseID,
		Object:  "response",
		Deleted: true,
	})
}

// ListResponseInputItems lists the locally retained input items used to create
// a stored response. Results follow the Responses API cursor envelope.
func (h *Handler) ListResponseInputItems(w http.ResponseWriter, r *http.Request) {
	responseID := strings.TrimSpace(chi.URLParam(r, "response_id"))
	if h == nil || h.responsesState == nil {
		writeResponseNotFound(w, responseID)
		return
	}

	_, inputItems, ok := h.responsesState.getCompleted(responseID)
	if !ok {
		writeResponseNotFound(w, responseID)
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
