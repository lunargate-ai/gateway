package api

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/go-chi/chi/v5"
	"github.com/google/uuid"
)

const (
	maxConversationItemsPerRequest = 20
	maxConversationMetadataPairs   = 16
	maxConversationMetadataKeyLen  = 64
	maxConversationMetadataValLen  = 512
)

type conversationCreateRequest struct {
	Items    []json.RawMessage `json:"items"`
	Metadata json.RawMessage   `json:"metadata"`
}

type conversationUpdateRequest struct {
	Metadata json.RawMessage `json:"metadata"`
}

type conversationItemsCreateRequest struct {
	Items []json.RawMessage `json:"items"`
}

type conversationDeletedObject struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Deleted bool   `json:"deleted"`
}

func (h *Handler) CreateConversation(w http.ResponseWriter, r *http.Request) {
	var req conversationCreateRequest
	if !decodeConversationRequest(w, r, &req) {
		return
	}
	if len(req.Items) > maxConversationItemsPerRequest {
		writeConversationInvalid(w, "items may contain at most 20 items", "items", "array_above_max_length")
		return
	}
	metadata, err := parseConversationMetadata(req.Metadata)
	if err != nil {
		writeConversationInvalid(w, err.Error(), "metadata", "invalid_metadata")
		return
	}
	items, err := prepareConversationItems(req.Items)
	if err != nil {
		writeConversationInvalid(w, err.Error(), "items", "invalid_conversation_item")
		return
	}
	if h == nil || h.conversationsState == nil {
		writeError(w, http.StatusServiceUnavailable, "conversation state is unavailable", "server_error")
		return
	}
	conversation, err := h.conversationsState.create(metadata, items)
	if err != nil {
		writeConversationStateError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, conversation)
}

func (h *Handler) GetConversation(w http.ResponseWriter, r *http.Request) {
	conversationID := strings.TrimSpace(chi.URLParam(r, "conversation_id"))
	if h == nil || h.conversationsState == nil {
		writeConversationNotFound(w, conversationID)
		return
	}
	conversation, ok := h.conversationsState.get(conversationID)
	if !ok {
		writeConversationNotFound(w, conversationID)
		return
	}
	writeJSON(w, http.StatusOK, conversation)
}

func (h *Handler) UpdateConversation(w http.ResponseWriter, r *http.Request) {
	conversationID := strings.TrimSpace(chi.URLParam(r, "conversation_id"))
	var req conversationUpdateRequest
	if !decodeConversationRequest(w, r, &req) {
		return
	}
	if len(req.Metadata) == 0 {
		writeConversationInvalid(w, "metadata is required", "metadata", "missing_required_parameter")
		return
	}
	metadata, err := parseConversationMetadata(req.Metadata)
	if err != nil {
		writeConversationInvalid(w, err.Error(), "metadata", "invalid_metadata")
		return
	}
	if h == nil || h.conversationsState == nil {
		writeConversationNotFound(w, conversationID)
		return
	}
	conversation, err := h.conversationsState.updateMetadata(conversationID, metadata)
	if err != nil {
		writeConversationStateErrorForID(w, err, conversationID, "")
		return
	}
	writeJSON(w, http.StatusOK, conversation)
}

func (h *Handler) DeleteConversation(w http.ResponseWriter, r *http.Request) {
	conversationID := strings.TrimSpace(chi.URLParam(r, "conversation_id"))
	if h == nil || h.conversationsState == nil {
		writeConversationNotFound(w, conversationID)
		return
	}
	conversation, ok := h.conversationsState.delete(conversationID)
	if !ok {
		writeConversationNotFound(w, conversationID)
		return
	}
	writeJSON(w, http.StatusOK, conversationDeletedObject{
		ID:      conversation.ID,
		Object:  "conversation.deleted",
		Deleted: true,
	})
}

func (h *Handler) CreateConversationItems(w http.ResponseWriter, r *http.Request) {
	conversationID := strings.TrimSpace(chi.URLParam(r, "conversation_id"))
	var req conversationItemsCreateRequest
	if !decodeConversationRequest(w, r, &req) {
		return
	}
	if len(req.Items) == 0 {
		writeConversationInvalid(w, "items is required", "items", "missing_required_parameter")
		return
	}
	if len(req.Items) > maxConversationItemsPerRequest {
		writeConversationInvalid(w, "items may contain at most 20 items", "items", "array_above_max_length")
		return
	}
	items, err := prepareConversationItems(req.Items)
	if err != nil {
		writeConversationInvalid(w, err.Error(), "items", "invalid_conversation_item")
		return
	}
	if h == nil || h.conversationsState == nil {
		writeConversationNotFound(w, conversationID)
		return
	}
	created, err := h.conversationsState.addItems(conversationID, items)
	if err != nil {
		writeConversationStateErrorForID(w, err, conversationID, "")
		return
	}
	writeJSON(w, http.StatusOK, newConversationItemList(created, false))
}

func (h *Handler) ListConversationItems(w http.ResponseWriter, r *http.Request) {
	conversationID := strings.TrimSpace(chi.URLParam(r, "conversation_id"))
	order := strings.ToLower(strings.TrimSpace(r.URL.Query().Get("order")))
	if order == "" {
		order = "desc"
	}
	if order != "asc" && order != "desc" {
		writeConversationInvalid(w, "order must be asc or desc", "order", "invalid_value")
		return
	}
	limit := 20
	if rawLimit := strings.TrimSpace(r.URL.Query().Get("limit")); rawLimit != "" {
		parsed, err := strconv.Atoi(rawLimit)
		if err != nil || parsed < 1 || parsed > 100 {
			writeConversationInvalid(w, "limit must be between 1 and 100", "limit", "invalid_value")
			return
		}
		limit = parsed
	}
	if h == nil || h.conversationsState == nil {
		writeConversationNotFound(w, conversationID)
		return
	}
	items, err := h.conversationsState.listItems(
		conversationID,
		strings.TrimSpace(r.URL.Query().Get("after")),
		order,
		limit,
	)
	if err != nil {
		if errors.Is(err, errConversationCursorNotFound) {
			writeConversationInvalid(w, "after must reference an item in the conversation", "after", "invalid_cursor")
			return
		}
		writeConversationStateErrorForID(w, err, conversationID, "")
		return
	}
	writeJSON(w, http.StatusOK, items)
}

func (h *Handler) GetConversationItem(w http.ResponseWriter, r *http.Request) {
	conversationID := strings.TrimSpace(chi.URLParam(r, "conversation_id"))
	itemID := strings.TrimSpace(chi.URLParam(r, "item_id"))
	if h == nil || h.conversationsState == nil {
		writeConversationNotFound(w, conversationID)
		return
	}
	item, err := h.conversationsState.getItem(conversationID, itemID)
	if err != nil {
		writeConversationStateErrorForID(w, err, conversationID, itemID)
		return
	}
	writeJSON(w, http.StatusOK, item)
}

func (h *Handler) DeleteConversationItem(w http.ResponseWriter, r *http.Request) {
	conversationID := strings.TrimSpace(chi.URLParam(r, "conversation_id"))
	itemID := strings.TrimSpace(chi.URLParam(r, "item_id"))
	if h == nil || h.conversationsState == nil {
		writeConversationNotFound(w, conversationID)
		return
	}
	conversation, err := h.conversationsState.deleteItem(conversationID, itemID)
	if err != nil {
		writeConversationStateErrorForID(w, err, conversationID, itemID)
		return
	}
	writeJSON(w, http.StatusOK, conversation)
}

func decodeConversationRequest(w http.ResponseWriter, r *http.Request, dst interface{}) bool {
	limitRequestBody(w, r)
	defer r.Body.Close()
	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeRequestReadError(w, err)
		return false
	}
	if err := decodeJSONStrict(bytes.NewReader(body), dst); err != nil {
		writeRequestDecodeError(w, err)
		return false
	}
	return true
}

func parseConversationMetadata(raw json.RawMessage) (map[string]string, error) {
	if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return map[string]string{}, nil
	}
	var metadata map[string]string
	if err := json.Unmarshal(raw, &metadata); err != nil {
		return nil, fmt.Errorf("metadata must be an object with string values")
	}
	if len(metadata) > maxConversationMetadataPairs {
		return nil, fmt.Errorf("metadata may contain at most 16 properties")
	}
	for key, value := range metadata {
		if utf8.RuneCountInString(key) > maxConversationMetadataKeyLen {
			return nil, fmt.Errorf("metadata keys may contain at most 64 characters")
		}
		if utf8.RuneCountInString(value) > maxConversationMetadataValLen {
			return nil, fmt.Errorf("metadata values may contain at most 512 characters")
		}
	}
	return metadata, nil
}

func prepareConversationItems(rawItems []json.RawMessage) ([]map[string]json.RawMessage, error) {
	items := make([]map[string]json.RawMessage, 0, len(rawItems))
	for index, raw := range rawItems {
		var item map[string]json.RawMessage
		if err := json.Unmarshal(raw, &item); err != nil || item == nil {
			return nil, fmt.Errorf("items[%d] must be an object", index)
		}
		itemType := parseJSONStringRaw(item["type"])
		if itemType == "" && parseJSONStringRaw(item["role"]) != "" {
			itemType = "message"
			item["type"] = json.RawMessage(`"message"`)
		}
		if itemType == "" {
			return nil, fmt.Errorf("items[%d].type is required", index)
		}
		if conversationItemID(item) == "" {
			item["id"] = mustJSONRawString(conversationItemPrefix(itemType) + uuid.NewString())
		}
		if len(item["status"]) == 0 && conversationItemHasStatus(itemType) {
			item["status"] = json.RawMessage(`"completed"`)
		}
		items = append(items, item)
	}
	return items, nil
}

func conversationItemPrefix(itemType string) string {
	switch strings.TrimSpace(itemType) {
	case "message":
		return "msg_"
	case "function_call", "function_call_output":
		return "fc_"
	case "reasoning":
		return "rs_"
	default:
		return "item_"
	}
}

func conversationItemHasStatus(itemType string) bool {
	switch strings.TrimSpace(itemType) {
	case "message", "function_call", "function_call_output", "reasoning":
		return true
	default:
		return false
	}
}

func mustJSONRawString(value string) json.RawMessage {
	raw, _ := json.Marshal(value)
	return raw
}

func newConversationItemList(items []map[string]json.RawMessage, hasMore bool) conversationItemList {
	list := conversationItemList{
		Object:  "list",
		Data:    cloneConversationItems(items),
		HasMore: hasMore,
	}
	if len(items) > 0 {
		firstID := conversationItemID(items[0])
		lastID := conversationItemID(items[len(items)-1])
		list.FirstID = &firstID
		list.LastID = &lastID
	}
	return list
}

func writeConversationInvalid(w http.ResponseWriter, message, param, code string) {
	writeErrorDetail(w, http.StatusBadRequest, message, "invalid_request_error", &param, &code)
}

func writeConversationNotFound(w http.ResponseWriter, conversationID string) {
	param := "conversation_id"
	code := "conversation_not_found"
	message := "conversation not found"
	if conversationID != "" {
		message = fmt.Sprintf("conversation %q not found", conversationID)
	}
	writeErrorDetail(w, http.StatusNotFound, message, "invalid_request_error", &param, &code)
}

func writeConversationItemNotFound(w http.ResponseWriter, itemID string) {
	param := "item_id"
	code := "conversation_item_not_found"
	message := "conversation item not found"
	if itemID != "" {
		message = fmt.Sprintf("conversation item %q not found", itemID)
	}
	writeErrorDetail(w, http.StatusNotFound, message, "invalid_request_error", &param, &code)
}

func writeConversationStateError(w http.ResponseWriter, err error) {
	if errors.Is(err, errConversationItemLimit) {
		writeConversationInvalid(w, "conversation contains too many items", "items", "conversation_item_limit_exceeded")
		return
	}
	writeError(w, http.StatusInsufficientStorage, "conversation state storage limit exceeded", "server_error")
}

func writeConversationStateErrorForID(w http.ResponseWriter, err error, conversationID, itemID string) {
	switch {
	case errors.Is(err, errConversationNotFound):
		writeConversationNotFound(w, conversationID)
	case errors.Is(err, errConversationItemNotFound):
		writeConversationItemNotFound(w, itemID)
	default:
		writeConversationStateError(w, err)
	}
}
