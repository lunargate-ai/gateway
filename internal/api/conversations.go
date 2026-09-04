package api

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"

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

type conversationItemInputError struct {
	message string
	param   string
	code    string
}

func (e *conversationItemInputError) Error() string {
	if e == nil {
		return "invalid conversation item"
	}
	return e.message
}

func (h *Handler) CreateConversation(w http.ResponseWriter, r *http.Request) {
	binding, native, err := h.conversationCreateBinding(r)
	if err != nil {
		writeConversationBindingResolutionError(w, err)
		return
	}
	if native {
		body, ok := readResponseOperationBody(w, r)
		if !ok {
			return
		}
		h.proxyNativeConversationCreate(w, r, binding, body)
		return
	}
	if rejectUnsupportedLocalConversationQuery(w, r) {
		return
	}
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
		writeConversationItemInputError(w, err)
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
	conversationID, ok := clientURLResourceID(w, r, "conversation_id")
	if !ok {
		return
	}
	binding, native, local, err := h.resolveConversationOwner(r, conversationID)
	if err != nil {
		writeConversationBindingResolutionError(w, err)
		return
	}
	if native {
		h.proxyNativeConversationRequest(w, r, binding, http.MethodGet, nativeConversationPath(conversationID), nil)
		return
	}
	if !local {
		writeConversationNotFound(w, conversationID)
		return
	}
	if rejectUnsupportedLocalConversationQuery(w, r) {
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
	conversationID, ok := clientURLResourceID(w, r, "conversation_id")
	if !ok {
		return
	}
	binding, native, local, err := h.resolveConversationOwner(r, conversationID)
	if err != nil {
		writeConversationBindingResolutionError(w, err)
		return
	}
	if native {
		body, ok := readResponseOperationBody(w, r)
		if !ok {
			return
		}
		h.proxyNativeConversationRequest(w, r, binding, http.MethodPost, nativeConversationPath(conversationID), body)
		return
	}
	if !local {
		writeConversationNotFound(w, conversationID)
		return
	}
	if rejectUnsupportedLocalConversationQuery(w, r) {
		return
	}
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
	conversation, err := h.conversationsState.updateMetadata(conversationID, metadata)
	if err != nil {
		writeConversationStateErrorForID(w, err, conversationID, "")
		return
	}
	writeJSON(w, http.StatusOK, conversation)
}

func (h *Handler) DeleteConversation(w http.ResponseWriter, r *http.Request) {
	conversationID, ok := clientURLResourceID(w, r, "conversation_id")
	if !ok {
		return
	}
	binding, native, local, err := h.resolveConversationOwner(r, conversationID)
	if err != nil {
		writeConversationBindingResolutionError(w, err)
		return
	}
	if native {
		h.deleteNativeConversation(w, r, binding, conversationID)
		return
	}
	if !local {
		writeConversationNotFound(w, conversationID)
		return
	}
	if rejectUnsupportedLocalConversationQuery(w, r) {
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
	conversationID, ok := clientURLResourceID(w, r, "conversation_id")
	if !ok {
		return
	}
	binding, native, local, err := h.resolveConversationOwner(r, conversationID)
	if err != nil {
		writeConversationBindingResolutionError(w, err)
		return
	}
	if native {
		body, ok := readResponseOperationBody(w, r)
		if !ok {
			return
		}
		h.proxyNativeConversationRequest(w, r, binding, http.MethodPost, nativeConversationItemsPath(conversationID), body)
		return
	}
	if !local {
		writeConversationNotFound(w, conversationID)
		return
	}
	if rejectUnsupportedLocalConversationQuery(w, r) {
		return
	}
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
		writeConversationItemInputError(w, err)
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
	conversationID, ok := clientURLResourceID(w, r, "conversation_id")
	if !ok {
		return
	}
	after, ok := clientOptionalResourceID(w, r.URL.Query().Get("after"), "after")
	if !ok {
		return
	}
	binding, native, local, err := h.resolveConversationOwner(r, conversationID)
	if err != nil {
		writeConversationBindingResolutionError(w, err)
		return
	}
	if native {
		h.proxyNativeConversationRequest(w, r, binding, http.MethodGet, nativeConversationItemsPath(conversationID), nil)
		return
	}
	if !local {
		writeConversationNotFound(w, conversationID)
		return
	}
	if rejectUnsupportedLocalConversationQuery(w, r, "after", "limit", "order") {
		return
	}
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
	items, err := h.conversationsState.listItems(
		conversationID,
		after,
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
	conversationID, ok := clientURLResourceID(w, r, "conversation_id")
	if !ok {
		return
	}
	itemID, ok := clientURLResourceID(w, r, "item_id")
	if !ok {
		return
	}
	binding, native, local, err := h.resolveConversationOwner(r, conversationID)
	if err != nil {
		writeConversationBindingResolutionError(w, err)
		return
	}
	if native {
		h.proxyNativeConversationRequest(w, r, binding, http.MethodGet, nativeConversationItemPath(conversationID, itemID), nil)
		return
	}
	if !local {
		writeConversationNotFound(w, conversationID)
		return
	}
	if rejectUnsupportedLocalConversationQuery(w, r) {
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
	conversationID, ok := clientURLResourceID(w, r, "conversation_id")
	if !ok {
		return
	}
	itemID, ok := clientURLResourceID(w, r, "item_id")
	if !ok {
		return
	}
	binding, native, local, err := h.resolveConversationOwner(r, conversationID)
	if err != nil {
		writeConversationBindingResolutionError(w, err)
		return
	}
	if native {
		h.proxyNativeConversationRequest(w, r, binding, http.MethodDelete, nativeConversationItemPath(conversationID, itemID), nil)
		return
	}
	if !local {
		writeConversationNotFound(w, conversationID)
		return
	}
	if rejectUnsupportedLocalConversationQuery(w, r) {
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
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(dst); err != nil {
		writeRequestDecodeError(w, err)
		return false
	}
	var extra json.RawMessage
	if err := decoder.Decode(&extra); err != io.EOF {
		writeRequestDecodeError(w, err)
		return false
	}
	return true
}

func rejectUnsupportedLocalConversationQuery(w http.ResponseWriter, r *http.Request, allowed ...string) bool {
	if r == nil || r.URL == nil || r.URL.RawQuery == "" {
		return false
	}
	allowedSet := make(map[string]struct{}, len(allowed))
	for _, key := range allowed {
		allowedSet[key] = struct{}{}
	}
	keys := make([]string, 0, len(r.URL.Query()))
	for key := range r.URL.Query() {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, rawKey := range keys {
		key := strings.TrimSpace(rawKey)
		if _, ok := allowedSet[key]; ok {
			continue
		}
		displayKey := key
		if key == "include[]" {
			displayKey = "include"
		}
		code := "unknown_parameter"
		message := fmt.Sprintf("query parameter %q is not supported", displayKey)
		if displayKey == "include" {
			code = "unsupported_feature"
			message = "include is not supported for locally stored conversations"
		}
		writeConversationInvalid(w, message, displayKey, code)
		return true
	}
	return false
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
	seenIDs := make(map[string]struct{}, len(rawItems))
	for index, raw := range rawItems {
		var item map[string]json.RawMessage
		if err := json.Unmarshal(raw, &item); err != nil || item == nil {
			return nil, newConversationItemInputError(index, "", "must be an object", "invalid_conversation_item")
		}
		itemType, hasType := conversationItemJSONString(item["type"])
		if hasType {
			if itemType == "" || itemType != strings.TrimSpace(itemType) {
				return nil, newConversationItemInputError(index, "type", "must be a non-empty string without surrounding whitespace", "invalid_value")
			}
			canonicalType := strings.ToLower(itemType)
			switch canonicalType {
			case "message", "function_call", "function_call_output", "reasoning", "item_reference":
				if itemType != canonicalType {
					return nil, newConversationItemInputError(index, "type", "must use the canonical lowercase item type", "invalid_value")
				}
			}
		} else if _, typeWasSupplied := item["type"]; typeWasSupplied {
			return nil, newConversationItemInputError(index, "type", "must be a non-empty string", "invalid_value")
		} else if _, hasRole := item["role"]; hasRole {
			itemType = "message"
			item["type"] = json.RawMessage(`"message"`)
		}
		if itemType == "" {
			return nil, newConversationItemInputError(index, "type", "is required", "invalid_conversation_item")
		}
		if itemType == "item_reference" {
			return nil, newConversationItemInputError(index, "type", "item_reference cannot be resolved by local conversation storage", "unsupported_feature")
		}
		if err := validateLocalConversationItem(index, itemType, item); err != nil {
			return nil, err
		}
		itemID, hasID, validID := suppliedConversationItemID(item)
		if hasID && !validID {
			return nil, newConversationItemInputError(index, "id", "must be a non-empty string", "invalid_value")
		}
		if !hasID {
			item["id"] = mustJSONRawString(conversationItemPrefix(itemType) + uuid.NewString())
			itemID = conversationItemID(item)
		}
		if _, exists := seenIDs[itemID]; exists {
			return nil, newConversationItemInputError(index, "id", "duplicates another item ID in this request", "duplicate_item_id")
		}
		seenIDs[itemID] = struct{}{}
		if len(item["status"]) == 0 && conversationItemHasStatus(itemType) {
			item["status"] = json.RawMessage(`"completed"`)
		}
		items = append(items, item)
	}
	return items, nil
}

// Local conversation storage interprets only these core Responses item kinds.
// Validate their stable structural contract before retaining them, while keeping
// every additive field verbatim. Other non-empty item types remain opaque so a
// newer Responses item can be stored and replayed without a gateway upgrade.
func validateLocalConversationItem(index int, itemType string, item map[string]json.RawMessage) error {
	switch itemType {
	case "message":
		return validateLocalConversationMessage(index, item)
	case "function_call":
		return validateLocalConversationFunctionCall(index, item)
	case "function_call_output":
		return validateLocalConversationFunctionCallOutput(index, item)
	case "reasoning":
		return validateLocalConversationReasoning(index, item)
	default:
		return nil
	}
}

func validateLocalConversationMessage(index int, item map[string]json.RawMessage) error {
	role, err := requiredConversationItemString(index, item, "role", "role", true)
	if err != nil {
		return err
	}
	switch role {
	case "user", "assistant", "system", "developer":
	default:
		return newConversationItemInputError(index, "role", "must be one of user, assistant, system, or developer", "invalid_value")
	}
	if err := validateConversationItemStatus(index, item); err != nil {
		return err
	}
	rawContent, exists := item["content"]
	if !exists {
		return newConversationItemInputError(index, "content", "is required", "invalid_conversation_item")
	}
	return validateConversationItemContent(index, "content", rawContent)
}

func validateLocalConversationFunctionCall(index int, item map[string]json.RawMessage) error {
	if _, err := requiredConversationItemString(index, item, "call_id", "call_id", true); err != nil {
		return err
	}
	if _, err := requiredConversationItemString(index, item, "name", "name", true); err != nil {
		return err
	}
	if _, err := requiredConversationItemString(index, item, "arguments", "arguments", false); err != nil {
		return err
	}
	return validateConversationItemStatus(index, item)
}

func validateLocalConversationFunctionCallOutput(index int, item map[string]json.RawMessage) error {
	// Local replay correlates function output exclusively by call_id. Newer native
	// variants may also carry caller/name/namespace, but those remain opaque fields
	// and cannot replace the correlation key on a locally managed conversation.
	if _, err := requiredConversationItemString(index, item, "call_id", "call_id", true); err != nil {
		return err
	}
	if err := validateConversationItemStatus(index, item); err != nil {
		return err
	}
	rawOutput, exists := item["output"]
	if !exists {
		return newConversationItemInputError(index, "output", "is required", "invalid_conversation_item")
	}
	return validateConversationItemContent(index, "output", rawOutput)
}

func validateLocalConversationReasoning(index int, item map[string]json.RawMessage) error {
	if _, err := requiredConversationItemString(index, item, "id", "id", true); err != nil {
		return err
	}
	if err := validateConversationItemStatus(index, item); err != nil {
		return err
	}
	rawSummary, exists := item["summary"]
	if !exists {
		return newConversationItemInputError(index, "summary", "is required", "invalid_conversation_item")
	}
	if err := validateConversationItemPartArray(index, "summary", rawSummary); err != nil {
		return err
	}
	if rawContent, exists := item["content"]; exists {
		if err := validateConversationItemPartArray(index, "content", rawContent); err != nil {
			return err
		}
	}
	return nil
}

func validateConversationItemStatus(index int, item map[string]json.RawMessage) error {
	rawStatus, exists := item["status"]
	if !exists {
		return nil
	}
	status, ok := conversationItemJSONString(rawStatus)
	if !ok {
		return newConversationItemInputError(index, "status", "must be a string", "invalid_value")
	}
	switch status {
	case "in_progress", "completed", "incomplete":
		return nil
	default:
		return newConversationItemInputError(index, "status", "must be one of in_progress, completed, or incomplete", "invalid_value")
	}
}

func validateConversationItemContent(index int, field string, raw json.RawMessage) error {
	if _, ok := conversationItemJSONString(raw); ok {
		return nil
	}
	return validateConversationItemPartArrayShape(
		index,
		field,
		raw,
		"must be a string or an array of content objects",
	)
}

func validateConversationItemPartArray(index int, field string, raw json.RawMessage) error {
	return validateConversationItemPartArrayShape(index, field, raw, "must be an array of content objects")
}

func validateConversationItemPartArrayShape(index int, field string, raw json.RawMessage, shapeError string) error {
	var parts []json.RawMessage
	if err := json.Unmarshal(raw, &parts); err != nil || parts == nil {
		return newConversationItemInputError(index, field, shapeError, "invalid_value")
	}
	for partIndex, rawPart := range parts {
		partField := fmt.Sprintf("%s[%d]", field, partIndex)
		var part map[string]json.RawMessage
		if err := json.Unmarshal(rawPart, &part); err != nil || part == nil {
			return newConversationItemInputError(index, partField, "must be an object", "invalid_value")
		}
		partType, err := requiredConversationItemString(index, part, "type", partField+".type", true)
		if err != nil {
			return err
		}
		if partType != strings.TrimSpace(partType) {
			return newConversationItemInputError(index, partField+".type", "must not contain surrounding whitespace", "invalid_value")
		}
		canonicalPartType := strings.ToLower(strings.TrimSpace(partType))
		switch canonicalPartType {
		case "input_text", "output_text", "text", "reasoning_text", "summary_text":
			if partType != canonicalPartType {
				return newConversationItemInputError(index, partField+".type", "must use the canonical lowercase content type", "invalid_value")
			}
			if _, err := requiredConversationItemString(index, part, "text", partField+".text", false); err != nil {
				return err
			}
		case "refusal":
			if partType != canonicalPartType {
				return newConversationItemInputError(index, partField+".type", "must use the canonical lowercase content type", "invalid_value")
			}
			if _, err := requiredConversationItemString(index, part, "refusal", partField+".refusal", false); err != nil {
				return err
			}
		}
	}
	return nil
}

func requiredConversationItemString(
	index int,
	item map[string]json.RawMessage,
	key string,
	param string,
	requireNonEmpty bool,
) (string, error) {
	raw, exists := item[key]
	if !exists {
		return "", newConversationItemInputError(index, param, "is required", "invalid_conversation_item")
	}
	value, ok := conversationItemJSONString(raw)
	if !ok {
		return "", newConversationItemInputError(index, param, "must be a string", "invalid_value")
	}
	if requireNonEmpty && strings.TrimSpace(value) == "" {
		return "", newConversationItemInputError(index, param, "must be a non-empty string", "invalid_value")
	}
	return value, nil
}

func conversationItemJSONString(raw json.RawMessage) (string, bool) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || trimmed[0] != '"' {
		return "", false
	}
	var value string
	if err := json.Unmarshal(trimmed, &value); err != nil {
		return "", false
	}
	return value, true
}

func suppliedConversationItemID(item map[string]json.RawMessage) (string, bool, bool) {
	raw, exists := item["id"]
	if !exists {
		return "", false, true
	}
	var id string
	if err := json.Unmarshal(raw, &id); err != nil || strings.TrimSpace(id) == "" || id != strings.TrimSpace(id) {
		return "", true, false
	}
	return id, true, true
}

func newConversationItemInputError(index int, field, message, code string) *conversationItemInputError {
	param := fmt.Sprintf("items[%d]", index)
	if field != "" {
		param += "." + field
	}
	return &conversationItemInputError{
		message: param + " " + message,
		param:   param,
		code:    code,
	}
}

func writeConversationItemInputError(w http.ResponseWriter, err error) {
	var inputErr *conversationItemInputError
	if errors.As(err, &inputErr) {
		writeConversationInvalid(w, inputErr.message, inputErr.param, inputErr.code)
		return
	}
	writeConversationInvalid(w, err.Error(), "items", "invalid_conversation_item")
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
	if errors.Is(err, errConversationItemIDConflict) {
		writeConversationInvalid(w, "conversation item ID already exists", "items", "duplicate_item_id")
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
