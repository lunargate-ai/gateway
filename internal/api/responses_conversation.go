package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

type responsesConversationAssociation struct {
	id       string
	rawInput json.RawMessage
}

type responsesConversationRequestError struct {
	message string
	param   string
	code    string
}

func (e *responsesConversationRequestError) Error() string {
	if e == nil {
		return "invalid conversation"
	}
	return e.message
}

func (h *Handler) resolveResponsesConversationPayload(
	payload map[string]json.RawMessage,
) (map[string]json.RawMessage, *responsesConversationAssociation, error) {
	resolved := cloneResponsesRawMap(payload)
	rawConversation, present := resolved["conversation"]
	if !present || bytes.Equal(bytes.TrimSpace(rawConversation), []byte("null")) {
		return resolved, nil, nil
	}
	if previousResponseID := parseJSONStringRaw(resolved["previous_response_id"]); previousResponseID != "" {
		return nil, nil, &responsesConversationRequestError{
			message: "conversation cannot be used together with previous_response_id",
			param:   "conversation",
			code:    "invalid_parameter_combination",
		}
	}

	conversationID, err := parseResponsesConversationID(rawConversation)
	if err != nil {
		return nil, nil, &responsesConversationRequestError{
			message: err.Error(),
			param:   "conversation",
			code:    "invalid_value",
		}
	}
	if h == nil || h.conversationsState == nil {
		return resolved, nil, nil
	}
	conversationItems, local := h.conversationsState.getItems(conversationID)
	if !local {
		// Native Responses providers may own conversation IDs which are not in
		// the gateway's bounded local state. Compatibility validation rejects
		// such IDs if routing selects a translated target.
		return resolved, nil, nil
	}

	history := make([]interface{}, 0, len(conversationItems))
	for _, item := range conversationItems {
		var decoded interface{}
		encoded, marshalErr := json.Marshal(item)
		if marshalErr != nil || json.Unmarshal(encoded, &decoded) != nil {
			return nil, nil, fmt.Errorf("failed to prepare conversation history")
		}
		history = append(history, decoded)
	}
	requestItems, err := responsesInputRawToItems(resolved["input"])
	if err != nil {
		return nil, nil, &responsesConversationRequestError{
			message: err.Error(),
			param:   "input",
			code:    "invalid_value",
		}
	}
	combined := make([]interface{}, 0, len(history)+len(requestItems))
	combined = append(combined, history...)
	combined = append(combined, requestItems...)
	if len(combined) > 0 {
		encoded, err := json.Marshal(combined)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to prepare conversation history")
		}
		resolved["input"] = encoded
	}
	delete(resolved, "conversation")

	return resolved, &responsesConversationAssociation{
		id:       conversationID,
		rawInput: cloneResponsesRawMessage(payload["input"]),
	}, nil
}

func parseResponsesConversationID(raw json.RawMessage) (string, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return "", fmt.Errorf("conversation must contain an id")
	}
	if trimmed[0] == '"' {
		var conversationID string
		if err := json.Unmarshal(trimmed, &conversationID); err != nil || strings.TrimSpace(conversationID) == "" {
			return "", fmt.Errorf("conversation must contain a non-empty id")
		}
		return strings.TrimSpace(conversationID), nil
	}
	var conversation struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal(trimmed, &conversation); err != nil || strings.TrimSpace(conversation.ID) == "" {
		return "", fmt.Errorf("conversation must be a string or an object with a non-empty id")
	}
	return strings.TrimSpace(conversation.ID), nil
}

func rawResponsesConversationID(raw json.RawMessage) string {
	if len(bytes.TrimSpace(raw)) == 0 {
		return ""
	}
	var payload map[string]json.RawMessage
	if err := json.Unmarshal(raw, &payload); err != nil {
		return ""
	}
	conversationID, _ := parseResponsesConversationID(payload["conversation"])
	return conversationID
}

func responsesConversationItems(
	rawInput json.RawMessage,
	completedResponse map[string]interface{},
) ([]map[string]json.RawMessage, error) {
	inputItems, err := responsesInputRawToItems(rawInput)
	if err != nil {
		return nil, err
	}
	combined := make([]interface{}, 0, len(inputItems)+4)
	combined = append(combined, inputItems...)
	if completedResponse != nil {
		if output, ok := completedResponse["output"].([]interface{}); ok {
			combined = append(combined, cloneResponsesInterfaceSlice(output)...)
		}
	}

	rawItems := make([]json.RawMessage, 0, len(combined))
	for index, item := range combined {
		encoded, err := json.Marshal(item)
		if err != nil {
			return nil, fmt.Errorf("failed to encode conversation item %d: %w", index, err)
		}
		rawItems = append(rawItems, encoded)
	}
	return prepareConversationItems(rawItems)
}

func (h *Handler) appendResponsesConversation(
	association *responsesConversationAssociation,
	completedResponse map[string]interface{},
) error {
	if association == nil || strings.TrimSpace(association.id) == "" {
		return nil
	}
	if h == nil || h.conversationsState == nil {
		return errConversationNotFound
	}
	items, err := responsesConversationItems(association.rawInput, completedResponse)
	if err != nil {
		return err
	}
	if len(items) == 0 {
		return nil
	}
	_, err = h.conversationsState.addItems(association.id, items)
	return err
}

func attachResponsesConversation(
	completedResponse map[string]interface{},
	association *responsesConversationAssociation,
) {
	if completedResponse == nil || association == nil || strings.TrimSpace(association.id) == "" {
		return
	}
	completedResponse["conversation"] = map[string]interface{}{"id": association.id}
}
