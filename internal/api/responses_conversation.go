package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/lunargate-ai/gateway/internal/modelid"
)

type responsesConversationAssociation struct {
	id            string
	rawInput      json.RawMessage
	native        bool
	nativeBinding conversationBinding
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
	r *http.Request,
	payload map[string]json.RawMessage,
) (map[string]json.RawMessage, *responsesConversationAssociation, error) {
	resolved := cloneResponsesRawMap(payload)
	rawConversation, present := resolved["conversation"]
	if !present || bytes.Equal(bytes.TrimSpace(rawConversation), []byte("null")) {
		return resolved, nil, nil
	}
	_, previousResponsePresent, previousResponseErr := optionalOpaqueResourceID(
		resolved["previous_response_id"],
		"previous_response_id",
	)
	if previousResponseErr != nil {
		return nil, nil, &responsesConversationRequestError{
			message: previousResponseErr.Error(),
			param:   "previous_response_id",
			code:    "invalid_value",
		}
	}
	if previousResponsePresent {
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
	binding, native, local, err := h.resolveConversationOwner(r, conversationID)
	if err != nil {
		return nil, nil, err
	}
	if native {
		if !h.providerSupportsResponseCapability(binding.Provider, responseNativeLifecycle) {
			return nil, nil, nativeResponsesConversationUnsupportedError(binding.Provider)
		}
		if err := validateNativeResponsesConversationProvider(r, resolved, binding.Provider); err != nil {
			return nil, nil, err
		}
		return resolved, &responsesConversationAssociation{
			id:            conversationID,
			native:        true,
			nativeBinding: binding,
		}, nil
	}
	if !local || h == nil || h.conversationsState == nil {
		return nil, nil, errConversationNotFound
	}
	conversationItems, local := h.conversationsState.getItems(conversationID)
	if !local {
		return nil, nil, errConversationNotFound
	}
	var background bool
	if json.Unmarshal(resolved["background"], &background) == nil && background {
		return nil, nil, &responsesConversationRequestError{
			message: "background responses are not supported with locally managed conversations",
			param:   "background",
			code:    "unsupported_feature",
		}
	}

	history := make([]interface{}, 0, len(conversationItems))
	for _, item := range conversationItems {
		var decoded interface{}
		encoded, marshalErr := json.Marshal(item)
		if marshalErr != nil || decodeJSONStrict(bytes.NewReader(encoded), &decoded) != nil {
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

func nativeResponsesConversationUnsupportedError(provider string) *responsesConversationRequestError {
	return &responsesConversationRequestError{
		message: fmt.Sprintf("provider %q does not enable native Responses for conversations", strings.TrimSpace(provider)),
		param:   "conversation",
		code:    "unsupported_feature",
	}
}

func validateNativeResponsesConversationProvider(
	r *http.Request,
	payload map[string]json.RawMessage,
	provider string,
) error {
	provider = strings.TrimSpace(provider)
	for _, selection := range []struct {
		value string
		param string
	}{
		{value: parseJSONStringRaw(payload["model"]), param: "model"},
		{value: strings.TrimSpace(r.Header.Get("X-LunarGate-Model")), param: "model"},
	} {
		selectedProvider, _, ok := modelid.SplitCanonical(selection.value)
		if !ok || strings.EqualFold(strings.TrimSpace(selectedProvider), "lunargate") {
			continue
		}
		if strings.TrimSpace(selectedProvider) != provider {
			return &responsesConversationRequestError{
				message: fmt.Sprintf("conversation belongs to provider %q, not model provider %q", provider, selectedProvider),
				param:   selection.param,
				code:    "invalid_value",
			}
		}
	}
	return nil
}

func parseResponsesConversationID(raw json.RawMessage) (string, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return "", fmt.Errorf("conversation must contain an id")
	}
	if trimmed[0] == '"' {
		var conversationID string
		if err := json.Unmarshal(trimmed, &conversationID); err != nil {
			return "", fmt.Errorf("conversation must contain a string id")
		}
		if !validOpaqueResourceID(conversationID) {
			return "", fmt.Errorf("conversation must contain a non-empty id")
		}
		return conversationID, nil
	}
	var conversation struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal(trimmed, &conversation); err != nil || !validOpaqueResourceID(conversation.ID) {
		return "", fmt.Errorf("conversation must be a string or an object with a non-empty id")
	}
	return conversation.ID, nil
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
	if association == nil || association.native || association.id == "" {
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
	if completedResponse == nil || association == nil || association.native || association.id == "" {
		return
	}
	completedResponse["conversation"] = map[string]interface{}{"id": association.id}
}
