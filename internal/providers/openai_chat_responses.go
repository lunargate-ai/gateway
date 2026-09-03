package providers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/lunargate-ai/gateway/pkg/models"
)

var openAIChatToResponsesTopLevelFields = map[string]struct{}{
	"max_completion_tokens": {},
	"max_tokens":            {},
	"messages":              {},
	"model":                 {},
	"previous_response_id":  {},
	"reasoning":             {},
	"reasoning_effort":      {},
	"store":                 {},
	"stream":                {},
	"stream_options":        {},
	"temperature":           {},
	"tool_choice":           {},
	"tools":                 {},
	"top_p":                 {},
	"user":                  {},
}

// ValidateRequestCompatibilityForUpstream rejects Chat Completions controls
// that would otherwise disappear when the request is translated to Responses.
func (t *OpenAITranslator) ValidateRequestCompatibilityForUpstream(
	providerID string,
	upstreamRequestType string,
	req *models.UnifiedRequest,
) error {
	if req == nil || !strings.EqualFold(strings.TrimSpace(upstreamRequestType), "responses") ||
		strings.EqualFold(strings.TrimSpace(req.SourceRequestType), "responses") {
		return nil
	}

	providerID = strings.TrimSpace(providerID)
	if providerID == "" {
		providerID = "openai"
	}
	unsupported := func(field string) error {
		return openAIChatToResponsesCompatibilityError(providerID, field)
	}
	requireResolvedToolCalls := isDeepSeekCompatibilityProfile(t.cfg)
	allowPriorResponseToolCalls := req.PreviousResponseID != ""

	switch {
	case req.Store != nil && *req.Store:
		return unsupported("store")
	case req.N != nil:
		return unsupported("n")
	case req.Stop != nil:
		return unsupported("stop")
	case req.FrequencyPenalty != nil:
		return unsupported("frequency_penalty")
	case req.PresencePenalty != nil:
		return unsupported("presence_penalty")
	case req.Seed != nil:
		return unsupported("seed")
	case req.ResponseFormat != nil:
		return unsupported("response_format")
	case req.LogitBias != nil:
		return unsupported("logit_bias")
	case req.TopK != nil:
		return unsupported("top_k")
	}
	if err := validateOpenAIChatToResponsesTypedMessages(
		req.Messages,
		providerID,
		requireResolvedToolCalls,
		allowPriorResponseToolCalls,
	); err != nil {
		return err
	}

	trimmedRaw := bytes.TrimSpace(req.RawJSON)
	if len(trimmedRaw) == 0 {
		return nil
	}
	var payload map[string]json.RawMessage
	if err := json.Unmarshal(trimmedRaw, &payload); err != nil || payload == nil {
		return unsupported("request")
	}
	if _, hasLegacy := payload["max_tokens"]; hasLegacy {
		if _, hasCurrent := payload["max_completion_tokens"]; hasCurrent {
			return unsupported("max_completion_tokens")
		}
	}
	if field := firstUnsupportedOpenAIChatToResponsesKey(payload, openAIChatToResponsesTopLevelFields, ""); field != "" {
		return unsupported(field)
	}
	if err := validateOpenAIChatToResponsesMessages(
		payload["messages"],
		providerID,
		requireResolvedToolCalls,
		allowPriorResponseToolCalls,
	); err != nil {
		return err
	}
	if rawStore, exists := payload["store"]; exists {
		trimmedStore := bytes.TrimSpace(rawStore)
		if !bytes.Equal(trimmedStore, []byte("null")) {
			var store bool
			if err := json.Unmarshal(trimmedStore, &store); err != nil || store {
				return unsupported("store")
			}
		}
	}
	if err := validateOpenAIChatToResponsesObject(
		payload["stream_options"],
		map[string]struct{}{"include_usage": {}},
		"stream_options",
		providerID,
	); err != nil {
		return err
	}
	if err := validateOpenAIChatToResponsesObject(
		payload["reasoning"],
		map[string]struct{}{"effort": {}},
		"reasoning",
		providerID,
	); err != nil {
		return err
	}
	if err := validateOpenAIChatToResponsesTools(payload["tools"], providerID); err != nil {
		return err
	}
	return validateOpenAIChatToResponsesToolChoice(payload["tool_choice"], providerID)
}

func validateOpenAIChatToResponsesTypedMessages(
	messages []models.Message,
	providerID string,
	requireResolvedToolCalls bool,
	allowPriorResponseToolCalls bool,
) error {
	// Tool-call indexes are streaming response metadata synthesized by request
	// normalization. The preserved client JSON is validated separately, so do
	// not mistake those internal indexes for lossy request fields.
	messagesCopy := append([]models.Message(nil), messages...)
	for messageIndex := range messagesCopy {
		if len(messagesCopy[messageIndex].ToolCalls) == 0 {
			continue
		}
		messagesCopy[messageIndex].ToolCalls = append([]models.ToolCall(nil), messagesCopy[messageIndex].ToolCalls...)
		for callIndex := range messagesCopy[messageIndex].ToolCalls {
			messagesCopy[messageIndex].ToolCalls[callIndex].Index = nil
		}
	}
	raw, err := json.Marshal(messagesCopy)
	if err != nil {
		return openAIChatToResponsesCompatibilityError(providerID, "messages")
	}
	return validateOpenAIChatToResponsesMessages(
		raw,
		providerID,
		requireResolvedToolCalls,
		allowPriorResponseToolCalls,
	)
}

type openAIChatToResponsesToolCall struct {
	id   string
	path string
}

func validateOpenAIChatToResponsesMessages(
	raw json.RawMessage,
	providerID string,
	requireResolvedToolCalls bool,
	allowPriorResponseToolCalls bool,
) error {
	if err := validateTranslatedChatRawMessages(providerID, raw); err != nil {
		return err
	}
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil
	}
	var messages []json.RawMessage
	if err := json.Unmarshal(trimmed, &messages); err != nil {
		return openAIChatToResponsesCompatibilityError(providerID, "messages")
	}
	toolCallsByID := make(map[string]openAIChatToResponsesToolCall)
	outstandingToolCalls := make(map[string]openAIChatToResponsesToolCall)
	resolvedToolCallIDs := make(map[string]struct{})
	toolCallOrder := make([]openAIChatToResponsesToolCall, 0)
	for messageIndex, rawMessage := range messages {
		messagePath := fmt.Sprintf("messages[%d]", messageIndex)
		var message map[string]json.RawMessage
		if err := json.Unmarshal(rawMessage, &message); err != nil || message == nil {
			return openAIChatToResponsesCompatibilityError(providerID, messagePath)
		}
		var role string
		if err := json.Unmarshal(message["role"], &role); err != nil || strings.TrimSpace(role) == "" {
			return openAIChatToResponsesCompatibilityError(providerID, messagePath+".role")
		}
		normalizedRole := strings.ToLower(strings.TrimSpace(role))
		if role != normalizedRole {
			return openAIChatToResponsesCompatibilityError(providerID, messagePath+".role")
		}
		role = normalizedRole
		switch role {
		case "assistant", "developer", "system", "tool", "user":
		default:
			return openAIChatToResponsesCompatibilityError(providerID, messagePath+".role")
		}
		for _, field := range []string{"name", "refusal", "reasoning_content"} {
			if value, exists := message[field]; exists && !bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
				return openAIChatToResponsesCompatibilityError(providerID, messagePath+"."+field)
			}
		}
		if value, exists := message["function_call"]; exists && !bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
			return openAIChatToResponsesCompatibilityError(providerID, messagePath+".function_call")
		}

		toolCalls, err := validateOpenAIChatToResponsesToolCalls(message["tool_calls"], messagePath+".tool_calls", providerID)
		if err != nil {
			return err
		}
		if role != "assistant" && len(toolCalls) > 0 {
			return openAIChatToResponsesCompatibilityError(providerID, messagePath+".tool_calls")
		}
		for _, toolCall := range toolCalls {
			if _, exists := toolCallsByID[toolCall.id]; exists {
				return openAIChatToResponsesCompatibilityError(providerID, toolCall.path)
			}
			if _, alreadyResolved := resolvedToolCallIDs[toolCall.id]; alreadyResolved {
				return openAIChatToResponsesCompatibilityError(providerID, toolCall.path)
			}
			toolCallsByID[toolCall.id] = toolCall
			outstandingToolCalls[toolCall.id] = toolCall
			toolCallOrder = append(toolCallOrder, toolCall)
		}

		rawToolCallID, hasToolCallID := message["tool_call_id"]
		hasToolCallID = hasToolCallID && !bytes.Equal(bytes.TrimSpace(rawToolCallID), []byte("null"))
		toolCallID, validToolCallID := openAIChatToResponsesString(rawToolCallID)
		if hasToolCallID && !validToolCallID {
			return openAIChatToResponsesCompatibilityError(providerID, messagePath+".tool_call_id")
		}
		if role == "tool" {
			if !hasToolCallID || strings.TrimSpace(toolCallID) == "" || toolCallID != strings.TrimSpace(toolCallID) {
				return openAIChatToResponsesCompatibilityError(providerID, messagePath+".tool_call_id")
			}
			if _, alreadyResolved := resolvedToolCallIDs[toolCallID]; alreadyResolved {
				return openAIChatToResponsesCompatibilityError(providerID, messagePath+".tool_call_id")
			}
			if _, exists := outstandingToolCalls[toolCallID]; exists {
				delete(outstandingToolCalls, toolCallID)
			} else if !allowPriorResponseToolCalls {
				return openAIChatToResponsesCompatibilityError(providerID, messagePath+".tool_call_id")
			}
			resolvedToolCallIDs[toolCallID] = struct{}{}
		} else if hasToolCallID {
			return openAIChatToResponsesCompatibilityError(providerID, messagePath+".tool_call_id")
		}

		hasContent, err := validateOpenAIChatToResponsesContent(message["content"], messagePath+".content", role, providerID)
		if err != nil {
			return err
		}
		if !hasContent && !(role == "assistant" && len(toolCalls) > 0) {
			return openAIChatToResponsesCompatibilityError(providerID, messagePath+".content")
		}
	}
	if requireResolvedToolCalls {
		for _, toolCall := range toolCallOrder {
			if _, outstanding := outstandingToolCalls[toolCall.id]; outstanding {
				return openAIChatToResponsesCompatibilityError(providerID, toolCall.path)
			}
		}
	}
	return nil
}

func validateOpenAIChatToResponsesToolCalls(
	raw json.RawMessage,
	path string,
	providerID string,
) ([]openAIChatToResponsesToolCall, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil, nil
	}
	var calls []json.RawMessage
	if err := json.Unmarshal(trimmed, &calls); err != nil {
		return nil, openAIChatToResponsesCompatibilityError(providerID, path)
	}
	validatedCalls := make([]openAIChatToResponsesToolCall, 0, len(calls))
	for callIndex, rawCall := range calls {
		callPath := fmt.Sprintf("%s[%d]", path, callIndex)
		var call map[string]json.RawMessage
		if err := json.Unmarshal(rawCall, &call); err != nil || call == nil {
			return nil, openAIChatToResponsesCompatibilityError(providerID, callPath)
		}
		id, hasID := openAIChatToResponsesString(call["id"])
		if !hasID || strings.TrimSpace(id) == "" || id != strings.TrimSpace(id) {
			return nil, openAIChatToResponsesCompatibilityError(providerID, callPath+".id")
		}
		callType, hasType := openAIChatToResponsesString(call["type"])
		if !hasType || callType != "function" {
			return nil, openAIChatToResponsesCompatibilityError(providerID, callPath+".type")
		}
		var function map[string]json.RawMessage
		if err := json.Unmarshal(call["function"], &function); err != nil || function == nil {
			return nil, openAIChatToResponsesCompatibilityError(providerID, callPath+".function")
		}
		name, hasName := openAIChatToResponsesString(function["name"])
		if !hasName || strings.TrimSpace(name) == "" || name != strings.TrimSpace(name) {
			return nil, openAIChatToResponsesCompatibilityError(providerID, callPath+".function.name")
		}
		if _, hasArguments := openAIChatToResponsesString(function["arguments"]); !hasArguments {
			return nil, openAIChatToResponsesCompatibilityError(providerID, callPath+".function.arguments")
		}
		validatedCalls = append(validatedCalls, openAIChatToResponsesToolCall{id: id, path: callPath + ".id"})
	}
	return validatedCalls, nil
}

func validateOpenAIChatToResponsesContent(raw json.RawMessage, path, role, providerID string) (bool, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return false, nil
	}
	if trimmed[0] == '"' {
		var text string
		if err := json.Unmarshal(trimmed, &text); err != nil {
			return false, openAIChatToResponsesCompatibilityError(providerID, path)
		}
		return true, nil
	}
	var parts []json.RawMessage
	if err := json.Unmarshal(trimmed, &parts); err != nil {
		return false, openAIChatToResponsesCompatibilityError(providerID, path)
	}
	if len(parts) == 0 {
		return false, openAIChatToResponsesCompatibilityError(providerID, path)
	}
	for partIndex, rawPart := range parts {
		partPath := fmt.Sprintf("%s[%d]", path, partIndex)
		var part map[string]json.RawMessage
		if err := json.Unmarshal(rawPart, &part); err != nil || part == nil {
			return false, openAIChatToResponsesCompatibilityError(providerID, partPath)
		}
		var partType string
		if err := json.Unmarshal(part["type"], &partType); err != nil || strings.TrimSpace(partType) == "" {
			return false, openAIChatToResponsesCompatibilityError(providerID, partPath+".type")
		}
		switch partType {
		case "text":
			var text string
			if err := json.Unmarshal(part["text"], &text); err != nil {
				return false, openAIChatToResponsesCompatibilityError(providerID, partPath+".text")
			}
		case "image_url":
			if role != "user" {
				return false, openAIChatToResponsesCompatibilityError(providerID, partPath+".type")
			}
			var reference interface{}
			if err := json.Unmarshal(part["image_url"], &reference); err != nil {
				return false, openAIChatToResponsesCompatibilityError(providerID, partPath+".image_url")
			}
			_, _, field, ok := openAIChatImageReference(reference)
			if !ok {
				return false, openAIChatToResponsesCompatibilityError(providerID, partPath+".image_url"+field)
			}
		default:
			return false, openAIChatToResponsesCompatibilityError(providerID, partPath+".type")
		}
	}
	return true, nil
}

func openAIChatToResponsesString(raw json.RawMessage) (string, bool) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return "", false
	}
	var value string
	if err := json.Unmarshal(trimmed, &value); err != nil {
		return "", false
	}
	return value, true
}

func openAIChatImageReference(reference interface{}) (url, detail, invalidField string, ok bool) {
	switch typed := reference.(type) {
	case string:
		if strings.TrimSpace(typed) == "" {
			return "", "", "", false
		}
		return typed, "", "", true
	case map[string]interface{}:
		rawURL, urlOK := typed["url"].(string)
		if !urlOK || strings.TrimSpace(rawURL) == "" {
			return "", "", ".url", false
		}
		if rawDetail, exists := typed["detail"]; exists {
			var detailOK bool
			detail, detailOK = rawDetail.(string)
			if !detailOK {
				return "", "", ".detail", false
			}
		}
		return rawURL, detail, "", true
	default:
		return "", "", "", false
	}
}

func openAIChatToResponsesCompatibilityError(providerID, field string) *models.CompatibilityError {
	return &models.CompatibilityError{
		Field:    field,
		Provider: providerID,
		Reason:   "Chat Completions field has no faithful mapping to OpenAI Responses",
	}
}

func firstUnsupportedOpenAIChatToResponsesKey(
	payload map[string]json.RawMessage,
	allowed map[string]struct{},
	prefix string,
) string {
	keys := make([]string, 0, len(payload))
	for key := range payload {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		if _, ok := allowed[key]; ok {
			continue
		}
		if prefix == "" {
			return key
		}
		return prefix + "." + key
	}
	return ""
}

func validateOpenAIChatToResponsesObject(
	raw json.RawMessage,
	allowed map[string]struct{},
	path string,
	providerID string,
) error {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil
	}
	var object map[string]json.RawMessage
	if err := json.Unmarshal(trimmed, &object); err != nil || object == nil {
		return openAIChatToResponsesCompatibilityError(providerID, path)
	}
	if field := firstUnsupportedOpenAIChatToResponsesKey(object, allowed, path); field != "" {
		return openAIChatToResponsesCompatibilityError(providerID, field)
	}
	return nil
}

func validateOpenAIChatToResponsesTools(raw json.RawMessage, providerID string) error {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil
	}
	var tools []json.RawMessage
	if err := json.Unmarshal(trimmed, &tools); err != nil {
		return openAIChatToResponsesCompatibilityError(providerID, "tools")
	}
	for index, rawTool := range tools {
		path := fmt.Sprintf("tools[%d]", index)
		var tool map[string]json.RawMessage
		if err := json.Unmarshal(rawTool, &tool); err != nil || tool == nil {
			return openAIChatToResponsesCompatibilityError(providerID, path)
		}
		if field := firstUnsupportedOpenAIChatToResponsesKey(
			tool,
			map[string]struct{}{"function": {}, "type": {}},
			path,
		); field != "" {
			return openAIChatToResponsesCompatibilityError(providerID, field)
		}
		var toolType string
		if err := json.Unmarshal(tool["type"], &toolType); err != nil || !strings.EqualFold(strings.TrimSpace(toolType), "function") {
			return openAIChatToResponsesCompatibilityError(providerID, path+".type")
		}
		var function map[string]json.RawMessage
		if err := json.Unmarshal(tool["function"], &function); err != nil || function == nil {
			return openAIChatToResponsesCompatibilityError(providerID, path+".function")
		}
		if field := firstUnsupportedOpenAIChatToResponsesKey(
			function,
			map[string]struct{}{"description": {}, "name": {}, "parameters": {}, "strict": {}},
			path+".function",
		); field != "" {
			return openAIChatToResponsesCompatibilityError(providerID, field)
		}
		if rawStrict, exists := function["strict"]; exists {
			// Chat Completions defines strict as boolean | null. Responses
			// requires the field but also accepts null; the typed mapping turns
			// both null and absence into its equivalent false default.
			if !bytes.Equal(bytes.TrimSpace(rawStrict), []byte("null")) {
				var strict bool
				if err := json.Unmarshal(rawStrict, &strict); err != nil {
					return openAIChatToResponsesCompatibilityError(providerID, path+".function.strict")
				}
			}
		}
	}
	return nil
}

func validateOpenAIChatToResponsesToolChoice(raw json.RawMessage, providerID string) error {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil
	}
	if trimmed[0] == '"' {
		var choice string
		if err := json.Unmarshal(trimmed, &choice); err != nil {
			return openAIChatToResponsesCompatibilityError(providerID, "tool_choice")
		}
		switch strings.ToLower(strings.TrimSpace(choice)) {
		case "auto", "none", "required":
			return nil
		default:
			return openAIChatToResponsesCompatibilityError(providerID, "tool_choice")
		}
	}

	var choice map[string]json.RawMessage
	if err := json.Unmarshal(trimmed, &choice); err != nil || choice == nil {
		return openAIChatToResponsesCompatibilityError(providerID, "tool_choice")
	}
	if field := firstUnsupportedOpenAIChatToResponsesKey(
		choice,
		map[string]struct{}{"function": {}, "type": {}},
		"tool_choice",
	); field != "" {
		return openAIChatToResponsesCompatibilityError(providerID, field)
	}
	var choiceType string
	if err := json.Unmarshal(choice["type"], &choiceType); err != nil || !strings.EqualFold(strings.TrimSpace(choiceType), "function") {
		return openAIChatToResponsesCompatibilityError(providerID, "tool_choice.type")
	}
	return validateOpenAIChatToResponsesObject(
		choice["function"],
		map[string]struct{}{"name": {}},
		"tool_choice.function",
		providerID,
	)
}
