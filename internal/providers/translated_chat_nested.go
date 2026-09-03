package providers

import (
	"bytes"
	"encoding/json"
	"sort"
	"strconv"
	"strings"

	"github.com/lunargate-ai/gateway/pkg/models"
)

func validateTranslatedChatRawControls(providerID string, req *models.UnifiedRequest) error {
	if req == nil || strings.EqualFold(strings.TrimSpace(req.SourceRequestType), "responses") {
		return nil
	}
	raw := bytes.TrimSpace(req.RawJSON)
	if len(raw) == 0 {
		return nil
	}

	var payload map[string]json.RawMessage
	if err := json.Unmarshal(raw, &payload); err != nil || payload == nil {
		return translatedChatNestedCompatibilityError(providerID, "request", "Chat Completions request cannot be validated")
	}
	if err := validateTranslatedChatRawObject(
		providerID,
		payload["stream_options"],
		"stream_options",
		map[string]struct{}{"include_usage": {}},
	); err != nil {
		return err
	}
	if err := validateTranslatedChatRawObject(
		providerID,
		payload["reasoning"],
		"reasoning",
		map[string]struct{}{"effort": {}},
	); err != nil {
		return err
	}
	if err := validateTranslatedChatRawResponseFormat(providerID, payload["response_format"]); err != nil {
		return err
	}
	if err := validateTranslatedChatRawMessages(providerID, payload["messages"]); err != nil {
		return err
	}
	if err := validateTranslatedChatRawTools(providerID, payload["tools"], "tools", true); err != nil {
		return err
	}
	if err := validateTranslatedChatRawTools(providerID, payload["functions"], "functions", false); err != nil {
		return err
	}
	if err := validateTranslatedChatRawToolChoice(providerID, payload["tool_choice"], "tool_choice"); err != nil {
		return err
	}
	if err := validateTranslatedChatRawToolChoice(providerID, payload["function_call"], "function_call"); err != nil {
		return err
	}

	reasoning, ok := decodeTranslatedChatRawObject(payload["reasoning"])
	if !ok {
		return nil
	}
	rawNestedEffort, hasNestedEffort := reasoning["effort"]
	rawTopLevelEffort, hasTopLevelEffort := payload["reasoning_effort"]
	if !hasNestedEffort || !hasTopLevelEffort {
		return nil
	}
	var nestedEffort, topLevelEffort string
	if json.Unmarshal(rawNestedEffort, &nestedEffort) != nil || json.Unmarshal(rawTopLevelEffort, &topLevelEffort) != nil {
		return nil
	}
	if !strings.EqualFold(strings.TrimSpace(nestedEffort), strings.TrimSpace(topLevelEffort)) {
		return translatedChatNestedCompatibilityError(
			providerID,
			"reasoning.effort",
			"reasoning.effort conflicts with reasoning_effort and would otherwise be ignored",
		)
	}
	return nil
}

func validateTranslatedChatRawResponseFormat(providerID string, raw json.RawMessage) error {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil
	}
	format, ok := decodeTranslatedChatRawObject(trimmed)
	if !ok {
		return translatedChatNestedCompatibilityError(providerID, "response_format", "expected a response format object")
	}
	var formatType string
	_ = json.Unmarshal(format["type"], &formatType)
	allowed := map[string]struct{}{"type": {}}
	if strings.EqualFold(strings.TrimSpace(formatType), "json_schema") {
		allowed["json_schema"] = struct{}{}
	}
	if field := firstUnsupportedTranslatedChatRawKey(format, allowed, "response_format"); field != "" {
		return translatedChatNestedCompatibilityError(providerID, field, "response format field has no faithful mapping")
	}
	if !strings.EqualFold(strings.TrimSpace(formatType), "json_schema") {
		return nil
	}
	return validateTranslatedChatRawObject(
		providerID,
		format["json_schema"],
		"response_format.json_schema",
		map[string]struct{}{"description": {}, "name": {}, "schema": {}, "strict": {}},
	)
}

func translatedChatAnnotatedJSONSchema(
	providerID string,
	format *models.JSONSchemaResponseFormat,
) (map[string]interface{}, error) {
	if format == nil || format.Schema == nil {
		return nil, translatedChatNestedCompatibilityError(
			providerID,
			"response_format.json_schema.schema",
			"a JSON schema object is required",
		)
	}
	encoded, err := json.Marshal(format.Schema)
	if err != nil {
		return nil, translatedChatNestedCompatibilityError(
			providerID,
			"response_format.json_schema.schema",
			"JSON schema cannot be represented as an object",
		)
	}
	var schema map[string]interface{}
	if err := decodeJSONPreserveNumbers(encoded, &schema); err != nil || schema == nil {
		return nil, translatedChatNestedCompatibilityError(
			providerID,
			"response_format.json_schema.schema",
			"JSON schema must be an object",
		)
	}
	if err := annotateTranslatedChatJSONSchema(
		providerID,
		schema,
		"title",
		strings.TrimSpace(format.Name),
		"response_format.json_schema.name",
	); err != nil {
		return nil, err
	}
	if err := annotateTranslatedChatJSONSchema(
		providerID,
		schema,
		"description",
		format.Description,
		"response_format.json_schema.description",
	); err != nil {
		return nil, err
	}
	return schema, nil
}

func annotateTranslatedChatJSONSchema(
	providerID string,
	schema map[string]interface{},
	annotation string,
	wrapperValue string,
	wrapperPath string,
) error {
	if wrapperValue == "" {
		return nil
	}
	existing, exists := schema[annotation]
	if !exists {
		schema[annotation] = wrapperValue
		return nil
	}
	existingString, ok := existing.(string)
	if !ok || existingString != wrapperValue {
		return translatedChatNestedCompatibilityError(
			providerID,
			wrapperPath,
			"wrapper annotation conflicts with the same JSON schema annotation",
		)
	}
	return nil
}

func validateTranslatedChatRawTools(providerID string, raw json.RawMessage, path string, wrapped bool) error {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil
	}
	var tools []json.RawMessage
	if err := json.Unmarshal(trimmed, &tools); err != nil {
		return translatedChatNestedCompatibilityError(providerID, path, "expected an array of function definitions")
	}
	for index, rawTool := range tools {
		toolPath := indexedTranslatedChatPath(path, index)
		tool, ok := decodeTranslatedChatRawObject(rawTool)
		if !ok {
			return translatedChatNestedCompatibilityError(providerID, toolPath, "expected a function definition object")
		}
		function := tool
		functionPath := toolPath
		if wrapped {
			if field := firstUnsupportedTranslatedChatRawKey(
				tool,
				map[string]struct{}{"function": {}, "type": {}},
				toolPath,
			); field != "" {
				return translatedChatNestedCompatibilityError(providerID, field, "tool field has no faithful mapping")
			}
			if rawType, exists := tool["type"]; exists {
				var toolType string
				if json.Unmarshal(rawType, &toolType) != nil ||
					(strings.TrimSpace(toolType) != "" && !strings.EqualFold(strings.TrimSpace(toolType), "function")) {
					return translatedChatNestedCompatibilityError(providerID, toolPath+".type", "translated targets only support function tools")
				}
			}
			function, ok = decodeTranslatedChatRawObject(tool["function"])
			if !ok {
				return translatedChatNestedCompatibilityError(providerID, toolPath+".function", "expected a function definition object")
			}
			functionPath += ".function"
		}
		if field := firstUnsupportedTranslatedChatRawKey(
			function,
			map[string]struct{}{"description": {}, "name": {}, "parameters": {}, "strict": {}},
			functionPath,
		); field != "" {
			return translatedChatNestedCompatibilityError(providerID, field, "function field has no faithful mapping")
		}
	}
	return nil
}

func validateTranslatedChatRawToolChoice(providerID string, raw json.RawMessage, path string) error {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) || trimmed[0] == '"' {
		return nil
	}
	choice, ok := decodeTranslatedChatRawObject(trimmed)
	if !ok {
		return translatedChatNestedCompatibilityError(providerID, path, "expected a string or tool-choice object")
	}
	allowed := map[string]struct{}{"function": {}, "type": {}}
	if path == "function_call" {
		allowed = map[string]struct{}{"name": {}}
	}
	if field := firstUnsupportedTranslatedChatRawKey(choice, allowed, path); field != "" {
		return translatedChatNestedCompatibilityError(providerID, field, "tool choice field has no faithful mapping")
	}
	if path == "function_call" {
		return nil
	}
	var choiceType string
	if rawType, exists := choice["type"]; !exists || json.Unmarshal(rawType, &choiceType) != nil || strings.TrimSpace(choiceType) == "" {
		return translatedChatNestedCompatibilityError(providerID, path+".type", "tool choice objects require a string type")
	}
	choiceType = strings.ToLower(strings.TrimSpace(choiceType))
	rawFunction, hasFunction := choice["function"]
	if choiceType != "function" {
		if hasFunction {
			return translatedChatNestedCompatibilityError(
				providerID,
				path+".function",
				"function is only valid when tool choice type is function",
			)
		}
		return translatedChatNestedCompatibilityError(
			providerID,
			path+".type",
			"tool choice objects only support type function; use a string for other modes",
		)
	}
	if !hasFunction {
		return translatedChatNestedCompatibilityError(providerID, path+".function", "function tool choice requires a function object")
	}
	if err := validateTranslatedChatRawObject(
		providerID,
		rawFunction,
		path+".function",
		map[string]struct{}{"name": {}},
	); err != nil {
		return err
	}
	return nil
}

func validateTranslatedChatTypedToolChoice(providerID string, choice interface{}) error {
	if choice == nil {
		return nil
	}
	encoded, err := json.Marshal(choice)
	if err != nil {
		return translatedChatNestedCompatibilityError(providerID, "tool_choice", "tool choice cannot be represented as JSON")
	}
	return validateTranslatedChatRawToolChoice(providerID, encoded, "tool_choice")
}

func validateTranslatedChatRawMessages(providerID string, raw json.RawMessage) error {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil
	}
	var messages []json.RawMessage
	if err := json.Unmarshal(trimmed, &messages); err != nil {
		return translatedChatNestedCompatibilityError(providerID, "messages", "expected an array of message objects")
	}
	messageFields := map[string]struct{}{
		"content":           {},
		"function_call":     {},
		"name":              {},
		"reasoning_content": {},
		"refusal":           {},
		"role":              {},
		"tool_call_id":      {},
		"tool_calls":        {},
	}
	for index, rawMessage := range messages {
		path := indexedTranslatedChatPath("messages", index)
		message, ok := decodeTranslatedChatRawObject(rawMessage)
		if !ok {
			return translatedChatNestedCompatibilityError(providerID, path, "expected a message object")
		}
		if field := firstUnsupportedTranslatedChatRawKey(message, messageFields, path); field != "" {
			return translatedChatNestedCompatibilityError(
				providerID,
				field,
				"message field has no faithful mapping to this translated target",
			)
		}
		if err := validateTranslatedChatRawContent(providerID, message["content"], path+".content"); err != nil {
			return err
		}
		if err := validateTranslatedChatRawToolCalls(providerID, message["tool_calls"], path+".tool_calls"); err != nil {
			return err
		}
		if err := validateTranslatedChatRawFunctionCall(providerID, message["function_call"], path+".function_call"); err != nil {
			return err
		}
	}
	return nil
}

func validateTranslatedChatRawContent(providerID string, raw json.RawMessage, path string) error {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) || trimmed[0] == '"' {
		return nil
	}
	var parts []json.RawMessage
	if err := json.Unmarshal(trimmed, &parts); err != nil {
		// The typed compatibility pass reports unsupported scalar content.
		return nil
	}
	for index, rawPart := range parts {
		partPath := indexedTranslatedChatPath(path, index)
		part, ok := decodeTranslatedChatRawObject(rawPart)
		if !ok {
			return translatedChatNestedCompatibilityError(providerID, partPath, "expected a content part object")
		}
		var partType string
		_ = json.Unmarshal(part["type"], &partType)
		var allowed map[string]struct{}
		switch strings.ToLower(strings.TrimSpace(partType)) {
		case "text", "input_text", "output_text":
			allowed = map[string]struct{}{"text": {}, "type": {}}
		case "image_url":
			allowed = map[string]struct{}{"image_url": {}, "type": {}}
		case "image", "input_image":
			allowed = map[string]struct{}{"detail": {}, "image_url": {}, "type": {}, "url": {}}
		default:
			// Provider-specific validation reports the unsupported content type.
			continue
		}
		if field := firstUnsupportedTranslatedChatRawKey(part, allowed, partPath); field != "" {
			return translatedChatNestedCompatibilityError(
				providerID,
				field,
				"content part field has no faithful mapping to this translated target",
			)
		}
		if rawImageURL, exists := part["image_url"]; exists {
			if err := validateTranslatedChatRawImageReference(providerID, rawImageURL, partPath+".image_url"); err != nil {
				return err
			}
		}
	}
	return nil
}

func validateTranslatedChatRawImageReference(providerID string, raw json.RawMessage, path string) error {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) || trimmed[0] == '"' {
		return nil
	}
	return validateTranslatedChatRawObject(
		providerID,
		trimmed,
		path,
		map[string]struct{}{"detail": {}, "url": {}},
	)
}

func validateTranslatedChatRawToolCalls(providerID string, raw json.RawMessage, path string) error {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil
	}
	var calls []json.RawMessage
	if err := json.Unmarshal(trimmed, &calls); err != nil {
		return translatedChatNestedCompatibilityError(providerID, path, "expected an array of tool calls")
	}
	for index, rawCall := range calls {
		callPath := indexedTranslatedChatPath(path, index)
		call, ok := decodeTranslatedChatRawObject(rawCall)
		if !ok {
			return translatedChatNestedCompatibilityError(providerID, callPath, "expected a tool call object")
		}
		if field := firstUnsupportedTranslatedChatRawKey(
			call,
			map[string]struct{}{"function": {}, "id": {}, "type": {}},
			callPath,
		); field != "" {
			return translatedChatNestedCompatibilityError(providerID, field, "tool call field has no faithful mapping")
		}
		if err := validateTranslatedChatRawObject(
			providerID,
			call["function"],
			callPath+".function",
			map[string]struct{}{"arguments": {}, "name": {}},
		); err != nil {
			return err
		}
	}
	return nil
}

func validateTranslatedChatRawFunctionCall(providerID string, raw json.RawMessage, path string) error {
	return validateTranslatedChatRawObject(
		providerID,
		raw,
		path,
		map[string]struct{}{"arguments": {}, "name": {}},
	)
}

func indexedTranslatedChatPath(path string, index int) string {
	return path + "[" + strconv.Itoa(index) + "]"
}

func validateTranslatedChatRawObject(
	providerID string,
	raw json.RawMessage,
	path string,
	allowed map[string]struct{},
) error {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil
	}
	object, ok := decodeTranslatedChatRawObject(trimmed)
	if !ok {
		return translatedChatNestedCompatibilityError(providerID, path, "expected a JSON object")
	}
	if field := firstUnsupportedTranslatedChatRawKey(object, allowed, path); field != "" {
		return translatedChatNestedCompatibilityError(
			providerID,
			field,
			"Chat Completions field has no faithful mapping to this translated target",
		)
	}
	return nil
}

func decodeTranslatedChatRawObject(raw json.RawMessage) (map[string]json.RawMessage, bool) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil, false
	}
	var object map[string]json.RawMessage
	if err := json.Unmarshal(trimmed, &object); err != nil || object == nil {
		return nil, false
	}
	return object, true
}

func firstUnsupportedTranslatedChatRawKey(
	object map[string]json.RawMessage,
	allowed map[string]struct{},
	path string,
) string {
	keys := make([]string, 0, len(object))
	for key := range object {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		if _, ok := allowed[key]; ok {
			continue
		}
		if path == "" {
			return key
		}
		return path + "." + key
	}
	return ""
}

func firstUnsupportedTranslatedChatField(object map[string]interface{}, allowed ...string) string {
	allowedSet := make(map[string]struct{}, len(allowed))
	for _, field := range allowed {
		allowedSet[field] = struct{}{}
	}
	fields := make([]string, 0, len(object))
	for field := range object {
		if _, ok := allowedSet[field]; !ok {
			fields = append(fields, field)
		}
	}
	if len(fields) == 0 {
		return ""
	}
	sort.Strings(fields)
	return fields[0]
}

func translatedChatNestedCompatibilityError(providerID, field, reason string) *models.CompatibilityError {
	providerID = strings.TrimSpace(providerID)
	if providerID == "" {
		providerID = "translated"
	}
	return &models.CompatibilityError{Field: field, Provider: providerID, Reason: reason}
}
