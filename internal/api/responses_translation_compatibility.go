package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/lunargate-ai/gateway/internal/routing"
	"github.com/lunargate-ai/gateway/pkg/models"
)

// translatedResponsesTopLevelFields is intentionally fail-closed. It contains
// only current Responses create fields that the gateway maps to its unified
// Chat Completions contract or resolves locally before target selection.
var translatedResponsesTopLevelFields = map[string]struct{}{
	"conversation":         {},
	"input":                {},
	"instructions":         {},
	"max_output_tokens":    {},
	"model":                {},
	"previous_response_id": {},
	"reasoning":            {},
	"store":                {},
	"stream":               {},
	"temperature":          {},
	"text":                 {},
	"tool_choice":          {},
	"tools":                {},
	"top_p":                {},
	"user":                 {},
}

func validateTranslatedResponsesCompatibility(
	target routing.Target,
	providerID string,
	providerType string,
	req *models.UnifiedRequest,
) error {
	if req == nil || !strings.EqualFold(strings.TrimSpace(req.SourceRequestType), requestTypeResponses) ||
		strings.EqualFold(strings.TrimSpace(target.UpstreamRequestType), requestTypeResponses) ||
		len(bytes.TrimSpace(req.RawJSON)) == 0 {
		return nil
	}

	var payload map[string]json.RawMessage
	if err := json.Unmarshal(req.RawJSON, &payload); err != nil || payload == nil {
		return nil
	}
	if field := firstUnsupportedRawKey(payload, translatedResponsesTopLevelFields, ""); field != "" {
		return translatedResponsesFieldError(providerID, field)
	}
	if err := validateTranslatedResponsesInput(payload["input"], "input", providerID, providerType); err != nil {
		return err
	}
	if raw, exists := payload["instructions"]; exists {
		if err := validateTranslatedResponsesInput(raw, "instructions", providerID, providerType); err != nil {
			return err
		}
	}
	if err := validateTranslatedResponsesReasoning(payload["reasoning"], providerID); err != nil {
		return err
	}
	if err := validateTranslatedResponsesText(payload["text"], providerID); err != nil {
		return err
	}
	if err := validateTranslatedResponsesTools(payload["tools"], providerID); err != nil {
		return err
	}
	return validateTranslatedResponsesToolChoice(payload["tool_choice"], providerID)
}

func translatedResponsesFieldError(providerID, field string) error {
	return &models.CompatibilityError{
		Field:    field,
		Provider: providerID,
		Reason:   "Responses field has no faithful mapping to this translated target",
	}
}

func firstUnsupportedRawKey(payload map[string]json.RawMessage, allowed map[string]struct{}, prefix string) string {
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

func decodeRawObject(raw json.RawMessage) (map[string]json.RawMessage, bool) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil, true
	}
	var object map[string]json.RawMessage
	if err := json.Unmarshal(trimmed, &object); err != nil || object == nil {
		return nil, false
	}
	return object, true
}

func validateTranslatedResponsesReasoning(raw json.RawMessage, providerID string) error {
	if len(bytes.TrimSpace(raw)) == 0 {
		return nil
	}
	object, ok := decodeRawObject(raw)
	if !ok {
		return translatedResponsesFieldError(providerID, "reasoning")
	}
	if field := firstUnsupportedRawKey(object, map[string]struct{}{"effort": {}}, "reasoning"); field != "" {
		return translatedResponsesFieldError(providerID, field)
	}
	return nil
}

func validateTranslatedResponsesText(raw json.RawMessage, providerID string) error {
	if len(bytes.TrimSpace(raw)) == 0 {
		return nil
	}
	text, ok := decodeRawObject(raw)
	if !ok {
		return translatedResponsesFieldError(providerID, "text")
	}
	if field := firstUnsupportedRawKey(text, map[string]struct{}{"format": {}}, "text"); field != "" {
		return translatedResponsesFieldError(providerID, field)
	}
	rawFormat, exists := text["format"]
	if !exists || bytes.Equal(bytes.TrimSpace(rawFormat), []byte("null")) {
		return nil
	}
	format, ok := decodeRawObject(rawFormat)
	if !ok {
		return translatedResponsesFieldError(providerID, "text.format")
	}
	formatType := strings.ToLower(strings.TrimSpace(parseJSONStringRaw(format["type"])))
	switch formatType {
	case "text", "json_object":
		if field := firstUnsupportedRawKey(format, map[string]struct{}{"type": {}}, "text.format"); field != "" {
			return translatedResponsesFieldError(providerID, field)
		}
	case "json_schema":
		allowed := map[string]struct{}{
			"description": {},
			"name":        {},
			"schema":      {},
			"strict":      {},
			"type":        {},
		}
		if field := firstUnsupportedRawKey(format, allowed, "text.format"); field != "" {
			return translatedResponsesFieldError(providerID, field)
		}
		if rawSchema, exists := format["schema"]; !exists || bytes.Equal(bytes.TrimSpace(rawSchema), []byte("null")) {
			return translatedResponsesFieldError(providerID, "text.format.schema")
		}
	default:
		return translatedResponsesFieldError(providerID, "text.format.type")
	}
	return nil
}

func validateTranslatedResponsesTools(raw json.RawMessage, providerID string) error {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil
	}
	var tools []json.RawMessage
	if err := json.Unmarshal(trimmed, &tools); err != nil {
		return translatedResponsesFieldError(providerID, "tools")
	}
	allowed := map[string]struct{}{
		"description": {},
		"name":        {},
		"parameters":  {},
		"strict":      {},
		"type":        {},
	}
	for index, rawTool := range tools {
		path := fmt.Sprintf("tools[%d]", index)
		tool, ok := decodeRawObject(rawTool)
		if !ok {
			return translatedResponsesFieldError(providerID, path)
		}
		if !strings.EqualFold(strings.TrimSpace(parseJSONStringRaw(tool["type"])), "function") {
			return translatedResponsesFieldError(providerID, path+".type")
		}
		if field := firstUnsupportedRawKey(tool, allowed, path); field != "" {
			return translatedResponsesFieldError(providerID, field)
		}
		if rawStrict, exists := tool["strict"]; exists {
			// Responses function strictness is nullable. A null value does not
			// request strict enforcement and is equivalent to the absent/false
			// setting used by translated providers.
			if !bytes.Equal(bytes.TrimSpace(rawStrict), []byte("null")) {
				var strict bool
				if err := json.Unmarshal(rawStrict, &strict); err != nil {
					return translatedResponsesFieldError(providerID, path+".strict")
				}
			}
		}
	}
	return nil
}

func validateTranslatedResponsesToolChoice(raw json.RawMessage, providerID string) error {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil
	}
	if trimmed[0] == '"' {
		mode := strings.ToLower(strings.TrimSpace(parseJSONStringRaw(trimmed)))
		switch mode {
		case "none", "auto", "required":
			return nil
		default:
			return translatedResponsesFieldError(providerID, "tool_choice")
		}
	}
	choice, ok := decodeRawObject(trimmed)
	if !ok {
		return translatedResponsesFieldError(providerID, "tool_choice")
	}
	if !strings.EqualFold(strings.TrimSpace(parseJSONStringRaw(choice["type"])), "function") {
		return translatedResponsesFieldError(providerID, "tool_choice.type")
	}
	if field := firstUnsupportedRawKey(choice, map[string]struct{}{"name": {}, "type": {}}, "tool_choice"); field != "" {
		return translatedResponsesFieldError(providerID, field)
	}
	return nil
}

func validateTranslatedResponsesInput(raw json.RawMessage, path, providerID, providerType string) error {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil
	}
	if trimmed[0] == '"' {
		return nil
	}
	var items []json.RawMessage
	if err := json.Unmarshal(trimmed, &items); err != nil {
		return translatedResponsesFieldError(providerID, path)
	}
	for index, rawItem := range items {
		itemPath := fmt.Sprintf("%s[%d]", path, index)
		item, ok := decodeRawObject(rawItem)
		if !ok {
			return translatedResponsesFieldError(providerID, itemPath)
		}
		itemType := strings.ToLower(strings.TrimSpace(parseJSONStringRaw(item["type"])))
		switch itemType {
		case "", "message":
			allowed := map[string]struct{}{"content": {}, "id": {}, "phase": {}, "role": {}, "status": {}, "type": {}}
			if field := firstUnsupportedRawKey(item, allowed, itemPath); field != "" {
				return translatedResponsesFieldError(providerID, field)
			}
			if err := validateTranslatedResponsesLifecycleFields(item, itemPath, true, providerID); err != nil {
				return err
			}
			role := strings.ToLower(strings.TrimSpace(parseJSONStringRaw(item["role"])))
			switch role {
			case "user", "assistant", "system", "developer":
			default:
				return translatedResponsesFieldError(providerID, itemPath+".role")
			}
			// Translated Responses history may contain an empty assistant output
			// item. It has no model-visible content, and the converter deliberately
			// consumes it while retaining all substantive items. Other roles must
			// remain non-empty.
			if role != "assistant" || !translatedResponsesMessageContentIsEmpty(item["content"]) {
				if err := validateTranslatedResponsesMessageContent(item["content"], itemPath+".content", providerID, providerType); err != nil {
					return err
				}
			}
		case "function_call":
			allowed := map[string]struct{}{"arguments": {}, "call_id": {}, "id": {}, "name": {}, "status": {}, "type": {}}
			if field := firstUnsupportedRawKey(item, allowed, itemPath); field != "" {
				return translatedResponsesFieldError(providerID, field)
			}
			if err := validateTranslatedResponsesLifecycleFields(item, itemPath, false, providerID); err != nil {
				return err
			}
			if callID, ok := rawJSONString(item["call_id"]); !ok || !validOpaqueResourceID(callID) {
				return translatedResponsesFieldError(providerID, itemPath+".call_id")
			}
			if strings.TrimSpace(parseJSONStringRaw(item["name"])) == "" {
				return translatedResponsesFieldError(providerID, itemPath+".name")
			}
			if _, ok := rawJSONString(item["arguments"]); !ok {
				return translatedResponsesFieldError(providerID, itemPath+".arguments")
			}
		case "function_call_output":
			allowed := map[string]struct{}{"call_id": {}, "id": {}, "output": {}, "status": {}, "type": {}}
			if field := firstUnsupportedRawKey(item, allowed, itemPath); field != "" {
				return translatedResponsesFieldError(providerID, field)
			}
			if err := validateTranslatedResponsesLifecycleFields(item, itemPath, false, providerID); err != nil {
				return err
			}
			if callID, ok := rawJSONString(item["call_id"]); !ok || !validOpaqueResourceID(callID) {
				return translatedResponsesFieldError(providerID, itemPath+".call_id")
			}
			if _, ok := rawJSONString(item["output"]); !ok {
				return translatedResponsesFieldError(providerID, itemPath+".output")
			}
		default:
			return translatedResponsesFieldError(providerID, itemPath+".type")
		}
	}
	return nil
}

// Conversations assigns lifecycle metadata to stored items. IDs and supported
// phases do not control model generation, so translated targets may consume
// them without forwarding them. Status is different: treating an incomplete
// message, function call, or function output as completed would expose partial
// data as final Chat Completions history.
func validateTranslatedResponsesLifecycleFields(
	item map[string]json.RawMessage,
	path string,
	allowPhase bool,
	providerID string,
) error {
	if rawID, exists := item["id"]; exists {
		if id, ok := rawJSONString(rawID); !ok || !validOpaqueResourceID(id) {
			return translatedResponsesFieldError(providerID, path+".id")
		}
	}
	if rawStatus, exists := item["status"]; exists &&
		!strings.EqualFold(strings.TrimSpace(parseJSONStringRaw(rawStatus)), "completed") {
		return translatedResponsesFieldError(providerID, path+".status")
	}
	if rawPhase, exists := item["phase"]; exists {
		phase := strings.ToLower(strings.TrimSpace(parseJSONStringRaw(rawPhase)))
		if !allowPhase || (phase != "commentary" && phase != "final_answer") {
			return translatedResponsesFieldError(providerID, path+".phase")
		}
	}
	return nil
}

func translatedResponsesMessageContentIsEmpty(raw json.RawMessage) bool {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return true
	}
	if trimmed[0] == '"' {
		value, ok := rawJSONString(trimmed)
		return ok && strings.TrimSpace(value) == ""
	}
	var parts []json.RawMessage
	return json.Unmarshal(trimmed, &parts) == nil && len(parts) == 0
}

func validateTranslatedResponsesMessageContent(raw json.RawMessage, path, providerID, providerType string) error {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return translatedResponsesFieldError(providerID, path)
	}
	if trimmed[0] == '"' {
		if value, ok := rawJSONString(trimmed); !ok || strings.TrimSpace(value) == "" {
			return translatedResponsesFieldError(providerID, path)
		}
		return nil
	}
	var parts []json.RawMessage
	if err := json.Unmarshal(trimmed, &parts); err != nil {
		return translatedResponsesFieldError(providerID, path)
	}
	if len(parts) == 0 {
		return translatedResponsesFieldError(providerID, path)
	}
	for index, rawPart := range parts {
		partPath := fmt.Sprintf("%s[%d]", path, index)
		part, ok := decodeRawObject(rawPart)
		if !ok {
			return translatedResponsesFieldError(providerID, partPath)
		}
		partType := strings.ToLower(strings.TrimSpace(parseJSONStringRaw(part["type"])))
		switch partType {
		case "input_text", "output_text", "text":
			allowed := map[string]struct{}{"text": {}, "type": {}}
			if partType == "output_text" {
				// Responses output text always carries annotations and may carry
				// logprobs. Neither affects the model-visible text replayed through
				// Chat Completions, but both must be arrays when present.
				allowed["annotations"] = struct{}{}
				allowed["logprobs"] = struct{}{}
			}
			if field := firstUnsupportedRawKey(part, allowed, partPath); field != "" {
				return translatedResponsesFieldError(providerID, field)
			}
			if value, ok := rawJSONString(part["text"]); !ok || strings.TrimSpace(value) == "" {
				return translatedResponsesFieldError(providerID, partPath+".text")
			}
			for _, field := range []string{"annotations", "logprobs"} {
				if rawValue, exists := part[field]; exists {
					var values []json.RawMessage
					if err := json.Unmarshal(rawValue, &values); err != nil {
						return translatedResponsesFieldError(providerID, partPath+"."+field)
					}
				}
			}
		case "input_image":
			if !translatedTargetSupportsImageContent(providerType) {
				return translatedResponsesFieldError(providerID, partPath+".type")
			}
			allowed := map[string]struct{}{"detail": {}, "image_url": {}, "type": {}}
			if field := firstUnsupportedRawKey(part, allowed, partPath); field != "" {
				return translatedResponsesFieldError(providerID, field)
			}
			if strings.TrimSpace(parseJSONStringRaw(part["image_url"])) == "" {
				return translatedResponsesFieldError(providerID, partPath+".image_url")
			}
			if _, hasDetail := part["detail"]; hasDetail && !strings.EqualFold(strings.TrimSpace(providerType), "openai") {
				return translatedResponsesFieldError(providerID, partPath+".detail")
			}
		default:
			return translatedResponsesFieldError(providerID, partPath+".type")
		}
	}
	return nil
}

func translatedTargetSupportsImageContent(providerType string) bool {
	switch strings.ToLower(strings.TrimSpace(providerType)) {
	case "openai", "anthropic", "ollama":
		return true
	default:
		return false
	}
}

func rawJSONString(raw json.RawMessage) (string, bool) {
	if len(bytes.TrimSpace(raw)) == 0 {
		return "", false
	}
	var value string
	if err := json.Unmarshal(raw, &value); err != nil {
		return "", false
	}
	return value, true
}
