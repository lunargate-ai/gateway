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
