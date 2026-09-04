package api

import (
	"bytes"
	"encoding/json"
	"time"
)

// completeSyntheticResponsesEnvelope fills the stable Responses resource
// fields which are otherwise lost when a Chat Completions response is
// translated locally. Existing values win so callers can layer terminal state
// and locally emulated conversation data without dropping additive fields.
func completeSyntheticResponsesEnvelope(
	response map[string]interface{},
	requestPayload map[string]json.RawMessage,
	stripSDKOutputText bool,
) map[string]interface{} {
	if response == nil {
		response = make(map[string]interface{})
	}
	if stripSDKOutputText {
		// output_text is an SDK convenience accessor on the stable Responses
		// resource, not a field returned by the stable HTTP/SSE wire contract.
		delete(response, "output_text")
		completeSyntheticResponsesOutput(response)
	}

	setResponsesFieldIfMissing(response, "error", nil)
	setResponsesFieldIfMissing(response, "incomplete_details", nil)
	setResponsesFieldIfMissing(response, "instructions", decodedResponsesRequestField(requestPayload, "instructions", nil))
	setResponsesFieldIfMissing(response, "max_output_tokens", decodedResponsesRequestField(requestPayload, "max_output_tokens", nil))
	setResponsesFieldIfMissing(response, "metadata", map[string]interface{}{})
	setResponsesFieldIfMissing(response, "output", []interface{}{})
	setResponsesFieldIfMissing(response, "parallel_tool_calls", true)
	setResponsesFieldIfMissing(response, "previous_response_id", decodedResponsesRequestField(requestPayload, "previous_response_id", nil))
	setResponsesFieldIfMissing(response, "reasoning", syntheticResponsesReasoning(requestPayload))
	setResponsesFieldIfMissing(response, "store", decodedResponsesRequestField(requestPayload, "store", true))
	setResponsesFieldIfMissing(response, "temperature", decodedResponsesRequestField(requestPayload, "temperature", float64(1)))
	setResponsesFieldIfMissing(response, "text", decodedResponsesRequestField(requestPayload, "text", map[string]interface{}{
		"format": map[string]interface{}{"type": "text"},
	}))
	setResponsesFieldIfMissing(response, "tool_choice", decodedResponsesRequestField(requestPayload, "tool_choice", "auto"))
	setResponsesFieldIfMissing(response, "tools", decodedResponsesRequestField(requestPayload, "tools", []interface{}{}))
	setResponsesFieldIfMissing(response, "top_p", decodedResponsesRequestField(requestPayload, "top_p", float64(1)))
	setResponsesFieldIfMissing(response, "truncation", "disabled")
	setResponsesFieldIfMissing(response, "usage", nil)
	setResponsesFieldIfMissing(response, "user", decodedResponsesRequestField(requestPayload, "user", nil))

	if status, _ := response["status"].(string); status == "completed" {
		if _, exists := response["completed_at"]; !exists {
			completedAt := time.Now().Unix()
			if createdAt := syntheticResponsesTimestamp(response["created_at"]); completedAt < createdAt {
				completedAt = createdAt
			}
			response["completed_at"] = completedAt
		}
	}

	return response
}

func completeSyntheticResponsesOutput(response map[string]interface{}) {
	output, _ := response["output"].([]interface{})
	for _, rawItem := range output {
		item, _ := rawItem.(map[string]interface{})
		content, _ := item["content"].([]interface{})
		for _, rawPart := range content {
			part, _ := rawPart.(map[string]interface{})
			if part["type"] == "output_text" {
				setResponsesFieldIfMissing(part, "annotations", []interface{}{})
			}
		}
	}
}

func setResponsesFieldIfMissing(response map[string]interface{}, key string, value interface{}) {
	if _, exists := response[key]; !exists {
		response[key] = value
	}
}

func decodedResponsesRequestField(
	payload map[string]json.RawMessage,
	key string,
	fallback interface{},
) interface{} {
	raw, exists := payload[key]
	if !exists || len(bytes.TrimSpace(raw)) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return fallback
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var value interface{}
	if err := decoder.Decode(&value); err != nil {
		return fallback
	}
	return value
}

func syntheticResponsesReasoning(payload map[string]json.RawMessage) map[string]interface{} {
	effort := interface{}(nil)
	if rawReasoning := decodedResponsesRequestField(payload, "reasoning", nil); rawReasoning != nil {
		if reasoning, ok := rawReasoning.(map[string]interface{}); ok {
			effort = reasoning["effort"]
		}
	}
	return map[string]interface{}{
		"effort":  effort,
		"summary": nil,
	}
}

func syntheticResponsesTimestamp(value interface{}) int64 {
	switch typed := value.(type) {
	case json.Number:
		parsed, _ := typed.Int64()
		return parsed
	case int64:
		return typed
	case int:
		return int64(typed)
	case float64:
		return int64(typed)
	default:
		return 0
	}
}
