package models

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
)

type ResponsesRequest struct {
	Model           string      `json:"model"`
	Input           interface{} `json:"input"`
	Instructions    string      `json:"instructions,omitempty"`
	Temperature     *float64    `json:"temperature,omitempty"`
	TopP            *float64    `json:"top_p,omitempty"`
	MaxOutputTokens *int        `json:"max_output_tokens,omitempty"`
	Tools           []ResponsesTool `json:"tools,omitempty"`
	ToolChoice      interface{} `json:"tool_choice,omitempty"`
	Stream          bool        `json:"stream,omitempty"`
	User            string      `json:"user,omitempty"`
}

type ResponsesTool struct {
	Type        string        `json:"type"`
	Function    *ToolFunction `json:"function,omitempty"`
	Name        string        `json:"name,omitempty"`
	Description string        `json:"description,omitempty"`
	Parameters  interface{}   `json:"parameters,omitempty"`
}

type ResponsesResponse struct {
	ID         string            `json:"id"`
	Object     string            `json:"object"`
	CreatedAt  int64             `json:"created_at"`
	Status     string            `json:"status"`
	Model      string            `json:"model"`
	Output     []ResponsesOutput `json:"output"`
	OutputText string            `json:"output_text"`
	Usage      *ResponsesUsage   `json:"usage,omitempty"`
}

type ResponsesOutput struct {
	Type      string                 `json:"type"`
	ID        string                 `json:"id,omitempty"`
	Status    string                 `json:"status,omitempty"`
	Role      string                 `json:"role,omitempty"`
	Content   []ResponsesContentPart `json:"content,omitempty"`
	CallID    string                 `json:"call_id,omitempty"`
	Name      string                 `json:"name,omitempty"`
	Arguments string                 `json:"arguments,omitempty"`
}

type ResponsesContentPart struct {
	Type        string `json:"type"`
	Text        string `json:"text,omitempty"`
	Annotations []any  `json:"annotations,omitempty"`
}

type ResponsesUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

func ResponsesToUnifiedRequest(req *ResponsesRequest) (*UnifiedRequest, error) {
	if req == nil {
		return nil, fmt.Errorf("request is required")
	}
	if strings.TrimSpace(req.Model) == "" {
		return nil, fmt.Errorf("model is required")
	}

	messages, err := responsesInputToMessages(req.Input)
	if err != nil {
		return nil, err
	}
	if strings.TrimSpace(req.Instructions) != "" {
		messages = append([]Message{{Role: "system", Content: strings.TrimSpace(req.Instructions)}}, messages...)
	}

	tools, err := responsesToolsToUnified(req.Tools)
	if err != nil {
		return nil, err
	}

	toolChoice, err := normalizeResponsesToolChoice(req.ToolChoice)
	if err != nil {
		return nil, err
	}

	unified := &UnifiedRequest{
		Model:      strings.TrimSpace(req.Model),
		Messages:   messages,
		Tools:      tools,
		ToolChoice: toolChoice,
		Stream:     req.Stream,
		User:       req.User,
	}

	if req.Temperature != nil {
		unified.Temperature = req.Temperature
	}
	if req.TopP != nil {
		unified.TopP = req.TopP
	}
	if req.MaxOutputTokens != nil {
		unified.MaxTokens = req.MaxOutputTokens
	}

	return unified, nil
}

// UnifiedResponseToResponses maps unified chat-completions responses into
// Responses API shape. It prioritizes assistant text and tool calls; for
// non-string assistant content it preserves data by serializing to text.
func UnifiedResponseToResponses(resp *UnifiedResponse) *ResponsesResponse {
	if resp == nil {
		return nil
	}

	output := make([]ResponsesOutput, 0, len(resp.Choices)+1)
	outputText := make([]string, 0, len(resp.Choices))

	for i, choice := range resp.Choices {
		if choice.Message == nil {
			continue
		}

		msgID := fmt.Sprintf("msg_%s_%d", strings.TrimSpace(resp.ID), i)
		parts := make([]ResponsesContentPart, 0, 1)
		if content, ok := choice.Message.Content.(string); ok {
			text := strings.TrimSpace(content)
			if text != "" {
				parts = append(parts, ResponsesContentPart{Type: "output_text", Text: text, Annotations: []any{}})
				outputText = append(outputText, text)
			}
		} else if choice.Message.Content != nil {
			text := strings.TrimSpace(stringifyAny(choice.Message.Content))
			if text != "" {
				parts = append(parts, ResponsesContentPart{Type: "output_text", Text: text, Annotations: []any{}})
				outputText = append(outputText, text)
			}
		}

		output = append(output, ResponsesOutput{
			Type:    "message",
			ID:      msgID,
			Status:  "completed",
			Role:    "assistant",
			Content: parts,
		})

		for _, tc := range choice.Message.ToolCalls {
			output = append(output, ResponsesOutput{
				Type:      "function_call",
				ID:        tc.ID,
				CallID:    tc.ID,
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
				Status:    "completed",
			})
		}
	}

	out := &ResponsesResponse{
		ID:         resp.ID,
		Object:     "response",
		CreatedAt:  resp.Created,
		Status:     "completed",
		Model:      resp.Model,
		Output:     output,
		OutputText: strings.Join(outputText, "\n"),
	}

	if resp.Usage != nil {
		out.Usage = &ResponsesUsage{
			InputTokens:  resp.Usage.PromptTokens,
			OutputTokens: resp.Usage.CompletionTokens,
			TotalTokens:  resp.Usage.TotalTokens,
		}
	}

	return out
}

func responsesInputToMessages(input interface{}) ([]Message, error) {
	switch v := input.(type) {
	case string:
		if strings.TrimSpace(v) == "" {
			return nil, fmt.Errorf("input is required")
		}
		return []Message{{Role: "user", Content: v}}, nil
	case []interface{}:
		if len(v) == 0 {
			return nil, fmt.Errorf("input is required")
		}
		messages := make([]Message, 0, len(v))
		for _, item := range v {
			msg, err := responsesInputItemToMessage(item)
			if err != nil {
				return nil, err
			}
			if msg != nil {
				messages = append(messages, *msg)
			}
		}
		if len(messages) == 0 {
			return nil, fmt.Errorf("input is required")
		}
		return mergeAssistantToolCallMessages(messages), nil
	default:
		return nil, fmt.Errorf("unsupported input format")
	}
}

func mergeAssistantToolCallMessages(messages []Message) []Message {
	if len(messages) < 2 {
		return messages
	}

	merged := make([]Message, 0, len(messages))
	for i := range messages {
		m := messages[i]
		if len(merged) > 0 && isAssistantToolCallMessage(m) && isAssistantToolCallMessage(merged[len(merged)-1]) {
			merged[len(merged)-1].ToolCalls = append(merged[len(merged)-1].ToolCalls, m.ToolCalls...)
			continue
		}
		merged = append(merged, m)
	}
	return merged
}

func isAssistantToolCallMessage(m Message) bool {
	if m.Role != "assistant" || len(m.ToolCalls) == 0 {
		return false
	}
	if m.Content == nil {
		return true
	}
	s, ok := m.Content.(string)
	if !ok {
		return false
	}
	return strings.TrimSpace(s) == ""
}

func responsesInputItemToMessage(item interface{}) (*Message, error) {
	obj, ok := item.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid input item")
	}

	itemType := strings.TrimSpace(toString(obj["type"]))
	toolCallID := strings.TrimSpace(toString(obj["tool_call_id"]))
	if toolCallID == "" {
		toolCallID = strings.TrimSpace(toString(obj["call_id"]))
	}
	if toolCallID == "" {
		toolCallID = strings.TrimSpace(toString(obj["id"]))
	}
	if itemType == "function_call_output" {
		if toolCallID == "" {
			return nil, fmt.Errorf("function_call_output.call_id is required")
		}
		return &Message{
			Role:       "tool",
			ToolCallID: toolCallID,
			Content:    stringifyAny(obj["output"]),
		}, nil
	}

	if itemType == "function_call" {
		name := strings.TrimSpace(toString(obj["name"]))
		if name == "" {
			return nil, nil
		}
		callID := strings.TrimSpace(toString(obj["call_id"]))
		if callID == "" {
			callID = strings.TrimSpace(toString(obj["id"]))
		}
		args := stringifyAny(obj["arguments"])
		idx := 0
		return &Message{
			Role:    "assistant",
			Content: "",
			ToolCalls: []ToolCall{{
				Index: &idx,
				ID:    callID,
				Type:  "function",
				Function: ToolCallFunction{
					Name:      name,
					Arguments: args,
				},
			}},
		}, nil
	}

	role := strings.TrimSpace(toString(obj["role"]))
	if role == "" {
		role = "user"
	}

	content := obj["content"]
	if content == nil {
		text := strings.TrimSpace(toString(obj["text"]))
		if text == "" {
			return nil, nil
		}
		return &Message{Role: role, Content: text, ToolCallID: toolCallID}, nil
	}

	switch c := content.(type) {
	case string:
		if strings.TrimSpace(c) == "" {
			return nil, nil
		}
		return &Message{Role: role, Content: c, ToolCallID: toolCallID}, nil
	case []interface{}:
		parts := make([]map[string]interface{}, 0, len(c))
		for _, part := range c {
			partObj, ok := part.(map[string]interface{})
			if !ok {
				continue
			}
			partType := strings.TrimSpace(toString(partObj["type"]))
			if partType == "" {
				partType = "input_text"
			}
			if partType == "input_text" || partType == "output_text" || partType == "text" {
				text := strings.TrimSpace(toString(partObj["text"]))
				if text == "" {
					continue
				}
				parts = append(parts, map[string]interface{}{"type": "text", "text": text})
				continue
			}
			parts = append(parts, partObj)
		}

		if len(parts) == 0 {
			return nil, nil
		}

		allText := true
		var b strings.Builder
		for _, p := range parts {
			t, ok := p["type"].(string)
			if !ok || t != "text" {
				allText = false
				break
			}
			pt, ok := p["text"].(string)
			if !ok {
				allText = false
				break
			}
			b.WriteString(pt)
		}
		if allText {
			return &Message{Role: role, Content: b.String(), ToolCallID: toolCallID}, nil
		}
		return &Message{Role: role, Content: parts, ToolCallID: toolCallID}, nil
	default:
		return nil, fmt.Errorf("unsupported input content format")
	}
}

func responsesToolsToUnified(tools []ResponsesTool) ([]Tool, error) {
	if len(tools) == 0 {
		return nil, nil
	}
	out := make([]Tool, 0, len(tools))
	for i, t := range tools {
		toolType := strings.TrimSpace(t.Type)
		if toolType == "" {
			toolType = "function"
		}
		if toolType != "function" {
			name := normalizeToolName(strings.TrimSpace(t.Name))
			if name == "" {
				name = normalizeToolName(toolType)
			}
			if name == "" {
				name = "tool_" + strconv.Itoa(i)
			}

			description := strings.TrimSpace(t.Description)
			if description == "" {
				description = "Compatibility shim for Responses tool type: " + toolType
			}

			parameters := t.Parameters
			if parameters == nil {
				parameters = map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{},
					"additionalProperties": true,
				}
			} else {
				parameters = ensureObjectSchemaHasProperties(parameters)
			}

			out = append(out, Tool{
				Type: "function",
				Function: ToolFunction{
					Name:        name,
					Description: description,
					Parameters:  parameters,
				},
			})
			continue
		}

		fn := ToolFunction{}
		if t.Function != nil {
			fn = *t.Function
		}
		if strings.TrimSpace(fn.Name) == "" {
			fn.Name = strings.TrimSpace(t.Name)
		}
		if strings.TrimSpace(fn.Description) == "" {
			fn.Description = strings.TrimSpace(t.Description)
		}
		if fn.Parameters == nil {
			fn.Parameters = t.Parameters
		}

		if strings.TrimSpace(fn.Name) == "" {
			return nil, fmt.Errorf("tools[%d].name is required", i)
		}

		out = append(out, Tool{Type: "function", Function: fn})
	}
	return out, nil
}

func normalizeResponsesToolChoice(toolChoice interface{}) (interface{}, error) {
	if toolChoice == nil {
		return nil, nil
	}

	switch v := toolChoice.(type) {
	case string:
		return v, nil
	case map[string]interface{}:
		choiceType, _ := v["type"].(string)
		if strings.TrimSpace(choiceType) != "function" {
			name, _ := v["name"].(string)
			normalizedName := normalizeToolName(name)
			if normalizedName == "" {
				normalizedName = normalizeToolName(choiceType)
			}
			if normalizedName == "" {
				return "auto", nil
			}
			return map[string]interface{}{
				"type": "function",
				"function": map[string]interface{}{
					"name": normalizedName,
				},
			}, nil
		}

		if fnObj, ok := v["function"].(map[string]interface{}); ok {
			if name, _ := fnObj["name"].(string); strings.TrimSpace(name) != "" {
				return toolChoice, nil
			}
			if topName, _ := v["name"].(string); strings.TrimSpace(topName) != "" {
				fnObj["name"] = strings.TrimSpace(topName)
				return map[string]interface{}{
					"type": "function",
					"function": fnObj,
				}, nil
			}
			return toolChoice, nil
		}

		if name, _ := v["name"].(string); strings.TrimSpace(name) != "" {
			return map[string]interface{}{
				"type": "function",
				"function": map[string]interface{}{
					"name": strings.TrimSpace(name),
				},
			}, nil
		}
		return nil, fmt.Errorf("tool_choice.function.name is required")
	default:
		b, err := json.Marshal(toolChoice)
		if err != nil {
			return nil, err
		}
		var obj map[string]interface{}
		if err := json.Unmarshal(b, &obj); err != nil {
			return nil, err
		}
		return normalizeResponsesToolChoice(obj)
	}
}

func toString(v interface{}) string {
	s, _ := v.(string)
	return s
}

func normalizeToolName(name string) string {
	trimmed := strings.TrimSpace(name)
	if trimmed == "" {
		return ""
	}
	replacer := strings.NewReplacer("-", "_", ":", "_", "/", "_", " ", "_")
	normalized := replacer.Replace(trimmed)
	normalized = strings.Trim(normalized, "_")
	return normalized
}

func stringifyAny(v interface{}) string {
	if v == nil {
		return ""
	}
	if s, ok := v.(string); ok {
		return s
	}
	b, err := json.Marshal(v)
	if err != nil {
		return fmt.Sprintf("%v", v)
	}
	return string(b)
}

func ensureObjectSchemaHasProperties(schema interface{}) interface{} {
	obj, ok := schema.(map[string]interface{})
	if !ok {
		return schema
	}

	typeName, _ := obj["type"].(string)
	if strings.TrimSpace(typeName) != "object" {
		return schema
	}

	if _, ok := obj["properties"]; !ok {
		obj["properties"] = map[string]interface{}{}
		return obj
	}
	if obj["properties"] == nil {
		obj["properties"] = map[string]interface{}{}
	}
	return obj
}
