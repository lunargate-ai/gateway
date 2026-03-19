package models

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strconv"
	"strings"
)

// UnifiedRequest represents an OpenAI-compatible chat completion request.
// All provider translators convert FROM this format to their native format.
type UnifiedRequest struct {
	Model            string          `json:"model"`
	Messages         []Message       `json:"messages"`
	Temperature      *float64        `json:"temperature,omitempty"`
	TopP             *float64        `json:"top_p,omitempty"`
	N                *int            `json:"n,omitempty"`
	Stream           bool            `json:"stream,omitempty"`
	StreamOptions    *StreamOptions  `json:"stream_options,omitempty"`
	Stop             interface{}     `json:"stop,omitempty"`
	MaxTokens        *int            `json:"max_tokens,omitempty"`
	PresencePenalty  *float64        `json:"presence_penalty,omitempty"`
	FrequencyPenalty *float64        `json:"frequency_penalty,omitempty"`
	LogitBias        map[string]int  `json:"logit_bias,omitempty"`
	User             string          `json:"user,omitempty"`
	Tools            []Tool          `json:"tools,omitempty"`
	ToolChoice       interface{}     `json:"tool_choice,omitempty"`
	Functions        []ToolFunction  `json:"functions,omitempty"`
	FunctionCall     interface{}     `json:"function_call,omitempty"`
	ResponseFormat   *ResponseFormat `json:"response_format,omitempty"`
	Seed             *int            `json:"seed,omitempty"`
}

type StreamOptions struct {
	IncludeUsage bool `json:"include_usage,omitempty"`
}

type Message struct {
	Role       string      `json:"role,omitempty"`
	Content    interface{} `json:"content,omitempty"`
	ReasoningContent string `json:"reasoning_content,omitempty"`
	Name       string      `json:"name,omitempty"`
	ToolCalls  []ToolCall  `json:"tool_calls,omitempty"`
	ToolCallID string      `json:"tool_call_id,omitempty"`
	FunctionCall *FunctionCall `json:"function_call,omitempty"`
}

// ContentString returns the message content as a plain string.
// Handles both string content and array content (multimodal).
func (m *Message) ContentString() string {
	switch v := m.Content.(type) {
	case string:
		return v
	case nil:
		return ""
	default:
		return ""
	}
}

type Tool struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

type ToolFunction struct {
	Name        string      `json:"name"`
	Description string      `json:"description,omitempty"`
	Parameters  interface{} `json:"parameters,omitempty"`
}

type ToolCall struct {
	Index    *int             `json:"index,omitempty"`
	ID       string           `json:"id,omitempty"`
	Type     string           `json:"type,omitempty"`
	Function ToolCallFunction `json:"function"`
}

type ToolCallFunction struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

type FunctionCall struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

type ResponseFormat struct {
	Type string `json:"type"`
}

func NormalizeUnifiedRequest(req *UnifiedRequest) error {
	if req == nil {
		return nil
	}

	// Reject conflicting modes.
	if len(req.Tools) > 0 && len(req.Functions) > 0 {
		return &jsonUnsupportedCombinationError{message: "cannot specify both tools and functions"}
	}
	if req.ToolChoice != nil && req.FunctionCall != nil {
		return &jsonUnsupportedCombinationError{message: "cannot specify both tool_choice and function_call"}
	}

	if len(req.Tools) == 0 && len(req.Functions) > 0 {
		req.Tools = make([]Tool, 0, len(req.Functions))
		for _, fn := range req.Functions {
			req.Tools = append(req.Tools, Tool{Type: "function", Function: fn})
		}
		req.Functions = nil
	}

	if req.ToolChoice == nil && req.FunctionCall != nil {
		normalized, err := normalizeFunctionCallToToolChoice(req.FunctionCall)
		if err != nil {
			return err
		}
		req.ToolChoice = normalized
		req.FunctionCall = nil
	}

	for i := range req.Tools {
		if req.Tools[i].Type == "" {
			req.Tools[i].Type = "function"
		}
	}

	nameToToolCallID := make(map[string]string, 8)

	for mi := range req.Messages {
		m := &req.Messages[mi]
		normalizeMessageContentParts(m)

		if m.Role == "assistant" && m.FunctionCall != nil && len(m.ToolCalls) == 0 {
			id := stableToolCallID(m.FunctionCall.Name, m.FunctionCall.Arguments, mi)
			idx := 0
			m.ToolCalls = []ToolCall{{
				Index: &idx,
				ID:    id,
				Type:  "function",
				Function: ToolCallFunction{
					Name:      m.FunctionCall.Name,
					Arguments: m.FunctionCall.Arguments,
				},
			}}
			nameToToolCallID[m.FunctionCall.Name] = id
			m.FunctionCall = nil
		}

		if m.Role == "function" {
			m.Role = "tool"
		}

		for ti := range m.ToolCalls {
			tc := &m.ToolCalls[ti]
			if tc.Type == "" {
				tc.Type = "function"
			}
			if tc.Index == nil {
				idx := ti
				tc.Index = &idx
			}
			if tc.ID == "" {
				idx := ti
				tc.ID = stableToolCallID(tc.Function.Name, tc.Function.Arguments, idx)
			}
			if tc.Function.Name == "" && m.Name != "" {
				tc.Function.Name = m.Name
			}
			if tc.Function.Name != "" {
				nameToToolCallID[tc.Function.Name] = tc.ID
			}
		}

		if m.Role == "tool" {
			if m.ToolCallID == "" {
				if m.Name != "" {
					if id, ok := nameToToolCallID[m.Name]; ok {
						m.ToolCallID = id
					} else {
						m.ToolCallID = m.Name
					}
				}
			}
			if m.ToolCallID == "" && len(m.ToolCalls) == 1 {
				m.ToolCallID = m.ToolCalls[0].ID
			}
		}
	}

	return nil
}

func normalizeMessageContentParts(m *Message) {
	if m == nil || m.Content == nil {
		return
	}

	parts, ok := m.Content.([]interface{})
	if !ok {
		return
	}

	textChunks := make([]string, 0, len(parts))
	allText := true

	for i := range parts {
		obj, ok := parts[i].(map[string]interface{})
		if !ok {
			allText = false
			continue
		}
		pt, _ := obj["type"].(string)
		switch pt {
		case "input_text", "output_text":
			obj["type"] = "text"
			pt = "text"
		}

		if pt != "text" {
			allText = false
			continue
		}
		text, ok := obj["text"].(string)
		if !ok {
			allText = false
			continue
		}
		textChunks = append(textChunks, text)
	}

	if allText && len(textChunks) > 0 {
		m.Content = strings.Join(textChunks, "")
	}
}

func normalizeFunctionCallToToolChoice(functionCall interface{}) (interface{}, error) {
	switch v := functionCall.(type) {
	case string:
		if v == "auto" || v == "none" {
			return v, nil
		}
		return v, nil
	default:
		b, err := json.Marshal(functionCall)
		if err != nil {
			return nil, err
		}
		var obj struct {
			Name string `json:"name"`
		}
		if err := json.Unmarshal(b, &obj); err != nil {
			return nil, err
		}
		if obj.Name == "" {
			return nil, nil
		}
		return map[string]interface{}{
			"type": "function",
			"function": map[string]interface{}{
				"name": obj.Name,
			},
		}, nil
	}
}

type jsonUnsupportedCombinationError struct {
	message string
}

func (e *jsonUnsupportedCombinationError) Error() string {
	return e.message
}

func stableToolCallID(name, arguments string, idx int) string {
	h := sha256.Sum256([]byte(name + ":" + strconv.Itoa(idx) + ":" + arguments))
	return "call_" + hex.EncodeToString(h[:8])
}
