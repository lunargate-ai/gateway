package api

import (
	"encoding/json"
	"strconv"
	"strings"
	"time"

	"github.com/lunargate-ai/gateway/pkg/models"
)

const (
	maxObservedChatStreamBodyBytes = 16 << 20
	maxObservedChatStreamToolCalls = 128

	chatStreamBodyLimitReason = "observed_body_limit_exceeded"
	chatStreamToolLimitReason = "observed_tool_call_limit_exceeded"
)

// chatStreamObservation retains the optional response snapshot sent to the
// collector. Stream delivery, token accounting, and timing are intentionally
// handled outside this type so reaching an observation limit can never affect
// the client-facing stream.
type chatStreamObservation struct {
	shareResponses bool
	maxBodyBytes   int
	maxToolCalls   int

	capturedBytes    int
	truncated        bool
	truncationReason string

	text          strings.Builder
	reasoning     strings.Builder
	toolCallByKey map[string]*models.ToolCall
	toolCallOrder []string
	finishReason  *string
}

type chatStreamObservationLimits struct {
	MaxBytes     int `json:"max_bytes"`
	MaxToolCalls int `json:"max_tool_calls"`
}

type omittedChatStreamResponse struct {
	ID               string                      `json:"id"`
	Object           string                      `json:"object"`
	Model            string                      `json:"model"`
	ResponseOmitted  bool                        `json:"response_omitted"`
	Truncated        bool                        `json:"truncated"`
	TruncationReason string                      `json:"truncation_reason"`
	ObservationLimit chatStreamObservationLimits `json:"observation_limit"`
	Usage            *models.Usage               `json:"usage,omitempty"`
}

func newChatStreamObservation(shareResponses bool) *chatStreamObservation {
	return &chatStreamObservation{
		shareResponses: shareResponses,
		maxBodyBytes:   maxObservedChatStreamBodyBytes,
		maxToolCalls:   maxObservedChatStreamToolCalls,
	}
}

// disable permanently stops capture for this request and releases anything
// already retained. It is used if response sharing is disabled while a stream
// is in flight; capture never starts midway through a response.
func (o *chatStreamObservation) disable() {
	if o == nil || !o.shareResponses {
		return
	}
	o.shareResponses = false
	o.truncated = false
	o.truncationReason = ""
	o.clearCapturedResponse()
}

func (o *chatStreamObservation) isShared() bool {
	return o != nil && o.shareResponses
}

// observe returns whether the chunk carries response content so callers can
// maintain TTFT/TTLT even when response sharing is disabled or capture was
// truncated.
func (o *chatStreamObservation) observe(chunk *models.StreamChunk) bool {
	if chunk == nil {
		return false
	}

	hasContent := false
	for _, choice := range chunk.Choices {
		if o != nil && o.canCapture() && choice.FinishReason != nil {
			o.captureFinishReason(*choice.FinishReason)
		}
		if choice.Delta == nil {
			continue
		}

		if len(choice.Delta.ToolCalls) > 0 {
			hasContent = true
			if o != nil && o.canCapture() {
				for _, toolCall := range choice.Delta.ToolCalls {
					o.captureToolCall(toolCall)
					if !o.canCapture() {
						break
					}
				}
			}
		}

		if content, ok := choice.Delta.Content.(string); ok && content != "" {
			hasContent = true
			if o != nil && o.canCapture() {
				o.captureString(&o.text, content)
			}
		}
		if choice.Delta.ReasoningContent != "" {
			hasContent = true
			if o != nil && o.canCapture() {
				o.captureString(&o.reasoning, choice.Delta.ReasoningContent)
			}
		}
	}

	return hasContent
}

func (o *chatStreamObservation) canCapture() bool {
	return o != nil && o.shareResponses && !o.truncated
}

func (o *chatStreamObservation) captureString(builder *strings.Builder, value string) {
	if value == "" || !o.reserveBytes(len(value)) {
		return
	}
	_, _ = builder.WriteString(value)
}

func (o *chatStreamObservation) captureFinishReason(value string) {
	previousBytes := 0
	if o.finishReason != nil {
		previousBytes = len(*o.finishReason)
	}
	additionalBytes := len(value) - previousBytes
	if additionalBytes > 0 && !o.reserveBytes(additionalBytes) {
		return
	}
	if additionalBytes < 0 {
		o.capturedBytes += additionalBytes
	}
	copyValue := value
	o.finishReason = &copyValue
}

func (o *chatStreamObservation) captureToolCall(toolCall models.ToolCall) {
	key := chatStreamToolCallKey(toolCall)
	if key == "" {
		return
	}
	if o.toolCallByKey == nil {
		o.toolCallByKey = make(map[string]*models.ToolCall, 8)
	}

	existing := o.toolCallByKey[key]
	if existing == nil {
		if len(o.toolCallOrder) >= o.maxToolCalls {
			o.truncate(chatStreamToolLimitReason)
			return
		}
		additionalBytes := len(key) + len(toolCall.ID) + len(toolCall.Type) +
			len(toolCall.Function.Name) + len(toolCall.Function.Arguments)
		if !o.reserveBytes(additionalBytes) {
			return
		}
		copyToolCall := models.ToolCall{
			Index: copyChatStreamToolCallIndex(toolCall.Index),
			ID:    toolCall.ID,
			Type:  toolCall.Type,
			Function: models.ToolCallFunction{
				Name:      toolCall.Function.Name,
				Arguments: toolCall.Function.Arguments,
			},
		}
		o.toolCallByKey[key] = &copyToolCall
		o.toolCallOrder = append(o.toolCallOrder, key)
		return
	}

	additionalBytes := len(toolCall.Function.Arguments)
	if existing.ID == "" {
		additionalBytes += len(toolCall.ID)
	}
	if existing.Type == "" {
		additionalBytes += len(toolCall.Type)
	}
	if existing.Function.Name == "" {
		additionalBytes += len(toolCall.Function.Name)
	}
	if !o.reserveBytes(additionalBytes) {
		return
	}
	if existing.ID == "" {
		existing.ID = toolCall.ID
	}
	if existing.Type == "" {
		existing.Type = toolCall.Type
	}
	if existing.Index == nil {
		existing.Index = copyChatStreamToolCallIndex(toolCall.Index)
	}
	if existing.Function.Name == "" {
		existing.Function.Name = toolCall.Function.Name
	}
	if toolCall.Function.Arguments != "" {
		existing.Function.Arguments += toolCall.Function.Arguments
	}
}

func chatStreamToolCallKey(toolCall models.ToolCall) string {
	if toolCall.Index != nil {
		return "idx:" + strconv.Itoa(*toolCall.Index)
	}
	if toolCall.ID != "" {
		return toolCall.ID
	}
	return toolCall.Function.Name
}

func copyChatStreamToolCallIndex(index *int) *int {
	if index == nil {
		return nil
	}
	copyValue := *index
	return &copyValue
}

func (o *chatStreamObservation) reserveBytes(additionalBytes int) bool {
	if !o.canCapture() || additionalBytes < 0 {
		return false
	}
	if additionalBytes > o.maxBodyBytes-o.capturedBytes {
		o.truncate(chatStreamBodyLimitReason)
		return false
	}
	o.capturedBytes += additionalBytes
	return true
}

func (o *chatStreamObservation) truncate(reason string) {
	if o == nil || o.truncated {
		return
	}
	o.truncated = true
	o.truncationReason = reason
	o.clearCapturedResponse()
}

func (o *chatStreamObservation) clearCapturedResponse() {
	if o == nil {
		return
	}
	o.capturedBytes = 0
	o.text.Reset()
	o.reasoning.Reset()
	o.toolCallByKey = nil
	o.toolCallOrder = nil
	o.finishReason = nil
}

func (o *chatStreamObservation) collectorResponse(requestID, model string, tokenUsage models.TokenUsage) interface{} {
	if !o.isShared() {
		return nil
	}
	usage := observedChatStreamUsage(tokenUsage)
	if o.truncated {
		return o.omittedResponse(requestID, model, usage)
	}

	toolCalls := make([]models.ToolCall, 0, len(o.toolCallOrder))
	for _, key := range o.toolCallOrder {
		if toolCall := o.toolCallByKey[key]; toolCall != nil {
			toolCalls = append(toolCalls, *toolCall)
		}
	}
	if len(toolCalls) == 0 {
		toolCalls = nil
	}

	response := models.UnifiedResponse{
		ID:      requestID,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []models.Choice{{
			Index: 0,
			Message: &models.Message{
				Role:             "assistant",
				Content:          o.text.String(),
				ReasoningContent: o.reasoning.String(),
				ToolCalls:        toolCalls,
			},
			FinishReason: o.finishReason,
		}},
		Usage: usage,
	}

	// The incremental bound controls retained memory. Validate the serialized
	// response too, because JSON escaping and envelope fields can make the
	// collector body larger than the strings from which it was assembled.
	encoded, err := json.Marshal(response)
	if err != nil || len(encoded) > o.maxBodyBytes {
		o.truncate(chatStreamBodyLimitReason)
		return o.omittedResponse(requestID, model, usage)
	}
	return response
}

func (o *chatStreamObservation) omittedResponse(requestID, model string, usage *models.Usage) omittedChatStreamResponse {
	return omittedChatStreamResponse{
		ID:               requestID,
		Object:           "lunargate.chat.completion.observation",
		Model:            model,
		ResponseOmitted:  true,
		Truncated:        true,
		TruncationReason: o.truncationReason,
		ObservationLimit: chatStreamObservationLimits{
			MaxBytes:     o.maxBodyBytes,
			MaxToolCalls: o.maxToolCalls,
		},
		Usage: usage,
	}
}

func observedChatStreamUsage(tokenUsage models.TokenUsage) *models.Usage {
	tokenUsage = tokenUsage.Normalized()
	if tokenUsage.InputTokens == 0 && tokenUsage.OutputTokens == 0 {
		return nil
	}
	return &models.Usage{
		PromptTokens:            tokenUsage.InputTokens,
		CompletionTokens:        tokenUsage.OutputTokens,
		TotalTokens:             models.SaturatingTokenSum(tokenUsage.InputTokens, tokenUsage.OutputTokens),
		PromptTokensDetails:     inputTokenDetailsFromTokenUsage(tokenUsage),
		CompletionTokensDetails: completionTokenDetailsFromTokenUsage(tokenUsage),
	}
}
