package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strings"

	"github.com/lunargate-ai/gateway/internal/streaming"
	"github.com/lunargate-ai/gateway/pkg/models"
	"github.com/rs/zerolog/log"
)

const (
	responsesFallbackResponseID = "resp_lunargate"
	responsesFallbackModel      = "unknown"
	responsesEventIDPrefix      = "evt_lg_"
)

type responsesStreamProxy struct {
	target      http.ResponseWriter
	headers     http.Header
	statusCode  int
	headersSent bool
	native      bool
	buffer      bytes.Buffer

	responseID       string
	itemID           string
	reasoningItemID  string
	model            string
	created          int64
	usage            *models.Usage
	text             strings.Builder
	reasoningText    strings.Builder
	started          bool
	messageStarted   bool
	reasoningStarted bool
	completed        bool
	streamErr        error
	incompleteReason string
	eventSeq         int

	nextOutputIndex      int
	reasoningOutputIndex int
	toolCalls            map[string]*responsesToolCallState
	toolCallOrder        []string
	completedResponse    map[string]interface{}
	terminalResponse     map[string]interface{}
	beforeTerminal       func(map[string]interface{})
}

type responsesToolCallState struct {
	ItemID      string
	CallID      string
	Name        string
	Arguments   string
	OutputIndex int
	Added       bool
	Done        bool
}

type nativeResponsesStreamTerminal struct {
	eventType    string
	status       string
	responseID   string
	model        string
	createdAt    int64
	tokensInput  int
	tokensOutput int
	tokensTotal  int
	response     map[string]interface{}
}

type nativeResponsesStreamSink interface {
	enableNativePassthrough()
	recordNativeTerminal(*nativeResponsesStreamTerminal)
}

func newResponsesStreamProxy(target http.ResponseWriter) *responsesStreamProxy {
	return &responsesStreamProxy{
		target:               target,
		headers:              make(http.Header),
		statusCode:           http.StatusOK,
		nextOutputIndex:      1,
		reasoningOutputIndex: -1,
		toolCalls:            make(map[string]*responsesToolCallState),
		toolCallOrder:        make([]string, 0, 4),
	}
}

func (p *responsesStreamProxy) Header() http.Header {
	return p.headers
}

func (p *responsesStreamProxy) WriteHeader(statusCode int) {
	p.statusCode = statusCode
}

func (p *responsesStreamProxy) RecordStreamError(err error) {
	p.streamErr = err
}

func (p *responsesStreamProxy) enableNativePassthrough() {
	p.native = true
}

func (p *responsesStreamProxy) recordNativeTerminal(terminal *nativeResponsesStreamTerminal) {
	if terminal == nil {
		return
	}
	p.completed = true
	p.responseID = terminal.responseID
	p.model = terminal.model
	p.created = terminal.createdAt
	if terminal.tokensInput != 0 || terminal.tokensOutput != 0 || terminal.tokensTotal != 0 {
		p.usage = &models.Usage{
			PromptTokens:     terminal.tokensInput,
			CompletionTokens: terminal.tokensOutput,
			TotalTokens:      terminal.tokensTotal,
		}
	}
	if p.beforeTerminal != nil {
		p.beforeTerminal(terminal.response)
	}
	p.terminalResponse = terminal.response
	p.completedResponse = nil
	if terminal.eventType == "response.completed" && terminal.status == "completed" {
		p.completedResponse = terminal.response
	}
}

func (p *responsesStreamProxy) Flush() {
	_ = p.FlushError()
}

func (p *responsesStreamProxy) FlushError() error {
	if p.native && !p.headersSent {
		p.sendHeadersIfNeeded()
	}
	if !p.headersSent {
		return nil
	}
	return http.NewResponseController(p.target).Flush()
}

func (p *responsesStreamProxy) Write(b []byte) (int, error) {
	if p.native {
		p.sendHeadersIfNeeded()
		return p.target.Write(b)
	}
	if p.statusCode >= 400 {
		p.sendHeadersIfNeeded()
		return p.target.Write(b)
	}

	_, _ = p.buffer.Write(b)
	for {
		all := p.buffer.String()
		idx := strings.Index(all, "\n\n")
		if idx < 0 {
			break
		}
		frame := all[:idx]
		remaining := all[idx+2:]
		p.buffer.Reset()
		_, _ = p.buffer.WriteString(remaining)
		if err := p.processFrame(frame); err != nil {
			return len(b), err
		}
	}
	return len(b), nil
}

func (p *responsesStreamProxy) finalize() error {
	if p.native {
		if p.streamErr != nil {
			return p.streamErr
		}
		if !p.completed {
			return fmt.Errorf("%w: native responses sse", streaming.ErrUpstreamStreamIncomplete)
		}
		return nil
	}
	if p.statusCode >= 400 {
		if p.buffer.Len() > 0 {
			p.sendHeadersIfNeeded()
			if _, err := p.target.Write(p.buffer.Bytes()); err != nil {
				return err
			}
			p.buffer.Reset()
		}
		return nil
	}

	if p.buffer.Len() > 0 {
		if err := p.processFrame(p.buffer.String()); err != nil {
			return err
		}
		p.buffer.Reset()
	}
	if p.streamErr != nil && !p.completed {
		return p.emitFailed(p.streamErr)
	}

	if !p.completed {
		return p.emitFailed(streaming.ErrUpstreamStreamIncomplete)
	}
	return nil
}

func parseNativeResponsesStreamTerminal(event streaming.SSEEvent) (*nativeResponsesStreamTerminal, bool) {
	if len(bytes.TrimSpace(event.Data)) == 0 {
		return nil, false
	}
	var envelope struct {
		Type     string          `json:"type"`
		Response json.RawMessage `json:"response"`
	}
	if err := json.Unmarshal(event.Data, &envelope); err != nil {
		return nil, false
	}
	eventType := strings.ToLower(strings.TrimSpace(envelope.Type))
	if eventType == "" {
		eventType = strings.ToLower(strings.TrimSpace(event.Event))
	}
	switch eventType {
	case "response.completed", "response.incomplete", "response.failed", "response.cancelled", "response.canceled":
	default:
		return nil, false
	}

	decoder := json.NewDecoder(bytes.NewReader(envelope.Response))
	decoder.UseNumber()
	var response map[string]interface{}
	if err := decoder.Decode(&response); err != nil || response == nil {
		return nil, false
	}
	status, _ := response["status"].(string)
	status = strings.ToLower(strings.TrimSpace(status))
	eventStatus := strings.TrimPrefix(eventType, "response.")
	if eventStatus == "canceled" {
		eventStatus = "cancelled"
	}
	if status == "" {
		status = eventStatus
	} else if eventStatus != "completed" && status == "completed" {
		// The terminal event type is authoritative. A failed/incomplete/cancelled
		// event must never be promoted to a completed local response by a
		// contradictory payload field.
		status = eventStatus
	}
	switch status {
	case "completed", "incomplete", "failed", "cancelled":
	case "canceled":
		status = "cancelled"
	default:
		status = "unknown"
	}
	terminal := &nativeResponsesStreamTerminal{
		eventType:  eventType,
		status:     status,
		responseID: nativeResponsesString(response["id"]),
		model:      nativeResponsesString(response["model"]),
		createdAt:  nativeResponsesInteger(response["created_at"]),
		response:   response,
	}
	if usage, ok := response["usage"].(map[string]interface{}); ok {
		terminal.tokensInput = int(nativeResponsesInteger(usage["input_tokens"]))
		terminal.tokensOutput = int(nativeResponsesInteger(usage["output_tokens"]))
		terminal.tokensTotal = int(nativeResponsesInteger(usage["total_tokens"]))
	}
	return terminal, true
}

func nativeResponsesString(value interface{}) string {
	text, _ := value.(string)
	return strings.TrimSpace(text)
}

func nativeResponsesInteger(value interface{}) int64 {
	switch typed := value.(type) {
	case json.Number:
		parsed, _ := typed.Int64()
		return parsed
	case float64:
		return int64(typed)
	case int64:
		return typed
	case int:
		return int64(typed)
	default:
		return 0
	}
}

func (p *responsesStreamProxy) emitFailed(streamErr error) error {
	if p.completed {
		return nil
	}
	if err := p.ensureStarted(); err != nil {
		return err
	}
	p.completed = true

	message := "upstream stream failed before completion"
	if streamErr != nil {
		message = streamErr.Error()
	}
	response := map[string]interface{}{
		"id":         p.responseID,
		"object":     "response",
		"created_at": p.created,
		"status":     "failed",
		"model":      p.model,
		"output":     []interface{}{},
		"error": map[string]interface{}{
			"code":    "upstream_stream_error",
			"message": message,
		},
	}
	return p.writeEvent(map[string]interface{}{
		"type":     "response.failed",
		"response": response,
	})
}

func (p *responsesStreamProxy) processToolCallDelta(tc models.ToolCall) error {
	key := ""
	if tc.Index != nil {
		key = fmt.Sprintf("idx_%d", *tc.Index)
	} else if strings.TrimSpace(tc.ID) != "" {
		key = strings.TrimSpace(tc.ID)
	} else if strings.TrimSpace(tc.Function.Name) != "" {
		key = strings.TrimSpace(tc.Function.Name)
	}
	if key == "" {
		key = fmt.Sprintf("anon_%d", len(p.toolCallOrder))
	}

	st, ok := p.toolCalls[key]
	if !ok {
		rawID := strings.TrimSpace(tc.ID)
		itemID := responsesCanonicalFunctionItemID(rawID)
		callID := responsesCanonicalFunctionCallID(rawID)
		if itemID == "" {
			itemID = fmt.Sprintf("fc_%s_%d", p.responseID, p.nextOutputIndex)
		}
		if callID == "" {
			callID = responsesCanonicalFunctionCallID(itemID)
		}
		name := strings.TrimSpace(tc.Function.Name)
		if name == "" {
			name = fmt.Sprintf("tool_call_%d", p.nextOutputIndex)
		}
		st = &responsesToolCallState{
			ItemID:      itemID,
			CallID:      callID,
			Name:        name,
			OutputIndex: p.nextOutputIndex,
		}
		p.nextOutputIndex++
		p.toolCalls[key] = st
		p.toolCallOrder = append(p.toolCallOrder, key)
		log.Debug().
			Str("response_id", p.responseID).
			Str("tool_key", key).
			Str("item_id", st.ItemID).
			Str("call_id", st.CallID).
			Str("tool_name", st.Name).
			Int("output_index", st.OutputIndex).
			Msg("responses stream proxy started tool call")
	}

	if name := strings.TrimSpace(tc.Function.Name); name != "" {
		st.Name = name
	}
	if id := strings.TrimSpace(tc.ID); id != "" {
		if !st.Added {
			if canonicalItemID := responsesCanonicalFunctionItemID(id); canonicalItemID != "" {
				st.ItemID = canonicalItemID
			}
		}
		if canonicalCallID := responsesCanonicalFunctionCallID(id); canonicalCallID != "" && st.CallID == "" {
			st.CallID = canonicalCallID
		}
	}
	if st.ItemID == "" {
		st.ItemID = responsesCanonicalFunctionItemID(st.CallID)
	}
	if st.CallID == "" {
		st.CallID = responsesCanonicalFunctionCallID(st.ItemID)
	}

	if !st.Added {
		if err := p.writeEvent(map[string]interface{}{
			"type":         "response.output_item.added",
			"response_id":  p.responseID,
			"output_index": st.OutputIndex,
			"item": map[string]interface{}{
				"id":        st.ItemID,
				"type":      "function_call",
				"status":    "in_progress",
				"call_id":   st.CallID,
				"name":      st.Name,
				"arguments": "",
			},
		}); err != nil {
			return err
		}
		st.Added = true
	}

	if delta := tc.Function.Arguments; delta != "" {
		st.Arguments += delta
		log.Debug().
			Str("response_id", p.responseID).
			Str("call_id", st.CallID).
			Int("delta_len", len(delta)).
			Int("arguments_len", len(st.Arguments)).
			Msg("responses stream proxy accumulated tool arguments")
		if err := p.writeEvent(map[string]interface{}{
			"type":         "response.function_call_arguments.delta",
			"response_id":  p.responseID,
			"item_id":      st.ItemID,
			"output_index": st.OutputIndex,
			"delta":        delta,
		}); err != nil {
			return err
		}
	}

	return nil
}

func (p *responsesStreamProxy) processFrame(frame string) error {
	lines := strings.Split(frame, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "" {
			continue
		}
		if payload == "[DONE]" {
			if !p.completed {
				return p.emitTerminal()
			}
			return nil
		}

		var chunk models.StreamChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			continue
		}
		if p.responseID == "" {
			p.responseID = strings.TrimSpace(chunk.ID)
		}
		if p.model == "" {
			p.model = strings.TrimSpace(chunk.Model)
		}
		if p.created == 0 {
			p.created = chunk.Created
		}
		if chunk.Usage != nil {
			p.mergeUsage(chunk.Usage)
		}
		if err := p.ensureStarted(); err != nil {
			return err
		}

		for _, c := range chunk.Choices {
			p.recordFinishReason(c.FinishReason)
			if c.Delta == nil {
				continue
			}
			if reasoningDelta := c.Delta.ReasoningContent; reasoningDelta != "" {
				emittedReasoningDelta := p.mergeReasoningDelta(reasoningDelta)
				if emittedReasoningDelta != "" {
					if err := p.ensureReasoningStarted(); err != nil {
						return err
					}
					if err := p.writeEvent(map[string]interface{}{
						"type":          "response.reasoning_summary_text.delta",
						"response_id":   p.responseID,
						"item_id":       p.reasoningItemID,
						"output_index":  p.reasoningOutputIndex,
						"summary_index": 0,
						"delta":         emittedReasoningDelta,
					}); err != nil {
						return err
					}
				}
			}
			if len(c.Delta.ToolCalls) > 0 {
				for _, tc := range c.Delta.ToolCalls {
					if err := p.processToolCallDelta(tc); err != nil {
						return err
					}
				}
			}
			delta, ok := c.Delta.Content.(string)
			if !ok || delta == "" {
				continue
			}
			emittedDelta := p.mergeTextDelta(delta)
			if emittedDelta == "" {
				continue
			}
			if err := p.ensureMessageStarted(); err != nil {
				return err
			}
			event := map[string]interface{}{
				"type":          "response.output_text.delta",
				"response_id":   p.responseID,
				"item_id":       p.itemID,
				"output_index":  0,
				"content_index": 0,
				"delta":         emittedDelta,
			}
			if err := p.writeEvent(event); err != nil {
				return err
			}
		}
	}
	return nil
}

func (p *responsesStreamProxy) recordFinishReason(finishReason *string) {
	if finishReason == nil {
		return
	}
	mapped := models.ResponsesIncompleteReasonForFinishReason(*finishReason)
	if mapped == "" {
		return
	}
	if p.incompleteReason == "" || mapped == models.ResponsesIncompleteReasonContentFilter {
		p.incompleteReason = mapped
	}
}

func (p *responsesStreamProxy) mergeUsage(update *models.Usage) {
	if update == nil {
		return
	}
	if p.usage == nil {
		p.usage = &models.Usage{}
	}
	if update.PromptTokens > p.usage.PromptTokens {
		p.usage.PromptTokens = update.PromptTokens
	}
	if update.CompletionTokens > p.usage.CompletionTokens {
		p.usage.CompletionTokens = update.CompletionTokens
	}
	if update.TotalTokens > p.usage.TotalTokens {
		p.usage.TotalTokens = update.TotalTokens
	}
	if componentTotal := p.usage.PromptTokens + p.usage.CompletionTokens; componentTotal > p.usage.TotalTokens {
		p.usage.TotalTokens = componentTotal
	}
}

func (p *responsesStreamProxy) mergeTextDelta(delta string) string {
	if delta == "" {
		return ""
	}
	p.text.WriteString(delta)
	return delta
}

func (p *responsesStreamProxy) mergeReasoningDelta(delta string) string {
	if delta == "" {
		return ""
	}
	p.reasoningText.WriteString(delta)
	return delta
}

func (p *responsesStreamProxy) ensureStarted() error {
	if p.started {
		return nil
	}
	p.started = true

	if p.responseID == "" {
		// Compatibility fallback for chunks that don't expose an ID.
		p.responseID = responsesFallbackResponseID
	}
	if p.itemID == "" {
		p.itemID = "msg_" + p.responseID + "_0"
	}
	if p.model == "" {
		// Compatibility fallback for providers omitting model in stream chunks.
		p.model = responsesFallbackModel
	}

	if err := p.writeEvent(map[string]interface{}{
		"type": "response.created",
		"response": map[string]interface{}{
			"id":         p.responseID,
			"object":     "response",
			"created_at": p.created,
			"status":     "in_progress",
			"model":      p.model,
			"output":     []interface{}{},
		},
	}); err != nil {
		return err
	}

	return nil
}

func (p *responsesStreamProxy) ensureReasoningStarted() error {
	if p.reasoningStarted {
		return nil
	}
	p.reasoningStarted = true

	if p.reasoningOutputIndex < 0 {
		p.reasoningOutputIndex = p.nextOutputIndex
		p.nextOutputIndex++
	}
	if p.reasoningItemID == "" {
		p.reasoningItemID = fmt.Sprintf("rs_%s_%d", p.responseID, p.reasoningOutputIndex)
	}

	if err := p.writeEvent(map[string]interface{}{
		"type":         "response.output_item.added",
		"response_id":  p.responseID,
		"output_index": p.reasoningOutputIndex,
		"item": map[string]interface{}{
			"id":      p.reasoningItemID,
			"type":    "reasoning",
			"status":  "in_progress",
			"summary": []interface{}{},
		},
	}); err != nil {
		return err
	}

	if err := p.writeEvent(map[string]interface{}{
		"type":          "response.reasoning_summary_part.added",
		"response_id":   p.responseID,
		"item_id":       p.reasoningItemID,
		"output_index":  p.reasoningOutputIndex,
		"summary_index": 0,
		"part": map[string]interface{}{
			"type": "summary_text",
			"text": "",
		},
	}); err != nil {
		return err
	}

	return nil
}

func (p *responsesStreamProxy) ensureMessageStarted() error {
	if p.messageStarted {
		return nil
	}
	p.messageStarted = true

	if err := p.writeEvent(map[string]interface{}{
		"type":         "response.output_item.added",
		"response_id":  p.responseID,
		"output_index": 0,
		"item": map[string]interface{}{
			"id":      p.itemID,
			"type":    "message",
			"role":    "assistant",
			"status":  "in_progress",
			"content": []interface{}{},
		},
	}); err != nil {
		return err
	}

	if err := p.writeEvent(map[string]interface{}{
		"type":          "response.content_part.added",
		"response_id":   p.responseID,
		"item_id":       p.itemID,
		"output_index":  0,
		"content_index": 0,
		"part": map[string]interface{}{
			"type": "output_text",
			"text": "",
		},
	}); err != nil {
		return err
	}
	return nil
}

func (p *responsesStreamProxy) emitCompleted() error {
	return p.emitFinal("completed", "response.completed", "")
}

func (p *responsesStreamProxy) emitTerminal() error {
	if p.incompleteReason != "" {
		return p.emitFinal("incomplete", "response.incomplete", p.incompleteReason)
	}
	return p.emitCompleted()
}

func (p *responsesStreamProxy) emitFinal(status, eventType, incompleteReason string) error {
	if p.completed {
		return nil
	}
	p.completed = true
	if err := p.ensureStarted(); err != nil {
		return err
	}

	text := p.text.String()
	reasoning := p.reasoningText.String()

	type indexedOutputItem struct {
		outputIndex int
		item        interface{}
	}
	indexedOutputItems := make([]indexedOutputItem, 0, 2+len(p.toolCallOrder))
	if p.messageStarted {
		if err := p.writeEvent(map[string]interface{}{
			"type":          "response.output_text.done",
			"response_id":   p.responseID,
			"item_id":       p.itemID,
			"output_index":  0,
			"content_index": 0,
			"text":          text,
		}); err != nil {
			return err
		}

		if err := p.writeEvent(map[string]interface{}{
			"type":          "response.content_part.done",
			"response_id":   p.responseID,
			"item_id":       p.itemID,
			"output_index":  0,
			"content_index": 0,
			"part": map[string]interface{}{
				"type": "output_text",
				"text": text,
			},
		}); err != nil {
			return err
		}

		messageItem := map[string]interface{}{
			"id":     p.itemID,
			"type":   "message",
			"role":   "assistant",
			"status": status,
			"content": []map[string]interface{}{
				{
					"type": "output_text",
					"text": text,
				},
			},
		}

		if err := p.writeEvent(map[string]interface{}{
			"type":         "response.output_item.done",
			"response_id":  p.responseID,
			"output_index": 0,
			"item":         messageItem,
		}); err != nil {
			return err
		}
		indexedOutputItems = append(indexedOutputItems, indexedOutputItem{
			outputIndex: 0,
			item:        messageItem,
		})
	}

	if p.reasoningStarted {
		if err := p.writeEvent(map[string]interface{}{
			"type":          "response.reasoning_summary_text.done",
			"response_id":   p.responseID,
			"item_id":       p.reasoningItemID,
			"output_index":  p.reasoningOutputIndex,
			"summary_index": 0,
			"text":          reasoning,
		}); err != nil {
			return err
		}
		if err := p.writeEvent(map[string]interface{}{
			"type":          "response.reasoning_summary_part.done",
			"response_id":   p.responseID,
			"item_id":       p.reasoningItemID,
			"output_index":  p.reasoningOutputIndex,
			"summary_index": 0,
			"part": map[string]interface{}{
				"type": "summary_text",
				"text": reasoning,
			},
		}); err != nil {
			return err
		}

		reasoningItem := map[string]interface{}{
			"id":     p.reasoningItemID,
			"type":   "reasoning",
			"status": status,
			"summary": []map[string]interface{}{
				{
					"type": "summary_text",
					"text": reasoning,
				},
			},
		}
		if err := p.writeEvent(map[string]interface{}{
			"type":         "response.output_item.done",
			"response_id":  p.responseID,
			"output_index": p.reasoningOutputIndex,
			"item":         reasoningItem,
		}); err != nil {
			return err
		}
		indexedOutputItems = append(indexedOutputItems, indexedOutputItem{
			outputIndex: p.reasoningOutputIndex,
			item:        reasoningItem,
		})
	}

	for _, key := range p.toolCallOrder {
		st := p.toolCalls[key]
		if st == nil || st.Done {
			continue
		}
		if !st.Added {
			if err := p.writeEvent(map[string]interface{}{
				"type":         "response.output_item.added",
				"response_id":  p.responseID,
				"output_index": st.OutputIndex,
				"item": map[string]interface{}{
					"id":        st.ItemID,
					"type":      "function_call",
					"status":    "in_progress",
					"call_id":   st.CallID,
					"name":      st.Name,
					"arguments": "",
				},
			}); err != nil {
				return err
			}
			st.Added = true
		}
		if err := p.writeEvent(map[string]interface{}{
			"type":         "response.function_call_arguments.done",
			"response_id":  p.responseID,
			"item_id":      st.ItemID,
			"output_index": st.OutputIndex,
			"name":         st.Name,
			"arguments":    st.Arguments,
		}); err != nil {
			return err
		}

		fcItem := map[string]interface{}{
			"id":        st.ItemID,
			"type":      "function_call",
			"status":    status,
			"call_id":   st.CallID,
			"name":      st.Name,
			"arguments": st.Arguments,
		}

		if err := p.writeEvent(map[string]interface{}{
			"type":         "response.output_item.done",
			"response_id":  p.responseID,
			"output_index": st.OutputIndex,
			"item":         fcItem,
		}); err != nil {
			return err
		}

		indexedOutputItems = append(indexedOutputItems, indexedOutputItem{
			outputIndex: st.OutputIndex,
			item:        fcItem,
		})
		st.Done = true
		log.Debug().
			Str("response_id", p.responseID).
			Str("call_id", st.CallID).
			Str("tool_name", st.Name).
			Int("arguments_len", len(st.Arguments)).
			Msg("responses stream proxy finalized tool call")
	}

	sort.Slice(indexedOutputItems, func(i, j int) bool {
		return indexedOutputItems[i].outputIndex < indexedOutputItems[j].outputIndex
	})
	outputItems := make([]interface{}, 0, len(indexedOutputItems))
	for _, out := range indexedOutputItems {
		outputItems = append(outputItems, out.item)
	}

	resp := map[string]interface{}{
		"id":          p.responseID,
		"object":      "response",
		"created_at":  p.created,
		"status":      status,
		"model":       p.model,
		"output_text": text,
		"output":      outputItems,
	}
	if p.usage != nil {
		resp["usage"] = models.ResponsesUsage{
			InputTokens:  p.usage.PromptTokens,
			OutputTokens: p.usage.CompletionTokens,
			TotalTokens:  p.usage.TotalTokens,
		}
	}
	if p.reasoningStarted {
		resp["reasoning"] = map[string]interface{}{
			"effort": nil,
			"summary": []map[string]interface{}{
				{
					"type": "summary_text",
					"text": reasoning,
				},
			},
		}
	}
	if incompleteReason != "" {
		resp["incomplete_details"] = map[string]interface{}{
			"reason": incompleteReason,
		}
	}

	if p.responseID == "" {
		resp["id"] = responsesFallbackResponseID
	}
	if p.model == "" {
		resp["model"] = responsesFallbackModel
	}
	if p.beforeTerminal != nil {
		p.beforeTerminal(resp)
	}
	p.terminalResponse = resp
	if status == "completed" {
		p.completedResponse = resp
	}

	if err := p.writeEvent(map[string]interface{}{
		"type":     eventType,
		"response": resp,
	}); err != nil {
		return err
	}

	return nil
}

func (p *responsesStreamProxy) writeEvent(event map[string]interface{}) error {
	p.sendHeadersIfNeeded()
	p.eventSeq++
	event["sequence_number"] = p.eventSeq
	event["event_id"] = fmt.Sprintf("%s%d", responsesEventIDPrefix, p.eventSeq)
	b, err := json.Marshal(event)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(p.target, "data: %s\n\n", string(b)); err != nil {
		return err
	}
	if err := p.FlushError(); err != nil {
		return fmt.Errorf("failed to flush responses stream event: %w", err)
	}
	return nil
}

func (p *responsesStreamProxy) sendHeadersIfNeeded() {
	if p.headersSent {
		return
	}
	p.headersSent = true

	for key, values := range p.headers {
		if strings.EqualFold(key, "Content-Length") {
			continue
		}
		for _, value := range values {
			p.target.Header().Add(key, value)
		}
	}
	if p.statusCode < 400 && (!p.native || strings.TrimSpace(p.target.Header().Get("Content-Type")) == "") {
		p.target.Header().Set("Content-Type", "text/event-stream")
	}
	p.target.WriteHeader(p.statusCode)
}

func responsesCanonicalFunctionItemID(raw string) string {
	id := strings.TrimSpace(raw)
	if id == "" {
		return ""
	}
	if strings.HasPrefix(id, "fc_") {
		return id
	}
	if strings.HasPrefix(id, "call_") {
		return "fc_" + strings.TrimPrefix(id, "call_")
	}
	if strings.HasPrefix(id, "fc") {
		return id
	}
	return "fc_" + id
}

func responsesCanonicalFunctionCallID(raw string) string {
	id := strings.TrimSpace(raw)
	if id == "" {
		return ""
	}
	if strings.HasPrefix(id, "call_") {
		return id
	}
	if strings.HasPrefix(id, "fc_") {
		return "call_" + strings.TrimPrefix(id, "fc_")
	}
	if strings.HasPrefix(id, "fc") {
		return "call_" + strings.TrimPrefix(id, "fc")
	}
	return "call_" + id
}
