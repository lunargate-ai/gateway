package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"

	"github.com/lunargate-ai/gateway/pkg/models"
)

const (
	responsesFallbackResponseID = "resp_lunargate"
	responsesFallbackModel      = "unknown"
	responsesEventIDPrefix      = "evt_lg_"
)

func parseResponsesRequest(w http.ResponseWriter, r *http.Request) (*models.ResponsesRequest, bool) {
	const maxRequestBodyBytes int64 = 10 << 20
	r.Body = http.MaxBytesReader(w, r.Body, maxRequestBodyBytes)
	defer r.Body.Close()

	decoder := json.NewDecoder(r.Body)
	var req models.ResponsesRequest
	if err := decoder.Decode(&req); err != nil {
		writeRequestDecodeError(w, err)
		return nil, false
	}
	var extra json.RawMessage
	if err := decoder.Decode(&extra); err != io.EOF {
		writeRequestDecodeError(w, err)
		return nil, false
	}
	return &req, true
}

func copyHeaders(dst http.Header, src http.Header) {
	for key, values := range src {
		if strings.EqualFold(key, "Content-Length") {
			continue
		}
		if _, exists := dst[key]; exists {
			continue
		}
		copied := make([]string, 0, len(values))
		for _, value := range values {
			copied = append(copied, value)
		}
		dst[key] = copied
	}
}

type responsesStreamProxy struct {
	target      http.ResponseWriter
	headers     http.Header
	statusCode  int
	headersSent bool
	buffer      bytes.Buffer

	responseID string
	itemID     string
	model      string
	created    int64
	text       strings.Builder
	started    bool
	completed  bool
	eventSeq   int

	nextOutputIndex int
	toolCalls       map[string]*responsesToolCallState
	toolCallOrder   []string
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

func newResponsesStreamProxy(target http.ResponseWriter) *responsesStreamProxy {
	return &responsesStreamProxy{
		target:     target,
		headers:    make(http.Header),
		statusCode: http.StatusOK,
		nextOutputIndex: 1,
		toolCalls:       make(map[string]*responsesToolCallState),
		toolCallOrder:   make([]string, 0, 4),
	}
}

func (p *responsesStreamProxy) Header() http.Header {
	return p.headers
}

func (p *responsesStreamProxy) WriteHeader(statusCode int) {
	p.statusCode = statusCode
}

func (p *responsesStreamProxy) Flush() {
	if f, ok := p.target.(http.Flusher); ok {
		f.Flush()
	}
}

func (p *responsesStreamProxy) Write(b []byte) (int, error) {
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

	if !p.completed {
		return p.emitCompleted()
	}
	return nil
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
		itemID := strings.TrimSpace(tc.ID)
		if itemID == "" {
			itemID = fmt.Sprintf("fc_%s_%d", p.responseID, p.nextOutputIndex)
		}
		name := strings.TrimSpace(tc.Function.Name)
		if name == "" {
			name = fmt.Sprintf("tool_call_%d", p.nextOutputIndex)
		}
		st = &responsesToolCallState{
			ItemID:      itemID,
			CallID:      itemID,
			Name:        name,
			OutputIndex: p.nextOutputIndex,
		}
		p.nextOutputIndex++
		p.toolCalls[key] = st
		p.toolCallOrder = append(p.toolCallOrder, key)
	}

	if name := strings.TrimSpace(tc.Function.Name); name != "" {
		st.Name = name
	}
	if id := strings.TrimSpace(tc.ID); id != "" {
		st.ItemID = id
		st.CallID = id
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
				return p.emitCompleted()
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
		if err := p.ensureStarted(); err != nil {
			return err
		}

		for _, c := range chunk.Choices {
			if c.Delta == nil {
				continue
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
			p.text.WriteString(delta)
			event := map[string]interface{}{
				"type":          "response.output_text.delta",
				"response_id":   p.responseID,
				"item_id":       p.itemID,
				"output_index":  0,
				"content_index": 0,
				"delta":         delta,
			}
			if err := p.writeEvent(event); err != nil {
				return err
			}
		}
	}
	return nil
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

	if err := p.writeEvent(map[string]interface{}{
		"type":        "response.output_item.added",
		"response_id": p.responseID,
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
		"type":         "response.content_part.added",
		"response_id":  p.responseID,
		"item_id":      p.itemID,
		"output_index": 0,
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
	if p.completed {
		return nil
	}
	p.completed = true
	if err := p.ensureStarted(); err != nil {
		return err
	}

	text := p.text.String()
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
		"type":         "response.content_part.done",
		"response_id":  p.responseID,
		"item_id":      p.itemID,
		"output_index": 0,
		"content_index": 0,
		"part": map[string]interface{}{
			"type": "output_text",
			"text": text,
		},
	}); err != nil {
		return err
	}

	item := map[string]interface{}{
		"id":     p.itemID,
		"type":   "message",
		"role":   "assistant",
		"status": "completed",
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
		"item":         item,
	}); err != nil {
		return err
	}

	toolItems := make([]interface{}, 0, len(p.toolCallOrder))
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
			"arguments":    st.Arguments,
		}); err != nil {
			return err
		}

		fcItem := map[string]interface{}{
			"id":        st.ItemID,
			"type":      "function_call",
			"status":    "completed",
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

		toolItems = append(toolItems, fcItem)
		st.Done = true
	}

	resp := map[string]interface{}{
		"id":         p.responseID,
		"object":     "response",
		"created_at": p.created,
		"status":     "completed",
		"model":      p.model,
		"output_text": text,
		"output":     append([]interface{}{item}, toolItems...),
	}

	if p.responseID == "" {
		resp["id"] = responsesFallbackResponseID
	}
	if p.model == "" {
		resp["model"] = responsesFallbackModel
	}

	if err := p.writeEvent(map[string]interface{}{
		"type":     "response.completed",
		"response": resp,
	}); err != nil {
		return err
	}

	return nil
}

func (p *responsesStreamProxy) writeEvent(event map[string]interface{}) error {
	p.sendHeadersIfNeeded()
	p.eventSeq++
	event["event_id"] = fmt.Sprintf("%s%d", responsesEventIDPrefix, p.eventSeq)
	b, err := json.Marshal(event)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(p.target, "data: %s\n\n", string(b)); err != nil {
		return err
	}
	p.Flush()
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
	p.target.Header().Set("Content-Type", "text/event-stream")
	p.target.WriteHeader(p.statusCode)
}

func (h *Handler) Responses(w http.ResponseWriter, r *http.Request) {
	responsesReq, ok := parseResponsesRequest(w, r)
	if !ok {
		return
	}

	unifiedReq, err := models.ResponsesToUnifiedRequest(responsesReq)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), "invalid_request_error")
		return
	}
	if err := models.NormalizeUnifiedRequest(unifiedReq); err != nil {
		// Keep unified normalization as a safety net for request invariants
		// shared across all chat-completions flows.
		writeError(w, http.StatusBadRequest, "invalid tool/function calling payload", "invalid_request_error")
		return
	}

	body, err := json.Marshal(unifiedReq)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to prepare request", "internal_error")
		return
	}

	chatReq := r.Clone(r.Context())
	chatReq.URL.Path = "/v1/responses"
	chatReq.RequestURI = "/v1/responses"
	chatReq.Body = io.NopCloser(bytes.NewReader(body))
	chatReq.ContentLength = int64(len(body))
	chatReq.Header = r.Header.Clone()
	chatReq.Header.Set("Content-Type", "application/json")
	chatReq.Header.Set("X-LunarGate-Request-Type", "responses")

	if unifiedReq.Stream {
		proxy := newResponsesStreamProxy(w)
		h.ChatCompletions(proxy, chatReq)
		if err := proxy.finalize(); err != nil {
			writeError(w, http.StatusBadGateway, "failed to stream responses event payload", "provider_error")
		}
		return
	}

	rec := httptest.NewRecorder()
	h.ChatCompletions(rec, chatReq)

	copyHeaders(w.Header(), rec.Header())
	if rec.Code >= 400 {
		w.WriteHeader(rec.Code)
		_, _ = w.Write(rec.Body.Bytes())
		return
	}

	var unifiedResp models.UnifiedResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &unifiedResp); err != nil {
		writeError(w, http.StatusBadGateway, "failed to parse provider response", "provider_error")
		return
	}

	resp := models.UnifiedResponseToResponses(&unifiedResp)
	writeJSON(w, rec.Code, resp)
}
