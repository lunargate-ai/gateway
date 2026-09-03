package streaming

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/pkg/models"
	"github.com/rs/zerolog/log"
)

// MaxStreamRecordBytes bounds one upstream SSE event or NDJSON record.
const MaxStreamRecordBytes = 4 << 20

const maxUpstreamErrorBodyBytes = 1 << 20

var (
	ErrUpstreamStreamIncomplete = errors.New("upstream stream ended before a terminal event")
	// ErrUpstreamStreamEmpty is returned when a provider does not supply a
	// readable successful streaming response.
	ErrUpstreamStreamEmpty = errors.New("provider returned an empty streaming response")
	// ErrStreamRecordTooLarge is returned before an oversized record is parsed
	// or forwarded downstream.
	ErrStreamRecordTooLarge = errors.New("upstream stream record exceeds 4 MiB limit")
)

func upstreamProviderError(status int, provider string, body []byte) *providers.ProviderError {
	trimmed := strings.TrimSpace(string(body))

	type parsedError struct {
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}

	var pe parsedError
	if err := json.Unmarshal(body, &pe); err == nil {
		t := strings.TrimSpace(pe.Error.Type)
		m := strings.TrimSpace(pe.Error.Message)
		if t != "" || m != "" {
			if m == "" {
				m = trimmed
			}
			if t == "" {
				t = "upstream_error"
			}
			return &providers.ProviderError{StatusCode: status, Provider: provider, Type: t, Message: m}
		}
	}

	if trimmed == "" {
		trimmed = http.StatusText(status)
	}
	return &providers.ProviderError{StatusCode: status, Provider: provider, Type: "upstream_error", Message: trimmed}
}

func readUpstreamProviderError(providerResp *http.Response, provider string) *providers.ProviderError {
	status := http.StatusBadGateway
	if providerResp != nil && providerResp.StatusCode > 0 {
		status = providerResp.StatusCode
	}
	if providerResp == nil || providerResp.Body == nil {
		return &providers.ProviderError{
			StatusCode: status,
			Provider:   provider,
			Type:       "upstream_error",
			Message:    "upstream returned an empty error response",
		}
	}
	defer providerResp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(providerResp.Body, maxUpstreamErrorBodyBytes+1))
	if err != nil {
		return &providers.ProviderError{
			StatusCode: status,
			Provider:   provider,
			Type:       "upstream_error",
			Message:    "failed to read upstream error response",
		}
	}
	if len(body) > maxUpstreamErrorBodyBytes {
		return &providers.ProviderError{
			StatusCode: status,
			Provider:   provider,
			Type:       "upstream_response_too_large",
			Message:    "upstream error response exceeds the 1 MiB limit",
		}
	}
	return upstreamProviderError(status, provider, body)
}

type ChunkObserver func(chunk *models.StreamChunk)

// SSEEvent is a parsed view used for observing or selectively transforming a
// raw SSE frame. Event and Data are copies for side-channel telemetry and state
// updates.
type SSEEvent struct {
	Event string
	Data  []byte
}

// SSEEventObserver reports whether an observed event is terminal.
type SSEEventObserver func(event SSEEvent) bool

// SSEEventDataTransformer may replace the data field of a complete SSE event
// before that event is forwarded and observed. A nil result preserves the raw
// frame byte-for-byte.
type SSEEventDataTransformer func(event SSEEvent) ([]byte, error)

// Handler manages SSE streaming between providers and clients.
type Handler struct{}

// NewHandler creates a new streaming handler.
func NewHandler() *Handler {
	return &Handler{}
}

// ProxySSE forwards a successful upstream SSE response without translating or
// reconstructing any frame. It flushes after each complete event, stops on
// downstream failures, and requires the observer to see a terminal event before
// a clean upstream EOF.
func (h *Handler) ProxySSE(
	ctx context.Context,
	w http.ResponseWriter,
	providerResp *http.Response,
	provider string,
	observer SSEEventObserver,
) error {
	return h.proxySSE(ctx, w, providerResp, provider, observer, nil)
}

// ProxySSEWithDataTransformer is ProxySSE with an optional event-data rewrite
// that runs before the frame is written and before the observer records it.
func (h *Handler) ProxySSEWithDataTransformer(
	ctx context.Context,
	w http.ResponseWriter,
	providerResp *http.Response,
	provider string,
	observer SSEEventObserver,
	transformer SSEEventDataTransformer,
) error {
	return h.proxySSE(ctx, w, providerResp, provider, observer, transformer)
}

func (h *Handler) proxySSE(
	ctx context.Context,
	w http.ResponseWriter,
	providerResp *http.Response,
	provider string,
	observer SSEEventObserver,
	transformer SSEEventDataTransformer,
) error {
	if providerResp == nil || providerResp.Body == nil {
		return errors.New("provider returned an empty SSE response")
	}

	if providerResp.StatusCode < http.StatusOK || providerResp.StatusCode >= http.StatusMultipleChoices {
		return readUpstreamProviderError(providerResp, provider)
	}
	defer providerResp.Body.Close()

	if strings.TrimSpace(w.Header().Get("Content-Type")) == "" {
		w.Header().Set("Content-Type", "text/event-stream")
	}
	if strings.TrimSpace(w.Header().Get("Cache-Control")) == "" {
		w.Header().Set("Cache-Control", "no-cache")
	}
	w.WriteHeader(providerResp.StatusCode)
	controller := http.NewResponseController(w)
	if err := controller.Flush(); err != nil {
		return fmt.Errorf("failed to flush native SSE headers: %w", err)
	}

	reader := bufio.NewReader(providerResp.Body)
	terminalSeen := false
	for {
		if err := ctx.Err(); err != nil {
			return err
		}

		rawFrame, readErr := readSSEFrame(reader)
		if readErr != nil {
			if errors.Is(readErr, io.EOF) {
				if err := ctx.Err(); err != nil {
					return err
				}
				if !terminalSeen {
					return fmt.Errorf("%w: native sse", ErrUpstreamStreamIncomplete)
				}
				return nil
			}
			return fmt.Errorf("native SSE read error: %w", readErr)
		}

		event := parseSSEEvent(rawFrame)
		if transformer != nil {
			transformedData, err := transformer(event)
			if err != nil {
				return fmt.Errorf("failed to transform native SSE event: %w", err)
			}
			if transformedData != nil {
				rawFrame, err = replaceSSEEventData(rawFrame, transformedData)
				if err != nil {
					return fmt.Errorf("failed to replace native SSE event data: %w", err)
				}
				event.Data = append([]byte(nil), transformedData...)
			}
		}
		if _, err := w.Write(rawFrame); err != nil {
			return fmt.Errorf("failed to write native SSE frame: %w", err)
		}
		if err := controller.Flush(); err != nil {
			return fmt.Errorf("failed to flush native SSE frame: %w", err)
		}
		if observer != nil && observer(event) {
			terminalSeen = true
		}
	}
}

func replaceSSEEventData(frame []byte, data []byte) ([]byte, error) {
	if len(data) > MaxStreamRecordBytes {
		return nil, ErrStreamRecordTooLarge
	}
	var output bytes.Buffer
	replaced := false
	remaining := frame
	for len(remaining) > 0 {
		line := remaining
		if newline := bytes.IndexByte(remaining, '\n'); newline >= 0 {
			line = remaining[:newline+1]
			remaining = remaining[newline+1:]
		} else {
			remaining = nil
		}

		content, lineEnding := splitSSELineEnding(line)
		if !isSSEDataField(content) {
			_, _ = output.Write(line)
			continue
		}
		if replaced {
			continue
		}

		prefix := []byte("data:")
		if bytes.HasPrefix(content, []byte("data: ")) {
			prefix = []byte("data: ")
		}
		dataLines := bytes.Split(data, []byte{'\n'})
		for _, dataLine := range dataLines {
			_, _ = output.Write(prefix)
			_, _ = output.Write(dataLine)
			_, _ = output.Write(lineEnding)
		}
		replaced = true
	}
	if !replaced {
		return nil, errors.New("SSE event has no data field")
	}
	result := output.Bytes()
	if err := validateSSEFrameSize(result); err != nil {
		return nil, err
	}
	return result, nil
}

func validateSSEFrameSize(frame []byte) error {
	if sseFrameRecordSize(frame) > MaxStreamRecordBytes {
		return ErrStreamRecordTooLarge
	}
	return nil
}

// sseFrameRecordSize counts the complete event, including every field and
// line ending, but excludes the final blank line that delimits the event.
func sseFrameRecordSize(frame []byte) int {
	if len(frame) == 0 {
		return 0
	}
	lastLineStart := bytes.LastIndexByte(frame[:len(frame)-1], '\n') + 1
	if isSSEBlankLine(frame[lastLineStart:]) {
		return lastLineStart
	}
	return len(frame)
}

func splitSSELineEnding(line []byte) ([]byte, []byte) {
	if bytes.HasSuffix(line, []byte("\r\n")) {
		return line[:len(line)-2], line[len(line)-2:]
	}
	if bytes.HasSuffix(line, []byte("\n")) {
		return line[:len(line)-1], line[len(line)-1:]
	}
	return line, nil
}

func isSSEDataField(line []byte) bool {
	return bytes.Equal(line, []byte("data")) || bytes.HasPrefix(line, []byte("data:"))
}

func isSSEBlankLine(line []byte) bool {
	return bytes.Equal(line, []byte{'\n'}) || bytes.Equal(line, []byte{'\r', '\n'})
}

func parseSSEEvent(frame []byte) SSEEvent {
	event := SSEEvent{}
	dataLines := make([]string, 0, 1)
	for _, rawLine := range bytes.Split(frame, []byte{'\n'}) {
		line := strings.TrimSuffix(string(rawLine), "\r")
		if line == "" || strings.HasPrefix(line, ":") {
			continue
		}
		field, value, hasColon := strings.Cut(line, ":")
		if !hasColon {
			value = ""
		}
		value = strings.TrimPrefix(value, " ")
		switch field {
		case "event":
			event.Event = value
		case "data":
			dataLines = append(dataLines, value)
		}
	}
	if len(dataLines) > 0 {
		event.Data = []byte(strings.Join(dataLines, "\n"))
	}
	return event
}

func readSSEEvent(reader *bufio.Reader) (SSEEvent, error) {
	frame, err := readSSEFrame(reader)
	if err != nil {
		return SSEEvent{}, err
	}
	return parseSSEEvent(frame), nil
}

func readSSEFrame(reader *bufio.Reader) ([]byte, error) {
	if reader == nil {
		return nil, errors.New("SSE reader is required")
	}

	var frame bytes.Buffer
	for {
		fragment, readErr := reader.ReadSlice('\n')
		if len(fragment) > 0 {
			if readErr == nil && isSSEBlankLine(fragment) {
				_, _ = frame.Write(fragment)
				return frame.Bytes(), nil
			}
			if len(fragment) > MaxStreamRecordBytes-frame.Len() {
				return nil, ErrStreamRecordTooLarge
			}
			_, _ = frame.Write(fragment)
		}

		if readErr == nil {
			continue
		}
		if errors.Is(readErr, bufio.ErrBufferFull) {
			continue
		}
		if errors.Is(readErr, io.EOF) {
			if frame.Len() == 0 {
				return nil, readErr
			}
			return nil, fmt.Errorf("%w: incomplete sse event", ErrUpstreamStreamIncomplete)
		}
		return nil, readErr
	}
}

// StreamResponse reads an SSE stream from a provider and forwards it to the client.
// It translates provider-specific chunks to OpenAI-compatible format using the translator.
func (h *Handler) StreamResponse(ctx context.Context, w http.ResponseWriter, providerResp *http.Response, translator models.ProviderTranslator) error {
	return h.streamResponse(ctx, w, providerResp, translator, nil, true)
}

func (h *Handler) StreamResponseWithObserver(
	ctx context.Context,
	w http.ResponseWriter,
	providerResp *http.Response,
	translator models.ProviderTranslator,
	observer ChunkObserver,
) error {
	return h.streamResponse(ctx, w, providerResp, translator, observer, true)
}

// StreamResponseWithObserverAndUsage controls whether upstream usage is
// exposed to a Chat Completions client while always reporting it to the
// gateway observer.
func (h *Handler) StreamResponseWithObserverAndUsage(
	ctx context.Context,
	w http.ResponseWriter,
	providerResp *http.Response,
	translator models.ProviderTranslator,
	observer ChunkObserver,
	includeUsage bool,
) error {
	return h.streamResponse(ctx, w, providerResp, translator, observer, includeUsage)
}

func (h *Handler) streamResponse(
	ctx context.Context,
	w http.ResponseWriter,
	providerResp *http.Response,
	translator models.ProviderTranslator,
	observer ChunkObserver,
	includeUsage bool,
) error {
	if providerResp != nil && providerResp.StatusCode != http.StatusOK {
		return readUpstreamProviderError(providerResp, translator.Name())
	}
	if providerResp == nil || providerResp.Body == nil {
		return ErrUpstreamStreamEmpty
	}
	defer providerResp.Body.Close()

	controller := http.NewResponseController(w)

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(http.StatusOK)
	if err := controller.Flush(); err != nil {
		return fmt.Errorf("failed to flush stream headers: %w", err)
	}

	reader := bufio.NewReader(providerResp.Body)
	envelope := newChatStreamEnvelopeNormalizer(translator.DefaultModel())

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		event, err := readSSEEvent(reader)
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			streamErr := fmt.Errorf("stream read error: %w", err)
			if errors.Is(err, ErrStreamRecordTooLarge) {
				return terminateChatStreamWithError(w, controller, streamErr)
			}
			return streamErr
		}
		if event.Data == nil {
			continue
		}
		data := string(event.Data)

		// Check for stream end
		if data == "[DONE]" {
			if err := writeSSEFrame(w, controller, []byte("[DONE]"), "done frame"); err != nil {
				return err
			}
			return nil
		}

		// Parse through the translator
		chunk, err := translator.ParseStreamChunk([]byte(data))
		streamDone := errors.Is(err, providers.ErrStreamDone)
		if err != nil && !streamDone {
			streamErr := fmt.Errorf("failed to parse stream chunk: %w", err)
			return terminateChatStreamWithError(w, controller, streamErr)
		}

		if chunk == nil && !streamDone {
			continue
		}

		if chunk != nil {
			chunk = envelope.normalize(chunk)
			if observer != nil {
				observer(chunk)
			}
			clientChunk := chunk
			if !includeUsage {
				clientChunk = withoutStreamUsage(chunk)
			}

			// Marshal to OpenAI-compatible format
			chunkJSON, err := marshalStreamChunk(clientChunk)
			if err != nil {
				return fmt.Errorf("failed to marshal stream chunk: %w", err)
			}
			if err := writeSSEFrame(w, controller, chunkJSON, "stream chunk"); err != nil {
				return err
			}
		}

		if streamDone {
			if err := writeSSEFrame(w, controller, []byte("[DONE]"), "done frame"); err != nil {
				return err
			}
			return nil
		}
	}

	if err := ctx.Err(); err != nil {
		return err
	}
	return fmt.Errorf("%w: sse", ErrUpstreamStreamIncomplete)
}

// StreamAnthropicResponse handles Anthropic's different SSE format.
// Anthropic sends "event: type\ndata: json\n\n" pairs.
func (h *Handler) StreamAnthropicResponse(ctx context.Context, w http.ResponseWriter, providerResp *http.Response, translator models.ProviderTranslator) error {
	return h.streamAnthropicResponse(ctx, w, providerResp, translator, nil, false)
}

func (h *Handler) StreamAnthropicResponseWithObserver(
	ctx context.Context,
	w http.ResponseWriter,
	providerResp *http.Response,
	translator models.ProviderTranslator,
	observer ChunkObserver,
) error {
	return h.streamAnthropicResponse(ctx, w, providerResp, translator, observer, false)
}

// StreamAnthropicResponseWithObserverAndUsage controls whether translated
// usage is exposed to a Chat Completions client while still reporting the
// original chunk to the gateway observer.
func (h *Handler) StreamAnthropicResponseWithObserverAndUsage(
	ctx context.Context,
	w http.ResponseWriter,
	providerResp *http.Response,
	translator models.ProviderTranslator,
	observer ChunkObserver,
	includeUsage bool,
) error {
	return h.streamAnthropicResponse(ctx, w, providerResp, translator, observer, includeUsage)
}

func (h *Handler) streamAnthropicResponse(
	ctx context.Context,
	w http.ResponseWriter,
	providerResp *http.Response,
	translator models.ProviderTranslator,
	observer ChunkObserver,
	includeUsage bool,
) error {
	if providerResp != nil && providerResp.StatusCode != http.StatusOK {
		return readUpstreamProviderError(providerResp, translator.Name())
	}
	if providerResp == nil || providerResp.Body == nil {
		return ErrUpstreamStreamEmpty
	}
	defer providerResp.Body.Close()

	controller := http.NewResponseController(w)

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(http.StatusOK)
	if err := controller.Flush(); err != nil {
		return fmt.Errorf("failed to flush stream headers: %w", err)
	}

	reader := bufio.NewReader(providerResp.Body)
	envelope := newChatStreamEnvelopeNormalizer(translator.DefaultModel())
	var usage streamUsageAccumulator

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		event, err := readSSEEvent(reader)
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			streamErr := fmt.Errorf("stream read error: %w", err)
			if errors.Is(err, ErrStreamRecordTooLarge) {
				return terminateChatStreamWithError(w, controller, streamErr)
			}
			return streamErr
		}
		if event.Data == nil {
			continue
		}

		payload := event.Data
		if event.Event != "" {
			trimmed := bytes.TrimSpace(payload)
			if len(trimmed) > 0 && trimmed[0] == '{' {
				var obj map[string]json.RawMessage
				if err := json.Unmarshal(trimmed, &obj); err == nil {
					if _, ok := obj["type"]; !ok {
						obj["type"] = json.RawMessage(fmt.Sprintf("%q", event.Event))
						if b, err := json.Marshal(obj); err == nil {
							payload = b
						}
					}
				}
			}
		}

		chunk, parseErr := translator.ParseStreamChunk(payload)
		streamDone := errors.Is(parseErr, providers.ErrStreamDone)
		if parseErr != nil && !streamDone {
			streamErr := fmt.Errorf("failed to parse anthropic stream chunk: %w", parseErr)
			return terminateChatStreamWithError(w, controller, streamErr)
		}

		if chunk == nil && !streamDone {
			continue
		}

		if chunk != nil {
			chunk = envelope.normalize(chunk)
			usage.add(chunk)
			if observer != nil {
				observer(chunk)
			}

			clientChunk := withoutStreamUsage(chunk)
			chunkJSON, marshalErr := json.Marshal(clientChunk)
			if marshalErr != nil {
				log.Error().Err(marshalErr).Msg("failed to marshal stream chunk")
			} else {
				if err := writeSSEFrame(w, controller, chunkJSON, "stream chunk"); err != nil {
					return err
				}
			}
		}

		if streamDone {
			if includeUsage {
				if err := writeCanonicalUsageTrailer(w, controller, envelope, usage); err != nil {
					return err
				}
			}
			if err := writeSSEFrame(w, controller, []byte("[DONE]"), "done frame"); err != nil {
				return err
			}
			return nil
		}
	}

	if err := ctx.Err(); err != nil {
		return err
	}
	return fmt.Errorf("%w: anthropic sse", ErrUpstreamStreamIncomplete)
}

func writeSSEFrame(w http.ResponseWriter, controller *http.ResponseController, payload []byte, frameName string) error {
	if _, err := fmt.Fprintf(w, "data: %s\n\n", payload); err != nil {
		return fmt.Errorf("failed to write %s: %w", frameName, err)
	}
	if err := controller.Flush(); err != nil {
		return fmt.Errorf("failed to flush %s: %w", frameName, err)
	}
	return nil
}

// IsStreamRequest checks if the request body has stream=true.
func IsStreamRequest(body []byte) bool {
	var req struct {
		Stream bool `json:"stream"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return false
	}
	return req.Stream
}

// ReadBody reads and returns the request body, allowing re-reading.
func ReadBody(r *http.Request) ([]byte, error) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read request body: %w", err)
	}
	defer r.Body.Close()
	// Reset the body for later reads
	r.Body = io.NopCloser(bytes.NewReader(body))
	return body, nil
}
