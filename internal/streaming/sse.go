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
	// ErrNativeSSEInvalidStatus prevents non-200 success responses from being
	// committed as an SSE stream.
	ErrNativeSSEInvalidStatus = errors.New("native SSE upstream must return HTTP 200")
	// ErrNativeSSEInvalidData marks a complete SSE data record that is not one
	// strict JSON object.
	ErrNativeSSEInvalidData = errors.New("native SSE data must be a JSON object")
	// ErrNativeSSEPreflightTooLarge bounds comments and empty records processed
	// while waiting for the first useful native SSE record.
	ErrNativeSSEPreflightTooLarge = errors.New("native SSE preflight exceeds resource limits")
	// ErrNativeSSEUpstreamRead distinguishes provider/body failures from client
	// cancellation and downstream write failures.
	ErrNativeSSEUpstreamRead = errors.New("native SSE upstream read failed")
	// ErrNativeSSETransform identifies event transformation failures.
	ErrNativeSSETransform = errors.New("native SSE event transformation failed")
	// ErrNativeSSEDownstream identifies client write and flush failures. Callers
	// must not synthesize additional terminal events after this error.
	ErrNativeSSEDownstream = errors.New("native SSE downstream write failed")
)

const (
	maxNativeSSEPreflightBytes  = MaxStreamRecordBytes + 2
	maxNativeSSEPreflightFrames = 1024
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

// ProxySSE forwards an HTTP 200 native Responses SSE stream without translating
// or reconstructing frames. It validates one JSON object before committing
// headers, flushes complete events, and stops as soon as the observer accepts a
// terminal event or a downstream operation fails.
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
		return ErrUpstreamStreamEmpty
	}

	if providerResp.StatusCode < http.StatusOK || providerResp.StatusCode >= http.StatusMultipleChoices {
		return readUpstreamProviderError(providerResp, provider)
	}
	if providerResp.StatusCode != http.StatusOK {
		_ = providerResp.Body.Close()
		return fmt.Errorf("%w: got %d", ErrNativeSSEInvalidStatus, providerResp.StatusCode)
	}
	defer providerResp.Body.Close()

	reader := bufio.NewReader(providerResp.Body)
	// Keep the raw preflight prefix in one byte-bounded allocation. Tracking
	// every leading comment or empty event as a separate frame would let an
	// upstream send millions of one-byte records while staying below the byte
	// limit and consume unbounded metadata memory.
	var preflight bytes.Buffer
	preflightFrames := 0
	var controller *http.ResponseController
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
				return ErrUpstreamStreamEmpty
			}
			return classifyNativeSSEReadError(ctx, readErr)
		}
		preflightFrames++
		if preflightFrames > maxNativeSSEPreflightFrames {
			return ErrNativeSSEPreflightTooLarge
		}
		if len(rawFrame) > maxNativeSSEPreflightBytes-preflight.Len() {
			return ErrNativeSSEPreflightTooLarge
		}
		frame, useful, err := prepareNativeSSEFrame(rawFrame, transformer)
		if err != nil {
			return err
		}
		if len(frame.raw) > maxNativeSSEPreflightBytes-preflight.Len() {
			return ErrNativeSSEPreflightTooLarge
		}
		_, _ = preflight.Write(frame.raw)
		if !useful {
			continue
		}

		controller, err = startNativeSSE(w)
		if err != nil {
			return err
		}
		// The leading frames contain no useful data, so only the first useful
		// event needs to reach the terminal observer. The raw bytes are still
		// forwarded byte-for-byte as one bounded prefix.
		frame.raw = preflight.Bytes()
		terminal, err := forwardNativeSSEFrame(w, controller, frame, observer)
		if err != nil {
			return err
		}
		if terminal {
			return nil
		}
		break
	}

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
				return fmt.Errorf("%w: native sse", ErrUpstreamStreamIncomplete)
			}
			return classifyNativeSSEReadError(ctx, readErr)
		}
		frame, _, err := prepareNativeSSEFrame(rawFrame, transformer)
		if err != nil {
			return err
		}
		terminal, err := forwardNativeSSEFrame(w, controller, frame, observer)
		if err != nil {
			return err
		}
		if terminal {
			return nil
		}
	}
}

type nativeSSEFrame struct {
	raw   []byte
	event SSEEvent
}

func prepareNativeSSEFrame(rawFrame []byte, transformer SSEEventDataTransformer) (nativeSSEFrame, bool, error) {
	event := parseSSEEvent(rawFrame)
	useful := len(bytes.TrimSpace(event.Data)) > 0
	if useful {
		if err := validateNativeSSEData(event.Data); err != nil {
			return nativeSSEFrame{}, false, err
		}
	}
	if transformer != nil {
		transformedData, err := transformer(event)
		if err != nil {
			return nativeSSEFrame{}, false, fmt.Errorf("%w: %w", ErrNativeSSETransform, err)
		}
		if transformedData != nil {
			rawFrame, err = replaceSSEEventData(rawFrame, transformedData)
			if err != nil {
				return nativeSSEFrame{}, false, fmt.Errorf("%w: %w", ErrNativeSSETransform, err)
			}
			if err := validateNativeSSEData(transformedData); err != nil {
				return nativeSSEFrame{}, false, fmt.Errorf("%w: %w", ErrNativeSSETransform, err)
			}
			event.Data = append([]byte(nil), transformedData...)
			useful = true
		}
	}
	return nativeSSEFrame{raw: rawFrame, event: event}, useful, nil
}

func validateNativeSSEData(data []byte) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	var object map[string]json.RawMessage
	if err := decoder.Decode(&object); err != nil {
		return fmt.Errorf("%w: %v", ErrNativeSSEInvalidData, err)
	}
	if object == nil {
		return fmt.Errorf("%w: expected object", ErrNativeSSEInvalidData)
	}
	var extra json.RawMessage
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		if err == nil {
			return fmt.Errorf("%w: multiple JSON values", ErrNativeSSEInvalidData)
		}
		return fmt.Errorf("%w: %v", ErrNativeSSEInvalidData, err)
	}
	return nil
}

func startNativeSSE(w http.ResponseWriter) (*http.ResponseController, error) {
	if strings.TrimSpace(w.Header().Get("Content-Type")) == "" {
		w.Header().Set("Content-Type", "text/event-stream")
	}
	if strings.TrimSpace(w.Header().Get("Cache-Control")) == "" {
		w.Header().Set("Cache-Control", "no-cache")
	}
	w.WriteHeader(http.StatusOK)
	controller := http.NewResponseController(w)
	if err := controller.Flush(); err != nil {
		return nil, fmt.Errorf("%w: flush headers: %w", ErrNativeSSEDownstream, err)
	}
	return controller, nil
}

func forwardNativeSSEFrame(
	w http.ResponseWriter,
	controller *http.ResponseController,
	frame nativeSSEFrame,
	observer SSEEventObserver,
) (bool, error) {
	written, err := w.Write(frame.raw)
	if err != nil {
		return false, fmt.Errorf("%w: write frame: %w", ErrNativeSSEDownstream, err)
	}
	if written != len(frame.raw) {
		return false, fmt.Errorf("%w: write frame: %w", ErrNativeSSEDownstream, io.ErrShortWrite)
	}
	if err := controller.Flush(); err != nil {
		return false, fmt.Errorf("%w: flush frame: %w", ErrNativeSSEDownstream, err)
	}
	return observer != nil && observer(frame.event), nil
}

func classifyNativeSSEReadError(ctx context.Context, readErr error) error {
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return err
		}
	}
	if errors.Is(readErr, ErrUpstreamStreamIncomplete) || errors.Is(readErr, ErrStreamRecordTooLarge) {
		return readErr
	}
	return fmt.Errorf("%w: %w", ErrNativeSSEUpstreamRead, readErr)
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

	output := newChatStreamOutput(ctx, w, translator.Name())

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
				if ctxErr := ctx.Err(); ctxErr != nil {
					return ctxErr
				}
				if !output.started {
					return output.fail(ErrUpstreamStreamEmpty)
				}
				return output.fail(fmt.Errorf("%w: sse", ErrUpstreamStreamIncomplete))
			}
			streamErr := fmt.Errorf("stream read error: %w", err)
			return output.fail(streamErr)
		}
		if len(bytes.TrimSpace(event.Data)) == 0 {
			continue
		}
		data := string(event.Data)

		// Check for stream end
		if data == "[DONE]" {
			if err := output.write([]byte("[DONE]"), "done frame"); err != nil {
				return err
			}
			return nil
		}

		// Parse through the translator
		chunk, err := translator.ParseStreamChunk([]byte(data))
		streamDone := errors.Is(err, providers.ErrStreamDone)
		if err != nil && !streamDone {
			streamErr := fmt.Errorf("failed to parse stream chunk: %w", err)
			var providerErr *providers.ProviderError
			if errors.As(err, &providerErr) && providerErr != nil && !output.started {
				if startErr := output.start(); startErr != nil {
					return errors.Join(streamErr, startErr)
				}
			}
			return output.fail(streamErr)
		}
		if !streamDone && !hasChatStreamPayload(chunk) {
			continue
		}

		var chunkJSON []byte
		if chunk != nil {
			chunk = envelope.normalize(chunk)
			clientChunk := chunk
			if !includeUsage {
				clientChunk = withoutStreamUsage(chunk)
			}

			// Marshal to OpenAI-compatible format
			chunkJSON, err = marshalStreamChunk(clientChunk)
			if err != nil {
				return output.fail(fmt.Errorf("failed to marshal stream chunk: %w", err))
			}
		}
		if err := output.start(); err != nil {
			return err
		}
		if chunk != nil {
			if observer != nil {
				observer(chunk)
			}
			if err := writeSSEFrame(w, output.controller, chunkJSON, "stream chunk"); err != nil {
				return err
			}
		}

		if streamDone {
			if err := writeSSEFrame(w, output.controller, []byte("[DONE]"), "done frame"); err != nil {
				return err
			}
			return nil
		}
	}
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

	output := newChatStreamOutput(ctx, w, translator.Name())

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
				if ctxErr := ctx.Err(); ctxErr != nil {
					return ctxErr
				}
				if !output.started {
					return output.fail(ErrUpstreamStreamEmpty)
				}
				return output.fail(fmt.Errorf("%w: anthropic sse", ErrUpstreamStreamIncomplete))
			}
			streamErr := fmt.Errorf("stream read error: %w", err)
			return output.fail(streamErr)
		}
		if len(bytes.TrimSpace(event.Data)) == 0 {
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
			var providerErr *providers.ProviderError
			if errors.As(parseErr, &providerErr) && providerErr != nil && !output.started {
				if startErr := output.start(); startErr != nil {
					return errors.Join(streamErr, startErr)
				}
			}
			return output.fail(streamErr)
		}
		if !streamDone && !hasChatStreamPayload(chunk) {
			continue
		}

		var chunkJSON []byte
		if chunk != nil {
			chunk = envelope.normalize(chunk)
			usage.add(chunk)

			clientChunk := withoutStreamUsage(chunk)
			encoded, marshalErr := json.Marshal(clientChunk)
			if marshalErr != nil {
				return output.fail(fmt.Errorf("failed to marshal anthropic stream chunk: %w", marshalErr))
			}
			chunkJSON = encoded
		}
		if err := output.start(); err != nil {
			return err
		}
		if chunk != nil {
			if observer != nil {
				observer(chunk)
			}
			if err := writeSSEFrame(w, output.controller, chunkJSON, "stream chunk"); err != nil {
				return err
			}
		}

		if streamDone {
			if includeUsage {
				if err := writeCanonicalUsageTrailer(w, output.controller, envelope, usage); err != nil {
					return err
				}
			}
			if err := writeSSEFrame(w, output.controller, []byte("[DONE]"), "done frame"); err != nil {
				return err
			}
			return nil
		}
	}
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
