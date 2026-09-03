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

var ErrUpstreamStreamIncomplete = errors.New("upstream stream ended before a terminal event")

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

type ChunkObserver func(chunk *models.StreamChunk)

// SSEEvent is a parsed view used only for observing a raw SSE frame. The raw
// frame is forwarded unchanged; Event and Data are copies for side-channel
// telemetry and state updates.
type SSEEvent struct {
	Event string
	Data  []byte
}

// SSEEventObserver reports whether an observed event is terminal.
type SSEEventObserver func(event SSEEvent) bool

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
	if providerResp == nil || providerResp.Body == nil {
		return errors.New("provider returned an empty SSE response")
	}
	defer providerResp.Body.Close()

	if providerResp.StatusCode < http.StatusOK || providerResp.StatusCode >= http.StatusMultipleChoices {
		body, _ := io.ReadAll(providerResp.Body)
		return upstreamProviderError(providerResp.StatusCode, provider, body)
	}

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
	var frame bytes.Buffer
	terminalSeen := false
	for {
		if err := ctx.Err(); err != nil {
			return err
		}

		line, readErr := reader.ReadBytes('\n')
		if len(line) > 0 {
			_, _ = frame.Write(line)
			if isSSEBlankLine(line) {
				rawFrame := append([]byte(nil), frame.Bytes()...)
				if _, err := w.Write(rawFrame); err != nil {
					return fmt.Errorf("failed to write native SSE frame: %w", err)
				}
				if err := controller.Flush(); err != nil {
					return fmt.Errorf("failed to flush native SSE frame: %w", err)
				}
				if observer != nil && observer(parseSSEEvent(rawFrame)) {
					terminalSeen = true
				}
				frame.Reset()
			}
		}

		if readErr == nil {
			continue
		}
		if !errors.Is(readErr, io.EOF) {
			return fmt.Errorf("native SSE read error: %w", readErr)
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if frame.Len() > 0 {
			if _, err := w.Write(frame.Bytes()); err != nil {
				return fmt.Errorf("failed to write trailing native SSE bytes: %w", err)
			}
			if err := controller.Flush(); err != nil {
				return fmt.Errorf("failed to flush trailing native SSE bytes: %w", err)
			}
		}
		if !terminalSeen {
			return fmt.Errorf("%w: native sse", ErrUpstreamStreamIncomplete)
		}
		return nil
	}
}

func isSSEBlankLine(line []byte) bool {
	return len(bytes.TrimRight(line, "\r\n")) == 0
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

// StreamResponse reads an SSE stream from a provider and forwards it to the client.
// It translates provider-specific chunks to OpenAI-compatible format using the translator.
func (h *Handler) StreamResponse(ctx context.Context, w http.ResponseWriter, providerResp *http.Response, translator models.ProviderTranslator) error {
	return h.streamResponse(ctx, w, providerResp, translator, nil)
}

func (h *Handler) StreamResponseWithObserver(
	ctx context.Context,
	w http.ResponseWriter,
	providerResp *http.Response,
	translator models.ProviderTranslator,
	observer ChunkObserver,
) error {
	return h.streamResponse(ctx, w, providerResp, translator, observer)
}

func (h *Handler) streamResponse(
	ctx context.Context,
	w http.ResponseWriter,
	providerResp *http.Response,
	translator models.ProviderTranslator,
	observer ChunkObserver,
) error {
	if providerResp != nil && providerResp.StatusCode != http.StatusOK {
		defer providerResp.Body.Close()
		b, _ := io.ReadAll(providerResp.Body)
		return upstreamProviderError(providerResp.StatusCode, translator.Name(), b)
	}
	defer providerResp.Body.Close()

	controller := http.NewResponseController(w)

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Transfer-Encoding", "chunked")
	w.WriteHeader(http.StatusOK)
	if err := controller.Flush(); err != nil {
		return fmt.Errorf("failed to flush stream headers: %w", err)
	}

	scanner := bufio.NewScanner(providerResp.Body)
	// Increase buffer size for large chunks
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	envelope := newChatStreamEnvelopeNormalizer(translator.DefaultModel())

	for scanner.Scan() {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		line := scanner.Text()

		// Skip empty lines and comments
		if line == "" || strings.HasPrefix(line, ":") {
			continue
		}

		// SSE format: "data: <json>"
		if !strings.HasPrefix(line, "data: ") {
			// For Anthropic which uses "event: " lines
			if strings.HasPrefix(line, "event: ") {
				continue
			}
			continue
		}

		data := strings.TrimPrefix(line, "data: ")

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
			return fmt.Errorf("failed to parse stream chunk: %w", err)
		}

		if chunk == nil && !streamDone {
			continue
		}

		if chunk != nil {
			chunk = envelope.normalize(chunk)
			if observer != nil {
				observer(chunk)
			}

			// Marshal to OpenAI-compatible format
			chunkJSON, err := json.Marshal(chunk)
			if err != nil {
				log.Error().Err(err).Msg("failed to marshal stream chunk")
			} else {
				if err := writeSSEFrame(w, controller, chunkJSON, "stream chunk"); err != nil {
					return err
				}
			}
		}

		if streamDone {
			if err := writeSSEFrame(w, controller, []byte("[DONE]"), "done frame"); err != nil {
				return err
			}
			return nil
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("stream scanner error: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	return fmt.Errorf("%w: sse", ErrUpstreamStreamIncomplete)
}

// StreamAnthropicResponse handles Anthropic's different SSE format.
// Anthropic sends "event: type\ndata: json\n\n" pairs.
func (h *Handler) StreamAnthropicResponse(ctx context.Context, w http.ResponseWriter, providerResp *http.Response, translator models.ProviderTranslator) error {
	return h.streamAnthropicResponse(ctx, w, providerResp, translator, nil, true)
}

func (h *Handler) StreamAnthropicResponseWithObserver(
	ctx context.Context,
	w http.ResponseWriter,
	providerResp *http.Response,
	translator models.ProviderTranslator,
	observer ChunkObserver,
) error {
	return h.streamAnthropicResponse(ctx, w, providerResp, translator, observer, true)
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
		defer providerResp.Body.Close()
		b, _ := io.ReadAll(providerResp.Body)
		return upstreamProviderError(providerResp.StatusCode, translator.Name(), b)
	}
	defer providerResp.Body.Close()

	controller := http.NewResponseController(w)

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Transfer-Encoding", "chunked")
	w.WriteHeader(http.StatusOK)
	if err := controller.Flush(); err != nil {
		return fmt.Errorf("failed to flush stream headers: %w", err)
	}

	reader := bufio.NewReader(providerResp.Body)
	var eventType string
	envelope := newChatStreamEnvelopeNormalizer(translator.DefaultModel())

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		line, err := reader.ReadString('\n')
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return fmt.Errorf("stream read error: %w", err)
		}

		line = strings.TrimRight(line, "\r\n")

		if line == "" {
			eventType = ""
			continue
		}

		if strings.HasPrefix(line, "event: ") {
			eventType = strings.TrimPrefix(line, "event: ")
			continue
		}

		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")

			payload := []byte(data)
			if eventType != "" {
				trimmed := bytes.TrimSpace(payload)
				if len(trimmed) > 0 && trimmed[0] == '{' {
					var obj map[string]json.RawMessage
					if err := json.Unmarshal(trimmed, &obj); err == nil {
						if _, ok := obj["type"]; !ok {
							obj["type"] = json.RawMessage(fmt.Sprintf("%q", eventType))
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
				return fmt.Errorf("failed to parse anthropic stream chunk: %w", parseErr)
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
				if !includeUsage && chunk.Usage != nil {
					copyChunk := *chunk
					copyChunk.Usage = nil
					clientChunk = &copyChunk
				}
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
				if err := writeSSEFrame(w, controller, []byte("[DONE]"), "done frame"); err != nil {
					return err
				}
				return nil
			}
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
