package streaming

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/pkg/models"
	"github.com/rs/zerolog/log"
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

type ChunkObserver func(chunk *models.StreamChunk)

// Handler manages SSE streaming between providers and clients.
type Handler struct{}

// NewHandler creates a new streaming handler.
func NewHandler() *Handler {
	return &Handler{}
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

	flusher, ok := w.(http.Flusher)
	if !ok {
		return fmt.Errorf("response writer does not support flushing")
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Transfer-Encoding", "chunked")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	defer providerResp.Body.Close()

	scanner := bufio.NewScanner(providerResp.Body)
	// Increase buffer size for large chunks
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

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
			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
			return nil
		}

		// Parse through the translator
		chunk, err := translator.ParseStreamChunk([]byte(data))
		if err != nil {
			if err == providers.ErrStreamDone {
				fmt.Fprintf(w, "data: [DONE]\n\n")
				flusher.Flush()
				return nil
			}
			return fmt.Errorf("failed to parse stream chunk: %w", err)
		}

		if chunk == nil {
			continue
		}

		if observer != nil {
			observer(chunk)
		}

		// Marshal to OpenAI-compatible format
		chunkJSON, err := json.Marshal(chunk)
		if err != nil {
			log.Error().Err(err).Msg("failed to marshal stream chunk")
			continue
		}

		fmt.Fprintf(w, "data: %s\n\n", chunkJSON)
		flusher.Flush()
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("stream scanner error: %w", err)
	}

	return nil
}

// StreamAnthropicResponse handles Anthropic's different SSE format.
// Anthropic sends "event: type\ndata: json\n\n" pairs.
func (h *Handler) StreamAnthropicResponse(ctx context.Context, w http.ResponseWriter, providerResp *http.Response, translator models.ProviderTranslator) error {
	return h.streamAnthropicResponse(ctx, w, providerResp, translator, nil)
}

func (h *Handler) StreamAnthropicResponseWithObserver(
	ctx context.Context,
	w http.ResponseWriter,
	providerResp *http.Response,
	translator models.ProviderTranslator,
	observer ChunkObserver,
) error {
	return h.streamAnthropicResponse(ctx, w, providerResp, translator, observer)
}

func (h *Handler) streamAnthropicResponse(
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

	flusher, ok := w.(http.Flusher)
	if !ok {
		return fmt.Errorf("response writer does not support flushing")
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Transfer-Encoding", "chunked")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	defer providerResp.Body.Close()

	reader := bufio.NewReader(providerResp.Body)
	var eventType string

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
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
			if parseErr != nil {
				if parseErr == providers.ErrStreamDone {
					fmt.Fprintf(w, "data: [DONE]\n\n")
					flusher.Flush()
					return nil
				}
				return fmt.Errorf("failed to parse anthropic stream chunk: %w", parseErr)
			}

			if chunk == nil {
				continue
			}

			if observer != nil {
				observer(chunk)
			}

			chunkJSON, marshalErr := json.Marshal(chunk)
			if marshalErr != nil {
				log.Error().Err(marshalErr).Msg("failed to marshal stream chunk")
				continue
			}

			fmt.Fprintf(w, "data: %s\n\n", chunkJSON)
			flusher.Flush()
		}
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
