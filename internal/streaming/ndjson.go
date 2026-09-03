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

	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/pkg/models"
	"github.com/rs/zerolog/log"
)

func (h *Handler) StreamNDJSONResponse(ctx context.Context, w http.ResponseWriter, providerResp *http.Response, translator models.ProviderTranslator) error {
	return h.streamNDJSONResponse(ctx, w, providerResp, translator, nil, true)
}

func (h *Handler) StreamNDJSONResponseWithObserver(
	ctx context.Context,
	w http.ResponseWriter,
	providerResp *http.Response,
	translator models.ProviderTranslator,
	observer ChunkObserver,
) error {
	return h.streamNDJSONResponse(ctx, w, providerResp, translator, observer, true)
}

// StreamNDJSONResponseWithObserverAndUsage controls whether translated usage
// is exposed to a Chat Completions client while always reporting it to the
// gateway observer.
func (h *Handler) StreamNDJSONResponseWithObserverAndUsage(
	ctx context.Context,
	w http.ResponseWriter,
	providerResp *http.Response,
	translator models.ProviderTranslator,
	observer ChunkObserver,
	includeUsage bool,
) error {
	return h.streamNDJSONResponse(ctx, w, providerResp, translator, observer, includeUsage)
}

func (h *Handler) streamNDJSONResponse(
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
	w.WriteHeader(http.StatusOK)
	if err := controller.Flush(); err != nil {
		return fmt.Errorf("failed to flush stream headers: %w", err)
	}

	scanner := bufio.NewScanner(providerResp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	envelope := newChatStreamEnvelopeNormalizer(translator.DefaultModel())

	for scanner.Scan() {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}

		chunk, err := translator.ParseStreamChunk(line)
		streamDone := errors.Is(err, providers.ErrStreamDone)
		if err != nil && !streamDone {
			return fmt.Errorf("failed to parse ndjson stream chunk: %w", err)
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

			chunkJSON, err := json.Marshal(clientChunk)
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

		isDone := false
		for _, c := range chunk.Choices {
			if c.FinishReason != nil {
				isDone = true
				break
			}
		}
		if isDone {
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
	return fmt.Errorf("%w: ndjson", ErrUpstreamStreamIncomplete)
}
