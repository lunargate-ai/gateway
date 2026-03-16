package streaming

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/pkg/models"
	"github.com/rs/zerolog/log"
)

func (h *Handler) StreamNDJSONResponse(ctx context.Context, w http.ResponseWriter, providerResp *http.Response, translator models.ProviderTranslator) error {
	return h.streamNDJSONResponse(ctx, w, providerResp, translator, nil)
}

func (h *Handler) StreamNDJSONResponseWithObserver(
	ctx context.Context,
	w http.ResponseWriter,
	providerResp *http.Response,
	translator models.ProviderTranslator,
	observer ChunkObserver,
) error {
	return h.streamNDJSONResponse(ctx, w, providerResp, translator, observer)
}

func (h *Handler) streamNDJSONResponse(
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
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)

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
		if err != nil {
			if err == providers.ErrStreamDone {
				fmt.Fprintf(w, "data: [DONE]\n\n")
				flusher.Flush()
				return nil
			}
			return fmt.Errorf("failed to parse ndjson stream chunk: %w", err)
		}
		if chunk == nil {
			continue
		}

		if observer != nil {
			observer(chunk)
		}

		chunkJSON, err := json.Marshal(chunk)
		if err != nil {
			log.Error().Err(err).Msg("failed to marshal stream chunk")
			continue
		}

		fmt.Fprintf(w, "data: %s\n\n", chunkJSON)
		flusher.Flush()

		isDone := false
		for _, c := range chunk.Choices {
			if c.FinishReason != nil {
				isDone = true
				break
			}
		}
		if isDone {
			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
			return nil
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("stream scanner error: %w", err)
	}

	return nil
}
