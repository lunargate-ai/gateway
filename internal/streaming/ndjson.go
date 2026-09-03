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
)

func (h *Handler) StreamNDJSONResponse(ctx context.Context, w http.ResponseWriter, providerResp *http.Response, translator models.ProviderTranslator) error {
	return h.streamNDJSONResponse(ctx, w, providerResp, translator, nil, false)
}

func (h *Handler) StreamNDJSONResponseWithObserver(
	ctx context.Context,
	w http.ResponseWriter,
	providerResp *http.Response,
	translator models.ProviderTranslator,
	observer ChunkObserver,
) error {
	return h.streamNDJSONResponse(ctx, w, providerResp, translator, observer, false)
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

		line, readErr := readNDJSONRecord(reader)
		if readErr != nil {
			if errors.Is(readErr, io.EOF) {
				if ctxErr := ctx.Err(); ctxErr != nil {
					return ctxErr
				}
				if !output.started {
					return output.fail(ErrUpstreamStreamEmpty)
				}
				return output.fail(fmt.Errorf("%w: ndjson", ErrUpstreamStreamIncomplete))
			}
			streamErr := fmt.Errorf("ndjson stream read error: %w", readErr)
			return output.fail(streamErr)
		}

		line = bytes.TrimSpace(line)
		if len(line) == 0 {
			continue
		}

		chunk, err := translator.ParseStreamChunk(line)
		streamDone := errors.Is(err, providers.ErrStreamDone)
		if err != nil && !streamDone {
			streamErr := fmt.Errorf("failed to parse ndjson stream chunk: %w", err)
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
			usage.add(chunk)
			clientChunk := withoutStreamUsage(chunk)

			chunkJSON, err = json.Marshal(clientChunk)
			if err != nil {
				return output.fail(fmt.Errorf("failed to marshal ndjson stream chunk: %w", err))
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

		isDone := false
		for _, c := range chunk.Choices {
			if c.FinishReason != nil {
				isDone = true
				break
			}
		}
		if isDone {
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

func readNDJSONRecord(reader *bufio.Reader) ([]byte, error) {
	if reader == nil {
		return nil, errors.New("NDJSON reader is required")
	}

	var record bytes.Buffer
	for {
		fragment, readErr := reader.ReadSlice('\n')
		if readErr == nil {
			delimiterBytes := 1
			if len(fragment) >= 2 && fragment[len(fragment)-2] == '\r' {
				delimiterBytes = 2
			}
			fragment = fragment[:len(fragment)-delimiterBytes]
		}
		if len(fragment) > MaxStreamRecordBytes-record.Len() {
			return nil, ErrStreamRecordTooLarge
		}
		_, _ = record.Write(fragment)

		if readErr == nil {
			return record.Bytes(), nil
		}
		if errors.Is(readErr, bufio.ErrBufferFull) {
			continue
		}
		if errors.Is(readErr, io.EOF) {
			if record.Len() == 0 {
				return nil, readErr
			}
			return record.Bytes(), nil
		}
		return nil, readErr
	}
}
