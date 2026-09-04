package streaming

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"

	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/pkg/models"
)

const (
	ChatStreamErrorMessage = "The upstream provider failed while streaming the response."
	ChatStreamErrorType    = "upstream_error"
	ChatStreamErrorCode    = "upstream_stream_error"
)

// chatStreamOutput delays committing the downstream response until a complete
// upstream record has been parsed successfully. Once committed, upstream
// failures are represented by the canonical Chat error frame and [DONE].
type chatStreamOutput struct {
	ctx        context.Context
	w          http.ResponseWriter
	controller *http.ResponseController
	provider   string
	started    bool
}

func newChatStreamOutput(ctx context.Context, w http.ResponseWriter, provider string) *chatStreamOutput {
	return &chatStreamOutput{
		ctx:        ctx,
		w:          w,
		controller: http.NewResponseController(w),
		provider:   provider,
	}
}

func (o *chatStreamOutput) start() error {
	if o.started {
		return nil
	}
	if err := o.ctx.Err(); err != nil {
		return err
	}
	o.w.Header().Set("Content-Type", "text/event-stream")
	o.w.Header().Set("Cache-Control", "no-cache")
	o.w.WriteHeader(http.StatusOK)
	o.started = true
	if err := o.controller.Flush(); err != nil {
		return fmt.Errorf("failed to flush stream headers: %w", err)
	}
	return nil
}

func (o *chatStreamOutput) write(payload []byte, frameName string) error {
	if err := o.start(); err != nil {
		return err
	}
	return writeSSEFrame(o.w, o.controller, payload, frameName)
}

func (o *chatStreamOutput) fail(streamErr error) error {
	if streamErr == nil {
		return nil
	}
	if err := o.ctx.Err(); err != nil {
		return err
	}
	if !o.started {
		return errors.Join(streamErr, &providers.ProviderError{
			StatusCode: http.StatusBadGateway,
			Provider:   o.provider,
			Type:       "streaming_error",
			Message:    ChatStreamErrorMessage,
		})
	}
	return terminateChatStreamWithError(o.w, o.controller, streamErr)
}

func hasChatStreamPayload(chunk *models.StreamChunk) bool {
	if chunk == nil {
		return false
	}
	if len(chunk.Choices) > 0 ||
		chunk.Usage != nil ||
		chunk.ID != "" ||
		chunk.Object != "" ||
		chunk.Model != "" ||
		chunk.Created != 0 ||
		chunk.SystemFingerprint != "" {
		return true
	}

	// OpenAI-compatible providers may add extension-only events. Preserve them
	// while keeping a truly empty JSON object from committing the response.
	var rawFields map[string]json.RawMessage
	raw := bytes.TrimSpace(chunk.RawJSON)
	return len(raw) > 0 && json.Unmarshal(raw, &rawFields) == nil && len(rawFields) > 0
}

func terminateChatStreamWithError(
	w http.ResponseWriter,
	controller *http.ResponseController,
	streamErr error,
) error {
	code := ChatStreamErrorCode
	payload, marshalErr := json.Marshal(models.ErrorResponse{
		Error: models.ErrorDetail{
			Message: ChatStreamErrorMessage,
			Type:    ChatStreamErrorType,
			Code:    &code,
		},
	})
	if marshalErr != nil {
		return errors.Join(streamErr, fmt.Errorf("failed to marshal stream error frame: %w", marshalErr))
	}
	if err := writeSSEFrame(w, controller, payload, "error frame"); err != nil {
		return errors.Join(streamErr, err)
	}
	if err := writeSSEFrame(w, controller, []byte("[DONE]"), "done frame"); err != nil {
		return errors.Join(streamErr, err)
	}
	return streamErr
}
