package streaming

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"

	"github.com/lunargate-ai/gateway/pkg/models"
)

const (
	ChatStreamErrorMessage = "The upstream provider failed while streaming the response."
	ChatStreamErrorType    = "upstream_error"
	ChatStreamErrorCode    = "upstream_stream_error"
)

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
