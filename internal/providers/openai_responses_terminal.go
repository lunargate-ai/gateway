package providers

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/lunargate-ai/gateway/pkg/models"
)

type openAIResponsesError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

func decodeOpenAIResponsesResponse(data []byte) (*models.ResponsesResponse, *openAIResponsesError, error) {
	var response models.ResponsesResponse
	if err := json.Unmarshal(data, &response); err != nil {
		return nil, nil, err
	}

	var terminal struct {
		Error *openAIResponsesError `json:"error"`
	}
	if err := json.Unmarshal(data, &terminal); err != nil {
		return nil, nil, err
	}
	return &response, terminal.Error, nil
}

func openAIResponsesTerminalFinishReason(
	response *models.ResponsesResponse,
	failure *openAIResponsesError,
	statusOverride string,
) (*string, error) {
	if response == nil {
		return nil, openAIResponsesInvalidStatusError("missing response object")
	}

	embeddedStatus := normalizeOpenAIResponsesStatus(response.Status)
	status := normalizeOpenAIResponsesStatus(statusOverride)
	if status == "" {
		status = embeddedStatus
	} else if embeddedStatus != "" && embeddedStatus != status {
		return nil, openAIResponsesInvalidStatusError(fmt.Sprintf(
			"terminal event status %q conflicts with response status %q",
			status,
			embeddedStatus,
		))
	}

	switch status {
	case "completed":
		finishReason := "stop"
		for i := range response.Output {
			if strings.EqualFold(strings.TrimSpace(response.Output[i].Type), "function_call") {
				finishReason = "tool_calls"
				break
			}
		}
		return &finishReason, nil
	case "incomplete":
		if response.IncompleteDetails == nil {
			return nil, openAIResponsesInvalidStatusError("incomplete response is missing incomplete_details.reason")
		}
		switch strings.ToLower(strings.TrimSpace(response.IncompleteDetails.Reason)) {
		case models.ResponsesIncompleteReasonMaxOutputTokens, models.ResponsesIncompleteReasonMaxMessages:
			finishReason := "length"
			return &finishReason, nil
		case models.ResponsesIncompleteReasonContentFilter:
			finishReason := "content_filter"
			return &finishReason, nil
		default:
			return nil, openAIResponsesInvalidStatusError(fmt.Sprintf(
				"unsupported incomplete_details.reason %q",
				response.IncompleteDetails.Reason,
			))
		}
	case "failed":
		return nil, openAIResponsesFailureError("response_failed", "OpenAI Responses request failed", failure)
	case "cancelled":
		return nil, openAIResponsesFailureError("response_cancelled", "OpenAI Responses request was cancelled", failure)
	case "":
		return nil, openAIResponsesInvalidStatusError("response is missing a terminal status")
	default:
		return nil, openAIResponsesInvalidStatusError(fmt.Sprintf("response ended with non-terminal or unknown status %q", status))
	}
}

func normalizeOpenAIResponsesStatus(status string) string {
	status = strings.ToLower(strings.TrimSpace(status))
	if status == "canceled" {
		return "cancelled"
	}
	return status
}

func openAIResponsesFailureError(errorType, fallbackMessage string, failure *openAIResponsesError) *ProviderError {
	message := fallbackMessage
	if failure != nil {
		if upstreamMessage := strings.TrimSpace(failure.Message); upstreamMessage != "" {
			message = upstreamMessage
		} else if code := strings.TrimSpace(failure.Code); code != "" {
			message = fallbackMessage + " (" + code + ")"
		}
	}
	return &ProviderError{
		StatusCode: http.StatusBadGateway,
		Message:    message,
		Type:       errorType,
		Provider:   "openai",
	}
}

func openAIResponsesInvalidStatusError(detail string) *ProviderError {
	return &ProviderError{
		StatusCode: http.StatusBadGateway,
		Message:    "invalid OpenAI Responses terminal state: " + detail,
		Type:       "invalid_response_status",
		Provider:   "openai",
	}
}
