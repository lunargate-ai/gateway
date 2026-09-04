package api

import (
	"errors"
	"net/http"
	"strings"

	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/internal/resilience"
	"github.com/lunargate-ai/gateway/pkg/models"
)

func requestFailureFromError(err error) (int, string, string, bool) {
	if !resilience.IsRequestError(err) {
		return 0, "", "", false
	}

	status := http.StatusBadRequest
	errType := "invalid_request_error"
	message := strings.TrimSpace(err.Error())
	if message == "" {
		message = "invalid provider request"
	}

	var compatibilityErr *models.CompatibilityError
	if errors.As(err, &compatibilityErr) && compatibilityErr != nil {
		return status, errType, compatibilityErr.Error(), true
	}

	var providerErr *providers.ProviderError
	if errors.As(err, &providerErr) && providerErr != nil {
		if providerErr.StatusCode >= http.StatusBadRequest && providerErr.StatusCode < http.StatusInternalServerError {
			status = providerErr.StatusCode
		}
		if value := strings.TrimSpace(providerErr.Type); value != "" {
			errType = value
		}
		if value := strings.TrimSpace(providerErr.Message); value != "" {
			message = value
		}
	}

	return status, errType, message, true
}
