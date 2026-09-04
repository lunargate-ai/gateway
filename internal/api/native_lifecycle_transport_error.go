package api

import (
	"context"
	"errors"
	"net/http"
	"strings"

	"github.com/rs/zerolog/log"
)

func writeNativeLifecycleTransportError(
	w http.ResponseWriter,
	parent context.Context,
	provider string,
	err error,
	logMessage string,
	genericMessage string,
) {
	provider = strings.TrimSpace(provider)
	status, message, errType, clientCancelled := classifyNativeLifecycleTransportError(parent, err, genericMessage)
	if clientCancelled {
		log.Info().Str("provider", provider).Msg(logMessage)
	} else {
		log.Error().Err(err).Str("provider", provider).Msg(logMessage)
	}
	writeError(w, status, message, errType)
}

func classifyNativeLifecycleTransportError(
	parent context.Context,
	err error,
	genericMessage string,
) (status int, message string, errType string, clientCancelled bool) {
	switch {
	case isUpstreamTTFTTimeout(err):
		return http.StatusBadGateway, "provider timed out waiting for first byte", "upstream_timeout", false
	case isUpstreamTotalTimeout(err):
		return http.StatusBadGateway, "provider timed out before full response completed", "upstream_timeout", false
	case isParentContextError(parent, err):
		return 499, "client disconnected", "client_cancelled", true
	default:
		return http.StatusBadGateway, genericMessage, "provider_error", false
	}
}

func isParentContextError(parent context.Context, err error) bool {
	if parent == nil || err == nil {
		return false
	}
	parentErr := parent.Err()
	return parentErr != nil &&
		(errors.Is(parentErr, context.Canceled) || errors.Is(parentErr, context.DeadlineExceeded)) &&
		errors.Is(err, parentErr)
}
