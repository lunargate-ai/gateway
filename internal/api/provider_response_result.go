package api

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/pkg/models"
)

type parsedProviderResponseKey struct{}

type parsedProviderResponse struct {
	provider                string
	chat                    *models.UnifiedResponse
	chatCompletionBindingID string
	embeddings              *models.EmbeddingsResponse
}

func parseChatProviderResponse(
	translator models.ProviderTranslator,
	resp *http.Response,
) (*models.UnifiedResponse, error) {
	parsed, err := translator.ParseResponse(resp)
	if err != nil {
		return nil, &providerResponseParseError{cause: err}
	}
	if parsed == nil {
		return nil, &providerResponseParseError{cause: errors.New("provider translator returned no chat response")}
	}
	return parsed, nil
}

func parseEmbeddingsProviderResponse(
	translator embeddingsTranslator,
	resp *http.Response,
) (*models.EmbeddingsResponse, error) {
	parsed, err := translator.ParseEmbeddingsResponse(resp)
	if err != nil {
		return nil, &providerResponseParseError{cause: err}
	}
	if parsed == nil {
		return nil, &providerResponseParseError{cause: errors.New("provider translator returned no embeddings response")}
	}
	return parsed, nil
}

func responseWithParsedChat(
	resp *http.Response,
	provider string,
	parsed *models.UnifiedResponse,
	chatCompletionBindingID string,
) *http.Response {
	if resp == nil || resp.Request == nil || parsed == nil {
		return resp
	}
	value := parsedProviderResponse{
		provider:                strings.TrimSpace(provider),
		chat:                    parsed,
		chatCompletionBindingID: chatCompletionBindingID,
	}
	ctx := context.WithValue(resp.Request.Context(), parsedProviderResponseKey{}, value)
	resp.Request = resp.Request.WithContext(ctx)
	return resp
}

func parsedChatFromResponse(resp *http.Response, provider string) (*models.UnifiedResponse, bool) {
	value, ok := parsedResponseFromResponse(resp, provider)
	return value.chat, ok && value.chat != nil
}

func parsedChatCompletionBindingID(resp *http.Response, provider string) string {
	value, ok := parsedResponseFromResponse(resp, provider)
	if !ok {
		return ""
	}
	return value.chatCompletionBindingID
}

func responseWithParsedEmbeddings(resp *http.Response, provider string, parsed *models.EmbeddingsResponse) *http.Response {
	if resp == nil || resp.Request == nil || parsed == nil {
		return resp
	}
	value := parsedProviderResponse{provider: strings.TrimSpace(provider), embeddings: parsed}
	ctx := context.WithValue(resp.Request.Context(), parsedProviderResponseKey{}, value)
	resp.Request = resp.Request.WithContext(ctx)
	return resp
}

func parsedEmbeddingsFromResponse(resp *http.Response, provider string) (*models.EmbeddingsResponse, bool) {
	value, ok := parsedResponseFromResponse(resp, provider)
	return value.embeddings, ok && value.embeddings != nil
}

func parsedResponseFromResponse(resp *http.Response, provider string) (parsedProviderResponse, bool) {
	if resp == nil || resp.Request == nil {
		return parsedProviderResponse{}, false
	}
	value, ok := resp.Request.Context().Value(parsedProviderResponseKey{}).(parsedProviderResponse)
	if !ok || value.provider != strings.TrimSpace(provider) {
		return parsedProviderResponse{}, false
	}
	return value, true
}

type providerResponseParseError struct {
	cause error
}

func (e *providerResponseParseError) Error() string {
	if e == nil || e.cause == nil {
		return "failed to parse provider response"
	}
	return "failed to parse provider response: " + e.cause.Error()
}

func (e *providerResponseParseError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.cause
}

func invalidProviderResponseStatus(provider string, status int) *providers.ProviderError {
	return &providers.ProviderError{
		StatusCode: http.StatusBadGateway,
		Provider:   strings.TrimSpace(provider),
		Type:       "invalid_response_status",
		Message:    fmt.Sprintf("upstream returned status %d; expected 200", status),
	}
}

func classifyProviderResponseError(err error) (status int, errType string, message string, ok bool) {
	if isUpstreamTTFTTimeout(err) {
		return http.StatusBadGateway, "upstream_timeout", "provider timed out waiting for first byte", true
	}
	if isUpstreamTotalTimeout(err) {
		return http.StatusBadGateway, "upstream_timeout", "provider timed out before full response completed", true
	}

	var providerErr *providers.ProviderError
	if errors.As(err, &providerErr) && providerErr != nil {
		status = providerErr.StatusCode
		if status == 0 {
			status = http.StatusBadGateway
		}
		errType = strings.TrimSpace(providerErr.Type)
		if errType == "" {
			errType = "provider_error"
		}
		message = strings.TrimSpace(providerErr.Message)
		if message == "" {
			message = providerErr.Error()
		}
		return status, errType, message, true
	}

	var parseErr *providerResponseParseError
	if errors.As(err, &parseErr) && parseErr != nil {
		return http.StatusBadGateway, "provider_error", parseErr.Error(), true
	}
	return 0, "", "", false
}
