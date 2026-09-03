package api

import (
	"bytes"
	"context"
	"crypto/subtle"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/streaming"
	"github.com/rs/zerolog/log"
)

type responseNativeCapability int

const (
	responseNativeLifecycle responseNativeCapability = iota
	responseNativeCancellation
	responseNativeCompaction
	responseNativeInputTokens
)

func responseCapabilityName(capability responseNativeCapability) string {
	switch capability {
	case responseNativeCancellation:
		return "response_cancellation"
	case responseNativeCompaction:
		return "response_compaction"
	case responseNativeInputTokens:
		return "response_input_tokens"
	default:
		return "responses_lifecycle"
	}
}

func providerHasResponseCapability(capabilities config.ProviderCapabilities, capability responseNativeCapability) bool {
	switch capability {
	case responseNativeCancellation:
		return capabilities.ResponseCancellation
	case responseNativeCompaction:
		return capabilities.ResponseCompaction
	case responseNativeInputTokens:
		return capabilities.ResponseInputTokens
	default:
		return capabilities.ResponsesLifecycle
	}
}

func (h *Handler) providerSupportsResponseCapability(provider string, capability responseNativeCapability) bool {
	if h == nil || h.registry == nil {
		return false
	}
	capabilities, ok := h.registry.Capabilities(strings.TrimSpace(provider))
	return ok && providerHasResponseCapability(capabilities, capability)
}

func responseBindingFromHeaders(headers http.Header) responseBinding {
	return responseBinding{
		Provider:            strings.TrimSpace(headers.Get("X-LunarGate-Provider")),
		Route:               strings.TrimSpace(headers.Get("X-LunarGate-Route")),
		Model:               strings.TrimSpace(headers.Get("X-LunarGate-Model")),
		UpstreamRequestType: requestTypeResponses,
	}
}

func (h *Handler) retainNativeResponseBinding(responseID string, headers http.Header) bool {
	if h == nil || h.responseBindings == nil {
		return false
	}
	binding := responseBindingFromHeaders(headers)
	if binding.Provider == "" || !h.providerSupportsResponseCapability(binding.Provider, responseNativeLifecycle) {
		return false
	}
	fingerprint, ok := h.responseAccountFingerprint(binding.Provider)
	if !ok {
		return false
	}
	binding.AccountFingerprint = fingerprint
	return h.responseBindings.put(responseID, binding)
}

func (h *Handler) responseAccountFingerprint(provider string) (string, bool) {
	provider = strings.TrimSpace(provider)
	if provider == "" || h == nil || h.registry == nil || h.providerClients == nil {
		return "", false
	}
	providerSnapshot, ok := h.registry.Snapshot(provider)
	if !ok || providerSnapshot.Translator == nil {
		return "", false
	}
	_, providerConfig, ok := h.providerClients.Snapshot(provider)
	if !ok {
		return "", false
	}
	providerType := strings.TrimSpace(providerConfig.Type)
	if providerType == "" {
		providerType = strings.TrimSpace(providerSnapshot.ProviderType)
	}
	baseURL := strings.TrimSpace(providerConfig.BaseURL)
	if baseURL == "" {
		baseURL = strings.TrimSpace(providerSnapshot.Translator.BaseURL())
	}
	return conversationAccountFingerprint(
		providerType,
		baseURL,
		providerConfig.Organization,
		providerConfig.APIKey,
	), true
}

func responseBindingHeaders(w http.ResponseWriter, binding responseBinding) {
	if provider := strings.TrimSpace(binding.Provider); provider != "" {
		w.Header().Set("X-LunarGate-Provider", provider)
	}
	if route := strings.TrimSpace(binding.Route); route != "" {
		w.Header().Set("X-LunarGate-Route", route)
	}
	if model := strings.TrimSpace(binding.Model); model != "" {
		w.Header().Set("X-LunarGate-Model", model)
	}
}

type responseBindingResolutionError struct {
	message string
	param   string
	code    string
}

func (e *responseBindingResolutionError) Error() string {
	if e == nil {
		return "response provider binding is invalid"
	}
	return e.message
}

func (h *Handler) boundResponseBinding(r *http.Request, responseID string, capability responseNativeCapability) (responseBinding, bool, error) {
	if h != nil && h.responseBindings != nil {
		if binding, ok := h.responseBindings.get(responseID); ok {
			requestedProvider := strings.TrimSpace(r.Header.Get("X-LunarGate-Provider"))
			if requestedProvider != "" && requestedProvider != binding.Provider {
				return responseBinding{}, false, &responseBindingResolutionError{
					message: fmt.Sprintf("response %q belongs to provider %q, not %q", responseID, binding.Provider, requestedProvider),
					param:   "provider",
					code:    "invalid_value",
				}
			}
			if !h.providerSupportsResponseCapability(binding.Provider, capability) {
				return responseBinding{}, false, &responseBindingResolutionError{
					message: fmt.Sprintf("provider %q no longer enables %s", binding.Provider, responseCapabilityName(capability)),
					param:   "provider",
					code:    "unsupported_feature",
				}
			}
			currentFingerprint, fingerprintOK := h.responseAccountFingerprint(binding.Provider)
			if !fingerprintOK {
				return responseBinding{}, false, &responseBindingResolutionError{
					message: fmt.Sprintf("provider %q no longer has an HTTP account configuration", binding.Provider),
					param:   "provider",
					code:    "provider_not_found",
				}
			}
			if subtle.ConstantTimeCompare(
				[]byte(binding.AccountFingerprint),
				[]byte(currentFingerprint),
			) != 1 {
				return responseBinding{}, false, &responseBindingResolutionError{
					message: fmt.Sprintf("provider account configuration changed for response %q", responseID),
					param:   "provider",
					code:    "provider_binding_stale",
				}
			}
			return binding, true, nil
		}
	}
	return responseBinding{}, false, nil
}

func (h *Handler) explicitResponseBinding(r *http.Request, capability responseNativeCapability) (responseBinding, bool, error) {
	provider := strings.TrimSpace(r.Header.Get("X-LunarGate-Provider"))
	if provider == "" {
		return responseBinding{}, false, nil
	}
	if h == nil || h.registry == nil {
		return responseBinding{}, false, &responseBindingResolutionError{
			message: "response provider registry is unavailable",
			param:   "provider",
			code:    "provider_not_found",
		}
	}
	if _, ok := h.registry.Snapshot(provider); !ok {
		return responseBinding{}, false, &responseBindingResolutionError{
			message: fmt.Sprintf("requested provider %q is not configured", provider),
			param:   "provider",
			code:    "provider_not_found",
		}
	}
	if !h.providerSupportsResponseCapability(provider, capability) {
		return responseBinding{}, false, &responseBindingResolutionError{
			message: fmt.Sprintf("provider %q does not enable %s", provider, responseCapabilityName(capability)),
			param:   "provider",
			code:    "unsupported_feature",
		}
	}
	model := strings.TrimSpace(r.Header.Get("X-LunarGate-Model"))
	fingerprint, ok := h.responseAccountFingerprint(provider)
	if !ok {
		return responseBinding{}, false, &responseBindingResolutionError{
			message: fmt.Sprintf("provider %q has no HTTP account configuration", provider),
			param:   "provider",
			code:    "provider_not_found",
		}
	}
	return responseBinding{
		Provider:            provider,
		Model:               model,
		UpstreamRequestType: requestTypeResponses,
		AccountFingerprint:  fingerprint,
	}, true, nil
}

func (h *Handler) nativeResponseRequest(
	ctx context.Context,
	method string,
	binding responseBinding,
	path string,
	rawQuery string,
	body []byte,
	inboundHeaders http.Header,
) (*http.Response, error) {
	provider := strings.TrimSpace(binding.Provider)
	if provider == "" || h == nil || h.registry == nil {
		return nil, fmt.Errorf("native response provider is required")
	}
	providerSnapshot, ok := h.registry.Snapshot(provider)
	if !ok || providerSnapshot.Translator == nil {
		return nil, fmt.Errorf("native response provider %q is not configured", provider)
	}
	if h.providerClients == nil {
		return nil, fmt.Errorf("native response provider %q has no HTTP configuration", provider)
	}
	clientCfg, providerCfg, ok := h.providerClients.Snapshot(provider)
	if !ok {
		return nil, fmt.Errorf("native response provider %q has no HTTP configuration", provider)
	}

	baseURL := strings.TrimRight(strings.TrimSpace(providerCfg.BaseURL), "/")
	if baseURL == "" {
		baseURL = strings.TrimRight(strings.TrimSpace(providerSnapshot.Translator.BaseURL()), "/")
	}
	if baseURL == "" {
		return nil, fmt.Errorf("native response provider %q has no base URL", provider)
	}
	endpoint, err := url.Parse(baseURL + "/" + strings.TrimLeft(path, "/"))
	if err != nil {
		return nil, fmt.Errorf("failed to build native response URL for %s: %w", provider, err)
	}
	endpoint.RawQuery = rawQuery

	request, err := http.NewRequestWithContext(ctx, method, endpoint.String(), bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create native response request for %s: %w", provider, err)
	}
	if len(body) > 0 {
		request.Header.Set("Content-Type", "application/json")
	}
	for _, header := range []string{"Accept", "OpenAI-Beta", "Idempotency-Key"} {
		if value := strings.TrimSpace(inboundHeaders.Get(header)); value != "" {
			request.Header.Set(header, value)
		}
	}
	providerType := strings.ToLower(strings.TrimSpace(providerCfg.Type))
	if providerType == "" {
		providerType = strings.ToLower(strings.TrimSpace(providerSnapshot.ProviderType))
	}
	switch providerType {
	case "anthropic":
		request.Header.Set("x-api-key", providerCfg.APIKey)
		if version := strings.TrimSpace(providerCfg.APIVersion); version != "" {
			request.Header.Set("anthropic-version", version)
		}
	default:
		if apiKey := strings.TrimSpace(providerCfg.APIKey); apiKey != "" {
			request.Header.Set("Authorization", "Bearer "+apiKey)
		}
		if organization := strings.TrimSpace(providerCfg.Organization); organization != "" {
			request.Header.Set("OpenAI-Organization", organization)
		}
	}

	startedAt := time.Now()
	// Native lifecycle and utility operations are stateful. Follow-up requests
	// must stay pinned to the selected provider and must never be replayed by
	// net/http against a redirect target.
	singleHopClient := *clientCfg.client
	singleHopClient.CheckRedirect = func(_ *http.Request, _ []*http.Request) error {
		return http.ErrUseLastResponse
	}
	response, err := singleHopClient.Do(request)
	if err != nil {
		if isHTTPTimeoutError(err) {
			if clientCfg.mode == upstreamTimeoutModeTotal {
				return nil, fmt.Errorf("%w: provider %s", errUpstreamTotalTimeout, provider)
			}
			return nil, fmt.Errorf("%w: provider %s", errUpstreamTTFTTimeout, provider)
		}
		return nil, fmt.Errorf("failed to call native response provider %s: %w", provider, err)
	}
	if response.Request == nil {
		response.Request = request
	}

	remaining := clientCfg.timeout - time.Since(startedAt)
	if transport, ok := clientCfg.client.Transport.(*http.Transport); ok && transport.ResponseHeaderTimeout > 0 {
		remaining = transport.ResponseHeaderTimeout - time.Since(startedAt)
	}
	if remaining <= 0 {
		response.Body.Close()
		if clientCfg.mode == upstreamTimeoutModeTotal {
			return nil, fmt.Errorf("%w: provider %s", errUpstreamTotalTimeout, provider)
		}
		return nil, fmt.Errorf("%w: provider %s", errUpstreamTTFTTimeout, provider)
	}
	if clientCfg.mode == upstreamTimeoutModeTotal {
		response.Body = wrapBodyWithTotalTimeout(response.Body, remaining)
	} else {
		response.Body = wrapBodyWithTTFTTimeout(response.Body, remaining)
	}
	return response, nil
}

func (h *Handler) proxyNativeResponse(w http.ResponseWriter, r *http.Request, binding responseBinding, response *http.Response) {
	if response == nil || response.Body == nil {
		writeError(w, http.StatusBadGateway, "provider returned an empty response", "provider_error")
		return
	}
	responseBindingHeaders(w, binding)
	copyHeaders(w.Header(), response.Header)

	contentType := strings.ToLower(strings.TrimSpace(response.Header.Get("Content-Type")))
	if response.StatusCode >= http.StatusOK && response.StatusCode < http.StatusMultipleChoices && strings.HasPrefix(contentType, "text/event-stream") {
		streamer := h.streamer
		if streamer == nil {
			streamer = streaming.NewHandler()
		}
		err := streamer.ProxySSE(r.Context(), w, response, binding.Provider, func(event streaming.SSEEvent) bool {
			_, terminal := parseNativeResponsesStreamTerminal(event)
			return terminal
		})
		if err != nil {
			log.Warn().Err(err).Str("provider", binding.Provider).Msg("native response lifecycle stream terminated")
		}
		return
	}

	defer response.Body.Close()
	w.WriteHeader(response.StatusCode)
	if _, err := io.Copy(w, response.Body); err != nil {
		log.Warn().Err(err).Str("provider", binding.Provider).Msg("failed to proxy native response lifecycle body")
	}
}
