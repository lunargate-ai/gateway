package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/safeurl"
	"github.com/lunargate-ai/gateway/internal/streaming"
	"github.com/rs/zerolog/log"
)

type responseNativeCapability int

const maxNativeLifecycleResponseBytes = 16 << 20

type nativeResponseBodyContract struct {
	expectedID     string
	expectedObject string
	requireID      bool
	requireJSON    bool
	requireDeleted bool
	onValidated    func()
}

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
	organization := providerConfig.Organization
	apiKey := providerConfig.APIKey
	switch strings.ToLower(providerType) {
	case "anthropic":
		organization = ""
	case "ollama":
		organization = ""
		apiKey = ""
	}
	return conversationAccountFingerprint(
		providerType,
		baseURL,
		organization,
		apiKey,
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
	if h == nil || h.responseBindings == nil {
		return responseBinding{}, false, nil
	}
	binding, lookup := h.responseBindings.lookup(responseID)
	if lookup == ownerLookupConflict {
		if h.responsesState != nil {
			h.responsesState.discard(responseID)
		}
		// A caller may still recover a known native object by explicitly naming
		// its provider, but implicit resolution must never choose between owners.
		if r != nil && strings.TrimSpace(r.Header.Get("X-LunarGate-Provider")) != "" {
			return responseBinding{}, false, nil
		}
		return responseBinding{}, false, responseOwnerConflictError(responseID, "response_id")
	}
	if lookup != ownerLookupBound {
		return responseBinding{}, false, nil
	}
	if err := h.validateClaimedResponseOwner(r, responseID, binding, capability, !binding.LocalSnapshot); err != nil {
		return responseBinding{}, false, err
	}
	return binding, true, nil
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

	baseURL := strings.TrimSpace(providerCfg.BaseURL)
	if baseURL == "" {
		baseURL = strings.TrimSpace(providerSnapshot.Translator.BaseURL())
	}
	if baseURL == "" {
		return nil, fmt.Errorf("native response provider %q has no base URL", provider)
	}
	endpoint, err := safeurl.JoinHTTPPathAndRawQuery(baseURL, rawQuery, strings.TrimLeft(path, "/"))
	if err != nil {
		return nil, fmt.Errorf("failed to build native response endpoint for %s: %w", provider, err)
	}

	request, err := http.NewRequestWithContext(ctx, method, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create native response request for %s: %w", provider, err)
	}
	if len(body) > 0 {
		request.Header.Set("Content-Type", "application/json")
	}
	providerType := strings.ToLower(strings.TrimSpace(providerCfg.Type))
	if providerType == "" {
		providerType = strings.ToLower(strings.TrimSpace(providerSnapshot.ProviderType))
	}
	copyForwardedRequestHeader(request.Header, inboundHeaders, "Accept")
	switch providerType {
	case "anthropic":
		copyForwardedRequestHeader(request.Header, inboundHeaders, "Anthropic-Beta")
		request.Header.Set("x-api-key", providerCfg.APIKey)
		if version := strings.TrimSpace(providerCfg.APIVersion); version != "" {
			request.Header.Set("anthropic-version", version)
		}
	default:
		copyForwardedRequestHeader(request.Header, inboundHeaders, "Idempotency-Key")
		copyForwardedRequestHeader(request.Header, inboundHeaders, "OpenAI-Beta")
		if apiKey := strings.TrimSpace(providerCfg.APIKey); apiKey != "" {
			request.Header.Set("Authorization", "Bearer "+apiKey)
		}
		if organization := strings.TrimSpace(providerCfg.Organization); organization != "" {
			request.Header.Set("OpenAI-Organization", organization)
		}
	}

	// Native lifecycle and utility operations are stateful. Follow-up requests
	// must stay pinned to the selected provider and must never be replayed by
	// net/http against a redirect target.
	singleHopClient := *clientCfg.client
	singleHopClient.CheckRedirect = func(_ *http.Request, _ []*http.Request) error {
		return http.ErrUseLastResponse
	}
	clientCfg.client = &singleHopClient
	return doProviderRequest(request, clientCfg, provider, "failed to call native response provider")
}

func copyForwardedRequestHeader(destination http.Header, source http.Header, name string) {
	for headerName, values := range source {
		if !strings.EqualFold(headerName, name) {
			continue
		}
		for _, value := range values {
			if strings.TrimSpace(value) != "" {
				destination.Add(name, value)
			}
		}
	}
}

func (h *Handler) proxyNativeResponse(w http.ResponseWriter, r *http.Request, binding responseBinding, response *http.Response) {
	h.proxyNativeResponseWithContract(w, r, binding, response, nativeResponseBodyContract{})
}

func (h *Handler) proxyNativeResponseForID(
	w http.ResponseWriter,
	r *http.Request,
	binding responseBinding,
	response *http.Response,
	expectedResponseID string,
) {
	h.proxyNativeResponseWithContract(w, r, binding, response, nativeResponseBodyContract{
		expectedID:     expectedResponseID,
		expectedObject: "response",
		requireID:      true,
	})
}

func (h *Handler) proxyNativeResponseWithContract(
	w http.ResponseWriter,
	r *http.Request,
	binding responseBinding,
	response *http.Response,
	contract nativeResponseBodyContract,
) {
	if response == nil || response.Body == nil {
		writeError(w, http.StatusBadGateway, "provider returned an empty response", "provider_error")
		return
	}

	contentType := strings.ToLower(strings.TrimSpace(response.Header.Get("Content-Type")))
	if strings.HasPrefix(contentType, "text/event-stream") {
		if contract.requireJSON {
			_ = response.Body.Close()
			responseBindingHeaders(w, binding)
			writeError(w, http.StatusBadGateway, "unexpected streaming response for lifecycle request", "provider_error")
			return
		}
		if response.StatusCode != http.StatusOK {
			_ = response.Body.Close()
			responseBindingHeaders(w, binding)
			writeError(w, http.StatusBadGateway, "unexpected status for lifecycle stream", "provider_error")
			return
		}
		if !strings.EqualFold(strings.TrimSpace(binding.UpstreamRequestType), requestTypeResponses) {
			_ = response.Body.Close()
			responseBindingHeaders(w, binding)
			writeError(w, http.StatusBadGateway, "unexpected streaming response for lifecycle request", "provider_error")
			return
		}
		streamer := h.streamer
		if streamer == nil {
			streamer = streaming.NewHandler()
		}
		proxy := newResponsesStreamProxy(w)
		proxy.requestContext = r.Context()
		proxy.responseID = contract.expectedID
		proxy.enableNativePassthrough()
		responseBindingHeaders(proxy, binding)
		copyHeaders(proxy.Header(), response.Header)
		streamErr := streamer.ProxySSEWithDataTransformer(
			r.Context(),
			proxy,
			response,
			binding.Provider,
			func(event streaming.SSEEvent) bool {
				terminal, ok := parseNativeResponsesStreamTerminal(event)
				if ok {
					proxy.recordNativeTerminal(terminal)
				}
				return ok
			},
			proxy.transformNativeEventData,
		)
		if streamErr != nil {
			proxy.RecordStreamError(streamErr)
		}
		if err := proxy.finalize(); err != nil {
			if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				return
			}
			if !proxy.headersSent {
				responseBindingHeaders(w, binding)
				writeError(w, http.StatusBadGateway, "failed to stream responses event payload", "provider_error")
				return
			}
			log.Warn().Err(err).Str("provider", binding.Provider).Msg("native response lifecycle stream terminated")
		}
		return
	}

	responseBindingHeaders(w, binding)
	copyHeaders(w.Header(), response.Header)
	defer response.Body.Close()
	if contract.validatesJSON() &&
		response.StatusCode >= http.StatusOK && response.StatusCode < http.StatusMultipleChoices {
		body, err := readValidatedNativeLifecycleResponse(response.Body, contract)
		if err != nil {
			if r != nil && r.Context().Err() != nil {
				return
			}
			writeError(w, http.StatusBadGateway, "native response provider returned an invalid response object", "provider_error")
			return
		}
		if contract.onValidated != nil {
			contract.onValidated()
		}
		w.WriteHeader(response.StatusCode)
		if _, err := w.Write(body); err != nil {
			log.Warn().Err(err).Str("provider", binding.Provider).Msg("failed to write native response lifecycle payload")
		}
		return
	}
	w.WriteHeader(response.StatusCode)
	if _, err := io.Copy(w, response.Body); err != nil {
		log.Warn().Err(err).Str("provider", binding.Provider).Msg("failed to proxy native response lifecycle body")
	}
}

func (c nativeResponseBodyContract) validatesJSON() bool {
	return c.requireJSON || c.requireID || c.requireDeleted || c.expectedID != "" || strings.TrimSpace(c.expectedObject) != ""
}

func readValidatedNativeLifecycleResponse(body io.Reader, contract nativeResponseBodyContract) ([]byte, error) {
	payload, err := io.ReadAll(io.LimitReader(body, maxNativeLifecycleResponseBytes+1))
	if err != nil {
		return nil, err
	}
	if len(payload) > maxNativeLifecycleResponseBytes {
		return nil, errors.New("native response lifecycle payload exceeds 16 MiB limit")
	}

	var response map[string]json.RawMessage
	if err := decodeJSONStrict(bytes.NewReader(payload), &response); err != nil || response == nil {
		return nil, errors.New("native response lifecycle payload must contain one JSON object")
	}
	if expectedObject := strings.TrimSpace(contract.expectedObject); expectedObject != "" {
		object, present, err := responsesEventNonEmptyString(response, "object")
		if err != nil || !present || object != expectedObject {
			return nil, errors.New("native response lifecycle payload has an invalid object kind")
		}
	}
	if contract.requireID || contract.expectedID != "" {
		responseID, present, err := responsesEventNonEmptyString(response, "id")
		if err != nil || !present {
			return nil, errors.New("native response lifecycle payload requires a non-empty string id")
		}
		if contract.expectedID != "" && responseID != contract.expectedID {
			return nil, errors.New("native response lifecycle payload id does not match the requested response")
		}
	}
	if contract.requireDeleted {
		deletedRaw, present := response["deleted"]
		if !present {
			return nil, errors.New("native response lifecycle payload requires deleted=true")
		}
		var deleted bool
		if err := decodeJSONStrict(bytes.NewReader(deletedRaw), &deleted); err != nil || !deleted {
			return nil, errors.New("native response lifecycle payload requires deleted=true")
		}
	}
	return payload, nil
}
