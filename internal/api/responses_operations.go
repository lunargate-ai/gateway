package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/lunargate-ai/gateway/internal/modelid"
)

// CompactResponses proxies POST /v1/responses/compact to one explicitly and
// deterministically selected provider. It never uses the routing fallback or
// retry chain because compaction is stateful.
func (h *Handler) CompactResponses(w http.ResponseWriter, r *http.Request) {
	h.handleNativeResponseOperation(w, r, responseNativeCompaction, "responses/compact")
}

// CountResponseInputTokens proxies POST /v1/responses/input_tokens to one
// explicitly and deterministically selected provider.
func (h *Handler) CountResponseInputTokens(w http.ResponseWriter, r *http.Request) {
	h.handleNativeResponseOperation(w, r, responseNativeInputTokens, "responses/input_tokens")
}

func (h *Handler) handleNativeResponseOperation(
	w http.ResponseWriter,
	r *http.Request,
	capability responseNativeCapability,
	path string,
) {
	body, payload, ok := readResponseOperationPayload(w, r)
	if !ok {
		return
	}
	binding, upstreamBody, err := h.selectResponseOperationProvider(r, payload, body, capability)
	if err != nil {
		writeResponseBindingResolutionError(w, err)
		return
	}
	h.proxyResponseLifecycleRequest(w, r, binding, http.MethodPost, path, upstreamBody)
}

func readResponseOperationPayload(w http.ResponseWriter, r *http.Request) ([]byte, map[string]json.RawMessage, bool) {
	body, ok := readResponseOperationBody(w, r)
	if !ok {
		return nil, nil, false
	}
	var payload map[string]json.RawMessage
	if err := decodeJSONStrict(bytes.NewReader(body), &payload); err != nil {
		writeRequestDecodeError(w, err)
		return nil, nil, false
	}
	if payload == nil {
		writeError(w, http.StatusBadRequest, "request body must be a JSON object", "invalid_request_error")
		return nil, nil, false
	}
	return body, payload, true
}

func (h *Handler) selectResponseOperationProvider(
	r *http.Request,
	payload map[string]json.RawMessage,
	originalBody []byte,
	capability responseNativeCapability,
) (responseBinding, []byte, error) {
	if h == nil || h.registry == nil {
		return responseBinding{}, nil, responseOperationSelectionError("provider registry is unavailable", "provider", "provider_not_found")
	}

	headerProvider := strings.TrimSpace(r.Header.Get("X-LunarGate-Provider"))
	headerModel := strings.TrimSpace(r.Header.Get("X-LunarGate-Model"))
	bodyModel := parseJSONStringRaw(payload["model"])

	selectedProvider := headerProvider
	headerModelProvider, headerModelName, headerCanonical := modelid.SplitCanonical(headerModel)
	bodyModelProvider, bodyModelName, bodyCanonical := modelid.SplitCanonical(bodyModel)
	for _, provider := range []string{headerModelProvider, bodyModelProvider} {
		provider = strings.TrimSpace(provider)
		if provider == "" {
			continue
		}
		if selectedProvider != "" && selectedProvider != provider {
			return responseBinding{}, nil, responseOperationSelectionError(
				fmt.Sprintf("provider %q conflicts with canonical model provider %q", selectedProvider, provider),
				"provider",
				"invalid_value",
			)
		}
		selectedProvider = provider
	}
	if headerCanonical && bodyCanonical && (headerModelProvider != bodyModelProvider || headerModelName != bodyModelName) {
		return responseBinding{}, nil, responseOperationSelectionError(
			"X-LunarGate-Model conflicts with request model",
			"model",
			"invalid_value",
		)
	}
	normalizedHeaderModel := headerModel
	if headerCanonical {
		normalizedHeaderModel = headerModelName
	}
	normalizedBodyModel := bodyModel
	if bodyCanonical {
		normalizedBodyModel = bodyModelName
	}
	if normalizedHeaderModel != "" && normalizedBodyModel != "" && normalizedHeaderModel != normalizedBodyModel {
		return responseBinding{}, nil, responseOperationSelectionError(
			"X-LunarGate-Model conflicts with request model",
			"model",
			"invalid_value",
		)
	}

	if selectedProvider == "" {
		capable := make([]string, 0, 2)
		for _, provider := range h.registry.List() {
			if h.providerSupportsResponseCapability(provider, capability) {
				capable = append(capable, provider)
			}
		}
		switch len(capable) {
		case 0:
			return responseBinding{}, nil, responseOperationSelectionError(
				fmt.Sprintf("no configured provider enables %s", responseCapabilityName(capability)),
				"provider",
				"unsupported_feature",
			)
		case 1:
			selectedProvider = capable[0]
		default:
			return responseBinding{}, nil, responseOperationSelectionError(
				fmt.Sprintf("multiple providers enable %s; specify a canonical model or X-LunarGate-Provider", responseCapabilityName(capability)),
				"provider",
				"ambiguous_provider",
			)
		}
	}

	if _, ok := h.registry.Snapshot(selectedProvider); !ok {
		return responseBinding{}, nil, responseOperationSelectionError(
			fmt.Sprintf("requested provider %q is not configured", selectedProvider),
			"provider",
			"provider_not_found",
		)
	}
	if !h.providerSupportsResponseCapability(selectedProvider, capability) {
		return responseBinding{}, nil, responseOperationSelectionError(
			fmt.Sprintf("provider %q does not enable %s", selectedProvider, responseCapabilityName(capability)),
			"provider",
			"unsupported_feature",
		)
	}

	upstreamModel := bodyModel
	rewriteModel := false
	if bodyCanonical {
		upstreamModel = bodyModelName
		rewriteModel = true
	} else if bodyModel == "" && headerModel != "" {
		upstreamModel = headerModel
		if headerCanonical {
			upstreamModel = headerModelName
		}
		rewriteModel = true
	}

	upstreamBody := append([]byte(nil), originalBody...)
	if rewriteModel {
		rewritten := cloneResponsesRawMap(payload)
		encodedModel, err := json.Marshal(upstreamModel)
		if err != nil {
			return responseBinding{}, nil, fmt.Errorf("failed to encode response operation model: %w", err)
		}
		rewritten["model"] = encodedModel
		upstreamBody, err = json.Marshal(rewritten)
		if err != nil {
			return responseBinding{}, nil, fmt.Errorf("failed to encode response operation request: %w", err)
		}
	}

	canonicalModel := ""
	if upstreamModel != "" {
		canonicalModel = modelid.BuildCanonical(selectedProvider, upstreamModel)
	}
	return responseBinding{
		Provider:            selectedProvider,
		Model:               canonicalModel,
		UpstreamRequestType: requestTypeResponses,
	}, upstreamBody, nil
}

func responseOperationSelectionError(message string, param string, code string) error {
	return &responseBindingResolutionError{message: message, param: param, code: code}
}
