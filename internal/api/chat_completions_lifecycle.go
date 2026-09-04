package api

import (
	"crypto/subtle"
	"fmt"
	"net/http"
	"net/url"
	"sort"
	"strings"
)

type chatCompletionBindingResolutionError struct {
	message string
	param   string
	code    string
}

func (e *chatCompletionBindingResolutionError) Error() string {
	if e == nil {
		return "chat completion provider binding is invalid"
	}
	return e.message
}

func (h *Handler) providerSupportsChatCompletionsLifecycle(provider string) bool {
	if h == nil || h.registry == nil {
		return false
	}
	snapshot, ok := h.registry.Snapshot(strings.TrimSpace(provider))
	return ok &&
		strings.EqualFold(strings.TrimSpace(snapshot.ProviderType), "openai") &&
		snapshot.Capabilities.ChatCompletionsLifecycle
}

func (h *Handler) validateChatCompletionProvider(provider string) (chatCompletionBinding, error) {
	provider = strings.TrimSpace(provider)
	if provider == "" {
		return chatCompletionBinding{}, chatCompletionBindingError(
			"chat completion provider is required",
			"provider",
			"missing_required_parameter",
		)
	}
	if h == nil || h.registry == nil {
		return chatCompletionBinding{}, chatCompletionBindingError(
			"chat completion provider registry is unavailable",
			"provider",
			"provider_not_found",
		)
	}
	snapshot, ok := h.registry.Snapshot(provider)
	if !ok {
		return chatCompletionBinding{}, chatCompletionBindingError(
			fmt.Sprintf("requested provider %q is not configured", provider),
			"provider",
			"provider_not_found",
		)
	}
	if !strings.EqualFold(strings.TrimSpace(snapshot.ProviderType), "openai") {
		return chatCompletionBinding{}, chatCompletionBindingError(
			fmt.Sprintf("provider %q cannot proxy native Chat Completions lifecycle", provider),
			"provider",
			"unsupported_feature",
		)
	}
	if !snapshot.Capabilities.ChatCompletionsLifecycle {
		return chatCompletionBinding{}, chatCompletionBindingError(
			fmt.Sprintf("provider %q does not enable chat_completions_lifecycle", provider),
			"provider",
			"unsupported_feature",
		)
	}
	fingerprint, ok := h.responseAccountFingerprint(provider)
	if !ok {
		return chatCompletionBinding{}, chatCompletionBindingError(
			fmt.Sprintf("provider %q has no HTTP account configuration", provider),
			"provider",
			"provider_not_found",
		)
	}
	return chatCompletionBinding{
		Provider:           provider,
		AccountFingerprint: fingerprint,
	}, nil
}

func (h *Handler) retainNativeChatCompletionBinding(
	completionID string,
	clientRequestType string,
	upstreamRequestType string,
	headers http.Header,
) bool {
	if h == nil || h.chatCompletionBindings == nil ||
		canonicalAPIRequestType(clientRequestType) != requestTypeChatCompletions ||
		canonicalAPIRequestType(upstreamRequestType) != requestTypeChatCompletions {
		return false
	}

	binding, err := h.validateChatCompletionProvider(headers.Get("X-LunarGate-Provider"))
	if err != nil {
		return false
	}
	binding.Route = strings.TrimSpace(headers.Get("X-LunarGate-Route"))
	binding.Model = strings.TrimSpace(headers.Get("X-LunarGate-Model"))
	return h.chatCompletionBindings.claim(completionID, binding).retained()
}

func (h *Handler) boundChatCompletionBinding(r *http.Request, completionID string) (chatCompletionBinding, bool, error) {
	if h == nil || h.chatCompletionBindings == nil {
		return chatCompletionBinding{}, false, nil
	}
	requestedProvider := strings.TrimSpace(r.Header.Get("X-LunarGate-Provider"))
	binding, lookup := h.chatCompletionBindings.lookup(completionID)
	if lookup == ownerLookupConflict {
		if requestedProvider != "" {
			return chatCompletionBinding{}, false, nil
		}
		return chatCompletionBinding{}, false, chatCompletionBindingError(
			fmt.Sprintf("chat completion %q has conflicting provider ownership", completionID),
			"completion_id",
			"provider_binding_conflict",
		)
	}
	if lookup != ownerLookupBound {
		return chatCompletionBinding{}, false, nil
	}

	if requestedProvider != "" && requestedProvider != binding.Provider {
		return chatCompletionBinding{}, false, chatCompletionBindingError(
			fmt.Sprintf("chat completion %q belongs to provider %q, not %q", completionID, binding.Provider, requestedProvider),
			"provider",
			"invalid_value",
		)
	}
	currentBinding, err := h.validateChatCompletionProvider(binding.Provider)
	if err != nil {
		return chatCompletionBinding{}, false, err
	}
	if subtle.ConstantTimeCompare(
		[]byte(binding.AccountFingerprint),
		[]byte(currentBinding.AccountFingerprint),
	) != 1 {
		return chatCompletionBinding{}, false, chatCompletionBindingError(
			fmt.Sprintf("provider account configuration changed for chat completion %q", completionID),
			"provider",
			"provider_binding_stale",
		)
	}
	return binding, true, nil
}

func (h *Handler) explicitChatCompletionBinding(r *http.Request) (chatCompletionBinding, bool, error) {
	provider := strings.TrimSpace(r.Header.Get("X-LunarGate-Provider"))
	if provider == "" {
		return chatCompletionBinding{}, false, nil
	}
	binding, err := h.validateChatCompletionProvider(provider)
	if err != nil {
		return chatCompletionBinding{}, false, err
	}
	binding.Model = strings.TrimSpace(r.Header.Get("X-LunarGate-Model"))
	return binding, true, nil
}

func (h *Handler) chatCompletionListBinding(r *http.Request) (chatCompletionBinding, error) {
	if binding, ok, err := h.explicitChatCompletionBinding(r); ok || err != nil {
		return binding, err
	}
	if h == nil || h.registry == nil {
		return chatCompletionBinding{}, chatCompletionBindingError(
			"chat completion provider registry is unavailable",
			"provider",
			"provider_not_found",
		)
	}

	providers := h.registry.List()
	sort.Strings(providers)
	capable := make([]string, 0, len(providers))
	for _, provider := range providers {
		if h.providerSupportsChatCompletionsLifecycle(provider) {
			capable = append(capable, provider)
		}
	}
	switch len(capable) {
	case 0:
		return chatCompletionBinding{}, chatCompletionBindingError(
			"no configured OpenAI provider enables chat_completions_lifecycle",
			"provider",
			"unsupported_feature",
		)
	case 1:
		return h.validateChatCompletionProvider(capable[0])
	default:
		return chatCompletionBinding{}, chatCompletionBindingError(
			"multiple providers enable chat_completions_lifecycle; select one with X-LunarGate-Provider",
			"provider",
			"ambiguous_provider",
		)
	}
}

// ListStoredChatCompletions proxies the native stored Chat Completions list
// to one explicitly and deterministically selected OpenAI provider account.
func (h *Handler) ListStoredChatCompletions(w http.ResponseWriter, r *http.Request) {
	if _, validAfter := clientOptionalResourceID(w, r.URL.Query().Get("after"), "after"); !validAfter {
		return
	}
	body, ok := readResponseOperationBody(w, r)
	if !ok {
		return
	}
	binding, err := h.chatCompletionListBinding(r)
	if err != nil {
		writeChatCompletionBindingResolutionError(w, err)
		return
	}
	h.proxyChatCompletionLifecycleRequest(w, r, binding, http.MethodGet, "chat/completions", body, false, "", "")
}

// RetrieveStoredChatCompletion proxies a stored Chat Completion lookup to its
// bound provider account, or to an explicit provider after the binding expires.
func (h *Handler) RetrieveStoredChatCompletion(w http.ResponseWriter, r *http.Request) {
	h.handleChatCompletionIDRequest(w, r, http.MethodGet, "", false)
}

// UpdateStoredChatCompletion proxies a metadata update without normalizing its
// body so additive upstream fields remain intact.
func (h *Handler) UpdateStoredChatCompletion(w http.ResponseWriter, r *http.Request) {
	h.handleChatCompletionIDRequest(w, r, http.MethodPost, "", false)
}

// DeleteStoredChatCompletion removes the owner binding only after the native
// provider confirms a successful deletion.
func (h *Handler) DeleteStoredChatCompletion(w http.ResponseWriter, r *http.Request) {
	h.handleChatCompletionIDRequest(w, r, http.MethodDelete, "", true)
}

// ListStoredChatCompletionMessages proxies the native messages collection for
// one stored Chat Completion.
func (h *Handler) ListStoredChatCompletionMessages(w http.ResponseWriter, r *http.Request) {
	h.handleChatCompletionIDRequest(w, r, http.MethodGet, "/messages", false)
}

func (h *Handler) handleChatCompletionIDRequest(
	w http.ResponseWriter,
	r *http.Request,
	method string,
	suffix string,
	deleteBindingOnSuccess bool,
) {
	completionID, validID := clientURLResourceID(w, r, "completion_id")
	if !validID {
		return
	}
	if suffix == "/messages" {
		if _, validAfter := clientOptionalResourceID(w, r.URL.Query().Get("after"), "after"); !validAfter {
			return
		}
	}
	body, ok := readResponseOperationBody(w, r)
	if !ok {
		return
	}

	binding, bound, err := h.boundChatCompletionBinding(r, completionID)
	if err != nil {
		writeChatCompletionBindingResolutionError(w, err)
		return
	}
	if !bound {
		binding, bound, err = h.explicitChatCompletionBinding(r)
		if err != nil {
			writeChatCompletionBindingResolutionError(w, err)
			return
		}
	}
	if !bound {
		writeChatCompletionNotFound(w, completionID)
		return
	}

	path := "chat/completions/" + url.PathEscape(completionID) + suffix
	h.proxyChatCompletionLifecycleRequest(w, r, binding, method, path, body, deleteBindingOnSuccess, completionID, suffix)
}

func (h *Handler) proxyChatCompletionLifecycleRequest(
	w http.ResponseWriter,
	r *http.Request,
	binding chatCompletionBinding,
	method string,
	path string,
	body []byte,
	deleteBindingOnSuccess bool,
	completionID string,
	suffix string,
) {
	rawQuery := ""
	if r != nil && r.URL != nil {
		rawQuery = r.URL.RawQuery
	}
	responseBinding := responseBinding{
		Provider:            binding.Provider,
		Route:               binding.Route,
		Model:               binding.Model,
		UpstreamRequestType: requestTypeChatCompletions,
		AccountFingerprint:  binding.AccountFingerprint,
	}
	response, err := h.nativeResponseRequest(r.Context(), method, responseBinding, path, rawQuery, body, r.Header)
	if err != nil {
		writeNativeLifecycleTransportError(
			w,
			r.Context(),
			binding.Provider,
			err,
			"native Chat Completions lifecycle request failed",
			"upstream Chat Completions provider request failed",
		)
		return
	}
	contract := nativeResponseBodyContract{}
	if completionID != "" && suffix == "" {
		contract = nativeResponseBodyContract{
			expectedID:     completionID,
			expectedObject: "chat.completion",
			requireID:      true,
			requireJSON:    true,
		}
		if deleteBindingOnSuccess {
			contract.expectedObject = "chat.completion.deleted"
			contract.requireDeleted = true
			contract.onValidated = func() {
				if h != nil && h.chatCompletionBindings != nil {
					h.chatCompletionBindings.deleteIfOwned(completionID, binding)
				}
			}
		}
	}
	h.proxyNativeResponseWithContract(w, r, responseBinding, response, contract)
}

func chatCompletionBindingError(message string, param string, code string) error {
	return &chatCompletionBindingResolutionError{message: message, param: param, code: code}
}

func writeChatCompletionBindingResolutionError(w http.ResponseWriter, err error) {
	resolutionErr, ok := err.(*chatCompletionBindingResolutionError)
	if !ok || resolutionErr == nil {
		writeError(w, http.StatusBadRequest, err.Error(), "invalid_request_error")
		return
	}
	param := resolutionErr.param
	code := resolutionErr.code
	writeErrorDetail(w, http.StatusBadRequest, resolutionErr.message, "invalid_request_error", &param, &code)
}

func writeChatCompletionNotFound(w http.ResponseWriter, completionID string) {
	param := "completion_id"
	code := "completion_not_found"
	writeErrorDetail(
		w,
		http.StatusNotFound,
		fmt.Sprintf("chat completion %q was not found", completionID),
		"invalid_request_error",
		&param,
		&code,
	)
}
