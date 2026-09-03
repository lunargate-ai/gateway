package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"

	"github.com/rs/zerolog/log"
)

const maxNativeConversationResponseBytes = 16 << 20

type nativeConversationResponseContract struct {
	object         string
	requireID      bool
	expectedID     string
	requireDeleted bool
}

// resolveConversationOwner deliberately checks retained native ownership before
// local state, then permits an explicit provider recovery path. It never picks
// a provider implicitly for an unknown or expired conversation ID.
func (h *Handler) resolveConversationOwner(
	r *http.Request,
	conversationID string,
) (conversationBinding, bool, bool, error) {
	if binding, ok, err := h.boundConversationBinding(r, conversationID); err != nil {
		return conversationBinding{}, false, false, err
	} else if ok {
		return binding, true, false, nil
	}
	if h != nil && h.conversationsState != nil {
		if _, ok := h.conversationsState.get(conversationID); ok {
			return conversationBinding{}, false, true, nil
		}
	}
	if binding, ok, err := h.explicitConversationBinding(r); err != nil {
		return conversationBinding{}, false, false, err
	} else if ok {
		return binding, true, false, nil
	}
	return conversationBinding{}, false, false, nil
}

func nativeConversationPath(conversationID string) string {
	return "conversations/" + url.PathEscape(conversationID)
}

func nativeConversationItemsPath(conversationID string) string {
	return nativeConversationPath(conversationID) + "/items"
}

func nativeConversationItemPath(conversationID, itemID string) string {
	return nativeConversationItemsPath(conversationID) + "/" + url.PathEscape(itemID)
}

func (h *Handler) makeNativeConversationRequest(
	r *http.Request,
	binding conversationBinding,
	method string,
	path string,
	body []byte,
) (*http.Response, error) {
	rawQuery := ""
	if r != nil && r.URL != nil {
		rawQuery = r.URL.RawQuery
	}
	return h.nativeResponseRequest(
		r.Context(),
		method,
		responseBinding{Provider: binding.Provider, UpstreamRequestType: requestTypeResponses},
		path,
		rawQuery,
		body,
		r.Header,
	)
}

func (h *Handler) proxyNativeConversationRequest(
	w http.ResponseWriter,
	r *http.Request,
	binding conversationBinding,
	method string,
	path string,
	body []byte,
) {
	response, err := h.makeNativeConversationRequest(r, binding, method, path, body)
	if err != nil {
		writeNativeConversationTransportError(w, r.Context(), binding, err)
		return
	}
	if contract, ok := nativeConversationRequestContract(method, path); ok {
		h.proxyValidatedNativeConversationResponse(w, r.Context(), binding, response, contract)
		return
	}
	h.proxyNativeConversationResponse(w, binding, response)
}

func (h *Handler) proxyNativeConversationCreate(
	w http.ResponseWriter,
	r *http.Request,
	binding conversationBinding,
	body []byte,
) {
	response, err := h.makeNativeConversationRequest(r, binding, http.MethodPost, "conversations", body)
	if err != nil {
		writeNativeConversationTransportError(w, r.Context(), binding, err)
		return
	}
	_, conversationID, valid := h.proxyValidatedNativeConversationResponse(
		w,
		r.Context(),
		binding,
		response,
		nativeConversationResponseContract{object: "conversation", requireID: true},
	)
	if !valid {
		return
	}
	h.retainNativeConversationBinding(conversationID, binding)
}

func (h *Handler) deleteNativeConversation(
	w http.ResponseWriter,
	r *http.Request,
	binding conversationBinding,
	conversationID string,
) {
	response, err := h.makeNativeConversationRequest(r, binding, http.MethodDelete, nativeConversationPath(conversationID), nil)
	if err != nil {
		writeNativeConversationTransportError(w, r.Context(), binding, err)
		return
	}
	_, _, valid := h.proxyValidatedNativeConversationResponse(
		w,
		r.Context(),
		binding,
		response,
		nativeConversationResponseContract{
			object:         "conversation.deleted",
			requireID:      true,
			expectedID:     conversationID,
			requireDeleted: true,
		},
	)
	if valid && h != nil && h.conversationBindings != nil {
		h.conversationBindings.deleteIfOwned(conversationID, binding)
	}
}

func nativeConversationRequestContract(method string, path string) (nativeConversationResponseContract, bool) {
	parts := strings.Split(path, "/")
	if len(parts) < 2 || len(parts) > 4 || parts[0] != "conversations" {
		return nativeConversationResponseContract{}, false
	}
	conversationID, ok := nativeConversationPathID(parts[1])
	if !ok {
		return nativeConversationResponseContract{}, false
	}

	switch len(parts) {
	case 2:
		if method != http.MethodGet && method != http.MethodPost {
			return nativeConversationResponseContract{}, false
		}
		return nativeConversationResponseContract{
			object:     "conversation",
			requireID:  true,
			expectedID: conversationID,
		}, true
	case 3:
		if parts[2] != "items" || (method != http.MethodGet && method != http.MethodPost) {
			return nativeConversationResponseContract{}, false
		}
		return nativeConversationResponseContract{object: "list"}, true
	case 4:
		if parts[2] != "items" {
			return nativeConversationResponseContract{}, false
		}
		itemID, ok := nativeConversationPathID(parts[3])
		if !ok {
			return nativeConversationResponseContract{}, false
		}
		switch method {
		case http.MethodGet:
			return nativeConversationResponseContract{
				requireID:  true,
				expectedID: itemID,
			}, true
		case http.MethodDelete:
			return nativeConversationResponseContract{
				object:     "conversation",
				requireID:  true,
				expectedID: conversationID,
			}, true
		}
	}
	return nativeConversationResponseContract{}, false
}

func nativeConversationPathID(encodedID string) (string, bool) {
	if encodedID == "" {
		return "", false
	}
	resourceID, err := url.PathUnescape(encodedID)
	if err != nil {
		return "", false
	}
	return resourceID, validOpaqueResourceID(resourceID)
}

func (h *Handler) proxyValidatedNativeConversationResponse(
	w http.ResponseWriter,
	parent context.Context,
	binding conversationBinding,
	response *http.Response,
	contract nativeConversationResponseContract,
) (int, string, bool) {
	if response == nil || response.Body == nil {
		conversationBindingHeaders(w, binding)
		writeError(w, http.StatusBadGateway, "provider returned an empty response", "provider_error")
		return http.StatusBadGateway, "", false
	}
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		status, _ := h.proxyNativeConversationResponse(w, binding, response)
		return status, "", false
	}
	defer response.Body.Close()

	payload, conversationID, err := readValidatedNativeConversationResponse(response.Body, contract)
	if err != nil {
		if parent != nil && parent.Err() != nil {
			return 499, "", false
		}
		conversationBindingHeaders(w, binding)
		writeError(w, http.StatusBadGateway, "native conversation provider returned an invalid response object", "provider_error")
		log.Warn().Err(err).Str("provider", binding.Provider).Msg("native conversation response validation failed")
		return http.StatusBadGateway, "", false
	}

	conversationBindingHeaders(w, binding)
	copyHeaders(w.Header(), response.Header)
	w.WriteHeader(response.StatusCode)
	if _, err := w.Write(payload); err != nil {
		log.Warn().Err(err).Str("provider", binding.Provider).Msg("failed to write native conversation payload")
		return response.StatusCode, "", false
	}
	return response.StatusCode, conversationID, true
}

func readValidatedNativeConversationResponse(
	body io.Reader,
	contract nativeConversationResponseContract,
) ([]byte, string, error) {
	payload, err := io.ReadAll(io.LimitReader(body, maxNativeConversationResponseBytes+1))
	if err != nil {
		return nil, "", fmt.Errorf("failed to read native conversation response: %w", err)
	}
	if len(payload) > maxNativeConversationResponseBytes {
		return nil, "", errors.New("native conversation response exceeds 16 MiB limit")
	}

	var envelope map[string]json.RawMessage
	if err := decodeJSONStrict(bytes.NewReader(payload), &envelope); err != nil || envelope == nil {
		return nil, "", errors.New("native conversation response must contain one JSON object")
	}
	conversationID := ""
	if contract.requireID || contract.expectedID != "" {
		conversationID, err = requiredNativeConversationString(envelope, "id")
		if err != nil {
			return nil, "", err
		}
		if trimmedID := strings.TrimSpace(conversationID); trimmedID == "" || conversationID != trimmedID {
			return nil, "", errors.New("native conversation response requires a non-empty string id without surrounding whitespace")
		}
	}
	if contract.object != "" {
		object, err := requiredNativeConversationString(envelope, "object")
		if err != nil || object != contract.object {
			return nil, "", fmt.Errorf("native conversation response object must be %q", contract.object)
		}
	}
	if contract.expectedID != "" && conversationID != contract.expectedID {
		return nil, "", errors.New("native conversation response id does not match the requested conversation")
	}
	if contract.requireDeleted {
		deletedRaw, ok := envelope["deleted"]
		if !ok {
			return nil, "", errors.New("native conversation delete response requires deleted=true")
		}
		var deleted bool
		if err := decodeJSONStrict(bytes.NewReader(deletedRaw), &deleted); err != nil || !deleted {
			return nil, "", errors.New("native conversation delete response requires deleted=true")
		}
	}
	return payload, conversationID, nil
}

func requiredNativeConversationString(envelope map[string]json.RawMessage, field string) (string, error) {
	raw, ok := envelope[field]
	if !ok {
		return "", fmt.Errorf("native conversation response requires string %s", field)
	}
	var value string
	if err := decodeJSONStrict(bytes.NewReader(raw), &value); err != nil {
		return "", fmt.Errorf("native conversation response requires string %s", field)
	}
	return value, nil
}

func (h *Handler) proxyNativeConversationResponse(
	w http.ResponseWriter,
	binding conversationBinding,
	response *http.Response,
) (int, error) {
	if response == nil || response.Body == nil {
		writeError(w, http.StatusBadGateway, "provider returned an empty response", "provider_error")
		return http.StatusBadGateway, nil
	}
	defer response.Body.Close()
	conversationBindingHeaders(w, binding)
	copyHeaders(w.Header(), response.Header)
	w.WriteHeader(response.StatusCode)

	_, err := io.Copy(w, response.Body)
	if err != nil {
		log.Warn().Err(err).Str("provider", binding.Provider).Msg("failed to proxy native conversation body")
	}
	return response.StatusCode, err
}

func writeNativeConversationTransportError(w http.ResponseWriter, parent context.Context, binding conversationBinding, err error) {
	writeNativeResponseTransportError(w, parent, responseBinding{Provider: binding.Provider}, err)
}
