package api

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/url"
	"strings"

	"github.com/rs/zerolog/log"
)

const maxNativeConversationCreateCaptureBytes = 1 << 20

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
	return "conversations/" + url.PathEscape(strings.TrimSpace(conversationID))
}

func nativeConversationItemsPath(conversationID string) string {
	return nativeConversationPath(conversationID) + "/items"
}

func nativeConversationItemPath(conversationID, itemID string) string {
	return nativeConversationItemsPath(conversationID) + "/" + url.PathEscape(strings.TrimSpace(itemID))
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
		writeNativeConversationTransportError(w, binding, err)
		return
	}
	h.proxyNativeConversationResponse(w, binding, response, nil)
}

func (h *Handler) proxyNativeConversationCreate(
	w http.ResponseWriter,
	r *http.Request,
	binding conversationBinding,
	body []byte,
) {
	response, err := h.makeNativeConversationRequest(r, binding, http.MethodPost, "conversations", body)
	if err != nil {
		writeNativeConversationTransportError(w, binding, err)
		return
	}
	capture := &boundedCaptureWriter{limit: maxNativeConversationCreateCaptureBytes}
	status, copyErr := h.proxyNativeConversationResponse(w, binding, response, capture)
	if copyErr != nil || status < http.StatusOK || status >= http.StatusMultipleChoices || capture.truncated {
		return
	}
	if conversationID := nativeConversationID(capture.Bytes()); conversationID != "" {
		h.retainNativeConversationBinding(conversationID, binding)
	}
}

func (h *Handler) deleteNativeConversation(
	w http.ResponseWriter,
	r *http.Request,
	binding conversationBinding,
	conversationID string,
) {
	response, err := h.makeNativeConversationRequest(r, binding, http.MethodDelete, nativeConversationPath(conversationID), nil)
	if err != nil {
		writeNativeConversationTransportError(w, binding, err)
		return
	}
	status := response.StatusCode
	h.proxyNativeConversationResponse(w, binding, response, nil)
	if status >= http.StatusOK && status < http.StatusMultipleChoices && h != nil && h.conversationBindings != nil {
		h.conversationBindings.delete(conversationID)
	}
}

func (h *Handler) proxyNativeConversationResponse(
	w http.ResponseWriter,
	binding conversationBinding,
	response *http.Response,
	capture io.Writer,
) (int, error) {
	if response == nil || response.Body == nil {
		writeError(w, http.StatusBadGateway, "provider returned an empty response", "provider_error")
		return http.StatusBadGateway, nil
	}
	defer response.Body.Close()
	conversationBindingHeaders(w, binding)
	copyHeaders(w.Header(), response.Header)
	w.WriteHeader(response.StatusCode)

	destination := io.Writer(w)
	if capture != nil {
		destination = io.MultiWriter(w, capture)
	}
	_, err := io.Copy(destination, response.Body)
	if err != nil {
		log.Warn().Err(err).Str("provider", binding.Provider).Msg("failed to proxy native conversation body")
	}
	return response.StatusCode, err
}

func writeNativeConversationTransportError(w http.ResponseWriter, binding conversationBinding, err error) {
	writeNativeResponseTransportError(w, responseBinding{Provider: binding.Provider}, err)
}

func nativeConversationID(body []byte) string {
	var envelope struct {
		ID string `json:"id"`
	}
	decoder := json.NewDecoder(bytes.NewReader(body))
	if err := decoder.Decode(&envelope); err != nil {
		return ""
	}
	var extra json.RawMessage
	if err := decoder.Decode(&extra); err != io.EOF {
		return ""
	}
	if !validNativeConversationID(envelope.ID) {
		return ""
	}
	return strings.TrimSpace(envelope.ID)
}

type boundedCaptureWriter struct {
	bytes.Buffer
	limit     int
	truncated bool
}

func (w *boundedCaptureWriter) Write(p []byte) (int, error) {
	if w == nil {
		return len(p), nil
	}
	remaining := w.limit - w.Len()
	if remaining <= 0 {
		if len(p) > 0 {
			w.truncated = true
		}
		return len(p), nil
	}
	writeLen := len(p)
	if writeLen > remaining {
		writeLen = remaining
		w.truncated = true
	}
	if writeLen > 0 {
		_, _ = w.Buffer.Write(p[:writeLen])
	}
	return len(p), nil
}
