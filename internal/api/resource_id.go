package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/go-chi/chi/v5"
)

// validOpaqueResourceID reports whether an API resource identifier can be
// used verbatim. IDs are opaque: internal whitespace is significant and must
// never be normalized into a different resource key.
func validOpaqueResourceID(id string) bool {
	return id != "" && id == strings.TrimSpace(id)
}

// optionalOpaqueResourceID decodes an optional JSON resource identifier
// without changing it. An omitted or null field is absent; every supplied
// string must be non-empty and free of surrounding whitespace.
func optionalOpaqueResourceID(raw json.RawMessage, param string) (string, bool, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return "", false, nil
	}
	var id string
	if err := json.Unmarshal(trimmed, &id); err != nil {
		return "", true, fmt.Errorf("%s must be a string", param)
	}
	if !validOpaqueResourceID(id) {
		return "", true, fmt.Errorf("%s must be a non-empty identifier without surrounding whitespace", param)
	}
	return id, true, nil
}

func clientURLResourceID(w http.ResponseWriter, r *http.Request, param string) (string, bool) {
	id := chi.URLParam(r, param)
	if validOpaqueResourceID(id) {
		return id, true
	}
	code := "invalid_value"
	writeErrorDetail(
		w,
		http.StatusBadRequest,
		fmt.Sprintf("%s must be a non-empty identifier without surrounding whitespace", param),
		"invalid_request_error",
		&param,
		&code,
	)
	return "", false
}

func clientOptionalResourceID(w http.ResponseWriter, id string, param string) (string, bool) {
	if id == "" || validOpaqueResourceID(id) {
		return id, true
	}
	code := "invalid_value"
	writeErrorDetail(
		w,
		http.StatusBadRequest,
		fmt.Sprintf("%s must be an identifier without surrounding whitespace", param),
		"invalid_request_error",
		&param,
		&code,
	)
	return "", false
}
