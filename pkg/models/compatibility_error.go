package models

import (
	"fmt"
	"strings"
)

// CompatibilityError reports a client-requested field that cannot be
// represented faithfully by the resolved provider contract.
type CompatibilityError struct {
	Field    string
	Provider string
	Reason   string
}

func (e *CompatibilityError) Error() string {
	if e == nil {
		return "unsupported provider feature"
	}
	field := strings.TrimSpace(e.Field)
	provider := strings.TrimSpace(e.Provider)
	reason := strings.TrimSpace(e.Reason)
	if reason != "" {
		return fmt.Sprintf("field %q is not supported by provider %q: %s", field, provider, reason)
	}
	return fmt.Sprintf("field %q is not supported by provider %q", field, provider)
}
