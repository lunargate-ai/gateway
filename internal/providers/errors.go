package providers

import (
	"errors"
	"fmt"
)

// ErrStreamDone signals that an SSE stream has completed.
var ErrStreamDone = errors.New("stream done")

// ProviderError represents an error returned by an LLM provider.
type ProviderError struct {
	StatusCode int
	Message    string
	Type       string
	Provider   string
}

func (e *ProviderError) Error() string {
	return fmt.Sprintf("provider %s returned %d: %s", e.Provider, e.StatusCode, e.Message)
}

// IsRetryable returns true if the error is worth retrying.
func (e *ProviderError) IsRetryable() bool {
	switch e.StatusCode {
	case 429, 500, 502, 503, 504:
		return true
	default:
		return false
	}
}
