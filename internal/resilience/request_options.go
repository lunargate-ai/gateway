package resilience

import "context"

type retryDisabledContextKey struct{}

// WithRetryDisabled marks a request context so the retrier performs only one
// attempt while keeping normal retryable-status handling for fallback logic.
func WithRetryDisabled(ctx context.Context) context.Context {
	return context.WithValue(ctx, retryDisabledContextKey{}, true)
}

func retryDisabled(ctx context.Context) bool {
	disabled, _ := ctx.Value(retryDisabledContextKey{}).(bool)
	return disabled
}
