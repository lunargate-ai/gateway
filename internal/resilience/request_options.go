package resilience

import "context"

type retryDisabledContextKey struct{}
type fallbackDisabledContextKey struct{}

// WithRetryDisabled marks a request context so the retrier performs only one
// attempt while keeping normal retryable-status handling for fallback logic.
func WithRetryDisabled(ctx context.Context) context.Context {
	return context.WithValue(ctx, retryDisabledContextKey{}, true)
}

func retryDisabled(ctx context.Context) bool {
	disabled, _ := ctx.Value(retryDisabledContextKey{}).(bool)
	return disabled
}

// WithFallbackDisabled pins an operation to its primary target. It is used for
// stateful or otherwise non-idempotent operations that must not be duplicated
// on another provider after an ambiguous failure.
func WithFallbackDisabled(ctx context.Context) context.Context {
	return context.WithValue(ctx, fallbackDisabledContextKey{}, true)
}

func fallbackDisabled(ctx context.Context) bool {
	disabled, _ := ctx.Value(fallbackDisabledContextKey{}).(bool)
	return disabled
}
