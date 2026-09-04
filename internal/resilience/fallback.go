package resilience

import (
	"context"
	"errors"
	"fmt"
	"net/http"

	"github.com/lunargate-ai/gateway/internal/routing"
	"github.com/rs/zerolog/log"
)

// FallbackExecutor runs a request against a primary target and cascades to fallbacks on failure.
type FallbackExecutor struct {
	retrier *Retrier
	cbm     *CircuitBreakerManager
}

// NewFallbackExecutor creates a new fallback executor.
func NewFallbackExecutor(retrier *Retrier, cbm *CircuitBreakerManager) *FallbackExecutor {
	return &FallbackExecutor{
		retrier: retrier,
		cbm:     cbm,
	}
}

// ExecuteFunc is the function that actually calls the provider.
// It receives the target and must return the raw HTTP response.
type ExecuteFunc func(ctx context.Context, target routing.Target) (*http.Response, error)

// Execute runs the request against the primary target, then fallbacks on failure.
// Returns the response, the target that succeeded, and any error.
func (f *FallbackExecutor) Execute(ctx context.Context, primary routing.Target, fallbacks []routing.Target, fn ExecuteFunc) (*http.Response, routing.Target, bool, int, string, error) {
	// Try primary target with retries
	resp, retryCount, cbState, err := f.executeWithCircuitBreaker(ctx, primary, fn)
	totalRetryCount := retryCount
	lastCBState := cbState
	lastTarget := primary
	fallbackAttempted := false
	if err == nil {
		return resp, primary, false, totalRetryCount, cbState, nil
	}
	if isTerminalFallbackError(ctx, err) {
		return nil, primary, false, totalRetryCount, cbState, err
	}
	if fallbackDisabled(ctx) {
		return nil, primary, false, totalRetryCount, cbState, err
	}

	log.Warn().
		Err(err).
		Str("provider", primary.Provider).
		Str("model", primary.Model).
		Msg("primary target failed, trying fallbacks")

	// Try each fallback in order
	for i, fb := range fallbacks {
		fallbackAttempted = true
		log.Info().
			Str("provider", fb.Provider).
			Str("model", fb.Model).
			Int("fallback_index", i).
			Msg("attempting fallback")

		resp, retryCount, cbState, err = f.executeWithCircuitBreaker(ctx, fb, fn)
		totalRetryCount += retryCount
		lastCBState = cbState
		lastTarget = fb
		if err == nil {
			return resp, fb, true, totalRetryCount, cbState, nil
		}
		if isTerminalFallbackError(ctx, err) {
			return nil, fb, true, totalRetryCount, cbState, err
		}

		log.Warn().
			Err(err).
			Str("provider", fb.Provider).
			Str("model", fb.Model).
			Int("fallback_index", i).
			Msg("fallback target failed")
	}

	return nil, lastTarget, fallbackAttempted, totalRetryCount, lastCBState, fmt.Errorf("all targets failed (primary + %d fallbacks): %w", len(fallbacks), err)
}

func isTerminalFallbackError(ctx context.Context, err error) bool {
	return IsRequestError(err) ||
		errors.Is(err, context.Canceled) ||
		errors.Is(err, context.DeadlineExceeded) ||
		ctx.Err() != nil
}

type execResult struct {
	resp       *http.Response
	retryCount int
}

func (f *FallbackExecutor) executeWithCircuitBreaker(ctx context.Context, target routing.Target, fn ExecuteFunc) (*http.Response, int, string, error) {
	lastRetryCount := 0
	breakerKey := target.CircuitBreakerKey()
	result, state, err := f.cbm.executeForKeyWithState(breakerKey, target.Provider, func() (interface{}, error) {
		resp, retryCount, err := f.retrier.Do(ctx, func(ctx context.Context) (*http.Response, error) {
			return fn(ctx, target)
		})
		lastRetryCount = retryCount
		if err != nil {
			return nil, err
		}
		return &execResult{resp: resp, retryCount: retryCount}, nil
	})
	cbState := state.String()

	if err != nil {
		return nil, lastRetryCount, cbState, err
	}

	res := result.(*execResult)
	return res.resp, res.retryCount, cbState, nil
}
