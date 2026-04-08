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
	lastRetryCount := retryCount
	lastCBState := cbState
	lastTarget := primary
	if err == nil {
		return resp, primary, false, retryCount, cbState, nil
	}
	if errors.Is(err, context.Canceled) || ctx.Err() != nil {
		return nil, primary, false, retryCount, cbState, err
	}

	log.Warn().
		Err(err).
		Str("provider", primary.Provider).
		Str("model", primary.Model).
		Msg("primary target failed, trying fallbacks")

	// Try each fallback in order
	for i, fb := range fallbacks {
		log.Info().
			Str("provider", fb.Provider).
			Str("model", fb.Model).
			Int("fallback_index", i).
			Msg("attempting fallback")

		resp, retryCount, cbState, err = f.executeWithCircuitBreaker(ctx, fb, fn)
		lastRetryCount = retryCount
		lastCBState = cbState
		lastTarget = fb
		if err == nil {
			return resp, fb, true, retryCount, cbState, nil
		}

		log.Warn().
			Err(err).
			Str("provider", fb.Provider).
			Str("model", fb.Model).
			Int("fallback_index", i).
			Msg("fallback target failed")
	}

	return nil, lastTarget, true, lastRetryCount, lastCBState, fmt.Errorf("all targets failed (primary + %d fallbacks): %w", len(fallbacks), err)
}

type execResult struct {
	resp       *http.Response
	retryCount int
}

func (f *FallbackExecutor) executeWithCircuitBreaker(ctx context.Context, target routing.Target, fn ExecuteFunc) (*http.Response, int, string, error) {
	result, err := f.cbm.Execute(target.Provider, func() (interface{}, error) {
		resp, retryCount, err := f.retrier.Do(ctx, func(ctx context.Context) (*http.Response, error) {
			return fn(ctx, target)
		})
		if err != nil {
			return nil, err
		}
		return &execResult{resp: resp, retryCount: retryCount}, nil
	})

	if err != nil {
		return nil, 0, "", err
	}

	res := result.(*execResult)
	cbState := f.cbm.State(target.Provider).String()
	return res.resp, res.retryCount, cbState, nil
}
