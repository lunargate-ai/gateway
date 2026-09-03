package resilience

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"net/http"
	"sync/atomic"
	"time"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/rs/zerolog/log"
)

// Retrier handles retry logic with exponential backoff and jitter.
type Retrier struct {
	cfg atomic.Value
}

// RetryableStatusError captures upstream HTTP status codes that make a target
// fail so callers can preserve the original status across fallback handling.
type RetryableStatusError struct {
	StatusCode int
}

func (e *RetryableStatusError) Error() string {
	return fmt.Sprintf("provider returned status %d", e.StatusCode)
}

// RequestError marks a failure produced before an upstream request can be
// sent. It is terminal for retry and fallback, and it does not reflect
// provider health.
type RequestError struct {
	cause error
}

func (e *RequestError) Error() string {
	if e == nil || e.cause == nil {
		return "invalid provider request"
	}
	return e.cause.Error()
}

func (e *RequestError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.cause
}

// NewRequestError classifies request validation or translation failures.
func NewRequestError(err error) error {
	if err == nil || IsRequestError(err) || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return err
	}
	return &RequestError{cause: err}
}

// IsRequestError reports whether an error is terminal client-request work.
func IsRequestError(err error) bool {
	var requestErr *RequestError
	return errors.As(err, &requestErr)
}

// NewRetrier creates a new retrier from config.
func NewRetrier(cfg config.RetryConfig) *Retrier {
	r := &Retrier{}
	r.cfg.Store(cfg)
	return r
}

// UpdateConfig hot-reloads the retry configuration.
func (r *Retrier) UpdateConfig(cfg config.RetryConfig) {
	r.cfg.Store(cfg)
	log.Info().Msg("retry config updated")
}

func (r *Retrier) currentConfig() config.RetryConfig {
	cfg, _ := r.cfg.Load().(config.RetryConfig)
	return cfg
}

// DoFunc is the function signature for retryable operations.
type DoFunc func(ctx context.Context) (*http.Response, error)

// Do executes the given function with retry logic.
// Returns the response from the first successful attempt or the last error.
func (r *Retrier) Do(ctx context.Context, fn DoFunc) (*http.Response, int, error) {
	cfg := r.currentConfig()
	maxAttempts := 1
	if cfg.Enabled {
		maxAttempts = cfg.MaxAttempts
		if maxAttempts < 1 {
			maxAttempts = 1
		}
	}
	if retryDisabled(ctx) && maxAttempts > 1 {
		maxAttempts = 1
	}

	var lastErr error
	for attempt := 0; attempt < maxAttempts; attempt++ {
		resp, err := fn(ctx)

		if err != nil {
			if IsRequestError(err) || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				return nil, attempt, err
			}
			lastErr = err
			if !cfg.Enabled {
				return nil, attempt, err
			}
		} else if resp != nil {
			retryableStatus := cfg.Enabled && isConfiguredRetryableStatus(cfg, resp.StatusCode)
			if !retryableStatus && !isProviderFailureStatus(resp.StatusCode) {
				return resp, attempt, nil
			}

			lastErr = &RetryableStatusError{StatusCode: resp.StatusCode}
			// Close responses that will not be returned to the handler.
			if resp.Body != nil {
				_ = resp.Body.Close()
			}
			if !retryableStatus {
				return nil, attempt, lastErr
			}
		} else {
			lastErr = errors.New("provider returned neither a response nor an error")
			if !cfg.Enabled {
				return nil, attempt, lastErr
			}
		}

		if attempt < maxAttempts-1 {
			delay := r.calculateDelay(attempt)
			log.Debug().
				Int("attempt", attempt+1).
				Int("max_attempts", maxAttempts).
				Dur("delay", delay).
				Err(lastErr).
				Msg("retrying request")

			select {
			case <-ctx.Done():
				return nil, attempt, ctx.Err()
			case <-time.After(delay):
			}
		}
	}

	return nil, maxAttempts, fmt.Errorf("max retries (%d) exceeded: %w", maxAttempts, lastErr)
}

func isConfiguredRetryableStatus(cfg config.RetryConfig, code int) bool {
	for _, retryable := range cfg.RetryableErrors {
		if code == retryable {
			return true
		}
	}
	return false
}

func isProviderFailureStatus(code int) bool {
	return code >= http.StatusInternalServerError && code <= 599
}

func (r *Retrier) calculateDelay(attempt int) time.Duration {
	cfg := r.currentConfig()
	delay := float64(cfg.InitialDelay) * math.Pow(cfg.Multiplier, float64(attempt))

	if delay > float64(cfg.MaxDelay) {
		delay = float64(cfg.MaxDelay)
	}

	// Add jitter: delay * (1 +/- jitterFactor/2)
	jitter := delay * cfg.JitterFactor * (rand.Float64() - 0.5)
	result := delay + jitter

	if result < 0 {
		result = float64(cfg.InitialDelay)
	}

	return time.Duration(result)
}
