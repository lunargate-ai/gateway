package resilience

import (
	"context"
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
	if !cfg.Enabled {
		resp, err := fn(ctx)
		return resp, 0, err
	}

	maxAttempts := cfg.MaxAttempts
	if retryDisabled(ctx) && maxAttempts > 1 {
		maxAttempts = 1
	}

	var lastErr error
	for attempt := 0; attempt < maxAttempts; attempt++ {
		resp, err := fn(ctx)

		if err == nil && resp != nil && !r.isRetryableStatus(resp.StatusCode) {
			return resp, attempt, nil
		}

		if err != nil {
			lastErr = err
		} else if resp != nil {
			lastErr = fmt.Errorf("provider returned status %d", resp.StatusCode)
			// Close the body of retryable responses to avoid leaking
			resp.Body.Close()
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

func (r *Retrier) isRetryableStatus(code int) bool {
	cfg := r.currentConfig()
	for _, retryable := range cfg.RetryableErrors {
		if code == retryable {
			return true
		}
	}
	return false
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
