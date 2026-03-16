package resilience

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"net/http"
	"time"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/rs/zerolog/log"
)

// Retrier handles retry logic with exponential backoff and jitter.
type Retrier struct {
	cfg config.RetryConfig
}

// NewRetrier creates a new retrier from config.
func NewRetrier(cfg config.RetryConfig) *Retrier {
	return &Retrier{cfg: cfg}
}

// DoFunc is the function signature for retryable operations.
type DoFunc func(ctx context.Context) (*http.Response, error)

// Do executes the given function with retry logic.
// Returns the response from the first successful attempt or the last error.
func (r *Retrier) Do(ctx context.Context, fn DoFunc) (*http.Response, int, error) {
	if !r.cfg.Enabled {
		resp, err := fn(ctx)
		return resp, 0, err
	}

	var lastErr error
	for attempt := 0; attempt < r.cfg.MaxAttempts; attempt++ {
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

		if attempt < r.cfg.MaxAttempts-1 {
			delay := r.calculateDelay(attempt)
			log.Debug().
				Int("attempt", attempt+1).
				Int("max_attempts", r.cfg.MaxAttempts).
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

	return nil, r.cfg.MaxAttempts, fmt.Errorf("max retries (%d) exceeded: %w", r.cfg.MaxAttempts, lastErr)
}

func (r *Retrier) isRetryableStatus(code int) bool {
	for _, retryable := range r.cfg.RetryableErrors {
		if code == retryable {
			return true
		}
	}
	return false
}

func (r *Retrier) calculateDelay(attempt int) time.Duration {
	delay := float64(r.cfg.InitialDelay) * math.Pow(r.cfg.Multiplier, float64(attempt))

	if delay > float64(r.cfg.MaxDelay) {
		delay = float64(r.cfg.MaxDelay)
	}

	// Add jitter: delay * (1 +/- jitterFactor/2)
	jitter := delay * r.cfg.JitterFactor * (rand.Float64() - 0.5)
	result := delay + jitter

	if result < 0 {
		result = float64(r.cfg.InitialDelay)
	}

	return time.Duration(result)
}
