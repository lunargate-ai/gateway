package resilience

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/rs/zerolog/log"
)

const retryableStatusBodyLimit = 1 << 20

// Retrier handles retry logic with exponential backoff and jitter.
type Retrier struct {
	cfg atomic.Value
}

// RetryableStatusError captures upstream HTTP status codes that make a target
// fail so callers can preserve the original status across fallback handling.
type RetryableStatusError struct {
	StatusCode int
	Headers    http.Header
	Body       []byte
	Truncated  bool
}

func (e *RetryableStatusError) Error() string {
	return fmt.Sprintf("provider returned status %d", e.StatusCode)
}

func snapshotRetryableStatus(resp *http.Response) *RetryableStatusError {
	snapshot := &RetryableStatusError{
		StatusCode: resp.StatusCode,
		Headers:    resp.Header.Clone(),
	}
	if resp.Body == nil {
		return snapshot
	}

	body, readErr := io.ReadAll(io.LimitReader(resp.Body, retryableStatusBodyLimit+1))
	_ = resp.Body.Close()
	if len(body) > retryableStatusBodyLimit {
		snapshot.Body = make([]byte, retryableStatusBodyLimit)
		copy(snapshot.Body, body[:retryableStatusBodyLimit])
		snapshot.Truncated = true
	} else {
		snapshot.Body = make([]byte, len(body))
		copy(snapshot.Body, body)
	}
	if readErr != nil {
		// A partial read cannot be treated as a complete upstream envelope.
		snapshot.Truncated = true
	}
	return snapshot
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
		} else if maxAttempts > config.MaxRetryAttempts {
			maxAttempts = config.MaxRetryAttempts
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

			lastErr = snapshotRetryableStatus(resp)
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
			delay := calculateRetryDelay(cfg, attempt, lastErr, time.Now())
			log.Debug().
				Int("attempt", attempt+1).
				Int("max_attempts", maxAttempts).
				Dur("delay", delay).
				Err(lastErr).
				Msg("retrying request")

			timer := time.NewTimer(delay)
			select {
			case <-ctx.Done():
				if !timer.Stop() {
					select {
					case <-timer.C:
					default:
					}
				}
				return nil, attempt, ctx.Err()
			case <-timer.C:
			}
		}
	}

	return nil, maxAttempts - 1, fmt.Errorf("max attempts (%d) exhausted: %w", maxAttempts, lastErr)
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

func calculateRetryDelay(cfg config.RetryConfig, attempt int, lastErr error, now time.Time) time.Duration {
	maxDelay := cfg.MaxDelay
	if maxDelay < 0 {
		maxDelay = 0
	}
	initialDelay := cfg.InitialDelay
	if initialDelay < 0 {
		initialDelay = 0
	}
	multiplier := cfg.Multiplier
	if math.IsNaN(multiplier) || math.IsInf(multiplier, -1) || multiplier < 1 {
		multiplier = 1
	}
	jitterFactor := cfg.JitterFactor
	if math.IsNaN(jitterFactor) || math.IsInf(jitterFactor, 0) || jitterFactor < 0 {
		jitterFactor = 0
	} else if jitterFactor > 1 {
		jitterFactor = 1
	}

	delay := float64(initialDelay)
	if delay > 0 && attempt > 0 {
		delay *= math.Pow(multiplier, float64(attempt))
	}
	// Add jitter: delay * (1 +/- jitterFactor/2).
	delay *= 1 + jitterFactor*(rand.Float64()-0.5)
	result := saturatingRetryDuration(delay, maxDelay)

	var statusErr *RetryableStatusError
	if errors.As(lastErr, &statusErr) {
		if retryAfter, ok := parseRetryAfter(statusErr.Headers.Get("Retry-After"), now); ok && retryAfter > result {
			result = retryAfter
		}
	}
	if result > maxDelay {
		result = maxDelay
	}
	return result
}

func saturatingRetryDuration(value float64, maximum time.Duration) time.Duration {
	if maximum <= 0 || math.IsNaN(value) || value <= 0 {
		return 0
	}
	// Compare before conversion: float64(MaxInt64) rounds to 1<<63, which
	// would wrap negative if converted directly to time.Duration.
	if math.IsInf(value, 1) || value >= float64(maximum) {
		return maximum
	}
	return time.Duration(value)
}

func parseRetryAfter(value string, now time.Time) (time.Duration, bool) {
	value = strings.TrimSpace(value)
	if value == "" {
		return 0, false
	}
	if seconds, err := strconv.ParseInt(value, 10, 64); err == nil {
		if seconds < 0 {
			return 0, false
		}
		const maxDurationSeconds = int64((1<<63 - 1) / time.Second)
		if seconds > maxDurationSeconds {
			return time.Duration(1<<63 - 1), true
		}
		return time.Duration(seconds) * time.Second, true
	}
	deadline, err := http.ParseTime(value)
	if err != nil {
		return 0, false
	}
	delay := deadline.Sub(now)
	if delay < 0 {
		delay = 0
	}
	return delay, true
}
