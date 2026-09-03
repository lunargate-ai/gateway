package resilience

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/lunargate-ai/gateway/internal/config"
)

type trackingResponseBody struct {
	io.Reader
	closed bool
}

func TestParseRetryAfterSupportsSecondsAndHTTPDate(t *testing.T) {
	now := time.Date(2026, time.September, 3, 10, 0, 0, 0, time.UTC)
	tests := []struct {
		value string
		want  time.Duration
		ok    bool
	}{
		{value: "7", want: 7 * time.Second, ok: true},
		{value: now.Add(9 * time.Second).Format(http.TimeFormat), want: 9 * time.Second, ok: true},
		{value: now.Add(-time.Second).Format(http.TimeFormat), want: 0, ok: true},
		{value: "-1"},
		{value: "not-a-date"},
	}
	for _, test := range tests {
		got, ok := parseRetryAfter(test.value, now)
		if ok != test.ok || got != test.want {
			t.Fatalf("parseRetryAfter(%q) = %s, %v; want %s, %v", test.value, got, ok, test.want, test.ok)
		}
	}
}

func TestRetrierHonorsRetryAfterBeforeNextAttempt(t *testing.T) {
	retrier := NewRetrier(config.RetryConfig{
		Enabled:         true,
		MaxAttempts:     2,
		InitialDelay:    0,
		MaxDelay:        2 * time.Second,
		Multiplier:      1,
		RetryableErrors: []int{http.StatusTooManyRequests},
	})
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	attempts := 0

	_, retryCount, err := retrier.Do(ctx, func(context.Context) (*http.Response, error) {
		attempts++
		return &http.Response{
			StatusCode: http.StatusTooManyRequests,
			Header:     http.Header{"Retry-After": []string{"1"}},
			Body:       http.NoBody,
		}, nil
	})

	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("error = %v, want context deadline", err)
	}
	if attempts != 1 {
		t.Fatalf("attempts = %d, want 1 before Retry-After elapsed", attempts)
	}
	if retryCount != 0 {
		t.Fatalf("retryCount = %d, want 0 completed retries", retryCount)
	}
}

func TestRetryAfterIsCappedByConfiguredMaximum(t *testing.T) {
	cfg := config.RetryConfig{
		InitialDelay: 10 * time.Millisecond,
		MaxDelay:     50 * time.Millisecond,
		Multiplier:   2,
	}
	err := &RetryableStatusError{Headers: http.Header{"Retry-After": []string{"30"}}}
	if got := calculateRetryDelay(cfg, 0, err, time.Now()); got != cfg.MaxDelay {
		t.Fatalf("delay = %s, want configured cap %s", got, cfg.MaxDelay)
	}
}

func TestCalculateRetryDelaySaturatesBeforeDurationConversion(t *testing.T) {
	cfg := config.RetryConfig{
		InitialDelay: time.Duration(math.MaxInt64 / 2),
		MaxDelay:     time.Duration(math.MaxInt64),
		Multiplier:   math.MaxFloat64,
	}

	got := calculateRetryDelay(cfg, 1, nil, time.Now())
	if got != cfg.MaxDelay {
		t.Fatalf("delay = %s, want saturated maximum %s", got, cfg.MaxDelay)
	}
	if got < 0 {
		t.Fatalf("delay wrapped negative: %s", got)
	}
}

func TestRetrierEnforcesAttemptCeilingForDirectConfig(t *testing.T) {
	retrier := NewRetrier(config.RetryConfig{
		Enabled:     true,
		MaxAttempts: config.MaxRetryAttempts + 100,
		Multiplier:  1,
	})
	calls := 0

	_, retryCount, err := retrier.Do(context.Background(), func(context.Context) (*http.Response, error) {
		calls++
		return nil, errors.New("temporary failure")
	})
	if err == nil {
		t.Fatal("expected retry exhaustion")
	}
	if calls != config.MaxRetryAttempts {
		t.Fatalf("calls = %d, want hard ceiling %d", calls, config.MaxRetryAttempts)
	}
	if retryCount != config.MaxRetryAttempts-1 {
		t.Fatalf("retryCount = %d, want %d", retryCount, config.MaxRetryAttempts-1)
	}
}

func (b *trackingResponseBody) Close() error {
	b.closed = true
	return nil
}

func TestRetrier_Do_WithRetryDisabledContext_UsesSingleAttempt(t *testing.T) {
	retrier := NewRetrier(config.RetryConfig{
		Enabled:         true,
		MaxAttempts:     3,
		RetryableErrors: []int{http.StatusInternalServerError},
	})

	attempts := 0
	resp, retryCount, err := retrier.Do(WithRetryDisabled(context.Background()), func(ctx context.Context) (*http.Response, error) {
		attempts++
		return &http.Response{
			StatusCode: http.StatusInternalServerError,
			Body:       http.NoBody,
		}, nil
	})

	if resp != nil {
		t.Fatalf("expected no response on exhausted retryable status, got %#v", resp)
	}
	if err == nil {
		t.Fatalf("expected an error after the single attempt")
	}
	if got, want := err.Error(), "max attempts (1) exhausted: provider returned status 500"; got != want {
		t.Fatalf("expected error %q, got %q", want, got)
	}
	if attempts != 1 {
		t.Fatalf("expected exactly one attempt, got %d", attempts)
	}
	if retryCount != 0 {
		t.Fatalf("expected retryCount=0 after one initial attempt, got %d", retryCount)
	}
}

func TestRetrier_Do_UsesConfiguredRetriesByDefault(t *testing.T) {
	retrier := NewRetrier(config.RetryConfig{
		Enabled:         true,
		MaxAttempts:     3,
		InitialDelay:    0,
		MaxDelay:        0,
		Multiplier:      1,
		RetryableErrors: []int{http.StatusInternalServerError},
	})

	attempts := 0
	resp, retryCount, err := retrier.Do(context.Background(), func(ctx context.Context) (*http.Response, error) {
		attempts++
		return nil, fmt.Errorf("boom")
	})

	if resp != nil {
		t.Fatalf("expected nil response on repeated errors, got %#v", resp)
	}
	if err == nil {
		t.Fatalf("expected final retry error")
	}
	if attempts != 3 {
		t.Fatalf("expected three attempts, got %d", attempts)
	}
	if retryCount != 2 {
		t.Fatalf("expected retryCount=2, got %d", retryCount)
	}
}

func TestRetrier_UpdateConfig_AppliesNewMaxAttempts(t *testing.T) {
	retrier := NewRetrier(config.RetryConfig{
		Enabled:         true,
		MaxAttempts:     3,
		InitialDelay:    0,
		MaxDelay:        0,
		Multiplier:      1,
		RetryableErrors: []int{http.StatusInternalServerError},
	})
	retrier.UpdateConfig(config.RetryConfig{
		Enabled:         true,
		MaxAttempts:     1,
		InitialDelay:    0,
		MaxDelay:        0,
		Multiplier:      1,
		RetryableErrors: []int{http.StatusInternalServerError},
	})

	attempts := 0
	_, retryCount, err := retrier.Do(context.Background(), func(ctx context.Context) (*http.Response, error) {
		attempts++
		return nil, fmt.Errorf("boom")
	})

	if err == nil {
		t.Fatalf("expected retry error after config update")
	}
	if attempts != 1 {
		t.Fatalf("expected one attempt after config update, got %d", attempts)
	}
	if retryCount != 0 {
		t.Fatalf("expected retryCount=0 after config update, got %d", retryCount)
	}
}

func TestRetrier_RequestErrorStopsBeforeRetry(t *testing.T) {
	retrier := NewRetrier(config.RetryConfig{
		Enabled:         true,
		MaxAttempts:     3,
		RetryableErrors: []int{http.StatusInternalServerError},
	})
	cause := errors.New("invalid translated payload")
	attempts := 0

	resp, retryCount, err := retrier.Do(context.Background(), func(context.Context) (*http.Response, error) {
		attempts++
		return nil, NewRequestError(cause)
	})

	if resp != nil {
		t.Fatalf("response = %#v, want nil", resp)
	}
	if !errors.Is(err, cause) || !IsRequestError(err) {
		t.Fatalf("error = %v, want classified request error", err)
	}
	if attempts != 1 {
		t.Fatalf("attempts = %d, want 1", attempts)
	}
	if retryCount != 0 {
		t.Fatalf("retryCount = %d, want 0", retryCount)
	}
}

func TestRetrier_NonRetryableClientStatusReturnsResponse(t *testing.T) {
	retrier := NewRetrier(config.RetryConfig{
		Enabled:         true,
		MaxAttempts:     3,
		RetryableErrors: []int{http.StatusTooManyRequests, http.StatusServiceUnavailable},
	})
	attempts := 0

	resp, retryCount, err := retrier.Do(context.Background(), func(context.Context) (*http.Response, error) {
		attempts++
		return &http.Response{StatusCode: http.StatusBadRequest, Body: http.NoBody}, nil
	})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp == nil || resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("response = %#v, want status 400", resp)
	}
	if attempts != 1 || retryCount != 0 {
		t.Fatalf("attempts/retryCount = %d/%d, want 1/0", attempts, retryCount)
	}
}

func TestRetrier_UnconfiguredServerStatusIsImmediateFailure(t *testing.T) {
	retrier := NewRetrier(config.RetryConfig{
		Enabled:         true,
		MaxAttempts:     3,
		RetryableErrors: []int{http.StatusTooManyRequests},
	})
	attempts := 0

	resp, retryCount, err := retrier.Do(context.Background(), func(context.Context) (*http.Response, error) {
		attempts++
		return &http.Response{StatusCode: http.StatusServiceUnavailable, Body: http.NoBody}, nil
	})

	if resp != nil {
		t.Fatalf("response = %#v, want nil", resp)
	}
	var statusErr *RetryableStatusError
	if !errors.As(err, &statusErr) || statusErr.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("error = %v, want status 503 failure", err)
	}
	if attempts != 1 || retryCount != 0 {
		t.Fatalf("attempts/retryCount = %d/%d, want 1/0", attempts, retryCount)
	}
}

func TestRetrier_RetryableStatusPreservesOnlyFinalSnapshot(t *testing.T) {
	retrier := NewRetrier(config.RetryConfig{
		Enabled:         true,
		MaxAttempts:     2,
		InitialDelay:    0,
		MaxDelay:        0,
		Multiplier:      1,
		RetryableErrors: []int{http.StatusTooManyRequests},
	})
	bodyValues := []string{`{"error":"first-attempt-secret"}`, `{"error":"final-rate-limit"}`}
	headerValues := []string{"first-header-secret", "final-header"}
	bodies := make([]*trackingResponseBody, 0, len(bodyValues))
	headers := make([]http.Header, 0, len(bodyValues))
	attempt := 0

	resp, retryCount, err := retrier.Do(context.Background(), func(context.Context) (*http.Response, error) {
		body := &trackingResponseBody{Reader: strings.NewReader(bodyValues[attempt])}
		header := http.Header{
			"Content-Type":       []string{"application/json"},
			"X-Upstream-Attempt": []string{headerValues[attempt]},
		}
		bodies = append(bodies, body)
		headers = append(headers, header)
		attempt++
		return &http.Response{
			StatusCode: http.StatusTooManyRequests,
			Header:     header,
			Body:       body,
		}, nil
	})

	if resp != nil {
		t.Fatalf("response = %#v, want nil", resp)
	}
	if retryCount != 1 {
		t.Fatalf("retryCount = %d, want 1", retryCount)
	}
	var statusErr *RetryableStatusError
	if !errors.As(err, &statusErr) {
		t.Fatalf("error = %v, want RetryableStatusError", err)
	}
	if statusErr.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("status = %d, want 429", statusErr.StatusCode)
	}
	if got, want := string(statusErr.Body), bodyValues[1]; got != want {
		t.Fatalf("body = %q, want %q", got, want)
	}
	if statusErr.Truncated {
		t.Fatal("final bounded response unexpectedly marked truncated")
	}
	if got, want := statusErr.Headers.Get("X-Upstream-Attempt"), headerValues[1]; got != want {
		t.Fatalf("header = %q, want %q", got, want)
	}
	for i, body := range bodies {
		if !body.closed {
			t.Fatalf("attempt %d response body was not closed", i+1)
		}
	}
	headers[1].Set("X-Upstream-Attempt", "mutated-after-snapshot")
	if got := statusErr.Headers.Get("X-Upstream-Attempt"); got != headerValues[1] {
		t.Fatalf("snapshot header aliased original response: %q", got)
	}
	if strings.Contains(err.Error(), "final-rate-limit") || strings.Contains(err.Error(), "final-header") || strings.Contains(err.Error(), "first-attempt-secret") {
		t.Fatalf("error string leaked upstream response data: %q", err.Error())
	}
}

func TestRetrier_RetryableStatusBoundsSnapshotBody(t *testing.T) {
	retrier := NewRetrier(config.RetryConfig{
		Enabled:         true,
		MaxAttempts:     1,
		RetryableErrors: []int{http.StatusBadGateway},
	})
	wantPrefix := bytes.Repeat([]byte("x"), retryableStatusBodyLimit)
	upstreamBody := append(append([]byte(nil), wantPrefix...), bytes.Repeat([]byte("y"), 128)...)
	body := &trackingResponseBody{Reader: bytes.NewReader(upstreamBody)}

	_, _, err := retrier.Do(context.Background(), func(context.Context) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusBadGateway,
			Header:     http.Header{"X-Upstream-Trace": []string{"sensitive-trace"}},
			Body:       body,
		}, nil
	})

	var statusErr *RetryableStatusError
	if !errors.As(err, &statusErr) {
		t.Fatalf("error = %v, want RetryableStatusError", err)
	}
	if !body.closed {
		t.Fatal("upstream response body was not closed")
	}
	if !statusErr.Truncated {
		t.Fatal("oversized response was not marked truncated")
	}
	if got := len(statusErr.Body); got != retryableStatusBodyLimit {
		t.Fatalf("snapshot body length = %d, want %d", got, retryableStatusBodyLimit)
	}
	if !bytes.Equal(statusErr.Body, wantPrefix) {
		t.Fatal("snapshot body does not contain the bounded upstream prefix")
	}
	if got := cap(statusErr.Body); got != retryableStatusBodyLimit {
		t.Fatalf("snapshot body capacity = %d, want bounded capacity %d", got, retryableStatusBodyLimit)
	}
	if strings.Contains(err.Error(), "sensitive-trace") {
		t.Fatalf("error string leaked upstream header: %q", err.Error())
	}
}
