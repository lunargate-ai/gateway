package resilience

import (
	"context"
	"fmt"
	"net/http"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
)

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
	if got, want := err.Error(), "max retries (1) exceeded: provider returned status 500"; got != want {
		t.Fatalf("expected error %q, got %q", want, got)
	}
	if attempts != 1 {
		t.Fatalf("expected exactly one attempt, got %d", attempts)
	}
	if retryCount != 1 {
		t.Fatalf("expected retryCount=1 after the single allowed attempt, got %d", retryCount)
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
	if retryCount != 3 {
		t.Fatalf("expected retryCount=3, got %d", retryCount)
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
	if retryCount != 1 {
		t.Fatalf("expected retryCount=1 after config update, got %d", retryCount)
	}
}
