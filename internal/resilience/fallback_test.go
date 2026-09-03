package resilience

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/routing"
)

func TestFallbackExecutor_NoFallbacks_PreservesRetryMetadataOnFailure(t *testing.T) {
	retrier := NewRetrier(config.RetryConfig{
		Enabled:         true,
		MaxAttempts:     3,
		InitialDelay:    0,
		MaxDelay:        0,
		Multiplier:      1,
		RetryableErrors: []int{http.StatusTooManyRequests},
	})
	fallback := NewFallbackExecutor(retrier, NewCircuitBreakerManager())

	primary := routing.Target{Provider: "venice", Model: "zai-org-glm-5"}
	resp, usedTarget, fallbackUsed, retryCount, cbState, err := fallback.Execute(
		context.Background(),
		primary,
		nil,
		func(ctx context.Context, target routing.Target) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusTooManyRequests,
				Body:       io.NopCloser(strings.NewReader(`{"error":"rate_limited"}`)),
			}, nil
		},
	)

	if resp != nil {
		t.Fatalf("expected nil response on exhausted retryable status, got %#v", resp)
	}
	if usedTarget != primary {
		t.Fatalf("expected used target to remain primary, got %#v", usedTarget)
	}
	if fallbackUsed {
		t.Fatalf("expected fallbackUsed=false when no fallback targets exist")
	}
	if retryCount != 3 {
		t.Fatalf("expected retryCount=3, got %d", retryCount)
	}
	if cbState == "" {
		t.Fatalf("expected non-empty circuit breaker state")
	}
	if err == nil {
		t.Fatalf("expected final error")
	}

	var statusErr *RetryableStatusError
	if !errors.As(err, &statusErr) {
		t.Fatalf("expected wrapped RetryableStatusError, got %v", err)
	}
	if statusErr.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("expected status code 429, got %d", statusErr.StatusCode)
	}
}

func TestFallbackExecutor_WithFallbackDisabledStaysOnPrimary(t *testing.T) {
	retrier := NewRetrier(config.RetryConfig{
		Enabled:         true,
		MaxAttempts:     3,
		RetryableErrors: []int{http.StatusInternalServerError},
	})
	fallback := NewFallbackExecutor(retrier, NewCircuitBreakerManager())
	primary := routing.Target{Provider: "primary", Model: "model-a"}
	backup := routing.Target{Provider: "backup", Model: "model-b"}
	primaryCalls := 0
	backupCalls := 0
	ctx := WithFallbackDisabled(WithRetryDisabled(context.Background()))

	_, usedTarget, fallbackUsed, retryCount, _, err := fallback.Execute(
		ctx,
		primary,
		[]routing.Target{backup},
		func(_ context.Context, target routing.Target) (*http.Response, error) {
			if target.Provider == "primary" {
				primaryCalls++
			} else {
				backupCalls++
			}
			return &http.Response{
				StatusCode: http.StatusInternalServerError,
				Body:       io.NopCloser(strings.NewReader(`{"error":"ambiguous"}`)),
			}, nil
		},
	)
	if err == nil {
		t.Fatal("expected primary failure")
	}
	if primaryCalls != 1 || backupCalls != 0 {
		t.Fatalf("calls primary=%d backup=%d, want 1/0", primaryCalls, backupCalls)
	}
	if usedTarget != primary || fallbackUsed {
		t.Fatalf("usedTarget=%#v fallbackUsed=%v", usedTarget, fallbackUsed)
	}
	if retryCount != 1 {
		t.Fatalf("retryCount=%d, want 1", retryCount)
	}
}
