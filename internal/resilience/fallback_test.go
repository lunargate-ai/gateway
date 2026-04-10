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

