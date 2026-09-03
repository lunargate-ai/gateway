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

func TestFallbackExecutor_RequestErrorIsTerminalAndHealthy(t *testing.T) {
	cbm := NewCircuitBreakerManager()
	fallback := NewFallbackExecutor(NewRetrier(config.RetryConfig{
		Enabled:         true,
		MaxAttempts:     3,
		RetryableErrors: []int{http.StatusInternalServerError},
	}), cbm)
	primary := routing.Target{Provider: "primary", Model: "model-a"}
	backup := routing.Target{Provider: "backup", Model: "model-b"}
	primaryCalls := 0
	backupCalls := 0
	cause := errors.New("invalid request translation")

	_, usedTarget, fallbackUsed, retryCount, _, err := fallback.Execute(
		context.Background(),
		primary,
		[]routing.Target{backup},
		func(_ context.Context, target routing.Target) (*http.Response, error) {
			if target.Provider == primary.Provider {
				primaryCalls++
				return nil, NewRequestError(cause)
			}
			backupCalls++
			return &http.Response{StatusCode: http.StatusOK, Body: http.NoBody}, nil
		},
	)

	if !errors.Is(err, cause) || !IsRequestError(err) {
		t.Fatalf("error = %v, want classified request error", err)
	}
	if usedTarget != primary || fallbackUsed {
		t.Fatalf("usedTarget=%#v fallbackUsed=%v, want primary/false", usedTarget, fallbackUsed)
	}
	if primaryCalls != 1 || backupCalls != 0 || retryCount != 0 {
		t.Fatalf("calls/retries = %d/%d/%d, want 1/0/0", primaryCalls, backupCalls, retryCount)
	}
	if counts := cbm.Get(primary.Provider).Counts(); counts.TotalFailures != 0 {
		t.Fatalf("request error provider failures = %d, want 0", counts.TotalFailures)
	}
}

func TestFallbackExecutor_Configured429FallsBackWithoutBreakerFailure(t *testing.T) {
	cbm := NewCircuitBreakerManager()
	fallback := NewFallbackExecutor(NewRetrier(config.RetryConfig{
		Enabled:         true,
		MaxAttempts:     1,
		RetryableErrors: []int{http.StatusTooManyRequests},
	}), cbm)
	primary := routing.Target{Provider: "primary", Model: "model-a"}
	backup := routing.Target{Provider: "backup", Model: "model-b"}

	resp, usedTarget, fallbackUsed, _, _, err := fallback.Execute(
		context.Background(),
		primary,
		[]routing.Target{backup},
		func(_ context.Context, target routing.Target) (*http.Response, error) {
			if target.Provider == primary.Provider {
				return &http.Response{
					StatusCode: http.StatusTooManyRequests,
					Body:       io.NopCloser(strings.NewReader(`{"error":"rate_limited"}`)),
				}, nil
			}
			return &http.Response{StatusCode: http.StatusOK, Body: http.NoBody}, nil
		},
	)

	if err != nil {
		t.Fatalf("unexpected fallback error: %v", err)
	}
	if resp == nil || resp.StatusCode != http.StatusOK || usedTarget != backup || !fallbackUsed {
		t.Fatalf("resp/target/fallback = %#v/%#v/%v, want 200/backup/true", resp, usedTarget, fallbackUsed)
	}
	if counts := cbm.Get(primary.Provider).Counts(); counts.TotalFailures != 0 {
		t.Fatalf("429 provider failures = %d, want 0", counts.TotalFailures)
	}
}

func TestFallbackExecutor_Unconfigured5xxFallsBackAndCountsFailure(t *testing.T) {
	cbm := NewCircuitBreakerManager()
	fallback := NewFallbackExecutor(NewRetrier(config.RetryConfig{
		Enabled:         true,
		MaxAttempts:     3,
		RetryableErrors: []int{http.StatusTooManyRequests},
	}), cbm)
	primary := routing.Target{Provider: "primary", Model: "model-a"}
	backup := routing.Target{Provider: "backup", Model: "model-b"}
	primaryCalls := 0

	resp, usedTarget, fallbackUsed, _, _, err := fallback.Execute(
		context.Background(),
		primary,
		[]routing.Target{backup},
		func(_ context.Context, target routing.Target) (*http.Response, error) {
			if target.Provider == primary.Provider {
				primaryCalls++
				return &http.Response{StatusCode: http.StatusServiceUnavailable, Body: http.NoBody}, nil
			}
			return &http.Response{StatusCode: http.StatusOK, Body: http.NoBody}, nil
		},
	)

	if err != nil {
		t.Fatalf("unexpected fallback error: %v", err)
	}
	if resp == nil || resp.StatusCode != http.StatusOK || usedTarget != backup || !fallbackUsed {
		t.Fatalf("resp/target/fallback = %#v/%#v/%v, want 200/backup/true", resp, usedTarget, fallbackUsed)
	}
	if primaryCalls != 1 {
		t.Fatalf("primary calls = %d, want 1 without configured retry", primaryCalls)
	}
	if counts := cbm.Get(primary.Provider).Counts(); counts.TotalFailures != 1 {
		t.Fatalf("503 provider failures = %d, want 1", counts.TotalFailures)
	}
}

func TestFallbackExecutor_PreservesOnlyFinalTargetSnapshot(t *testing.T) {
	tests := []struct {
		name        string
		finalStatus int
	}{
		{name: "rate limit", finalStatus: http.StatusTooManyRequests},
		{name: "server error", finalStatus: http.StatusBadGateway},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fallback := NewFallbackExecutor(NewRetrier(config.RetryConfig{
				Enabled:         true,
				MaxAttempts:     1,
				RetryableErrors: []int{http.StatusTooManyRequests, http.StatusBadGateway},
			}), NewCircuitBreakerManager())
			primary := routing.Target{Provider: "primary", Model: "model-a"}
			backup := routing.Target{Provider: "backup", Model: "model-b"}
			primaryBody := &trackingResponseBody{Reader: strings.NewReader(`{"error":"primary-503"}`)}
			finalBody := &trackingResponseBody{Reader: strings.NewReader(`{"error":"final-target"}`)}

			resp, usedTarget, fallbackUsed, _, _, err := fallback.Execute(
				context.Background(),
				primary,
				[]routing.Target{backup},
				func(_ context.Context, target routing.Target) (*http.Response, error) {
					if target.Provider == primary.Provider {
						return &http.Response{
							StatusCode: http.StatusServiceUnavailable,
							Header:     http.Header{"X-Upstream-Target": []string{"primary"}},
							Body:       primaryBody,
						}, nil
					}
					return &http.Response{
						StatusCode: tt.finalStatus,
						Header:     http.Header{"X-Upstream-Target": []string{"backup"}},
						Body:       finalBody,
					}, nil
				},
			)

			if resp != nil {
				t.Fatalf("response = %#v, want nil", resp)
			}
			if usedTarget != backup || !fallbackUsed {
				t.Fatalf("target/fallback = %#v/%v, want backup/true", usedTarget, fallbackUsed)
			}
			if err == nil {
				t.Fatal("expected final fallback error")
			}
			var statusErr *RetryableStatusError
			if !errors.As(err, &statusErr) {
				t.Fatalf("error = %v, want wrapped RetryableStatusError", err)
			}
			if statusErr.StatusCode != tt.finalStatus {
				t.Fatalf("snapshot status = %d, want %d", statusErr.StatusCode, tt.finalStatus)
			}
			if got, want := string(statusErr.Body), `{"error":"final-target"}`; got != want {
				t.Fatalf("snapshot body = %q, want %q", got, want)
			}
			if got := statusErr.Headers.Get("X-Upstream-Target"); got != "backup" {
				t.Fatalf("snapshot target header = %q, want backup", got)
			}
			if strings.Contains(string(statusErr.Body), "primary-503") || strings.Contains(err.Error(), "primary-503") {
				t.Fatalf("primary snapshot leaked into final error: %#v / %q", statusErr, err.Error())
			}
			if !primaryBody.closed || !finalBody.closed {
				t.Fatalf("closed primary/final = %v/%v, want true/true", primaryBody.closed, finalBody.closed)
			}
		})
	}
}
