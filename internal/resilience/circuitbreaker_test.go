package resilience

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"testing"

	"github.com/sony/gobreaker"
)

func TestCircuitBreakerFailureClassification(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want bool
	}{
		{name: "success", want: true},
		{name: "request translation", err: NewRequestError(errors.New("invalid request")), want: true},
		{name: "client cancellation", err: context.Canceled, want: true},
		{name: "client deadline", err: context.DeadlineExceeded, want: true},
		{name: "rate limit", err: fmt.Errorf("retry exhausted: %w", &RetryableStatusError{StatusCode: http.StatusTooManyRequests}), want: true},
		{name: "other client status", err: &RetryableStatusError{StatusCode: http.StatusBadRequest}, want: true},
		{name: "server status", err: &RetryableStatusError{StatusCode: http.StatusServiceUnavailable}, want: false},
		{name: "network failure", err: errors.New("connection reset by peer"), want: false},
		{name: "upstream timeout", err: errors.New("upstream timed out waiting for first token"), want: false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := isCircuitBreakerSuccess(tc.err); got != tc.want {
				t.Fatalf("isCircuitBreakerSuccess(%v) = %v, want %v", tc.err, got, tc.want)
			}
		})
	}
}

func TestCircuitBreakerConfigured429NeverTrips(t *testing.T) {
	manager := NewCircuitBreakerManager()
	for i := 0; i < 6; i++ {
		_, err := manager.Execute("rate-limited", func() (interface{}, error) {
			return nil, &RetryableStatusError{StatusCode: http.StatusTooManyRequests}
		})
		if err == nil {
			t.Fatal("expected the caller to receive the 429 failure")
		}
	}

	if state := manager.State("rate-limited"); state != gobreaker.StateClosed {
		t.Fatalf("429 circuit state = %s, want closed", state)
	}
	if counts := manager.Get("rate-limited").Counts(); counts.TotalFailures != 0 {
		t.Fatalf("429 provider failures = %d, want 0", counts.TotalFailures)
	}
}

func TestCircuitBreakerNetworkAnd5xxFailuresTrip(t *testing.T) {
	tests := []struct {
		name string
		err  error
	}{
		{name: "network", err: errors.New("connection reset by peer")},
		{name: "timeout", err: errors.New("upstream timed out waiting for first token")},
		{name: "server status", err: &RetryableStatusError{StatusCode: http.StatusBadGateway}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			manager := NewCircuitBreakerManager()
			for i := 0; i < 5; i++ {
				_, err := manager.Execute(tc.name, func() (interface{}, error) {
					return nil, tc.err
				})
				if err == nil {
					t.Fatal("expected provider failure")
				}
			}
			if state := manager.State(tc.name); state != gobreaker.StateOpen {
				t.Fatalf("circuit state = %s, want open", state)
			}
		})
	}
}
