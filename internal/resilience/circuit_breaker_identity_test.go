package resilience

import (
	"bytes"
	"context"
	"errors"
	"net/http"
	"strings"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/routing"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
	"github.com/sony/gobreaker"
)

func TestFallbackExecutorScopesCircuitStateToTargetIdentity(t *testing.T) {
	var logOutput bytes.Buffer
	previousLogger := log.Logger
	log.Logger = zerolog.New(&logOutput)
	t.Cleanup(func() { log.Logger = previousLogger })

	manager := NewCircuitBreakerManager()
	executor := NewFallbackExecutor(NewRetrier(config.RetryConfig{Enabled: false}), manager)
	const provider = "shared-provider"
	const oldKey = "old-opaque-account-fingerprint"
	const newKey = "new-opaque-account-fingerprint"
	oldTarget := routing.Target{Provider: provider, Model: "old-model"}.WithCircuitBreakerKey(oldKey)
	calls := 0
	fail := func(context.Context, routing.Target) (*http.Response, error) {
		calls++
		return nil, errors.New("connection reset by peer")
	}

	for i := 0; i < 5; i++ {
		_, _, _, _, _, err := executor.Execute(context.Background(), oldTarget, nil, fail)
		if err == nil {
			t.Fatal("expected provider failure")
		}
	}
	if calls != 5 {
		t.Fatalf("old identity calls = %d, want 5", calls)
	}
	if state := manager.stateForKey(oldKey, provider); state != gobreaker.StateOpen {
		t.Fatalf("old identity state = %s, want open", state)
	}

	modelOnlyChange := routing.Target{Provider: provider, Model: "new-model"}.WithCircuitBreakerKey(oldKey)
	_, _, _, _, _, err := executor.Execute(context.Background(), modelOnlyChange, nil, fail)
	if err == nil {
		t.Fatal("expected unchanged identity to remain open")
	}
	openErr := err
	if calls != 5 {
		t.Fatalf("open unchanged identity executed request; calls = %d", calls)
	}

	newCalls := 0
	newTarget := routing.Target{Provider: provider, Model: "new-model"}.WithCircuitBreakerKey(newKey)
	resp, used, fallbackUsed, _, state, err := executor.Execute(
		context.Background(),
		newTarget,
		nil,
		func(context.Context, routing.Target) (*http.Response, error) {
			newCalls++
			return &http.Response{StatusCode: http.StatusOK, Body: http.NoBody}, nil
		},
	)
	if err != nil {
		t.Fatalf("new identity request failed: %v", err)
	}
	if resp == nil || used != newTarget || fallbackUsed || state != gobreaker.StateClosed.String() {
		t.Fatalf("new identity result = %#v/%#v/%v/%q", resp, used, fallbackUsed, state)
	}
	if newCalls != 1 {
		t.Fatalf("new identity calls = %d, want 1", newCalls)
	}

	combinedOutput := logOutput.String() + openErr.Error()
	if strings.Contains(combinedOutput, oldKey) || strings.Contains(combinedOutput, newKey) {
		t.Fatal("logs or errors expose circuit-breaker identity")
	}
	if !strings.Contains(combinedOutput, provider) {
		t.Fatalf("safe provider label missing from diagnostics: %s", combinedOutput)
	}
}

func TestFallbackExecutorUsesIndependentFallbackIdentity(t *testing.T) {
	manager := NewCircuitBreakerManager()
	executor := NewFallbackExecutor(NewRetrier(config.RetryConfig{Enabled: false}), manager)
	const provider = "shared-provider"
	const primaryKey = "primary-identity"
	const fallbackKey = "fallback-identity"
	primary := routing.Target{Provider: provider, Model: "primary"}.WithCircuitBreakerKey(primaryKey)
	fallback := routing.Target{Provider: provider, Model: "fallback"}.WithCircuitBreakerKey(fallbackKey)

	for i := 0; i < 5; i++ {
		_, _, _, _, _, err := executor.Execute(
			WithFallbackDisabled(context.Background()),
			primary,
			nil,
			func(context.Context, routing.Target) (*http.Response, error) {
				return nil, errors.New("upstream unavailable")
			},
		)
		if err == nil {
			t.Fatal("expected primary failure")
		}
	}

	fallbackCalls := 0
	resp, used, fallbackUsed, _, _, err := executor.Execute(
		context.Background(),
		primary,
		[]routing.Target{fallback},
		func(_ context.Context, target routing.Target) (*http.Response, error) {
			if target.CircuitBreakerKey() != fallbackKey {
				t.Fatalf("open primary unexpectedly reached callback: %#v", target)
			}
			fallbackCalls++
			return &http.Response{StatusCode: http.StatusOK, Body: http.NoBody}, nil
		},
	)
	if err != nil {
		t.Fatalf("independent fallback failed: %v", err)
	}
	if resp == nil || used != fallback || !fallbackUsed || fallbackCalls != 1 {
		t.Fatalf("fallback result = %#v/%#v/%v calls=%d", resp, used, fallbackUsed, fallbackCalls)
	}
	if state := manager.stateForKey(primaryKey, provider); state != gobreaker.StateOpen {
		t.Fatalf("primary identity state = %s, want open", state)
	}
	if state := manager.stateForKey(fallbackKey, provider); state != gobreaker.StateClosed {
		t.Fatalf("fallback identity state = %s, want closed", state)
	}
}

func TestCircuitBreakerPublicLookupPreservesExactProviderAlias(t *testing.T) {
	manager := NewCircuitBreakerManager()
	spacedAlias := " shared-provider "
	plainAlias := "shared-provider"
	spacedBreaker := manager.Get(spacedAlias)
	plainBreaker := manager.Get(plainAlias)
	if spacedBreaker == plainBreaker {
		t.Fatal("distinct exact provider aliases share a public breaker lookup")
	}

	if _, err := manager.Execute(spacedAlias, func() (interface{}, error) {
		return nil, errors.New("upstream unavailable")
	}); err == nil {
		t.Fatal("expected spaced-alias provider failure")
	}
	if got := manager.Get(spacedAlias); got != spacedBreaker {
		t.Fatal("Get and Execute used different identities for the same exact alias")
	}
	if failures := spacedBreaker.Counts().TotalFailures; failures != 1 {
		t.Fatalf("spaced-alias failures = %d, want 1", failures)
	}
	if failures := plainBreaker.Counts().TotalFailures; failures != 0 {
		t.Fatalf("plain-alias failures = %d, want 0", failures)
	}
}

func TestCircuitBreakerIdentityCacheIsBoundedAndPreservesInFlightEntries(t *testing.T) {
	manager := newCircuitBreakerManager(2)
	started := make(chan struct{})
	release := make(chan struct{})
	done := make(chan error, 1)
	go func() {
		_, err := manager.executeForKey("active-key", "active-provider", func() (interface{}, error) {
			close(started)
			<-release
			return nil, nil
		})
		done <- err
	}()
	<-started

	if _, err := manager.executeForKey("idle-key", "idle-provider", func() (interface{}, error) {
		return nil, nil
	}); err != nil {
		t.Fatalf("seed idle breaker: %v", err)
	}
	if _, err := manager.executeForKey("new-key", "new-provider", func() (interface{}, error) {
		return nil, nil
	}); err != nil {
		t.Fatalf("create replacement breaker: %v", err)
	}

	manager.mu.Lock()
	_, activeRetained := manager.breakers["active-key"]
	_, idleRetained := manager.breakers["idle-key"]
	_, newRetained := manager.breakers["new-key"]
	entryCount := len(manager.breakers)
	manager.mu.Unlock()
	if !activeRetained || idleRetained || !newRetained || entryCount != 2 {
		t.Fatalf("bounded cache active/idle/new/count = %v/%v/%v/%d", activeRetained, idleRetained, newRetained, entryCount)
	}

	close(release)
	if err := <-done; err != nil {
		t.Fatalf("in-flight breaker failed after cache churn: %v", err)
	}

	for i := 0; i < 100; i++ {
		key := "rotated-" + strings.Repeat("x", i+1)
		if _, err := manager.executeForKey(key, "rotating-provider", func() (interface{}, error) {
			return nil, nil
		}); err != nil {
			t.Fatalf("execute rotated identity %d: %v", i, err)
		}
	}
	manager.mu.Lock()
	entryCount = len(manager.breakers)
	manager.mu.Unlock()
	if entryCount > 2 {
		t.Fatalf("circuit-breaker cache grew beyond limit: %d", entryCount)
	}
}

func TestCircuitBreakerIdentityCacheDoesNotEvictWhenAllEntriesAreInFlight(t *testing.T) {
	manager := newCircuitBreakerManager(2)
	release := make(chan struct{})
	started := make(chan struct{}, 2)
	done := make(chan error, 2)
	for _, key := range []string{"active-a", "active-b"} {
		key := key
		go func() {
			_, err := manager.executeForKey(key, "active-provider", func() (interface{}, error) {
				started <- struct{}{}
				<-release
				return nil, nil
			})
			done <- err
		}()
	}
	<-started
	<-started

	if _, err := manager.executeForKey("overflow", "overflow-provider", func() (interface{}, error) {
		return nil, nil
	}); err != nil {
		t.Fatalf("overflow execution failed: %v", err)
	}
	manager.mu.Lock()
	_, hasA := manager.breakers["active-a"]
	_, hasB := manager.breakers["active-b"]
	_, storedOverflow := manager.breakers["overflow"]
	entryCount := len(manager.breakers)
	manager.mu.Unlock()
	if !hasA || !hasB || storedOverflow || entryCount != 2 {
		t.Fatalf("all-active cache a/b/overflow/count = %v/%v/%v/%d", hasA, hasB, storedOverflow, entryCount)
	}

	close(release)
	for i := 0; i < 2; i++ {
		if err := <-done; err != nil {
			t.Fatalf("active execution failed: %v", err)
		}
	}
}
