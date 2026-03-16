package resilience

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/rs/zerolog/log"
	"github.com/sony/gobreaker"
)

// CircuitBreakerManager manages per-provider circuit breakers.
type CircuitBreakerManager struct {
	mu       sync.RWMutex
	breakers map[string]*gobreaker.CircuitBreaker
	settings gobreaker.Settings
}

// NewCircuitBreakerManager creates a new manager with default settings.
func NewCircuitBreakerManager() *CircuitBreakerManager {
	return &CircuitBreakerManager{
		breakers: make(map[string]*gobreaker.CircuitBreaker),
		settings: gobreaker.Settings{
			Timeout:    30 * time.Second,
			Interval:   60 * time.Second,
			MaxRequests: 3,
			IsSuccessful: func(err error) bool {
				if err == nil {
					return true
				}
				if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
					return true
				}
				return false
			},
			ReadyToTrip: func(counts gobreaker.Counts) bool {
				return counts.ConsecutiveFailures >= 5
			},
			OnStateChange: func(name string, from gobreaker.State, to gobreaker.State) {
				log.Warn().
					Str("provider", name).
					Str("from", from.String()).
					Str("to", to.String()).
					Msg("circuit breaker state change")
			},
		},
	}
}

// Get returns (or creates) the circuit breaker for a given provider.
func (m *CircuitBreakerManager) Get(provider string) *gobreaker.CircuitBreaker {
	m.mu.RLock()
	cb, ok := m.breakers[provider]
	m.mu.RUnlock()
	if ok {
		return cb
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	// Double-check after write lock
	if cb, ok = m.breakers[provider]; ok {
		return cb
	}

	settings := m.settings
	settings.Name = provider
	cb = gobreaker.NewCircuitBreaker(settings)
	m.breakers[provider] = cb

	log.Debug().Str("provider", provider).Msg("created circuit breaker")
	return cb
}

// Execute runs a function through the provider's circuit breaker.
func (m *CircuitBreakerManager) Execute(provider string, fn func() (interface{}, error)) (interface{}, error) {
	cb := m.Get(provider)
	result, err := cb.Execute(fn)
	if err != nil {
		return nil, fmt.Errorf("circuit breaker [%s]: %w", provider, err)
	}
	return result, nil
}

// State returns the current state of a provider's circuit breaker.
func (m *CircuitBreakerManager) State(provider string) gobreaker.State {
	cb := m.Get(provider)
	return cb.State()
}
