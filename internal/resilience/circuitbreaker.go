package resilience

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/rs/zerolog/log"
	"github.com/sony/gobreaker"
)

// CircuitBreakerManager manages circuit breakers by opaque provider identity.
// Provider aliases remain separate display labels and are never used to expose
// the internal identity key.
type CircuitBreakerManager struct {
	mu         sync.Mutex
	breakers   map[string]*circuitBreakerEntry
	settings   gobreaker.Settings
	maxEntries int
	sequence   uint64
}

type circuitBreakerEntry struct {
	breaker  *gobreaker.CircuitBreaker
	provider string
	lastUsed uint64
	inFlight int
}

const defaultMaxCircuitBreakers = 1024

// NewCircuitBreakerManager creates a new manager with default settings.
func NewCircuitBreakerManager() *CircuitBreakerManager {
	return newCircuitBreakerManager(defaultMaxCircuitBreakers)
}

func newCircuitBreakerManager(maxEntries int) *CircuitBreakerManager {
	if maxEntries < 1 {
		maxEntries = 1
	}
	return &CircuitBreakerManager{
		breakers:   make(map[string]*circuitBreakerEntry),
		maxEntries: maxEntries,
		settings: gobreaker.Settings{
			Timeout:     30 * time.Second,
			Interval:    60 * time.Second,
			MaxRequests: 3,
			IsSuccessful: func(err error) bool {
				return isCircuitBreakerSuccess(err)
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

func isCircuitBreakerSuccess(err error) bool {
	if err == nil || IsRequestError(err) {
		return true
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return true
	}

	var statusErr *RetryableStatusError
	if errors.As(err, &statusErr) && statusErr.StatusCode >= http.StatusBadRequest && statusErr.StatusCode < http.StatusInternalServerError {
		// Client errors, including configured 429 retry exhaustion, do not
		// indicate a broken provider and must not trip its circuit.
		return true
	}
	return false
}

// Get returns (or creates) the circuit breaker for a given provider.
func (m *CircuitBreakerManager) Get(provider string) *gobreaker.CircuitBreaker {
	_, provider = normalizeCircuitBreakerIdentity("", provider)
	m.mu.Lock()
	defer m.mu.Unlock()
	if entry := m.latestProviderEntryLocked(provider); entry != nil {
		m.touchLocked(entry)
		return entry.breaker
	}
	breaker, _ := m.getOrCreateLocked(provider, provider)
	return breaker
}

func (m *CircuitBreakerManager) getForKey(key string, provider string) *gobreaker.CircuitBreaker {
	key, provider = normalizeCircuitBreakerIdentity(key, provider)
	m.mu.Lock()
	defer m.mu.Unlock()
	breaker, _ := m.getOrCreateLocked(key, provider)
	return breaker
}

// Execute runs a function through the provider's circuit breaker.
func (m *CircuitBreakerManager) Execute(provider string, fn func() (interface{}, error)) (interface{}, error) {
	return m.executeForKey(provider, provider, fn)
}

func (m *CircuitBreakerManager) executeForKey(key string, provider string, fn func() (interface{}, error)) (interface{}, error) {
	result, _, err := m.executeForKeyWithState(key, provider, fn)
	return result, err
}

func (m *CircuitBreakerManager) executeForKeyWithState(key string, provider string, fn func() (interface{}, error)) (interface{}, gobreaker.State, error) {
	key, provider = normalizeCircuitBreakerIdentity(key, provider)
	cb, release := m.acquireForExecution(key, provider)
	defer release()
	result, err := cb.Execute(fn)
	state := cb.State()
	if err != nil {
		return nil, state, fmt.Errorf("circuit breaker [%s]: %w", provider, err)
	}
	return result, state, nil
}

// State returns the current state of a provider's circuit breaker.
func (m *CircuitBreakerManager) State(provider string) gobreaker.State {
	return m.Get(provider).State()
}

func (m *CircuitBreakerManager) stateForKey(key string, provider string) gobreaker.State {
	cb := m.getForKey(key, provider)
	return cb.State()
}

func normalizeCircuitBreakerIdentity(key string, provider string) (string, string) {
	rawProvider := provider
	if strings.TrimSpace(provider) == "" {
		provider = "unknown"
	}
	if strings.TrimSpace(key) == "" {
		key = rawProvider
		if strings.TrimSpace(key) == "" {
			key = provider
		}
	}
	return key, provider
}

func (m *CircuitBreakerManager) acquireForExecution(key string, provider string) (*gobreaker.CircuitBreaker, func()) {
	m.mu.Lock()
	breaker, tracked := m.getOrCreateLocked(key, provider)
	if tracked {
		m.breakers[key].inFlight++
	}
	m.mu.Unlock()

	if !tracked {
		return breaker, func() {}
	}
	return breaker, func() {
		m.mu.Lock()
		if entry, ok := m.breakers[key]; ok && entry.breaker == breaker && entry.inFlight > 0 {
			entry.inFlight--
			m.touchLocked(entry)
		}
		m.mu.Unlock()
	}
}

func (m *CircuitBreakerManager) getOrCreateLocked(key string, provider string) (*gobreaker.CircuitBreaker, bool) {
	if entry, ok := m.breakers[key]; ok {
		m.touchLocked(entry)
		return entry.breaker, true
	}

	if len(m.breakers) >= m.maxEntries && !m.evictOldestIdleLocked() {
		// Every retained breaker is executing. Use an untracked breaker for
		// this request rather than evicting active state or sharing state
		// between unrelated identities.
		return m.newBreaker(provider), false
	}

	entry := &circuitBreakerEntry{breaker: m.newBreaker(provider), provider: provider}
	m.touchLocked(entry)
	m.breakers[key] = entry
	return entry.breaker, true
}

func (m *CircuitBreakerManager) touchLocked(entry *circuitBreakerEntry) {
	m.sequence++
	entry.lastUsed = m.sequence
}

func (m *CircuitBreakerManager) latestProviderEntryLocked(provider string) *circuitBreakerEntry {
	var latest *circuitBreakerEntry
	for _, entry := range m.breakers {
		if entry.provider != provider || latest != nil && entry.lastUsed <= latest.lastUsed {
			continue
		}
		latest = entry
	}
	return latest
}

func (m *CircuitBreakerManager) evictOldestIdleLocked() bool {
	oldestKey := ""
	oldestSequence := ^uint64(0)
	for key, entry := range m.breakers {
		if entry.inFlight != 0 || entry.lastUsed >= oldestSequence {
			continue
		}
		oldestKey = key
		oldestSequence = entry.lastUsed
	}
	if oldestKey == "" {
		return false
	}
	delete(m.breakers, oldestKey)
	return true
}

func (m *CircuitBreakerManager) newBreaker(provider string) *gobreaker.CircuitBreaker {
	settings := m.settings
	settings.Name = provider
	breaker := gobreaker.NewCircuitBreaker(settings)
	log.Debug().Str("provider", provider).Msg("created circuit breaker")
	return breaker
}
