package api

import (
	"encoding/json"
	"sync"
	"time"
)

type responsesStateStore struct {
	mu      sync.RWMutex
	ttl     time.Duration
	entries map[string]responsesStateEntry
}

type responsesStateEntry struct {
	payload   map[string]json.RawMessage
	expiresAt time.Time
}

func newResponsesStateStore(ttl time.Duration) *responsesStateStore {
	if ttl <= 0 {
		ttl = 30 * time.Minute
	}
	return &responsesStateStore{
		ttl:     ttl,
		entries: make(map[string]responsesStateEntry),
	}
}

func (s *responsesStateStore) get(responseID string) (map[string]json.RawMessage, bool) {
	if s == nil || responseID == "" {
		return nil, false
	}

	now := time.Now()

	s.mu.RLock()
	entry, ok := s.entries[responseID]
	s.mu.RUnlock()
	if !ok {
		return nil, false
	}
	if now.After(entry.expiresAt) {
		s.mu.Lock()
		delete(s.entries, responseID)
		s.mu.Unlock()
		return nil, false
	}

	return cloneResponsesRawMap(entry.payload), true
}

func (s *responsesStateStore) put(responseID string, payload map[string]json.RawMessage) {
	if s == nil || responseID == "" || len(payload) == 0 {
		return
	}

	now := time.Now()
	expiresAt := now.Add(s.ttl)

	s.mu.Lock()
	s.cleanupExpiredLocked(now)
	s.entries[responseID] = responsesStateEntry{
		payload:   cloneResponsesRawMap(payload),
		expiresAt: expiresAt,
	}
	s.mu.Unlock()
}

func (s *responsesStateStore) delete(responseID string) {
	if s == nil || responseID == "" {
		return
	}
	s.mu.Lock()
	delete(s.entries, responseID)
	s.mu.Unlock()
}

func (s *responsesStateStore) cleanupExpiredLocked(now time.Time) {
	for responseID, entry := range s.entries {
		if now.After(entry.expiresAt) {
			delete(s.entries, responseID)
		}
	}
}
