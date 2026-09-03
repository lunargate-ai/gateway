package api

import (
	"container/list"
	"encoding/json"
	"sync"
	"time"
)

const (
	defaultResponsesStateMaxEntries = 1000
	defaultResponsesStateMaxBytes   = 64 << 20
)

type responsesStateStore struct {
	mu         sync.Mutex
	ttl        time.Duration
	maxEntries int
	maxBytes   int
	totalBytes int
	entries    map[string]*responsesStateEntry
	order      *list.List
}

type responsesStateEntry struct {
	payload   map[string]json.RawMessage
	expiresAt time.Time
	size      int
	element   *list.Element
}

func newResponsesStateStore(ttl time.Duration) *responsesStateStore {
	if ttl <= 0 {
		ttl = 30 * time.Minute
	}
	return &responsesStateStore{
		ttl:        ttl,
		maxEntries: defaultResponsesStateMaxEntries,
		maxBytes:   defaultResponsesStateMaxBytes,
		entries:    make(map[string]*responsesStateEntry),
		order:      list.New(),
	}
}

func (s *responsesStateStore) get(responseID string) (map[string]json.RawMessage, bool) {
	if s == nil || responseID == "" {
		return nil, false
	}

	now := time.Now()

	s.mu.Lock()
	defer s.mu.Unlock()
	entry, ok := s.entries[responseID]
	if !ok {
		return nil, false
	}
	if now.After(entry.expiresAt) {
		s.removeLocked(responseID, entry)
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
	cloned := cloneResponsesRawMap(payload)
	size := responsesStatePayloadSize(responseID, cloned)
	if size > s.maxBytes {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	s.cleanupExpiredLocked(now)
	if existing, ok := s.entries[responseID]; ok {
		s.removeLocked(responseID, existing)
	}
	for len(s.entries) >= s.maxEntries || s.totalBytes+size > s.maxBytes {
		if !s.removeOldestLocked() {
			break
		}
	}
	element := s.order.PushBack(responseID)
	s.entries[responseID] = &responsesStateEntry{
		payload:   cloned,
		expiresAt: expiresAt,
		size:      size,
		element:   element,
	}
	s.totalBytes += size
}

func (s *responsesStateStore) delete(responseID string) {
	if s == nil || responseID == "" {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if entry, ok := s.entries[responseID]; ok {
		s.removeLocked(responseID, entry)
	}
}

func (s *responsesStateStore) cleanupExpiredLocked(now time.Time) {
	for element := s.order.Front(); element != nil; {
		responseID, _ := element.Value.(string)
		entry := s.entries[responseID]
		if entry == nil || !now.After(entry.expiresAt) {
			return
		}
		next := element.Next()
		s.removeLocked(responseID, entry)
		element = next
	}
}

func (s *responsesStateStore) removeOldestLocked() bool {
	element := s.order.Front()
	if element == nil {
		return false
	}
	responseID, _ := element.Value.(string)
	entry := s.entries[responseID]
	if entry == nil {
		s.order.Remove(element)
		return true
	}
	s.removeLocked(responseID, entry)
	return true
}

func (s *responsesStateStore) removeLocked(responseID string, entry *responsesStateEntry) {
	delete(s.entries, responseID)
	if entry == nil {
		return
	}
	if entry.element != nil {
		s.order.Remove(entry.element)
	}
	s.totalBytes -= entry.size
	if s.totalBytes < 0 {
		s.totalBytes = 0
	}
}

func responsesStatePayloadSize(responseID string, payload map[string]json.RawMessage) int {
	size := len(responseID)
	for key, value := range payload {
		size += len(key) + len(value)
	}
	return size
}
