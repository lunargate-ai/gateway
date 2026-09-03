package api

import (
	"container/list"
	"strings"
	"sync"
	"time"
)

const (
	defaultResponseBindingMaxEntries = 1000
	defaultResponseBindingMaxBytes   = 1 << 20
)

// responseBinding identifies the configured provider account that owns a
// native Responses resource. Account-sensitive configuration is retained only
// as a one-way digest; no credential or endpoint is stored directly.
type responseBinding struct {
	Provider            string
	Route               string
	Model               string
	UpstreamRequestType string
	AccountFingerprint  string
}

type responseBindingStore struct {
	mu         sync.Mutex
	ttl        time.Duration
	maxEntries int
	maxBytes   int
	totalBytes int
	entries    map[string]*responseBindingEntry
	order      *list.List
}

type responseBindingEntry struct {
	binding   responseBinding
	expiresAt time.Time
	size      int
	element   *list.Element
}

func newResponseBindingStore(ttl time.Duration) *responseBindingStore {
	if ttl <= 0 {
		ttl = 30 * time.Minute
	}
	return &responseBindingStore{
		ttl:        ttl,
		maxEntries: defaultResponseBindingMaxEntries,
		maxBytes:   defaultResponseBindingMaxBytes,
		entries:    make(map[string]*responseBindingEntry),
		order:      list.New(),
	}
}

func (s *responseBindingStore) put(responseID string, binding responseBinding) bool {
	responseID = strings.TrimSpace(responseID)
	binding.Provider = strings.TrimSpace(binding.Provider)
	binding.Route = strings.TrimSpace(binding.Route)
	binding.Model = strings.TrimSpace(binding.Model)
	binding.UpstreamRequestType = strings.TrimSpace(binding.UpstreamRequestType)
	binding.AccountFingerprint = strings.TrimSpace(binding.AccountFingerprint)
	if s == nil || responseID == "" || binding.Provider == "" || binding.AccountFingerprint == "" {
		return false
	}
	size := responseBindingSize(responseID, binding)
	if size > s.maxBytes {
		return false
	}

	now := time.Now()
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cleanupExpiredLocked(now)
	if existing := s.entries[responseID]; existing != nil {
		s.removeLocked(responseID, existing)
	}
	for len(s.entries) >= s.maxEntries || s.totalBytes+size > s.maxBytes {
		if !s.removeOldestLocked() {
			return false
		}
	}
	element := s.order.PushBack(responseID)
	s.entries[responseID] = &responseBindingEntry{
		binding:   binding,
		expiresAt: now.Add(s.ttl),
		size:      size,
		element:   element,
	}
	s.totalBytes += size
	return true
}

func (s *responseBindingStore) get(responseID string) (responseBinding, bool) {
	responseID = strings.TrimSpace(responseID)
	if s == nil || responseID == "" {
		return responseBinding{}, false
	}

	now := time.Now()
	s.mu.Lock()
	defer s.mu.Unlock()
	entry := s.entries[responseID]
	if entry == nil {
		return responseBinding{}, false
	}
	if now.After(entry.expiresAt) {
		s.removeLocked(responseID, entry)
		return responseBinding{}, false
	}
	return entry.binding, true
}

func (s *responseBindingStore) delete(responseID string) bool {
	responseID = strings.TrimSpace(responseID)
	if s == nil || responseID == "" {
		return false
	}

	now := time.Now()
	s.mu.Lock()
	defer s.mu.Unlock()
	entry := s.entries[responseID]
	if entry == nil {
		return false
	}
	if now.After(entry.expiresAt) {
		s.removeLocked(responseID, entry)
		return false
	}
	s.removeLocked(responseID, entry)
	return true
}

func (s *responseBindingStore) cleanupExpiredLocked(now time.Time) {
	for element := s.order.Front(); element != nil; {
		responseID, _ := element.Value.(string)
		entry := s.entries[responseID]
		if entry == nil {
			next := element.Next()
			s.order.Remove(element)
			element = next
			continue
		}
		if !now.After(entry.expiresAt) {
			return
		}
		next := element.Next()
		s.removeLocked(responseID, entry)
		element = next
	}
}

func (s *responseBindingStore) removeOldestLocked() bool {
	element := s.order.Front()
	if element == nil {
		return false
	}
	responseID, _ := element.Value.(string)
	s.removeLocked(responseID, s.entries[responseID])
	return true
}

func (s *responseBindingStore) removeLocked(responseID string, entry *responseBindingEntry) {
	delete(s.entries, responseID)
	if entry != nil && entry.element != nil {
		s.order.Remove(entry.element)
	}
	if entry != nil {
		s.totalBytes -= entry.size
		if s.totalBytes < 0 {
			s.totalBytes = 0
		}
	}
}

func responseBindingSize(responseID string, binding responseBinding) int {
	return len(responseID) + len(binding.Provider) + len(binding.Route) + len(binding.Model) + len(binding.UpstreamRequestType) + len(binding.AccountFingerprint)
}
