package api

import (
	"container/list"
	"encoding/json"
	"strings"
	"sync"
	"time"

	"github.com/lunargate-ai/gateway/pkg/models"
)

const (
	defaultChatCompletionBindingMaxEntries = 1000
	defaultChatCompletionBindingMaxBytes   = 1 << 20
)

// chatCompletionBinding identifies the configured provider account that owns
// a stored native Chat Completion. Account-sensitive configuration is retained
// only as a one-way digest; no credential or endpoint is stored directly.
type chatCompletionBinding struct {
	Provider           string
	Route              string
	Model              string
	AccountFingerprint string
}

type chatCompletionBindingStore struct {
	mu         sync.Mutex
	ttl        time.Duration
	maxEntries int
	maxBytes   int
	totalBytes int
	entries    map[string]*chatCompletionBindingEntry
	order      *list.List
	now        func() time.Time
}

type chatCompletionBindingEntry struct {
	binding   chatCompletionBinding
	expiresAt time.Time
	size      int
	element   *list.Element
}

func newChatCompletionBindingStore(ttl time.Duration) *chatCompletionBindingStore {
	if ttl <= 0 {
		ttl = 30 * time.Minute
	}
	return &chatCompletionBindingStore{
		ttl:        ttl,
		maxEntries: defaultChatCompletionBindingMaxEntries,
		maxBytes:   defaultChatCompletionBindingMaxBytes,
		entries:    make(map[string]*chatCompletionBindingEntry),
		order:      list.New(),
		now:        time.Now,
	}
}

func (s *chatCompletionBindingStore) put(completionID string, binding chatCompletionBinding) bool {
	completionID = strings.TrimSpace(completionID)
	binding.Provider = strings.TrimSpace(binding.Provider)
	binding.Route = strings.TrimSpace(binding.Route)
	binding.Model = strings.TrimSpace(binding.Model)
	binding.AccountFingerprint = strings.TrimSpace(binding.AccountFingerprint)
	if s == nil || completionID == "" || binding.Provider == "" || binding.AccountFingerprint == "" || s.maxEntries <= 0 {
		return false
	}

	size := chatCompletionBindingSize(completionID, binding)
	if size > s.maxBytes {
		return false
	}

	now := s.currentTime()
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cleanupExpiredLocked(now)
	if existing := s.entries[completionID]; existing != nil {
		s.removeLocked(completionID, existing)
	}
	for len(s.entries) >= s.maxEntries || s.totalBytes+size > s.maxBytes {
		if !s.removeOldestLocked() {
			return false
		}
	}

	element := s.order.PushBack(completionID)
	s.entries[completionID] = &chatCompletionBindingEntry{
		binding:   binding,
		expiresAt: now.Add(s.ttl),
		size:      size,
		element:   element,
	}
	s.totalBytes += size
	return true
}

func (s *chatCompletionBindingStore) get(completionID string) (chatCompletionBinding, bool) {
	completionID = strings.TrimSpace(completionID)
	if s == nil || completionID == "" {
		return chatCompletionBinding{}, false
	}

	now := s.currentTime()
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cleanupExpiredLocked(now)
	entry := s.entries[completionID]
	if entry == nil {
		return chatCompletionBinding{}, false
	}
	s.order.MoveToBack(entry.element)
	return entry.binding, true
}

func (s *chatCompletionBindingStore) delete(completionID string) bool {
	completionID = strings.TrimSpace(completionID)
	if s == nil || completionID == "" {
		return false
	}

	now := s.currentTime()
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cleanupExpiredLocked(now)
	entry := s.entries[completionID]
	if entry == nil {
		return false
	}
	s.removeLocked(completionID, entry)
	return true
}

func (s *chatCompletionBindingStore) currentTime() time.Time {
	if s != nil && s.now != nil {
		return s.now()
	}
	return time.Now()
}

func (s *chatCompletionBindingStore) cleanupExpiredLocked(now time.Time) {
	for element := s.order.Front(); element != nil; {
		next := element.Next()
		completionID, _ := element.Value.(string)
		entry := s.entries[completionID]
		if entry == nil || !now.Before(entry.expiresAt) {
			s.removeLocked(completionID, entry)
		}
		element = next
	}
}

func (s *chatCompletionBindingStore) removeOldestLocked() bool {
	element := s.order.Front()
	if element == nil {
		return false
	}
	completionID, _ := element.Value.(string)
	s.removeLocked(completionID, s.entries[completionID])
	return true
}

func (s *chatCompletionBindingStore) removeLocked(completionID string, entry *chatCompletionBindingEntry) {
	delete(s.entries, completionID)
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

func chatCompletionBindingSize(completionID string, binding chatCompletionBinding) int {
	return len(completionID) + len(binding.Provider) + len(binding.Route) + len(binding.Model) + len(binding.AccountFingerprint)
}

// chatCompletionStreamBindingCandidate accepts a stream ID only when every
// non-empty chunk ID agrees. A malformed stream must not bind an arbitrary ID
// to a provider account.
type chatCompletionStreamBindingCandidate struct {
	id           string
	inconsistent bool
}

func (c *chatCompletionStreamBindingCandidate) observe(chunk *models.StreamChunk) {
	if c == nil || chunk == nil {
		return
	}
	id := strings.TrimSpace(chunk.ID)
	if len(chunk.RawJSON) > 0 {
		var envelope struct {
			ID string `json:"id"`
		}
		if err := json.Unmarshal(chunk.RawJSON, &envelope); err == nil {
			// Native stream normalization may synthesize or stabilize the typed
			// ID. Lifecycle binding must use the ID actually supplied by the
			// upstream account, never that client-facing normalization.
			id = strings.TrimSpace(envelope.ID)
		}
	}
	if id == "" {
		return
	}
	if c.id == "" {
		c.id = id
		return
	}
	if c.id != id {
		c.inconsistent = true
	}
}

func (c chatCompletionStreamBindingCandidate) completionID() string {
	if c.inconsistent {
		return ""
	}
	return c.id
}
