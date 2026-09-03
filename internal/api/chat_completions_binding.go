package api

import (
	"container/list"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
	"time"

	"github.com/lunargate-ai/gateway/pkg/models"
)

const (
	defaultChatCompletionBindingMaxEntries = 1000
	defaultChatCompletionBindingMaxBytes   = 1 << 20
	maxChatCompletionIDCaptureBytes        = 16 << 20
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
	ambiguous bool
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

func (s *chatCompletionBindingStore) claim(completionID string, binding chatCompletionBinding) ownerClaimResult {
	binding = normalizeChatCompletionBinding(binding)
	if s == nil || !validOpaqueResourceID(completionID) || binding.Provider == "" || binding.AccountFingerprint == "" || s.maxEntries <= 0 {
		return ownerClaimUnavailable
	}
	size := chatCompletionBindingSize(completionID, binding)
	if size > s.maxBytes {
		return ownerClaimUnavailable
	}

	now := s.currentTime()
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cleanupExpiredLocked(now)
	if existing := s.entries[completionID]; existing != nil {
		if existing.ambiguous {
			existing.expiresAt = now.Add(s.ttl)
			s.order.MoveToBack(existing.element)
			return ownerClaimConflict
		}
		if sameChatCompletionBindingOwner(existing.binding, binding) {
			existing.expiresAt = now.Add(s.ttl)
			s.order.MoveToBack(existing.element)
			return ownerClaimRefreshed
		}
		s.markConflictLocked(completionID, existing, now)
		return ownerClaimConflict
	}
	for len(s.entries) >= s.maxEntries || s.totalBytes+size > s.maxBytes {
		if !s.removeOldestLocked() {
			return ownerClaimUnavailable
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
	return ownerClaimed
}

func normalizeChatCompletionBinding(binding chatCompletionBinding) chatCompletionBinding {
	binding.Provider = strings.TrimSpace(binding.Provider)
	binding.Route = strings.TrimSpace(binding.Route)
	binding.Model = strings.TrimSpace(binding.Model)
	binding.AccountFingerprint = strings.TrimSpace(binding.AccountFingerprint)
	return binding
}

func sameChatCompletionBindingOwner(first, second chatCompletionBinding) bool {
	return first.Provider == second.Provider && first.AccountFingerprint == second.AccountFingerprint
}

func (s *chatCompletionBindingStore) markConflictLocked(
	completionID string,
	entry *chatCompletionBindingEntry,
	now time.Time,
) {
	if entry == nil {
		return
	}
	tombstoneSize := len(completionID)
	s.totalBytes += tombstoneSize - entry.size
	entry.binding = chatCompletionBinding{}
	entry.ambiguous = true
	entry.expiresAt = now.Add(s.ttl)
	entry.size = tombstoneSize
	s.order.MoveToBack(entry.element)
}

func (s *chatCompletionBindingStore) lookup(completionID string) (chatCompletionBinding, ownerLookupResult) {
	if s == nil || !validOpaqueResourceID(completionID) {
		return chatCompletionBinding{}, ownerLookupMissing
	}

	now := s.currentTime()
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cleanupExpiredLocked(now)
	entry := s.entries[completionID]
	if entry == nil {
		return chatCompletionBinding{}, ownerLookupMissing
	}
	s.order.MoveToBack(entry.element)
	if entry.ambiguous {
		return chatCompletionBinding{}, ownerLookupConflict
	}
	return entry.binding, ownerLookupBound
}

func (s *chatCompletionBindingStore) put(completionID string, binding chatCompletionBinding) bool {
	binding = normalizeChatCompletionBinding(binding)
	if s == nil || !validOpaqueResourceID(completionID) || binding.Provider == "" || binding.AccountFingerprint == "" || s.maxEntries <= 0 {
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
	binding, result := s.lookup(completionID)
	return binding, result == ownerLookupBound
}

// deleteIfOwned removes a binding only when it still belongs to the account
// observed before an upstream operation. In particular, it must not erase a
// conflict tombstone installed while that operation was in flight.
func (s *chatCompletionBindingStore) deleteIfOwned(completionID string, binding chatCompletionBinding) bool {
	binding = normalizeChatCompletionBinding(binding)
	if s == nil || !validOpaqueResourceID(completionID) || binding.Provider == "" || binding.AccountFingerprint == "" {
		return false
	}

	now := s.currentTime()
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cleanupExpiredLocked(now)
	entry := s.entries[completionID]
	if entry == nil || entry.ambiguous || !sameChatCompletionBindingOwner(entry.binding, binding) {
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
	invalid      bool
	inconsistent bool
}

func (c *chatCompletionStreamBindingCandidate) observe(chunk *models.StreamChunk) {
	if c == nil || chunk == nil {
		return
	}
	id := chunk.ID
	if len(chunk.RawJSON) > 0 {
		var envelope map[string]json.RawMessage
		if err := json.Unmarshal(chunk.RawJSON, &envelope); err != nil || envelope == nil {
			c.invalid = true
			return
		}
		rawID, present := envelope["id"]
		if !present {
			// Native stream normalization may synthesize or stabilize the typed
			// ID. Lifecycle binding must use only the ID actually supplied by the
			// upstream account.
			return
		}
		if err := json.Unmarshal(rawID, &id); err != nil || !validOpaqueResourceID(id) {
			c.invalid = true
			return
		}
	}
	if id == "" {
		return
	}
	if !validOpaqueResourceID(id) {
		c.invalid = true
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
	if c.invalid || c.inconsistent {
		return ""
	}
	return c.id
}

// chatCompletionResponseIDCapture observes the original non-stream upstream
// document while the provider parser consumes it. This keeps a synthetic ID
// added by compatibility normalization out of the lifecycle binding store.
// The bound mirrors the provider response-body limit and fails closed if that
// limit is ever exceeded or the original document is not valid JSON.
type chatCompletionResponseIDCapture struct {
	body      io.ReadCloser
	captured  []byte
	truncated bool
}

func newChatCompletionResponseIDCapture(body io.ReadCloser) *chatCompletionResponseIDCapture {
	if body == nil {
		return nil
	}
	return &chatCompletionResponseIDCapture{body: body}
}

func (c *chatCompletionResponseIDCapture) Read(p []byte) (int, error) {
	if c == nil || c.body == nil {
		return 0, io.EOF
	}
	n, err := c.body.Read(p)
	if n <= 0 {
		return n, err
	}
	remaining := maxChatCompletionIDCaptureBytes - len(c.captured)
	if remaining > 0 {
		captureBytes := n
		if captureBytes > remaining {
			captureBytes = remaining
		}
		c.captured = append(c.captured, p[:captureBytes]...)
	}
	if n > remaining {
		c.truncated = true
	}
	return n, err
}

func (c *chatCompletionResponseIDCapture) Close() error {
	if c == nil || c.body == nil {
		return nil
	}
	return c.body.Close()
}

func (c *chatCompletionResponseIDCapture) completionID() string {
	if c == nil || c.truncated || len(c.captured) == 0 {
		return ""
	}
	var envelope map[string]json.RawMessage
	if err := json.Unmarshal(c.captured, &envelope); err != nil || envelope == nil {
		return ""
	}
	rawID, idPresent := envelope["id"]
	rawObject, objectPresent := envelope["object"]
	if !idPresent || !objectPresent {
		return ""
	}
	var id string
	if err := json.Unmarshal(rawID, &id); err != nil || !validOpaqueResourceID(id) {
		return ""
	}
	var object string
	if err := json.Unmarshal(rawObject, &object); err != nil || object != "chat.completion" {
		return ""
	}
	return id
}

// storedChatCompletionStreamTranslator validates the upstream resource ID
// before the streaming layer commits its first downstream byte. It is used
// only for store:true Chat-to-Chat requests whose provider enables lifecycle
// operations; compatibility normalization remains available elsewhere.
type storedChatCompletionStreamTranslator struct {
	models.ProviderTranslator
	completionID string
}

func (t *storedChatCompletionStreamTranslator) ParseStreamChunk(data []byte) (*models.StreamChunk, error) {
	if t == nil || t.ProviderTranslator == nil {
		return nil, errors.New("stored Chat Completion stream translator is unavailable")
	}
	chunk, err := t.ProviderTranslator.ParseStreamChunk(data)
	if err != nil || chunk == nil {
		return chunk, err
	}

	var envelope map[string]json.RawMessage
	if err := json.Unmarshal(chunk.RawJSON, &envelope); err != nil || envelope == nil {
		return nil, errors.New("stored Chat Completion stream requires a JSON object")
	}
	rawID, present := envelope["id"]
	if !present {
		return nil, errors.New("stored Chat Completion stream requires a non-empty string id")
	}
	var completionID string
	if err := json.Unmarshal(rawID, &completionID); err != nil || !validOpaqueResourceID(completionID) {
		return nil, errors.New("stored Chat Completion stream requires a non-empty string id without surrounding whitespace")
	}
	rawObject, present := envelope["object"]
	if !present {
		return nil, errors.New("stored Chat Completion stream requires object=chat.completion.chunk")
	}
	var object string
	if err := json.Unmarshal(rawObject, &object); err != nil || object != "chat.completion.chunk" {
		return nil, errors.New("stored Chat Completion stream requires object=chat.completion.chunk")
	}
	if t.completionID != "" && completionID != t.completionID {
		return nil, fmt.Errorf("stored Chat Completion stream changed id from %q to %q", t.completionID, completionID)
	}
	t.completionID = completionID
	return chunk, nil
}
