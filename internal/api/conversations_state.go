package api

import (
	"container/list"
	"encoding/json"
	"errors"
	"sync"
	"time"

	"github.com/google/uuid"
)

const (
	defaultConversationsStateMaxEntries = 1000
	defaultConversationMaxItems         = 1000
	defaultConversationsStateMaxBytes   = 64 << 20
)

var (
	errConversationNotFound       = errors.New("conversation not found")
	errConversationItemNotFound   = errors.New("conversation item not found")
	errConversationItemLimit      = errors.New("conversation item limit exceeded")
	errConversationItemIDConflict = errors.New("conversation item ID already exists")
	errConversationStateTooLarge  = errors.New("conversation state exceeds storage limit")
	errConversationCursorNotFound = errors.New("conversation item cursor not found")
)

type conversationObject struct {
	ID        string            `json:"id"`
	Object    string            `json:"object"`
	CreatedAt int64             `json:"created_at"`
	Metadata  map[string]string `json:"metadata"`
}

type conversationItemList struct {
	Object  string                       `json:"object"`
	Data    []map[string]json.RawMessage `json:"data"`
	FirstID *string                      `json:"first_id"`
	LastID  *string                      `json:"last_id"`
	HasMore bool                         `json:"has_more"`
}

type conversationStateStore struct {
	mu         sync.Mutex
	ttl        time.Duration
	maxEntries int
	maxItems   int
	maxBytes   int
	totalBytes int
	entries    map[string]*conversationStateEntry
	order      *list.List
	now        func() time.Time
}

type conversationStateEntry struct {
	conversation conversationObject
	items        []map[string]json.RawMessage
	expiresAt    time.Time
	size         int
	element      *list.Element
}

func newConversationStateStore(ttl time.Duration) *conversationStateStore {
	if ttl <= 0 {
		ttl = 30 * time.Minute
	}
	return &conversationStateStore{
		ttl:        ttl,
		maxEntries: defaultConversationsStateMaxEntries,
		maxItems:   defaultConversationMaxItems,
		maxBytes:   defaultConversationsStateMaxBytes,
		entries:    make(map[string]*conversationStateEntry),
		order:      list.New(),
		now:        time.Now,
	}
}

func (s *conversationStateStore) create(metadata map[string]string, items []map[string]json.RawMessage) (conversationObject, error) {
	if s == nil {
		return conversationObject{}, errConversationStateTooLarge
	}
	if len(items) > s.maxItems {
		return conversationObject{}, errConversationItemLimit
	}
	if hasDuplicateConversationItemIDs(nil, items) {
		return conversationObject{}, errConversationItemIDConflict
	}

	now := s.currentTime()
	conversation := conversationObject{
		ID:        "conv_" + uuid.NewString(),
		Object:    "conversation",
		CreatedAt: now.Unix(),
		Metadata:  cloneConversationMetadata(metadata),
	}
	entry := &conversationStateEntry{
		conversation: conversation,
		items:        cloneConversationItems(items),
		expiresAt:    now.Add(s.ttl),
	}
	entry.size = conversationEntrySize(entry)
	if entry.size > s.maxBytes || s.maxEntries <= 0 {
		return conversationObject{}, errConversationStateTooLarge
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	s.cleanupExpiredLocked(now)
	if !s.makeRoomLocked(entry.size, "") {
		return conversationObject{}, errConversationStateTooLarge
	}
	entry.element = s.order.PushBack(conversation.ID)
	s.entries[conversation.ID] = entry
	s.totalBytes += entry.size
	return cloneConversationObject(conversation), nil
}

func (s *conversationStateStore) get(conversationID string) (conversationObject, bool) {
	if s == nil || conversationID == "" {
		return conversationObject{}, false
	}

	now := s.currentTime()
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cleanupExpiredLocked(now)
	entry, ok := s.entries[conversationID]
	if !ok {
		return conversationObject{}, false
	}
	s.touchLocked(entry)
	return cloneConversationObject(entry.conversation), true
}

func (s *conversationStateStore) getItems(conversationID string) ([]map[string]json.RawMessage, bool) {
	if s == nil || conversationID == "" {
		return nil, false
	}

	now := s.currentTime()
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cleanupExpiredLocked(now)
	entry, ok := s.entries[conversationID]
	if !ok {
		return nil, false
	}
	s.touchLocked(entry)
	return cloneConversationItems(entry.items), true
}

func (s *conversationStateStore) updateMetadata(conversationID string, metadata map[string]string) (conversationObject, error) {
	if s == nil || conversationID == "" {
		return conversationObject{}, errConversationNotFound
	}

	now := s.currentTime()
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cleanupExpiredLocked(now)
	entry, ok := s.entries[conversationID]
	if !ok {
		return conversationObject{}, errConversationNotFound
	}

	previousSize := entry.size
	previousMetadata := entry.conversation.Metadata
	entry.conversation.Metadata = cloneConversationMetadata(metadata)
	entry.size = conversationEntrySize(entry)
	if entry.size > s.maxBytes || !s.makeRoomLocked(entry.size-previousSize, conversationID) {
		entry.conversation.Metadata = previousMetadata
		entry.size = previousSize
		return conversationObject{}, errConversationStateTooLarge
	}
	s.totalBytes += entry.size - previousSize
	s.touchLocked(entry)
	return cloneConversationObject(entry.conversation), nil
}

func (s *conversationStateStore) delete(conversationID string) (conversationObject, bool) {
	if s == nil || conversationID == "" {
		return conversationObject{}, false
	}

	now := s.currentTime()
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cleanupExpiredLocked(now)
	entry, ok := s.entries[conversationID]
	if !ok {
		return conversationObject{}, false
	}
	conversation := cloneConversationObject(entry.conversation)
	s.removeLocked(conversationID, entry)
	return conversation, true
}

func (s *conversationStateStore) addItems(conversationID string, items []map[string]json.RawMessage) ([]map[string]json.RawMessage, error) {
	if s == nil || conversationID == "" {
		return nil, errConversationNotFound
	}

	now := s.currentTime()
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cleanupExpiredLocked(now)
	entry, ok := s.entries[conversationID]
	if !ok {
		return nil, errConversationNotFound
	}
	if len(entry.items)+len(items) > s.maxItems {
		return nil, errConversationItemLimit
	}
	if hasDuplicateConversationItemIDs(entry.items, items) {
		return nil, errConversationItemIDConflict
	}

	previousSize := entry.size
	previousItems := entry.items
	createdItems := cloneConversationItems(items)
	entry.items = append(cloneConversationItems(previousItems), createdItems...)
	entry.size = conversationEntrySize(entry)
	if entry.size > s.maxBytes || !s.makeRoomLocked(entry.size-previousSize, conversationID) {
		entry.items = previousItems
		entry.size = previousSize
		return nil, errConversationStateTooLarge
	}
	s.totalBytes += entry.size - previousSize
	s.touchLocked(entry)
	return cloneConversationItems(createdItems), nil
}

func (s *conversationStateStore) getItem(conversationID, itemID string) (map[string]json.RawMessage, error) {
	if s == nil || conversationID == "" {
		return nil, errConversationNotFound
	}

	now := s.currentTime()
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cleanupExpiredLocked(now)
	entry, ok := s.entries[conversationID]
	if !ok {
		return nil, errConversationNotFound
	}
	for _, item := range entry.items {
		if conversationItemID(item) == itemID {
			s.touchLocked(entry)
			return cloneResponsesRawMap(item), nil
		}
	}
	return nil, errConversationItemNotFound
}

func (s *conversationStateStore) deleteItem(conversationID, itemID string) (conversationObject, error) {
	if s == nil || conversationID == "" {
		return conversationObject{}, errConversationNotFound
	}

	now := s.currentTime()
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cleanupExpiredLocked(now)
	entry, ok := s.entries[conversationID]
	if !ok {
		return conversationObject{}, errConversationNotFound
	}
	for index, item := range entry.items {
		if conversationItemID(item) != itemID {
			continue
		}
		previousSize := entry.size
		entry.items = append(entry.items[:index:index], entry.items[index+1:]...)
		entry.size = conversationEntrySize(entry)
		s.totalBytes += entry.size - previousSize
		s.touchLocked(entry)
		return cloneConversationObject(entry.conversation), nil
	}
	return conversationObject{}, errConversationItemNotFound
}

func (s *conversationStateStore) listItems(conversationID, after, order string, limit int) (conversationItemList, error) {
	if s == nil || conversationID == "" {
		return conversationItemList{}, errConversationNotFound
	}

	now := s.currentTime()
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cleanupExpiredLocked(now)
	entry, ok := s.entries[conversationID]
	if !ok {
		return conversationItemList{}, errConversationNotFound
	}

	ordered := cloneConversationItems(entry.items)
	if order == "desc" {
		for left, right := 0, len(ordered)-1; left < right; left, right = left+1, right-1 {
			ordered[left], ordered[right] = ordered[right], ordered[left]
		}
	}
	start := 0
	if after != "" {
		found := false
		for index, item := range ordered {
			if conversationItemID(item) == after {
				start = index + 1
				found = true
				break
			}
		}
		if !found {
			return conversationItemList{}, errConversationCursorNotFound
		}
	}
	end := start + limit
	if end > len(ordered) {
		end = len(ordered)
	}
	data := ordered[start:end]
	result := conversationItemList{
		Object:  "list",
		Data:    data,
		HasMore: end < len(ordered),
	}
	if len(data) > 0 {
		firstID := conversationItemID(data[0])
		lastID := conversationItemID(data[len(data)-1])
		result.FirstID = &firstID
		result.LastID = &lastID
	}
	s.touchLocked(entry)
	return result, nil
}

func (s *conversationStateStore) currentTime() time.Time {
	if s != nil && s.now != nil {
		return s.now()
	}
	return time.Now()
}

func (s *conversationStateStore) cleanupExpiredLocked(now time.Time) {
	for element := s.order.Front(); element != nil; {
		next := element.Next()
		conversationID, _ := element.Value.(string)
		entry := s.entries[conversationID]
		if entry == nil {
			s.order.Remove(element)
		} else if !now.Before(entry.expiresAt) {
			s.removeLocked(conversationID, entry)
		}
		element = next
	}
}

func (s *conversationStateStore) makeRoomLocked(additionalBytes int, excludedID string) bool {
	if additionalBytes <= 0 {
		return true
	}
	for s.totalBytes+additionalBytes > s.maxBytes || (excludedID == "" && len(s.entries) >= s.maxEntries) {
		if !s.removeOldestLocked(excludedID) {
			return false
		}
	}
	return true
}

func (s *conversationStateStore) removeOldestLocked(excludedID string) bool {
	for element := s.order.Front(); element != nil; element = element.Next() {
		conversationID, _ := element.Value.(string)
		if conversationID == excludedID {
			continue
		}
		entry := s.entries[conversationID]
		if entry == nil {
			s.order.Remove(element)
			return true
		}
		s.removeLocked(conversationID, entry)
		return true
	}
	return false
}

func (s *conversationStateStore) touchLocked(entry *conversationStateEntry) {
	if entry != nil && entry.element != nil {
		s.order.MoveToBack(entry.element)
	}
}

func (s *conversationStateStore) removeLocked(conversationID string, entry *conversationStateEntry) {
	delete(s.entries, conversationID)
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

func conversationEntrySize(entry *conversationStateEntry) int {
	if entry == nil {
		return 0
	}
	size := len(entry.conversation.ID) + len(entry.conversation.Object) + 8
	for key, value := range entry.conversation.Metadata {
		size += len(key) + len(value)
	}
	for _, item := range entry.items {
		for key, value := range item {
			size += len(key) + len(value)
		}
	}
	return size
}

func cloneConversationObject(src conversationObject) conversationObject {
	src.Metadata = cloneConversationMetadata(src.Metadata)
	return src
}

func cloneConversationMetadata(src map[string]string) map[string]string {
	if src == nil {
		return map[string]string{}
	}
	dst := make(map[string]string, len(src))
	for key, value := range src {
		dst[key] = value
	}
	return dst
}

func cloneConversationItems(src []map[string]json.RawMessage) []map[string]json.RawMessage {
	if len(src) == 0 {
		return []map[string]json.RawMessage{}
	}
	dst := make([]map[string]json.RawMessage, 0, len(src))
	for _, item := range src {
		dst = append(dst, cloneResponsesRawMap(item))
	}
	return dst
}

func conversationItemID(item map[string]json.RawMessage) string {
	return parseJSONStringRaw(item["id"])
}

func hasDuplicateConversationItemIDs(existing, added []map[string]json.RawMessage) bool {
	seen := make(map[string]struct{}, len(existing)+len(added))
	for _, item := range existing {
		if id := conversationItemID(item); id != "" {
			seen[id] = struct{}{}
		}
	}
	for _, item := range added {
		id := conversationItemID(item)
		if id == "" {
			continue
		}
		if _, exists := seen[id]; exists {
			return true
		}
		seen[id] = struct{}{}
	}
	return false
}
