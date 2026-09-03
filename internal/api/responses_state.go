package api

import (
	"container/list"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"strings"
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
	payload    map[string]json.RawMessage
	response   json.RawMessage
	inputItems []json.RawMessage
	expiresAt  time.Time
	size       int
	element    *list.Element
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
	s.putEntry(responseID, payload, nil, nil)
}

func (s *responsesStateStore) putCompleted(
	responseID string,
	requestPayload map[string]json.RawMessage,
	completedResponse map[string]interface{},
) {
	if s == nil || responseID == "" || completedResponse == nil {
		return
	}

	response, err := json.Marshal(completedResponse)
	if err != nil {
		return
	}
	inputItems, err := responsesStateInputItems(responseID, requestPayload["input"])
	if err != nil {
		return
	}
	s.putEntry(
		responseID,
		withCompletedResponseHistory(requestPayload, completedResponse),
		response,
		inputItems,
	)
}

func (s *responsesStateStore) putEntry(
	responseID string,
	payload map[string]json.RawMessage,
	response json.RawMessage,
	inputItems []json.RawMessage,
) {
	if s == nil || responseID == "" || len(payload) == 0 {
		return
	}

	now := time.Now()
	expiresAt := now.Add(s.ttl)
	clonedPayload := cloneResponsesRawMap(payload)
	clonedResponse := cloneResponsesRawMessage(response)
	clonedInputItems := cloneResponsesRawMessages(inputItems)
	size := responsesStateEntrySize(responseID, clonedPayload, clonedResponse, clonedInputItems)
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
		payload:    clonedPayload,
		response:   clonedResponse,
		inputItems: clonedInputItems,
		expiresAt:  expiresAt,
		size:       size,
		element:    element,
	}
	s.totalBytes += size
}

func (s *responsesStateStore) getCompleted(responseID string) (json.RawMessage, []json.RawMessage, bool) {
	if s == nil || responseID == "" {
		return nil, nil, false
	}

	now := time.Now()
	s.mu.Lock()
	defer s.mu.Unlock()
	entry, ok := s.entries[responseID]
	if !ok {
		return nil, nil, false
	}
	if now.After(entry.expiresAt) {
		s.removeLocked(responseID, entry)
		return nil, nil, false
	}
	if len(entry.response) == 0 {
		return nil, nil, false
	}

	return cloneResponsesRawMessage(entry.response), cloneResponsesRawMessages(entry.inputItems), true
}

func (s *responsesStateStore) delete(responseID string) bool {
	if s == nil || responseID == "" {
		return false
	}
	now := time.Now()
	s.mu.Lock()
	defer s.mu.Unlock()
	entry, ok := s.entries[responseID]
	if !ok {
		return false
	}
	if now.After(entry.expiresAt) {
		s.removeLocked(responseID, entry)
		return false
	}
	if len(entry.response) == 0 {
		return false
	}
	s.removeLocked(responseID, entry)
	return true
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

func responsesStateEntrySize(
	responseID string,
	payload map[string]json.RawMessage,
	response json.RawMessage,
	inputItems []json.RawMessage,
) int {
	size := responsesStatePayloadSize(responseID, payload) + len(response)
	for _, item := range inputItems {
		size += len(item)
	}
	return size
}

func responsesStateInputItems(responseID string, rawInput json.RawMessage) ([]json.RawMessage, error) {
	items, err := responsesInputRawToItems(rawInput)
	if err != nil {
		return nil, err
	}

	stored := make([]json.RawMessage, 0, len(items))
	for index, rawItem := range items {
		item, ok := rawItem.(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("invalid response input item at index %d", index)
		}
		itemType := responsesStateString(item["type"])
		if itemType == "" && responsesStateString(item["role"]) != "" {
			itemType = "message"
			item["type"] = itemType
		}
		if itemType == "message" {
			if content, ok := item["content"].(string); ok {
				item["content"] = []interface{}{
					map[string]interface{}{"type": "input_text", "text": content},
				}
			}
		}
		if responsesStateString(item["id"]) == "" {
			item["id"] = responsesStateInputItemID(responseID, index, itemType)
		}
		encoded, err := json.Marshal(item)
		if err != nil {
			return nil, fmt.Errorf("failed to encode response input item %d: %w", index, err)
		}
		stored = append(stored, json.RawMessage(encoded))
	}
	return stored, nil
}

func responsesStateString(value interface{}) string {
	text, _ := value.(string)
	return strings.TrimSpace(text)
}

func responsesStateInputItemID(responseID string, index int, itemType string) string {
	prefix := "item"
	switch strings.TrimSpace(itemType) {
	case "message":
		prefix = "msg"
	case "function_call", "function_call_output":
		prefix = "fc"
	case "reasoning":
		prefix = "rs"
	}
	digest := sha256.Sum256([]byte(fmt.Sprintf("%s:%d", responseID, index)))
	return fmt.Sprintf("%s_%x", prefix, digest[:12])
}

func cloneResponsesRawMessages(src []json.RawMessage) []json.RawMessage {
	if len(src) == 0 {
		return nil
	}
	dst := make([]json.RawMessage, len(src))
	for index, item := range src {
		dst[index] = cloneResponsesRawMessage(item)
	}
	return dst
}
