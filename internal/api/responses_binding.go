package api

import (
	"container/list"
	"crypto/subtle"
	"fmt"
	"net/http"
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
	LocalSnapshot       bool
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
	ambiguous bool
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

// claim retains the first account owner of an ID. A conflicting account turns
// the ID into a tombstone so no later implicit lookup can select either owner.
// put remains available for test/bootstrap compatibility until all creation
// paths have been migrated to this fail-closed API.
func (s *responseBindingStore) claim(responseID string, binding responseBinding) ownerClaimResult {
	binding = normalizeResponseBinding(binding)
	if s == nil || !validOpaqueResourceID(responseID) || binding.Provider == "" || binding.AccountFingerprint == "" || s.maxEntries <= 0 {
		return ownerClaimUnavailable
	}
	size := responseBindingSize(responseID, binding)
	if size > s.maxBytes {
		return ownerClaimUnavailable
	}

	now := time.Now()
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cleanupExpiredLocked(now)
	if existing := s.entries[responseID]; existing != nil {
		if existing.ambiguous {
			existing.expiresAt = now.Add(s.ttl)
			s.order.MoveToBack(existing.element)
			return ownerClaimConflict
		}
		if sameResponseBindingOwner(existing.binding, binding) {
			existing.expiresAt = now.Add(s.ttl)
			s.order.MoveToBack(existing.element)
			return ownerClaimRefreshed
		}
		s.markConflictLocked(responseID, existing, now)
		return ownerClaimConflict
	}
	for len(s.entries) >= s.maxEntries || s.totalBytes+size > s.maxBytes {
		if !s.removeOldestLocked() {
			return ownerClaimUnavailable
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
	return ownerClaimed
}

func normalizeResponseBinding(binding responseBinding) responseBinding {
	binding.Provider = strings.TrimSpace(binding.Provider)
	binding.Route = strings.TrimSpace(binding.Route)
	binding.Model = strings.TrimSpace(binding.Model)
	binding.UpstreamRequestType = strings.TrimSpace(binding.UpstreamRequestType)
	binding.AccountFingerprint = strings.TrimSpace(binding.AccountFingerprint)
	return binding
}

func sameResponseBindingOwner(first, second responseBinding) bool {
	return first.LocalSnapshot == second.LocalSnapshot &&
		first.Provider == second.Provider &&
		first.AccountFingerprint == second.AccountFingerprint &&
		first.Route == second.Route &&
		first.Model == second.Model &&
		first.UpstreamRequestType == second.UpstreamRequestType
}

func (s *responseBindingStore) markConflictLocked(responseID string, entry *responseBindingEntry, now time.Time) {
	if entry == nil {
		return
	}
	tombstoneSize := len(responseID)
	s.totalBytes += tombstoneSize - entry.size
	entry.binding = responseBinding{}
	entry.ambiguous = true
	entry.expiresAt = now.Add(s.ttl)
	entry.size = tombstoneSize
	s.order.MoveToBack(entry.element)
}

func (s *responseBindingStore) lookup(responseID string) (responseBinding, ownerLookupResult) {
	if s == nil || !validOpaqueResourceID(responseID) {
		return responseBinding{}, ownerLookupMissing
	}

	now := time.Now()
	s.mu.Lock()
	defer s.mu.Unlock()
	entry := s.entries[responseID]
	if entry == nil {
		return responseBinding{}, ownerLookupMissing
	}
	if now.After(entry.expiresAt) {
		s.removeLocked(responseID, entry)
		return responseBinding{}, ownerLookupMissing
	}
	if entry.ambiguous {
		return responseBinding{}, ownerLookupConflict
	}
	return entry.binding, ownerLookupBound
}

func (s *responseBindingStore) put(responseID string, binding responseBinding) bool {
	binding = normalizeResponseBinding(binding)
	if s == nil || !validOpaqueResourceID(responseID) || binding.Provider == "" || binding.AccountFingerprint == "" {
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
	binding, result := s.lookup(responseID)
	return binding, result == ownerLookupBound
}

// claimResponseOwner binds a routed response to the account that produced it.
// Local snapshots and native resources intentionally use distinct ownership
// kinds so an upstream ID cannot silently switch lifecycle implementations.
func (h *Handler) claimResponseOwner(
	responseID string,
	headers http.Header,
	owner responseExecutionOwner,
	localSnapshot bool,
) (responseBinding, ownerClaimResult) {
	if h == nil || h.responseBindings == nil {
		return responseBinding{}, ownerClaimUnavailable
	}
	headerBinding := responseBindingFromHeaders(headers)
	owner.Provider = strings.TrimSpace(owner.Provider)
	owner.Route = strings.TrimSpace(owner.Route)
	owner.Model = strings.TrimSpace(owner.Model)
	owner.UpstreamRequestType = strings.TrimSpace(owner.UpstreamRequestType)
	if owner.Provider == "" || owner.Route == "" || owner.Model == "" ||
		owner.AccountFingerprint == "" || owner.UpstreamRequestType == "" {
		return responseBinding{}, ownerClaimUnavailable
	}
	if headerBinding.Provider != owner.Provider ||
		headerBinding.Route != owner.Route ||
		headerBinding.Model != owner.Model {
		return responseBinding{}, ownerClaimUnavailable
	}
	if !localSnapshot {
		if !strings.EqualFold(owner.UpstreamRequestType, requestTypeResponses) ||
			!h.providerSupportsResponseCapability(owner.Provider, responseNativeLifecycle) {
			return responseBinding{}, ownerClaimUnavailable
		}
	}
	binding := responseBinding{
		Provider:            owner.Provider,
		Route:               owner.Route,
		Model:               owner.Model,
		UpstreamRequestType: owner.UpstreamRequestType,
		AccountFingerprint:  owner.AccountFingerprint,
		LocalSnapshot:       localSnapshot,
	}
	return binding, h.responseBindings.claim(responseID, binding)
}

func responseOwnerConflictError(responseID string, param string) error {
	return &responseBindingResolutionError{
		message: fmt.Sprintf("response %q has conflicting provider ownership", responseID),
		param:   param,
		code:    "provider_binding_conflict",
	}
}

func (h *Handler) validateClaimedResponseOwner(
	r *http.Request,
	responseID string,
	binding responseBinding,
	capability responseNativeCapability,
	requireCapability bool,
) error {
	requestedProvider := ""
	if r != nil {
		requestedProvider = strings.TrimSpace(r.Header.Get("X-LunarGate-Provider"))
	}
	if requestedProvider != "" && requestedProvider != binding.Provider {
		return &responseBindingResolutionError{
			message: fmt.Sprintf("response %q belongs to provider %q, not %q", responseID, binding.Provider, requestedProvider),
			param:   "provider",
			code:    "invalid_value",
		}
	}
	if requireCapability && !h.providerSupportsResponseCapability(binding.Provider, capability) {
		return &responseBindingResolutionError{
			message: fmt.Sprintf("provider %q no longer enables %s", binding.Provider, responseCapabilityName(capability)),
			param:   "provider",
			code:    "unsupported_feature",
		}
	}
	currentFingerprint, fingerprintOK := h.responseAccountFingerprint(binding.Provider)
	if !fingerprintOK {
		return &responseBindingResolutionError{
			message: fmt.Sprintf("provider %q no longer has an HTTP account configuration", binding.Provider),
			param:   "provider",
			code:    "provider_not_found",
		}
	}
	if subtle.ConstantTimeCompare(
		[]byte(binding.AccountFingerprint),
		[]byte(currentFingerprint),
	) != 1 {
		return &responseBindingResolutionError{
			message: fmt.Sprintf("provider account configuration changed for response %q", responseID),
			param:   "provider",
			code:    "provider_binding_stale",
		}
	}
	return nil
}

// deleteIfOwned removes a binding only when it still matches the execution
// owner observed before an upstream operation. A concurrent conflicting claim
// therefore remains a fail-closed tombstone.
func (s *responseBindingStore) deleteIfOwned(responseID string, binding responseBinding) bool {
	binding = normalizeResponseBinding(binding)
	if s == nil || !validOpaqueResourceID(responseID) || binding.Provider == "" || binding.AccountFingerprint == "" {
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
	if entry.ambiguous || !sameResponseBindingOwner(entry.binding, binding) {
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
	size := len(responseID) + len(binding.Provider) + len(binding.Route) + len(binding.Model) + len(binding.UpstreamRequestType) + len(binding.AccountFingerprint)
	if binding.LocalSnapshot {
		size++
	}
	return size
}
