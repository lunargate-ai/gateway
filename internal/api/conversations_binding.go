package api

import (
	"container/list"
	"crypto/subtle"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/lunargate-ai/gateway/internal/modelid"
	"github.com/lunargate-ai/gateway/internal/provideridentity"
)

const (
	defaultConversationBindingMaxEntries = 1000
	defaultConversationBindingMaxBytes   = 1 << 20
)

// conversationBinding identifies the configured provider account that owns a
// native Conversation. Account-sensitive configuration is retained only as a
// one-way digest; no model, endpoint, organization, or credential is stored.
type conversationBinding struct {
	Provider           string
	AccountFingerprint string
	Local              bool
}

type conversationBindingStore struct {
	mu         sync.Mutex
	ttl        time.Duration
	maxEntries int
	maxBytes   int
	totalBytes int
	entries    map[string]*conversationBindingEntry
	order      *list.List
	now        func() time.Time
}

type conversationBindingEntry struct {
	binding   conversationBinding
	ambiguous bool
	expiresAt time.Time
	size      int
	element   *list.Element
}

func newConversationBindingStore(ttl time.Duration) *conversationBindingStore {
	if ttl <= 0 {
		ttl = 30 * time.Minute
	}
	return &conversationBindingStore{
		ttl:        ttl,
		maxEntries: defaultConversationBindingMaxEntries,
		maxBytes:   defaultConversationBindingMaxBytes,
		entries:    make(map[string]*conversationBindingEntry),
		order:      list.New(),
		now:        time.Now,
	}
}

func (s *conversationBindingStore) claim(conversationID string, binding conversationBinding) ownerClaimResult {
	binding = normalizeConversationBinding(binding)
	if s == nil || !validOpaqueResourceID(conversationID) || binding.Provider == "" || binding.AccountFingerprint == "" || s.maxEntries <= 0 {
		return ownerClaimUnavailable
	}
	size := conversationBindingSize(conversationID, binding)
	if size > s.maxBytes {
		return ownerClaimUnavailable
	}

	now := s.currentTime()
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cleanupExpiredLocked(now)
	if existing := s.entries[conversationID]; existing != nil {
		if existing.ambiguous {
			existing.expiresAt = now.Add(s.ttl)
			s.order.MoveToBack(existing.element)
			return ownerClaimConflict
		}
		if sameConversationBindingOwner(existing.binding, binding) {
			existing.expiresAt = now.Add(s.ttl)
			s.order.MoveToBack(existing.element)
			return ownerClaimRefreshed
		}
		s.markConflictLocked(conversationID, existing, now)
		return ownerClaimConflict
	}
	for len(s.entries) >= s.maxEntries || s.totalBytes+size > s.maxBytes {
		if !s.removeOldestLocked() {
			return ownerClaimUnavailable
		}
	}
	element := s.order.PushBack(conversationID)
	s.entries[conversationID] = &conversationBindingEntry{
		binding:   binding,
		expiresAt: now.Add(s.ttl),
		size:      size,
		element:   element,
	}
	s.totalBytes += size
	return ownerClaimed
}

func normalizeConversationBinding(binding conversationBinding) conversationBinding {
	binding.Provider = strings.TrimSpace(binding.Provider)
	binding.AccountFingerprint = strings.TrimSpace(binding.AccountFingerprint)
	return binding
}

func conversationBindingSize(conversationID string, binding conversationBinding) int {
	size := len(conversationID) + len(binding.Provider) + len(binding.AccountFingerprint)
	if binding.Local {
		size++
	}
	return size
}

func sameConversationBindingOwner(first, second conversationBinding) bool {
	return first.Local == second.Local &&
		first.Provider == second.Provider &&
		first.AccountFingerprint == second.AccountFingerprint
}

func (s *conversationBindingStore) markConflictLocked(
	conversationID string,
	entry *conversationBindingEntry,
	now time.Time,
) {
	if entry == nil {
		return
	}
	tombstoneSize := len(conversationID)
	s.totalBytes += tombstoneSize - entry.size
	entry.binding = conversationBinding{}
	entry.ambiguous = true
	entry.expiresAt = now.Add(s.ttl)
	entry.size = tombstoneSize
	s.order.MoveToBack(entry.element)
}

func (s *conversationBindingStore) lookup(conversationID string) (conversationBinding, ownerLookupResult) {
	if s == nil || !validOpaqueResourceID(conversationID) {
		return conversationBinding{}, ownerLookupMissing
	}

	now := s.currentTime()
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cleanupExpiredLocked(now)
	entry := s.entries[conversationID]
	if entry == nil {
		return conversationBinding{}, ownerLookupMissing
	}
	s.order.MoveToBack(entry.element)
	if entry.ambiguous {
		return conversationBinding{}, ownerLookupConflict
	}
	return entry.binding, ownerLookupBound
}

func (s *conversationBindingStore) put(conversationID string, binding conversationBinding) bool {
	binding = normalizeConversationBinding(binding)
	if s == nil || !validOpaqueResourceID(conversationID) || binding.Provider == "" || binding.AccountFingerprint == "" {
		return false
	}
	size := conversationBindingSize(conversationID, binding)
	if size > s.maxBytes || s.maxEntries <= 0 {
		return false
	}

	now := s.currentTime()
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cleanupExpiredLocked(now)
	if existing := s.entries[conversationID]; existing != nil {
		s.removeLocked(conversationID, existing)
	}
	for len(s.entries) >= s.maxEntries || s.totalBytes+size > s.maxBytes {
		if !s.removeOldestLocked() {
			return false
		}
	}
	element := s.order.PushBack(conversationID)
	s.entries[conversationID] = &conversationBindingEntry{
		binding:   binding,
		expiresAt: now.Add(s.ttl),
		size:      size,
		element:   element,
	}
	s.totalBytes += size
	return true
}

func (s *conversationBindingStore) get(conversationID string) (conversationBinding, bool) {
	binding, result := s.lookup(conversationID)
	return binding, result == ownerLookupBound
}

// deleteIfOwned removes a binding only when it still belongs to the account
// observed before an upstream operation. In particular, it must not erase a
// conflict tombstone installed while that operation was in flight.
func (s *conversationBindingStore) deleteIfOwned(conversationID string, binding conversationBinding) bool {
	binding = normalizeConversationBinding(binding)
	if s == nil || !validOpaqueResourceID(conversationID) || binding.Provider == "" || binding.AccountFingerprint == "" {
		return false
	}

	now := s.currentTime()
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cleanupExpiredLocked(now)
	entry := s.entries[conversationID]
	if entry == nil || entry.ambiguous || !sameConversationBindingOwner(entry.binding, binding) {
		return false
	}
	s.removeLocked(conversationID, entry)
	return true
}

func (s *conversationBindingStore) currentTime() time.Time {
	if s != nil && s.now != nil {
		return s.now()
	}
	return time.Now()
}

func (s *conversationBindingStore) cleanupExpiredLocked(now time.Time) {
	for element := s.order.Front(); element != nil; {
		next := element.Next()
		conversationID, _ := element.Value.(string)
		entry := s.entries[conversationID]
		if entry == nil || !now.Before(entry.expiresAt) {
			s.removeLocked(conversationID, entry)
		}
		element = next
	}
}

func (s *conversationBindingStore) removeOldestLocked() bool {
	element := s.order.Front()
	if element == nil {
		return false
	}
	conversationID, _ := element.Value.(string)
	s.removeLocked(conversationID, s.entries[conversationID])
	return true
}

func (s *conversationBindingStore) removeLocked(conversationID string, entry *conversationBindingEntry) {
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

type conversationBindingResolutionError struct {
	message string
	param   string
	code    string
}

func (e *conversationBindingResolutionError) Error() string {
	if e == nil {
		return "conversation provider binding is invalid"
	}
	return e.message
}

func (h *Handler) providerSupportsConversations(provider string) bool {
	if h == nil || h.registry == nil {
		return false
	}
	capabilities, ok := h.registry.Capabilities(strings.TrimSpace(provider))
	return ok && capabilities.Conversations
}

func (h *Handler) validateConversationProvider(provider string) (conversationBinding, error) {
	provider = strings.TrimSpace(provider)
	if provider == "" {
		return conversationBinding{}, &conversationBindingResolutionError{
			message: "conversation provider is required",
			param:   "provider",
			code:    "missing_required_parameter",
		}
	}
	if h == nil || h.registry == nil {
		return conversationBinding{}, &conversationBindingResolutionError{
			message: "conversation provider registry is unavailable",
			param:   "provider",
			code:    "provider_not_found",
		}
	}
	providerSnapshot, ok := h.registry.Snapshot(provider)
	if !ok {
		return conversationBinding{}, &conversationBindingResolutionError{
			message: fmt.Sprintf("requested provider %q is not configured", provider),
			param:   "provider",
			code:    "provider_not_found",
		}
	}
	if !h.providerSupportsConversations(provider) {
		return conversationBinding{}, &conversationBindingResolutionError{
			message: fmt.Sprintf("provider %q does not enable conversations", provider),
			param:   "provider",
			code:    "unsupported_feature",
		}
	}
	if h.providerClients == nil {
		return conversationBinding{}, &conversationBindingResolutionError{
			message: fmt.Sprintf("provider %q has no HTTP account configuration", provider),
			param:   "provider",
			code:    "provider_not_found",
		}
	}
	_, providerConfig, ok := h.providerClients.Snapshot(provider)
	if !ok {
		return conversationBinding{}, &conversationBindingResolutionError{
			message: fmt.Sprintf("provider %q has no HTTP account configuration", provider),
			param:   "provider",
			code:    "provider_not_found",
		}
	}
	providerType := strings.TrimSpace(providerConfig.Type)
	if providerType == "" {
		providerType = strings.TrimSpace(providerSnapshot.ProviderType)
	}
	baseURL := strings.TrimSpace(providerConfig.BaseURL)
	if baseURL == "" && providerSnapshot.Translator != nil {
		baseURL = strings.TrimSpace(providerSnapshot.Translator.BaseURL())
	}
	return conversationBinding{
		Provider: provider,
		AccountFingerprint: conversationAccountFingerprint(
			providerType,
			baseURL,
			providerConfig.Organization,
			providerConfig.APIKey,
		),
	}, nil
}

// conversationCreateBinding deterministically selects native storage for a
// newly created conversation. A canonical model header may select its provider
// account, but a model is never retained in the conversation binding.
func (h *Handler) conversationCreateBinding(r *http.Request) (conversationBinding, bool, error) {
	explicitProvider := strings.TrimSpace(r.Header.Get("X-LunarGate-Provider"))
	modelProvider := ""
	modelHeader := strings.TrimSpace(r.Header.Get("X-LunarGate-Model"))
	if provider, _, ok := modelid.SplitCanonical(modelHeader); ok && !strings.EqualFold(provider, "lunargate") {
		modelProvider = strings.TrimSpace(provider)
	}
	if explicitProvider != "" && modelProvider != "" && explicitProvider != modelProvider {
		return conversationBinding{}, false, &conversationBindingResolutionError{
			message: fmt.Sprintf("requested provider %q conflicts with model provider %q", explicitProvider, modelProvider),
			param:   "provider",
			code:    "invalid_value",
		}
	}
	selectedProvider := explicitProvider
	if selectedProvider == "" {
		selectedProvider = modelProvider
	}
	if selectedProvider != "" {
		binding, err := h.validateConversationProvider(selectedProvider)
		return binding, err == nil, err
	}

	if h == nil || h.registry == nil {
		return conversationBinding{}, false, nil
	}
	providers := h.registry.List()
	sort.Strings(providers)
	capable := make([]string, 0, len(providers))
	for _, provider := range providers {
		if h.providerSupportsConversations(provider) {
			capable = append(capable, provider)
		}
	}
	switch len(capable) {
	case 0:
		return conversationBinding{}, false, nil
	case 1:
		binding, err := h.validateConversationProvider(capable[0])
		return binding, err == nil, err
	default:
		return conversationBinding{}, false, &conversationBindingResolutionError{
			message: "multiple providers enable conversations; select one with X-LunarGate-Provider",
			param:   "provider",
			code:    "ambiguous_provider",
		}
	}
}

func (h *Handler) boundConversationBinding(r *http.Request, conversationID string) (conversationBinding, bool, error) {
	if h == nil || h.conversationBindings == nil {
		return conversationBinding{}, false, nil
	}
	requestedProvider := strings.TrimSpace(r.Header.Get("X-LunarGate-Provider"))
	binding, lookup := h.conversationBindings.lookup(conversationID)
	if lookup == ownerLookupConflict {
		if requestedProvider != "" {
			return conversationBinding{}, false, nil
		}
		return conversationBinding{}, false, &conversationBindingResolutionError{
			message: fmt.Sprintf("conversation %q has conflicting provider ownership", conversationID),
			param:   "conversation_id",
			code:    "provider_binding_conflict",
		}
	}
	if lookup != ownerLookupBound {
		return conversationBinding{}, false, nil
	}
	if requestedProvider != "" && requestedProvider != binding.Provider {
		return conversationBinding{}, false, &conversationBindingResolutionError{
			message: fmt.Sprintf("conversation %q belongs to provider %q, not %q", conversationID, binding.Provider, requestedProvider),
			param:   "provider",
			code:    "invalid_value",
		}
	}
	currentBinding, err := h.validateConversationProvider(binding.Provider)
	if err != nil {
		return conversationBinding{}, false, err
	}
	if subtle.ConstantTimeCompare(
		[]byte(binding.AccountFingerprint),
		[]byte(currentBinding.AccountFingerprint),
	) != 1 {
		return conversationBinding{}, false, &conversationBindingResolutionError{
			message: fmt.Sprintf("provider account configuration changed for conversation %q", conversationID),
			param:   "provider",
			code:    "provider_binding_stale",
		}
	}
	return binding, true, nil
}

func (h *Handler) explicitConversationBinding(r *http.Request) (conversationBinding, bool, error) {
	provider := strings.TrimSpace(r.Header.Get("X-LunarGate-Provider"))
	if provider == "" {
		return conversationBinding{}, false, nil
	}
	binding, err := h.validateConversationProvider(provider)
	return binding, err == nil, err
}

func (h *Handler) retainNativeConversationBinding(conversationID string, binding conversationBinding) bool {
	if h == nil || h.conversationBindings == nil || !validNativeConversationID(conversationID) {
		return false
	}
	return h.conversationBindings.claim(conversationID, binding).retained()
}

func validNativeConversationID(conversationID string) bool {
	return validOpaqueResourceID(conversationID) &&
		strings.HasPrefix(conversationID, "conv_") && len(conversationID) > len("conv_")
}

func conversationAccountFingerprint(providerType, baseURL, organization, apiKey string) string {
	return provideridentity.AccountFingerprint(providerType, baseURL, organization, apiKey)
}

func conversationBindingHeaders(w http.ResponseWriter, binding conversationBinding) {
	if provider := strings.TrimSpace(binding.Provider); provider != "" {
		w.Header().Set("X-LunarGate-Provider", provider)
	}
}

func writeConversationBindingResolutionError(w http.ResponseWriter, err error) {
	resolutionErr, ok := err.(*conversationBindingResolutionError)
	if !ok || resolutionErr == nil {
		writeError(w, http.StatusBadRequest, err.Error(), "invalid_request_error")
		return
	}
	param := resolutionErr.param
	code := resolutionErr.code
	writeErrorDetail(w, http.StatusBadRequest, resolutionErr.message, "invalid_request_error", &param, &code)
}
