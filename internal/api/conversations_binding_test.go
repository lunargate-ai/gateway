package api

import (
	"crypto/sha256"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/providers"
)

func TestConversationCreateBindingSelection(t *testing.T) {
	tests := []struct {
		name          string
		providers     map[string]config.ProviderConfig
		provider      string
		model         string
		wantNative    bool
		wantProvider  string
		wantErrorCode string
	}{
		{
			name: "no capable provider uses local storage",
			providers: map[string]config.ProviderConfig{
				"chat": conversationBindingProviderConfig(false),
			},
		},
		{
			name: "single capable provider is deterministic",
			providers: map[string]config.ProviderConfig{
				"chat":   conversationBindingProviderConfig(false),
				"native": conversationBindingProviderConfig(true),
			},
			wantNative:   true,
			wantProvider: "native",
		},
		{
			name: "multiple capable providers require selection",
			providers: map[string]config.ProviderConfig{
				"alpha": conversationBindingProviderConfig(true),
				"beta":  conversationBindingProviderConfig(true),
			},
			wantErrorCode: "ambiguous_provider",
		},
		{
			name: "explicit provider selects among capable providers",
			providers: map[string]config.ProviderConfig{
				"alpha": conversationBindingProviderConfig(true),
				"beta":  conversationBindingProviderConfig(true),
			},
			provider:     "beta",
			wantNative:   true,
			wantProvider: "beta",
		},
		{
			name: "canonical model selects provider account only",
			providers: map[string]config.ProviderConfig{
				"alpha": conversationBindingProviderConfig(true),
				"beta":  conversationBindingProviderConfig(true),
			},
			model:        "alpha/gpt-test",
			wantNative:   true,
			wantProvider: "alpha",
		},
		{
			name: "conflicting provider and model are rejected",
			providers: map[string]config.ProviderConfig{
				"alpha": conversationBindingProviderConfig(true),
				"beta":  conversationBindingProviderConfig(true),
			},
			provider:      "beta",
			model:         "alpha/gpt-test",
			wantErrorCode: "invalid_value",
		},
		{
			name: "explicit provider must declare capability",
			providers: map[string]config.ProviderConfig{
				"chat": conversationBindingProviderConfig(false),
			},
			provider:      "chat",
			wantErrorCode: "unsupported_feature",
		},
		{
			name: "explicit provider must exist",
			providers: map[string]config.ProviderConfig{
				"native": conversationBindingProviderConfig(true),
			},
			provider:      "missing",
			wantErrorCode: "provider_not_found",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			handler := NewHandler(providers.NewRegistry(test.providers), nil, nil, nil, nil, nil, nil, nil, nil)
			handler.UpdateProviderConfigs(test.providers)
			request := httptest.NewRequest(http.MethodPost, "/v1/conversations", nil)
			request.Header.Set("X-LunarGate-Provider", test.provider)
			request.Header.Set("X-LunarGate-Model", test.model)

			binding, native, err := handler.conversationCreateBinding(request)
			if test.wantErrorCode != "" {
				resolutionErr, ok := err.(*conversationBindingResolutionError)
				if !ok || resolutionErr.code != test.wantErrorCode {
					t.Fatalf("error = %#v, want code %q", err, test.wantErrorCode)
				}
				return
			}
			if err != nil {
				t.Fatalf("selection error: %v", err)
			}
			if native != test.wantNative || binding.Provider != test.wantProvider {
				t.Fatalf("binding = %#v, native = %v; want provider %q, native %v", binding, native, test.wantProvider, test.wantNative)
			}
		})
	}
}

func TestConversationBindingStoreIsBoundedAndExpires(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	store := newConversationBindingStore(time.Minute)
	store.now = func() time.Time { return now }
	store.maxEntries = 1
	store.maxBytes = 128

	if !store.put("conv_first", conversationBinding{Provider: "alpha", AccountFingerprint: "first"}) {
		t.Fatal("first binding was not retained")
	}
	if !store.put("conv_second", conversationBinding{Provider: "beta", AccountFingerprint: "second"}) {
		t.Fatal("second binding was not retained")
	}
	if _, ok := store.get("conv_first"); ok {
		t.Fatal("oldest binding was not evicted")
	}
	if binding, ok := store.get("conv_second"); !ok || binding.Provider != "beta" {
		t.Fatalf("second binding = %#v, ok = %v", binding, ok)
	}

	now = now.Add(time.Minute)
	if _, ok := store.get("conv_second"); ok {
		t.Fatal("expired binding remained available")
	}
}

func TestBoundConversationBindingNeverGuessesAfterExpiry(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	providerConfigs := map[string]config.ProviderConfig{
		"alpha": conversationBindingProviderConfig(true),
		"beta":  conversationBindingProviderConfig(true),
	}
	handler := NewHandler(providers.NewRegistry(providerConfigs), nil, nil, nil, nil, nil, nil, nil, nil)
	handler.UpdateProviderConfigs(providerConfigs)
	handler.conversationBindings = newConversationBindingStore(time.Minute)
	handler.conversationBindings.now = func() time.Time { return now }
	binding, err := handler.validateConversationProvider("alpha")
	if err != nil {
		t.Fatal(err)
	}
	handler.conversationBindings.put("conv_bound", binding)

	request := httptest.NewRequest(http.MethodGet, "/v1/conversations/conv_bound", nil)
	resolvedBinding, ok, err := handler.boundConversationBinding(request, "conv_bound")
	if err != nil || !ok || resolvedBinding.Provider != "alpha" {
		t.Fatalf("bound resolution = %#v, %v, %v", resolvedBinding, ok, err)
	}

	request.Header.Set("X-LunarGate-Provider", "beta")
	if _, _, err := handler.boundConversationBinding(request, "conv_bound"); err == nil {
		t.Fatal("conflicting explicit provider was accepted")
	}

	now = now.Add(time.Minute)
	request.Header.Del("X-LunarGate-Provider")
	if binding, ok, err := handler.boundConversationBinding(request, "conv_bound"); err != nil || ok || binding.Provider != "" {
		t.Fatalf("expired resolution = %#v, %v, %v; want unbound", binding, ok, err)
	}
}

func TestBoundConversationBindingRejectsChangedProviderAccount(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*config.ProviderConfig)
	}{
		{name: "API key", mutate: func(cfg *config.ProviderConfig) { cfg.APIKey = "rotated-key" }},
		{name: "base URL", mutate: func(cfg *config.ProviderConfig) { cfg.BaseURL = "https://other.example/v1" }},
		{name: "organization", mutate: func(cfg *config.ProviderConfig) { cfg.Organization = "org-other" }},
		{name: "provider type", mutate: func(cfg *config.ProviderConfig) { cfg.Type = "anthropic" }},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			original := conversationBindingProviderConfig(true)
			original.Organization = "org-initial"
			providerConfigs := map[string]config.ProviderConfig{"native": original}
			handler := NewHandler(providers.NewRegistry(providerConfigs), nil, nil, nil, nil, nil, nil, nil, nil)
			handler.UpdateProviderConfigs(providerConfigs)
			binding, err := handler.validateConversationProvider("native")
			if err != nil {
				t.Fatal(err)
			}
			if strings.Contains(binding.AccountFingerprint, original.APIKey) || len(binding.AccountFingerprint) != sha256.Size*2 {
				t.Fatalf("unsafe account fingerprint %q", binding.AccountFingerprint)
			}
			handler.conversationBindings.put("conv_account", binding)

			changed := original
			test.mutate(&changed)
			changedConfigs := map[string]config.ProviderConfig{"native": changed}
			handler.registry.UpdateProvidersConfig(changedConfigs)
			handler.UpdateProviderConfigs(changedConfigs)

			request := httptest.NewRequest(http.MethodGet, "/v1/conversations/conv_account", nil)
			_, _, err = handler.boundConversationBinding(request, "conv_account")
			resolutionErr, ok := err.(*conversationBindingResolutionError)
			if !ok || resolutionErr.code != "provider_binding_stale" {
				t.Fatalf("error = %#v, want provider_binding_stale", err)
			}
			for _, secret := range []string{original.APIKey, changed.APIKey, binding.AccountFingerprint} {
				if secret != "" && strings.Contains(resolutionErr.Error(), secret) {
					t.Fatalf("binding error leaked account identity: %q", resolutionErr.Error())
				}
			}
		})
	}
}

func conversationBindingProviderConfig(conversations bool) config.ProviderConfig {
	return config.ProviderConfig{
		Type:         "openai",
		APIKey:       "test-key",
		BaseURL:      "https://example.invalid/v1",
		DefaultModel: "gpt-test",
		Capabilities: config.ProviderCapabilities{Conversations: conversations},
	}
}
