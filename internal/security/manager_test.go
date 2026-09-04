package security

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
)

func TestManagerRejectsEnabledAuthWithoutProvider(t *testing.T) {
	for _, provider := range []string{"", "none", " NONE "} {
		t.Run(strings.TrimSpace(provider), func(t *testing.T) {
			_, err := NewManager(config.SecurityConfig{Enabled: true, Provider: provider})
			if err == nil {
				t.Fatal("NewManager returned nil error")
			}
			if !strings.Contains(err.Error(), "must not be none") {
				t.Fatalf("NewManager error = %q", err)
			}
		})
	}
}

func TestManagerKeepsAuthStateWhenReloadDisablesProvider(t *testing.T) {
	manager, err := NewManager(config.SecurityConfig{
		Enabled:  true,
		Provider: "api_key",
		APIKey: config.APIKeyAuthConfig{Keys: []config.APIKeyCredential{
			{Name: "active", Value: "lg_test_active"},
		}},
	})
	if err != nil {
		t.Fatalf("NewManager returned error: %v", err)
	}

	if err := manager.UpdateConfig(config.SecurityConfig{Enabled: true, Provider: "none"}); err == nil {
		t.Fatal("UpdateConfig returned nil error")
	}

	handler := manager.Middleware(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))
	unauthenticated := httptest.NewRecorder()
	handler.ServeHTTP(unauthenticated, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	if unauthenticated.Code != http.StatusUnauthorized {
		t.Fatalf("unauthenticated status = %d, want %d", unauthenticated.Code, http.StatusUnauthorized)
	}

	authenticatedRequest := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	authenticatedRequest.Header.Set("Authorization", "lg_test_active")
	authenticated := httptest.NewRecorder()
	handler.ServeHTTP(authenticated, authenticatedRequest)
	if authenticated.Code != http.StatusNoContent {
		t.Fatalf("authenticated status = %d, want %d", authenticated.Code, http.StatusNoContent)
	}
}

func TestManagerMiddlewareFailsClosedWithoutAuthenticator(t *testing.T) {
	manager := &Manager{}
	manager.state.Store(&runtimeState{enabled: true, provider: "api_key"})
	nextCalled := false
	handler := manager.Middleware(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		nextCalled = true
		w.WriteHeader(http.StatusNoContent)
	}))

	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/v1/models", nil))

	if nextCalled {
		t.Fatal("middleware called protected handler without an authenticator")
	}
	if recorder.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusInternalServerError)
	}
}

func TestManager_APIKeyAuthViaAuthorizationHeader(t *testing.T) {
	manager, err := NewManager(config.SecurityConfig{
		Enabled:  true,
		Provider: "api_key",
		APIKey: config.APIKeyAuthConfig{
			Header: "Authorization",
			Prefix: "Bearer",
			Keys: []config.APIKeyCredential{
				{Name: "dashboard", Value: "lg_test_123"},
			},
		},
	})
	if err != nil {
		t.Fatalf("NewManager returned error: %v", err)
	}

	var authSubject string
	handler := manager.Middleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		info, ok := AuthInfoFromContext(r.Context())
		if !ok {
			t.Fatalf("expected auth info in request context")
		}
		authSubject = info.Subject
		w.WriteHeader(http.StatusNoContent)
	}))

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	req.Header.Set("Authorization", "Bearer lg_test_123")
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusNoContent {
		t.Fatalf("expected status %d, got %d", http.StatusNoContent, rec.Code)
	}
	if authSubject != "dashboard" {
		t.Fatalf("expected auth subject %q, got %q", "dashboard", authSubject)
	}
}

func TestManager_APIKeyAuthViaXAPIKeyFallback(t *testing.T) {
	manager, err := NewManager(config.SecurityConfig{
		Enabled:  true,
		Provider: "api_key",
		APIKey: config.APIKeyAuthConfig{
			AllowXAPIKey: true,
			Keys: []config.APIKeyCredential{
				{Name: "cli", Value: "lg_test_456"},
			},
		},
	})
	if err != nil {
		t.Fatalf("NewManager returned error: %v", err)
	}

	handler := manager.Middleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusAccepted)
	}))

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	req.Header.Set("X-API-Key", "lg_test_456")
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusAccepted {
		t.Fatalf("expected status %d, got %d", http.StatusAccepted, rec.Code)
	}
}

func TestManager_MissingCredentialsReturnsUnauthorized(t *testing.T) {
	manager, err := NewManager(config.SecurityConfig{
		Enabled:  true,
		Provider: "api_key",
		APIKey: config.APIKeyAuthConfig{
			Keys: []config.APIKeyCredential{
				{Name: "app", Value: "lg_test_789"},
			},
		},
	})
	if err != nil {
		t.Fatalf("NewManager returned error: %v", err)
	}

	handler := manager.Middleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Fatalf("next handler should not be called for unauthorized request")
	}))

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("expected status %d, got %d", http.StatusUnauthorized, rec.Code)
	}
	if got := rec.Header().Get("WWW-Authenticate"); got == "" {
		t.Fatalf("expected WWW-Authenticate header to be set")
	}
}

func TestManager_InvalidCredentialsReturnsUnauthorized(t *testing.T) {
	manager, err := NewManager(config.SecurityConfig{
		Enabled:  true,
		Provider: "api_key",
		APIKey: config.APIKeyAuthConfig{
			Keys: []config.APIKeyCredential{
				{Name: "app", Value: "lg_valid"},
			},
		},
	})
	if err != nil {
		t.Fatalf("NewManager returned error: %v", err)
	}

	handler := manager.Middleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Fatalf("next handler should not be called for unauthorized request")
	}))

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	req.Header.Set("Authorization", "Bearer lg_invalid")
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("expected status %d, got %d", http.StatusUnauthorized, rec.Code)
	}
}
