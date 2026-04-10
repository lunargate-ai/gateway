package security

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
)

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
