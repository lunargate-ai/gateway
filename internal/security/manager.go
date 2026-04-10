package security

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"sync/atomic"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/pkg/models"
	"github.com/rs/zerolog/log"
)

var (
	ErrMissingCredentials = errors.New("missing authentication credentials")
	ErrInvalidCredentials = errors.New("invalid authentication credentials")
)

type contextKey string

const authInfoContextKey contextKey = "lunargate-auth-info"

type AuthInfo struct {
	Provider       string
	Subject        string
	CredentialName string
}

type Authenticator interface {
	Authenticate(r *http.Request) (*AuthInfo, error)
}

type Manager struct {
	state atomic.Value // stores *runtimeState
}

type runtimeState struct {
	enabled         bool
	provider        string
	credentialCount int
	authenticator   Authenticator
}

type apiKeyAuthenticator struct {
	header       string
	prefix       string
	allowXAPIKey bool
	keys         []apiKeyCredential
}

type apiKeyCredential struct {
	name  string
	value string
}

func NewManager(cfg config.SecurityConfig) (*Manager, error) {
	m := &Manager{}
	if err := m.UpdateConfig(cfg); err != nil {
		return nil, err
	}
	return m, nil
}

func (m *Manager) UpdateConfig(cfg config.SecurityConfig) error {
	state, err := buildRuntimeState(cfg)
	if err != nil {
		return err
	}

	m.state.Store(state)
	if !state.enabled {
		log.Info().Msg("inbound auth disabled")
		return nil
	}

	log.Info().
		Str("provider", state.provider).
		Int("credential_count", state.credentialCount).
		Msg("inbound auth config updated")
	return nil
}

func (m *Manager) Middleware(next http.Handler) http.Handler {
	if m == nil {
		return next
	}

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		state := m.currentState()
		if state == nil || !state.enabled || state.authenticator == nil {
			next.ServeHTTP(w, r)
			return
		}

		info, err := state.authenticator.Authenticate(r)
		if err != nil {
			writeAuthError(w, state.provider, err)
			return
		}

		next.ServeHTTP(w, r.WithContext(ContextWithAuthInfo(r.Context(), *info)))
	})
}

func ContextWithAuthInfo(ctx context.Context, info AuthInfo) context.Context {
	return context.WithValue(ctx, authInfoContextKey, info)
}

func AuthInfoFromContext(ctx context.Context) (AuthInfo, bool) {
	if ctx == nil {
		return AuthInfo{}, false
	}
	info, ok := ctx.Value(authInfoContextKey).(AuthInfo)
	return info, ok
}

func (m *Manager) currentState() *runtimeState {
	if m == nil {
		return nil
	}
	raw := m.state.Load()
	if raw == nil {
		return nil
	}
	return raw.(*runtimeState)
}

func buildRuntimeState(cfg config.SecurityConfig) (*runtimeState, error) {
	provider := strings.ToLower(strings.TrimSpace(cfg.Provider))
	if provider == "" {
		provider = "none"
	}

	state := &runtimeState{
		enabled:  cfg.Enabled,
		provider: provider,
	}

	if !cfg.Enabled || provider == "none" {
		return state, nil
	}

	switch provider {
	case "api_key":
		authenticator, err := newAPIKeyAuthenticator(cfg.APIKey)
		if err != nil {
			return nil, err
		}
		state.authenticator = authenticator
		state.credentialCount = len(cfg.APIKey.Keys)
		return state, nil
	case "external":
		return nil, errors.New("security.provider=external is not implemented yet")
	default:
		return nil, errors.New("unsupported inbound auth provider")
	}
}

func newAPIKeyAuthenticator(cfg config.APIKeyAuthConfig) (*apiKeyAuthenticator, error) {
	if len(cfg.Keys) == 0 {
		return nil, errors.New("security.api_key.keys must contain at least one key")
	}

	authenticator := &apiKeyAuthenticator{
		header:       strings.TrimSpace(cfg.Header),
		prefix:       strings.TrimSpace(cfg.Prefix),
		allowXAPIKey: cfg.AllowXAPIKey,
		keys:         make([]apiKeyCredential, 0, len(cfg.Keys)),
	}
	if authenticator.header == "" {
		authenticator.header = "Authorization"
	}

	for _, key := range cfg.Keys {
		value := strings.TrimSpace(key.Value)
		if value == "" {
			return nil, errors.New("security.api_key.keys contains an empty key value")
		}
		authenticator.keys = append(authenticator.keys, apiKeyCredential{
			name:  strings.TrimSpace(key.Name),
			value: value,
		})
	}

	return authenticator, nil
}

func (a *apiKeyAuthenticator) Authenticate(r *http.Request) (*AuthInfo, error) {
	if a == nil {
		return nil, nil
	}

	credential, presented := a.extractCredential(r)
	if !presented {
		return nil, ErrMissingCredentials
	}

	for _, key := range a.keys {
		if subtle.ConstantTimeCompare([]byte(credential), []byte(key.value)) == 1 {
			name := key.name
			if name == "" {
				name = "api-key"
			}
			return &AuthInfo{
				Provider:       "api_key",
				Subject:        name,
				CredentialName: name,
			}, nil
		}
	}

	return nil, ErrInvalidCredentials
}

func (a *apiKeyAuthenticator) extractCredential(r *http.Request) (string, bool) {
	if a == nil {
		return "", false
	}

	if value, presented := extractHeaderCredential(r.Header.Get(a.header), a.prefix); presented {
		if value != "" {
			return value, true
		}
	}

	if a.allowXAPIKey {
		xAPIKey := strings.TrimSpace(r.Header.Get("X-API-Key"))
		if xAPIKey != "" {
			return xAPIKey, true
		}
	}

	raw := strings.TrimSpace(r.Header.Get(a.header))
	return "", raw != ""
}

func extractHeaderCredential(raw, prefix string) (string, bool) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return "", false
	}
	if prefix == "" {
		return raw, true
	}

	parts := strings.Fields(raw)
	if len(parts) < 2 {
		return "", true
	}
	if !strings.EqualFold(parts[0], prefix) {
		return "", true
	}
	return strings.Join(parts[1:], " "), true
}

func writeAuthError(w http.ResponseWriter, provider string, err error) {
	status := http.StatusUnauthorized
	message := "authentication required"
	errType := "authentication_error"

	switch {
	case errors.Is(err, ErrMissingCredentials):
		message = "missing API key"
	case errors.Is(err, ErrInvalidCredentials):
		message = "invalid API key"
		errType = "invalid_api_key"
	default:
		status = http.StatusInternalServerError
		message = "failed to authenticate request"
		errType = "internal_error"
	}

	if provider == "api_key" {
		w.Header().Set("WWW-Authenticate", `Bearer realm="lunargate"`)
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(models.ErrorResponse{
		Error: models.ErrorDetail{
			Message: message,
			Type:    errType,
		},
	}); err != nil {
		log.Error().Err(err).Msg("failed to encode auth error response")
	}
}
