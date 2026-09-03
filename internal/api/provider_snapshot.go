package api

import (
	"context"
	"net/http"
	"strings"

	"github.com/lunargate-ai/gateway/internal/providers"
)

type providerRequestSnapshotKey struct{}
type responseExecutionOwnerKey struct{}

type providerRequestSnapshot struct {
	provider string
	snapshot providers.ProviderSnapshot
}

type responseExecutionOwner struct {
	Provider            string
	Route               string
	Model               string
	UpstreamRequestType string
	AccountFingerprint  string
}

type responseExecutionOwnerSink interface {
	setResponseExecutionOwner(responseExecutionOwner)
}

func withProviderRequestSnapshot(ctx context.Context, provider string, snapshot providers.ProviderSnapshot) context.Context {
	return context.WithValue(ctx, providerRequestSnapshotKey{}, providerRequestSnapshot{
		provider: strings.TrimSpace(provider),
		snapshot: snapshot,
	})
}

func providerRequestSnapshotFromResponse(resp *http.Response, provider string) (providers.ProviderSnapshot, bool) {
	if resp == nil || resp.Request == nil {
		return providers.ProviderSnapshot{}, false
	}
	value, ok := resp.Request.Context().Value(providerRequestSnapshotKey{}).(providerRequestSnapshot)
	if !ok || value.provider != strings.TrimSpace(provider) || value.snapshot.Translator == nil {
		return providers.ProviderSnapshot{}, false
	}
	return value.snapshot, true
}

func withResponseExecutionOwner(ctx context.Context, owner responseExecutionOwner) context.Context {
	return context.WithValue(ctx, responseExecutionOwnerKey{}, owner)
}

func responseExecutionOwnerFromResponse(resp *http.Response, provider string) (responseExecutionOwner, bool) {
	if resp == nil || resp.Request == nil {
		return responseExecutionOwner{}, false
	}
	owner, ok := resp.Request.Context().Value(responseExecutionOwnerKey{}).(responseExecutionOwner)
	if !ok || owner.Provider != strings.TrimSpace(provider) || owner.AccountFingerprint == "" {
		return responseExecutionOwner{}, false
	}
	return owner, true
}

func responseExecutionOwnerFromRequest(
	provider string,
	snapshot providers.ProviderSnapshot,
	request *http.Request,
) (responseExecutionOwner, bool) {
	provider = strings.TrimSpace(provider)
	if provider == "" || snapshot.Translator == nil || request == nil {
		return responseExecutionOwner{}, false
	}
	providerType := strings.ToLower(strings.TrimSpace(snapshot.ProviderType))
	organization := ""
	apiKey := ""
	switch providerType {
	case "openai":
		organization = request.Header.Get("OpenAI-Organization")
		authorization := request.Header.Get("Authorization")
		if len(authorization) >= len("Bearer ") && strings.EqualFold(authorization[:len("Bearer ")], "Bearer ") {
			apiKey = authorization[len("Bearer "):]
		} else {
			apiKey = authorization
		}
	case "anthropic":
		apiKey = request.Header.Get("x-api-key")
	case "ollama":
		// Ollama's native HTTP contract does not use ProviderConfig.APIKey.
	default:
		return responseExecutionOwner{}, false
	}
	return responseExecutionOwner{
		Provider: provider,
		AccountFingerprint: conversationAccountFingerprint(
			providerType,
			snapshot.Translator.BaseURL(),
			organization,
			apiKey,
		),
	}, true
}

func setResponseExecutionOwner(w http.ResponseWriter, owner responseExecutionOwner) {
	if sink, ok := w.(responseExecutionOwnerSink); ok {
		sink.setResponseExecutionOwner(owner)
	}
}
