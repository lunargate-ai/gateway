package api

import (
	"context"
	"net/http"
	"strings"

	"github.com/lunargate-ai/gateway/internal/providers"
)

type providerRequestSnapshotKey struct{}

type providerRequestSnapshot struct {
	provider string
	snapshot providers.ProviderSnapshot
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
