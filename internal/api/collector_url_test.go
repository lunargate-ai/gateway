package api

import (
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/providers"
)

func TestEnrichCollectorTagsSanitizesProviderBaseURL(t *testing.T) {
	tests := []struct {
		name    string
		baseURL string
		want    string
		wantTag bool
	}{
		{
			name:    "credentials query and fragment",
			baseURL: "https://alice:secret@example.com/v1?api-key=secret#x",
			want:    "https://example.com/v1",
			wantTag: true,
		},
		{
			name:    "Azure deployment path",
			baseURL: "https://resource.openai.azure.com/openai/deployments/gpt-4o?api-version=secret",
			want:    "https://resource.openai.azure.com/openai/deployments/gpt-4o",
			wantTag: true,
		},
		{
			name:    "empty forced query",
			baseURL: "https://example.com/v1?",
			want:    "https://example.com/v1",
			wantTag: true,
		},
		{name: "invalid", baseURL: "://not-a-url"},
		{name: "relative", baseURL: "/v1/providers/openai"},
		{name: "unsupported scheme", baseURL: "ftp://example.com/v1"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			registry := providers.NewRegistry(map[string]config.ProviderConfig{
				"provider": {
					Type:         "openai",
					BaseURL:      test.baseURL,
					DefaultModel: "gpt-test",
				},
			})
			handler := &Handler{registry: registry}

			tags := handler.enrichCollectorTags(nil, "provider", "provider/gpt-test", false)
			got, ok := tags["x-lunargate-upstream-base-url"]
			if ok != test.wantTag {
				t.Fatalf("upstream base URL tag present = %v, want %v; value=%q", ok, test.wantTag, got)
			}
			if got != test.want {
				t.Fatalf("upstream base URL tag = %q, want %q", got, test.want)
			}
		})
	}
}
