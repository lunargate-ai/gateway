package api

import (
	"net/http"
	"net/url"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestRequestContextWithRetryPolicyRetainsOnlyProviderControlHeaders(t *testing.T) {
	request := &http.Request{
		Header: http.Header{
			"Authorization":       {"Bearer client-secret"},
			"Cookie":              {"session=client-secret"},
			"Idempotency-Key":     {"request-one", "request-two"},
			"OpenAI-Beta":         {"responses=v1"},
			"Anthropic-Beta":      {"prompt-caching-2024-07-31"},
			"OpenAI-Organization": {"client-organization"},
			"X-Api-Key":           {"client-secret"},
		},
	}

	translator := providers.NewOpenAITranslator(config.ProviderConfig{
		APIKey:       "configured-secret",
		BaseURL:      "https://api.openai.com/v1",
		Organization: "configured-organization",
	})
	upstream, err := translator.TranslateRequest(requestContextWithRetryPolicy(request), &models.UnifiedRequest{
		Model:    "gpt-5.4",
		Messages: []models.Message{{Role: "user", Content: "hello"}},
	})
	if err != nil {
		t.Fatalf("TranslateRequest: %v", err)
	}
	if got := upstream.Header.Values("Idempotency-Key"); len(got) != 2 || got[0] != "request-one" || got[1] != "request-two" {
		t.Fatalf("Idempotency-Key = %#v", got)
	}
	if got := upstream.Header.Values("OpenAI-Beta"); len(got) != 1 || got[0] != "responses=v1" {
		t.Fatalf("OpenAI-Beta = %#v", got)
	}
	if got := upstream.Header.Get("Authorization"); got != "Bearer configured-secret" {
		t.Fatalf("Authorization = %q", got)
	}
	if got := upstream.Header.Get("OpenAI-Organization"); got != "configured-organization" {
		t.Fatalf("OpenAI-Organization = %q", got)
	}
	for _, name := range []string{"Anthropic-Beta", "Cookie", "X-Api-Key"} {
		if got := upstream.Header.Values(name); len(got) != 0 {
			t.Fatalf("unsafe header %s = %#v", name, got)
		}
	}
}

func TestCopyForwardedRequestHeaderPreservesValuesCaseInsensitively(t *testing.T) {
	destination := make(http.Header)
	source := http.Header{
		"openai-beta": {"responses=v1", "assistants=v2"},
		"Cookie":      {"secret=value"},
	}

	copyForwardedRequestHeader(destination, source, "OpenAI-Beta")
	if got := destination.Values("OpenAI-Beta"); len(got) != 2 || got[0] != "responses=v1" || got[1] != "assistants=v2" {
		t.Fatalf("OpenAI-Beta = %#v", got)
	}
	if got := destination.Values("Cookie"); len(got) != 0 {
		t.Fatalf("Cookie = %#v", got)
	}
}

func TestResponsesWebSocketCreateDoesNotReuseHandshakeIdempotencyKey(t *testing.T) {
	headers := make(http.Header)
	headers.Add("Idempotency-Key", "websocket-handshake")
	headers.Add("OpenAI-Beta", "responses=v1")
	baseRequest := &http.Request{
		Method: http.MethodGet,
		URL:    &url.URL{Path: "/v1/responses"},
		Header: headers,
	}

	request := makeResponsesWebSocketHTTPRequest(baseRequest, []byte(`{"model":"gpt-5.4"}`), "session-one")
	if got := request.Header.Values("Idempotency-Key"); len(got) != 0 {
		t.Fatalf("Idempotency-Key = %#v", got)
	}
	if got := request.Header.Values("OpenAI-Beta"); len(got) != 1 || got[0] != "responses=v1" {
		t.Fatalf("OpenAI-Beta = %#v", got)
	}
	if got := request.Header.Get("X-LunarGate-Sessionid"); got != "session-one" {
		t.Fatalf("session ID = %q", got)
	}
}
