package observability

import (
	"bytes"
	"context"
	"errors"
	"net/http"
	"strings"
	"testing"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

func TestCollectorTransportErrorRedactsBackendURL(t *testing.T) {
	const backendURL = "https://url-user:url-password@private-collector.example/root/v1?api_key=query-secret#fragment-secret"
	client := &CollectorClient{
		httpClient: &http.Client{Transport: collectorURLRoundTripFunc(func(request *http.Request) (*http.Response, error) {
			if got, want := request.URL.Path, "/root/v1/collector"; got != want {
				t.Fatalf("collector path = %q, want %q", got, want)
			}
			if got, want := request.URL.RawQuery, "api_key=query-secret"; got != want {
				t.Fatalf("collector query = %q, want %q", got, want)
			}
			return nil, errors.New("connection-refused-category")
		})},
		cfg: collectorRuntimeConfig{
			enabled:    true,
			backendURL: backendURL,
			apiKey:     "collector-header-secret",
		},
	}
	item := collectorItem{
		requestID: "request-1",
		payload:   []byte(`{"events":[]}`),
		identity: collectorIdentity{
			backendURL: backendURL,
			apiKey:     "collector-header-secret",
		},
	}

	err := client.send(context.Background(), item)
	if err == nil {
		t.Fatal("collector transport failure returned no error")
	}
	var logs bytes.Buffer
	previousLogger := log.Logger
	log.Logger = zerolog.New(&logs)
	t.Cleanup(func() { log.Logger = previousLogger })
	client.logSendError(item.requestID, err)

	for source, value := range map[string]string{"error": err.Error(), "log": logs.String()} {
		for _, secret := range []string{"url-user", "url-password", "query-secret", "fragment-secret", "collector-header-secret"} {
			if strings.Contains(value, secret) {
				t.Fatalf("collector %s leaked %q: %s", source, secret, value)
			}
		}
		for _, useful := range []string{"connection-refused-category", "private-collector.example", "/root/v1/collector"} {
			if !strings.Contains(value, useful) {
				t.Fatalf("collector %s lost %q: %s", source, useful, value)
			}
		}
	}
}

type collectorURLRoundTripFunc func(*http.Request) (*http.Response, error)

func (f collectorURLRoundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return f(request)
}
