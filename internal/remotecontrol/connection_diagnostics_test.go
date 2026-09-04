package remotecontrol

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"testing"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

func TestWebsocketURLDoesNotEchoInvalidBackendURL(t *testing.T) {
	const secret = "invalid-url-secret"
	client := &Client{backendURL: "https://example.test/%zz?token=" + secret}

	_, err := client.websocketURL()
	if err == nil {
		t.Fatal("websocketURL returned nil error for invalid backend URL")
	}
	if strings.Contains(err.Error(), secret) {
		t.Fatalf("websocketURL error leaked backend URL: %v", err)
	}
}

func TestWebsocketURLPreservesTransportQueryAndDropsFragment(t *testing.T) {
	client := &Client{
		backendURL: "HTTPS://url-user:url-password@private.example.test/root/v1/?token=query-secret#fragment-secret",
	}

	endpoint, err := client.websocketURL()
	if err != nil {
		t.Fatalf("websocketURL returned error: %v", err)
	}
	parsed, err := url.Parse(endpoint)
	if err != nil {
		t.Fatalf("parse WebSocket URL: %v", err)
	}
	if got, want := parsed.Scheme, "wss"; got != want {
		t.Fatalf("scheme = %q, want %q", got, want)
	}
	if got, want := parsed.Path, "/root/v1/remote-control/ws/gateway"; got != want {
		t.Fatalf("path = %q, want %q", got, want)
	}
	if got, want := parsed.RawQuery, "token=query-secret"; got != want {
		t.Fatalf("query = %q, want preserved transport query", got)
	}
	if parsed.Fragment != "" {
		t.Fatalf("fragment = %q, want discarded", parsed.Fragment)
	}
}

func newSecretHandshakeResponse(secret string) *http.Response {
	return &http.Response{
		StatusCode: http.StatusUnauthorized,
		Body:       io.NopCloser(strings.NewReader(`{"detail":"` + secret + `"}`)),
	}
}

func TestDialErrorAndLogRedactWebSocketURL(t *testing.T) {
	const rawURL = "wss://url-user:url-password@private.example.test/root/v1/remote-control/ws/gateway?token=query-secret#fragment-secret"
	requestURL, err := url.Parse(rawURL)
	if err != nil {
		t.Fatalf("parse request URL: %v", err)
	}
	category := errors.New("connection-refused-category")
	causes := []struct {
		name string
		err  error
	}{
		{name: "URL error", err: &url.Error{
			Op:  "dial",
			URL: rawURL,
			Err: fmt.Errorf("dial %s: %w", rawURL, category),
		}},
		{name: "plain error", err: fmt.Errorf("dial %s: %w", rawURL, category)},
	}
	for _, test := range causes {
		t.Run(test.name, func(t *testing.T) {
			classified := classifyDialError(test.err, nil, requestURL)
			if !errors.Is(classified, category) {
				t.Fatalf("dial error lost transport classification: %v", classified)
			}
			for _, secret := range []string{"url-user", "url-password", "query-secret", "fragment-secret"} {
				if strings.Contains(classified.Error(), secret) {
					t.Fatalf("error leaked %q: %s", secret, classified)
				}
			}
			if !strings.Contains(classified.Error(), "wss://private.example.test/root/v1/remote-control/ws/gateway") {
				t.Fatalf("error lost sanitized endpoint: %s", classified)
			}
		})
	}
	dialErr := classifyDialError(causes[1].err, nil, requestURL)

	var output bytes.Buffer
	previousLogger := log.Logger
	log.Logger = zerolog.New(&output)
	t.Cleanup(func() { log.Logger = previousLogger })
	client := &Client{}
	client.logConnectionIssue(dialErr)

	logged := output.String()
	for _, secret := range []string{"url-user", "url-password", "query-secret", "fragment-secret"} {
		if strings.Contains(logged, secret) {
			t.Fatalf("log leaked %q: %s", secret, logged)
		}
	}
	if !strings.Contains(logged, "wss://private.example.test/root/v1/remote-control/ws/gateway") {
		t.Fatalf("log lost sanitized endpoint: %s", logged)
	}
}

func TestHandshakeLogDoesNotIncludeBackendResponseBody(t *testing.T) {
	const secret = "backend-response-secret"
	requestURL, err := url.Parse("wss://private.example.test/v1/remote-control/ws/gateway?token=query-secret")
	if err != nil {
		t.Fatalf("parse request URL: %v", err)
	}
	dialErr := classifyDialError(
		errors.New("bad handshake"),
		newSecretHandshakeResponse(secret),
		requestURL,
	)

	var output bytes.Buffer
	previousLogger := log.Logger
	log.Logger = zerolog.New(&output)
	t.Cleanup(func() { log.Logger = previousLogger })
	client := &Client{}
	client.logConnectionIssue(dialErr)

	for source, value := range map[string]string{"error": dialErr.Error(), "log": output.String()} {
		if strings.Contains(value, secret) || strings.Contains(value, "query-secret") {
			t.Fatalf("%s leaked handshake diagnostics: %s", source, value)
		}
		if !strings.Contains(value, "401") {
			t.Fatalf("%s lost safe HTTP status: %s", source, value)
		}
	}
}
