package safeurl

import (
	"errors"
	"net/url"
	"strings"
	"testing"
)

func TestJoinHTTPPathPreservesBasePathAndQuery(t *testing.T) {
	endpoint, err := JoinHTTPPath(
		"HTTPS://url-user:url-password@example.test/root/v1/?api_key=query-secret#fragment-secret",
		"chat/completions",
	)
	if err != nil {
		t.Fatalf("JoinHTTPPath returned error: %v", err)
	}
	parsed, err := url.Parse(endpoint)
	if err != nil {
		t.Fatalf("parse endpoint: %v", err)
	}
	if got, want := parsed.Scheme, "https"; got != want {
		t.Fatalf("scheme = %q, want %q", got, want)
	}
	if got, want := parsed.Path, "/root/v1/chat/completions"; got != want {
		t.Fatalf("path = %q, want %q", got, want)
	}
	if got, want := parsed.RawQuery, "api_key=query-secret"; got != want {
		t.Fatalf("query = %q, want %q", got, want)
	}
	if parsed.User == nil || parsed.User.Username() != "url-user" {
		t.Fatalf("userinfo was not preserved for transport: %#v", parsed.User)
	}
	if got, _ := parsed.User.Password(); got != "url-password" {
		t.Fatalf("password = %q, want preserved transport credential", got)
	}
	if parsed.Fragment != "" {
		t.Fatalf("fragment = %q, want discarded", parsed.Fragment)
	}
}

func TestJoinHTTPPathAndRawQueryKeepsBaseQueryFirst(t *testing.T) {
	endpoint, err := JoinHTTPPathAndRawQuery(
		"https://example.test/root?api_key=configured-secret",
		"limit=20&after=item_1",
		"responses/resp_1/input_items",
	)
	if err != nil {
		t.Fatalf("JoinHTTPPathAndRawQuery returned error: %v", err)
	}
	parsed, err := url.Parse(endpoint)
	if err != nil {
		t.Fatalf("parse endpoint: %v", err)
	}
	if got, want := parsed.Path, "/root/responses/resp_1/input_items"; got != want {
		t.Fatalf("path = %q, want %q", got, want)
	}
	if got, want := parsed.RawQuery, "api_key=configured-secret&limit=20&after=item_1"; got != want {
		t.Fatalf("query = %q, want %q", got, want)
	}
}

func TestParseHTTPBaseURLDoesNotEchoInvalidInput(t *testing.T) {
	const secret = "invalid-url-secret"
	_, err := ParseHTTPBaseURL("https://example.test/%zz?token=" + secret)
	if !errors.Is(err, ErrInvalidHTTPBaseURL) {
		t.Fatalf("error = %v, want ErrInvalidHTTPBaseURL", err)
	}
	if strings.Contains(err.Error(), secret) {
		t.Fatalf("validation error leaked input: %v", err)
	}
}

func TestRedactTransportErrorPreservesClassificationWithoutURLSecrets(t *testing.T) {
	const rawURL = "https://url-user:url-password@private.example.test/root/v1?api_key=query-secret#fragment-secret"
	requestURL, err := url.Parse(rawURL)
	if err != nil {
		t.Fatalf("parse request URL: %v", err)
	}
	cause := &classifiedNetworkError{
		cause: errors.New("dial-category"),
		text:  "dial " + rawURL + ": connection refused",
	}
	original := &url.Error{Op: "Post", URL: rawURL, Err: cause}

	redacted := RedactTransportError(original, requestURL)
	if redacted == original {
		t.Fatal("transport error was not cloned")
	}
	for _, secret := range []string{"url-user", "url-password", "query-secret", "fragment-secret"} {
		if strings.Contains(redacted.Error(), secret) {
			t.Fatalf("redacted error leaked %q: %v", secret, redacted)
		}
	}
	for _, useful := range []string{"Post", "private.example.test", "/root/v1", "connection refused"} {
		if !strings.Contains(redacted.Error(), useful) {
			t.Fatalf("redacted error lost category %q: %v", useful, redacted)
		}
	}
	if !errors.Is(redacted, cause.cause) {
		t.Fatal("errors.Is no longer reaches the transport cause")
	}
	var gotURL *url.Error
	if !errors.As(redacted, &gotURL) {
		t.Fatal("errors.As no longer finds url.Error")
	}
	if gotURL.Op != original.Op {
		t.Fatalf("operation = %q, want %q", gotURL.Op, original.Op)
	}
	if gotURL.URL != "https://private.example.test/root/v1" {
		t.Fatalf("url.Error.URL = %q, want sanitized URL", gotURL.URL)
	}
	var gotCause *classifiedNetworkError
	if !errors.As(redacted, &gotCause) || gotCause != cause {
		t.Fatalf("errors.As transport cause = %#v, want original", gotCause)
	}
	netErr, ok := redacted.(interface {
		Timeout() bool
		Temporary() bool
	})
	if !ok || !netErr.Timeout() || !netErr.Temporary() {
		t.Fatalf("network classification was not preserved: %#v", redacted)
	}
	if strings.Contains(original.Error(), "[redacted") {
		t.Fatal("original error was mutated")
	}
}

type classifiedNetworkError struct {
	cause error
	text  string
}

func (e *classifiedNetworkError) Error() string   { return e.text }
func (e *classifiedNetworkError) Unwrap() error   { return e.cause }
func (e *classifiedNetworkError) Timeout() bool   { return true }
func (e *classifiedNetworkError) Temporary() bool { return true }
