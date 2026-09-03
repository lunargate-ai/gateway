package updatecheck

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/lunargate-ai/gateway/internal/config"
)

func TestCheckSendsOnlyVersionAndArchitecture(t *testing.T) {
	var captured map[string]any
	var userAgent string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("method = %s, want POST", r.Method)
		}
		userAgent = r.Header.Get("User-Agent")
		if err := json.NewDecoder(r.Body).Decode(&captured); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"version":"0.4.0"}`))
	}))
	defer server.Close()

	checker := newChecker(config.UpdateCheckConfig{
		Enabled:  true,
		Endpoint: server.URL,
		Interval: time.Hour,
		Timeout:  time.Second,
	}, "0.3.0", "arm64", server.Client())

	result, err := checker.Check(context.Background())
	if err != nil {
		t.Fatalf("Check returned error: %v", err)
	}
	if result.Version != "0.4.0" {
		t.Fatalf("version = %q, want 0.4.0", result.Version)
	}
	if userAgent != "" {
		t.Fatalf("User-Agent = %q, want empty", userAgent)
	}
	if len(captured) != 2 || captured["version"] != "0.3.0" || captured["arch"] != "arm64" {
		t.Fatalf("payload = %#v, want only version and arch", captured)
	}
}

func TestCheckHonorsDisabledConfig(t *testing.T) {
	checker := newChecker(config.UpdateCheckConfig{Enabled: false}, "0.3.0", "amd64", http.DefaultClient)

	if _, err := checker.Check(context.Background()); err != ErrDisabled {
		t.Fatalf("Check error = %v, want ErrDisabled", err)
	}
}

func TestCheckDoesNotFollowRedirects(t *testing.T) {
	redirectTargetCalled := false
	target := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		redirectTargetCalled = true
	}))
	defer target.Close()

	redirect := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, target.URL, http.StatusTemporaryRedirect)
	}))
	defer redirect.Close()

	client := &http.Client{CheckRedirect: func(_ *http.Request, _ []*http.Request) error {
		return http.ErrUseLastResponse
	}}
	checker := newChecker(config.UpdateCheckConfig{
		Enabled:  true,
		Endpoint: redirect.URL,
		Timeout:  time.Second,
	}, "0.3.0", "amd64", client)

	if _, err := checker.Check(context.Background()); err == nil {
		t.Fatal("Check returned nil error for redirect")
	}
	if redirectTargetCalled {
		t.Fatal("redirect target received update-check payload")
	}
}

func TestCheckAcceptsAdditiveResponseFields(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"version":"0.4.0","channel":"stable","metadata":{"published":true}}`))
	}))
	defer server.Close()

	checker := newChecker(config.UpdateCheckConfig{
		Enabled:  true,
		Endpoint: server.URL,
		Timeout:  time.Second,
	}, "0.3.0", "amd64", server.Client())

	result, err := checker.Check(context.Background())
	if err != nil {
		t.Fatalf("Check returned error: %v", err)
	}
	if result.Version != "0.4.0" {
		t.Fatalf("version = %q, want 0.4.0", result.Version)
	}
}

func TestCheckRejectsInvalidJSONDocuments(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{name: "malformed", body: `{"version":"0.4.0"`},
		{name: "multiple", body: `{"version":"0.4.0"} {"version":"0.5.0"}`},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(test.body))
			}))
			defer server.Close()

			checker := newChecker(config.UpdateCheckConfig{
				Enabled:  true,
				Endpoint: server.URL,
				Timeout:  time.Second,
			}, "0.3.0", "amd64", server.Client())

			if _, err := checker.Check(context.Background()); err == nil {
				t.Fatal("Check returned nil error")
			}
		})
	}
}

func TestIsUpdateAvailableUsesSemanticVersionOrder(t *testing.T) {
	tests := []struct {
		current string
		latest  string
		want    bool
	}{
		{current: "0.3.9", latest: "0.4.0", want: true},
		{current: "0.10.0", latest: "0.9.0", want: false},
		{current: "0.4.0", latest: "0.4.0", want: false},
		{current: "dev", latest: "0.4.0", want: false},
	}

	for _, test := range tests {
		if got := IsUpdateAvailable(test.current, test.latest); got != test.want {
			t.Errorf("IsUpdateAvailable(%q, %q) = %v, want %v", test.current, test.latest, got, test.want)
		}
	}
}
