package api

import (
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

func TestProviderHTTPClientDoesNotFollowRedirects(t *testing.T) {
	var redirectedCalls atomic.Int32
	redirectTarget := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		redirectedCalls.Add(1)
		w.WriteHeader(http.StatusOK)
	}))
	defer redirectTarget.Close()

	redirectSource := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Location", redirectTarget.URL+"/credential-target")
		w.WriteHeader(http.StatusTemporaryRedirect)
		_, _ = io.WriteString(w, `{"redirect":"not followed"}`)
	}))
	defer redirectSource.Close()

	resp, err := newProviderHTTPClient().Get(redirectSource.URL)
	if err != nil {
		t.Fatalf("provider request failed: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusTemporaryRedirect {
		t.Fatalf("status = %d, want %d", resp.StatusCode, http.StatusTemporaryRedirect)
	}
	if got := redirectedCalls.Load(); got != 0 {
		t.Fatalf("redirect target calls = %d, want zero", got)
	}
}
