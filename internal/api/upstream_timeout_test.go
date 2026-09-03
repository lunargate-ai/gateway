package api

import (
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/providers"
)

const shortUpstreamAttemptTimeout = 30 * time.Millisecond

func TestProviderAttemptTimeoutDuringDialUsesFallback(t *testing.T) {
	for _, api := range nonStreamingProtocolAPIs() {
		for _, mode := range []string{upstreamTimeoutModeTTFT, upstreamTimeoutModeTotal} {
			t.Run(api.name+"/"+mode, func(t *testing.T) {
				providerConfigs := map[string]config.ProviderConfig{
					"primary": {
						Type:        "openai",
						APIKey:      "test-primary",
						BaseURL:     "http://primary.invalid/v1",
						Timeout:     shortUpstreamAttemptTimeout,
						TimeoutMode: mode,
					},
					"fallback": {
						Type:    "openai",
						APIKey:  "test-fallback",
						BaseURL: "http://fallback.invalid/v1",
						Timeout: time.Second,
					},
				}
				handler, cbm, _ := newResilienceClassificationHandler(
					t,
					providerConfigs,
					config.RouteConfig{
						Name:     api.name,
						Match:    config.MatchConfig{Path: api.path},
						Targets:  []config.TargetConfig{{Provider: "primary", Model: api.primaryModel, Weight: 1}},
						Fallback: []config.TargetConfig{{Provider: "fallback", Model: api.fallbackModel, Weight: 1}},
					},
					config.RetryConfig{Enabled: false},
				)
				handler.UpdateProviderConfigs(providerConfigs)

				var primaryCalls atomic.Int32
				primaryTransport := contextBlockedDialTransport(&primaryCalls)
				t.Cleanup(primaryTransport.CloseIdleConnections)
				setProviderTransportForTest(t, handler, "primary", primaryTransport)

				var fallbackCalls, fallbackClosed, redirectCalls atomic.Int32
				setProviderTransportForTest(t, handler, "fallback", protocolResponseTransport(
					http.StatusOK,
					func() io.ReadCloser { return io.NopCloser(strings.NewReader(api.validFallback)) },
					&fallbackCalls,
					&fallbackClosed,
					&redirectCalls,
				))

				startedAt := time.Now()
				recorder := performNonStreamingProtocolRequest(t, api, handler)
				elapsed := time.Since(startedAt)

				if elapsed < shortUpstreamAttemptTimeout/2 || elapsed > time.Second {
					t.Fatalf("request duration = %s, want a short provider timeout", elapsed)
				}
				if recorder.Code != http.StatusOK || recorder.Body.String() != api.validFallback {
					t.Fatalf("fallback response = status %d body %s", recorder.Code, recorder.Body.String())
				}
				if got := recorder.Header().Get("X-LunarGate-Provider"); got != "fallback" {
					t.Fatalf("provider header = %q, want fallback", got)
				}
				if primaryCalls.Load() != 1 || fallbackCalls.Load() != 1 || fallbackClosed.Load() != 1 {
					t.Fatalf(
						"primary/fallback/closed calls = %d/%d/%d, want 1/1/1",
						primaryCalls.Load(),
						fallbackCalls.Load(),
						fallbackClosed.Load(),
					)
				}
				if failures := cbm.Get("primary").Counts().TotalFailures; failures != 1 {
					t.Fatalf("primary breaker failures = %d, want 1", failures)
				}
			})
		}
	}
}

func TestProviderAttemptTimeoutClosesBlockedResponseBody(t *testing.T) {
	testCases := []struct {
		mode      string
		isTimeout func(error) bool
	}{
		{mode: upstreamTimeoutModeTTFT, isTimeout: isUpstreamTTFTTimeout},
		{mode: upstreamTimeoutModeTotal, isTimeout: isUpstreamTotalTimeout},
	}

	for _, testCase := range testCases {
		t.Run(testCase.mode, func(t *testing.T) {
			body := newCloseBlockedReadBody()
			client := &http.Client{Transport: providerURLRoundTripFunc(func(request *http.Request) (*http.Response, error) {
				return &http.Response{
					StatusCode: http.StatusOK,
					Header:     make(http.Header),
					Body:       body,
					Request:    request,
				}, nil
			})}
			request, err := http.NewRequest(http.MethodGet, "http://provider.invalid/v1/models", nil)
			if err != nil {
				t.Fatalf("create request: %v", err)
			}

			startedAt := time.Now()
			response, err := doProviderRequest(request, providerClientConfig{
				client:  client,
				timeout: shortUpstreamAttemptTimeout,
				mode:    testCase.mode,
			}, "provider", "failed to call provider")
			if err != nil {
				t.Fatalf("provider request returned before body read: %v", err)
			}
			_, readErr := response.Body.Read(make([]byte, 1))
			elapsed := time.Since(startedAt)

			if !testCase.isTimeout(readErr) {
				t.Fatalf("body read error = %v, want %s timeout", readErr, testCase.mode)
			}
			if elapsed < shortUpstreamAttemptTimeout/2 || elapsed > time.Second {
				t.Fatalf("body read duration = %s, want a short provider timeout", elapsed)
			}
			if got := body.closeCalls.Load(); got != 1 {
				t.Fatalf("body close calls after timeout = %d, want 1", got)
			}
			_ = response.Body.Close()
			if got := body.closeCalls.Load(); got != 1 {
				t.Fatalf("body close calls after caller cleanup = %d, want 1", got)
			}
		})
	}
}

func TestNativeResponseAttemptTimeoutCoversDial(t *testing.T) {
	testCases := []struct {
		mode      string
		isTimeout func(error) bool
	}{
		{mode: upstreamTimeoutModeTTFT, isTimeout: isUpstreamTTFTTimeout},
		{mode: upstreamTimeoutModeTotal, isTimeout: isUpstreamTotalTimeout},
	}

	for _, testCase := range testCases {
		t.Run(testCase.mode, func(t *testing.T) {
			providerConfigs := map[string]config.ProviderConfig{
				"native": {
					Type:        "openai",
					APIKey:      "test-native",
					BaseURL:     "http://native.invalid/v1",
					Timeout:     shortUpstreamAttemptTimeout,
					TimeoutMode: testCase.mode,
				},
			}
			handler := &Handler{
				registry:        providers.NewRegistry(providerConfigs),
				providerClients: newProviderClientRegistry(providerConfigs),
			}
			var calls atomic.Int32
			transport := contextBlockedDialTransport(&calls)
			t.Cleanup(transport.CloseIdleConnections)
			setProviderTransportForTest(t, handler, "native", transport)

			startedAt := time.Now()
			response, err := handler.nativeResponseRequest(
				context.Background(),
				http.MethodGet,
				responseBinding{Provider: "native"},
				"responses/resp_timeout",
				"",
				nil,
				nil,
			)
			elapsed := time.Since(startedAt)

			if response != nil {
				response.Body.Close()
				t.Fatal("native response unexpectedly succeeded")
			}
			if !testCase.isTimeout(err) {
				t.Fatalf("native response error = %v, want %s timeout", err, testCase.mode)
			}
			if elapsed < shortUpstreamAttemptTimeout/2 || elapsed > time.Second {
				t.Fatalf("request duration = %s, want a short provider timeout", elapsed)
			}
			if calls.Load() != 1 {
				t.Fatalf("dial calls = %d, want 1", calls.Load())
			}
		})
	}
}

func TestProviderAttemptPreservesParentCancellation(t *testing.T) {
	parent, cancel := context.WithCancel(context.Background())
	started := make(chan struct{})
	client := &http.Client{Transport: providerURLRoundTripFunc(func(request *http.Request) (*http.Response, error) {
		close(started)
		<-request.Context().Done()
		return nil, request.Context().Err()
	})}
	request, err := http.NewRequestWithContext(parent, http.MethodGet, "http://provider.invalid/v1/models", nil)
	if err != nil {
		t.Fatalf("create request: %v", err)
	}

	errCh := make(chan error, 1)
	go func() {
		_, requestErr := doProviderRequest(request, providerClientConfig{
			client:  client,
			timeout: time.Second,
			mode:    upstreamTimeoutModeTTFT,
		}, "provider", "failed to call provider")
		errCh <- requestErr
	}()
	<-started
	cancel()

	select {
	case requestErr := <-errCh:
		if !errors.Is(requestErr, context.Canceled) {
			t.Fatalf("request error = %v, want context.Canceled", requestErr)
		}
		if isUpstreamTTFTTimeout(requestErr) || isUpstreamTotalTimeout(requestErr) {
			t.Fatalf("client cancellation was relabeled as provider timeout: %v", requestErr)
		}
	case <-time.After(time.Second):
		t.Fatal("provider request did not observe parent cancellation")
	}
}

func TestProviderAttemptPreservesParentDeadline(t *testing.T) {
	parent, cancel := context.WithTimeout(context.Background(), shortUpstreamAttemptTimeout)
	defer cancel()
	client := &http.Client{Transport: providerURLRoundTripFunc(func(request *http.Request) (*http.Response, error) {
		<-request.Context().Done()
		return nil, request.Context().Err()
	})}
	request, err := http.NewRequestWithContext(parent, http.MethodGet, "http://provider.invalid/v1/models", nil)
	if err != nil {
		t.Fatalf("create request: %v", err)
	}

	_, requestErr := doProviderRequest(request, providerClientConfig{
		client:  client,
		timeout: time.Second,
		mode:    upstreamTimeoutModeTTFT,
	}, "provider", "failed to call provider")
	if !errors.Is(requestErr, context.DeadlineExceeded) {
		t.Fatalf("request error = %v, want context.DeadlineExceeded", requestErr)
	}
	if isUpstreamTTFTTimeout(requestErr) || isUpstreamTotalTimeout(requestErr) {
		t.Fatalf("client deadline was relabeled as provider timeout: %v", requestErr)
	}
}

func contextBlockedDialTransport(calls *atomic.Int32) *http.Transport {
	return &http.Transport{
		DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
			calls.Add(1)
			<-ctx.Done()
			return nil, ctx.Err()
		},
		DisableKeepAlives: true,
	}
}

type closeBlockedReadBody struct {
	closed     chan struct{}
	closeOnce  sync.Once
	closeCalls atomic.Int32
}

func newCloseBlockedReadBody() *closeBlockedReadBody {
	return &closeBlockedReadBody{closed: make(chan struct{})}
}

func (b *closeBlockedReadBody) Read([]byte) (int, error) {
	<-b.closed
	return 0, errors.New("response body closed")
}

func (b *closeBlockedReadBody) Close() error {
	b.closeOnce.Do(func() {
		b.closeCalls.Add(1)
		close(b.closed)
	})
	return nil
}
