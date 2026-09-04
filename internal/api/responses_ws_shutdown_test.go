package api

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
)

func TestResponsesWebSocketRegistryCloseSendsGoingAwayAndCancels(t *testing.T) {
	registry := &responsesWebSocketRegistry{}
	attached := make(chan struct{})
	handlerDone := make(chan struct{})
	requestCanceled := make(chan struct{})

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer close(handlerDone)
		connectionCtx, cancel := context.WithCancel(r.Context())
		registration, ok := registry.register(cancel)
		if !ok {
			http.Error(w, errResponsesWebSocketShuttingDown.Error(), http.StatusServiceUnavailable)
			return
		}
		defer registry.unregister(registration)

		conn, err := responsesWebSocketUpgrader.Upgrade(w, r.Clone(connectionCtx), nil)
		if err != nil {
			return
		}
		if !registry.attach(registration, conn) {
			return
		}
		defer conn.Close()
		close(attached)
		<-connectionCtx.Done()
		close(requestCanceled)
	}))
	defer server.Close()

	conn := mustDialResponsesWebSocket(t, server.URL)
	defer conn.Close()
	select {
	case <-attached:
	case <-time.After(2 * time.Second):
		t.Fatal("websocket was not registered")
	}

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := registry.close(shutdownCtx); err != nil {
		t.Fatalf("close responses websocket registry: %v", err)
	}

	select {
	case <-requestCanceled:
	case <-time.After(time.Second):
		t.Fatal("connection context was not canceled")
	}
	select {
	case <-handlerDone:
	case <-time.After(time.Second):
		t.Fatal("websocket handler did not return")
	}

	_, _, readErr := conn.ReadMessage()
	var closeErr *websocket.CloseError
	if !errors.As(readErr, &closeErr) {
		t.Fatalf("client read error = %v, want websocket close error", readErr)
	}
	if closeErr.Code != websocket.CloseGoingAway {
		t.Fatalf("close code = %d, want %d", closeErr.Code, websocket.CloseGoingAway)
	}
	if !strings.Contains(closeErr.Text, responsesWebSocketShutdownReason) {
		t.Fatalf("close reason = %q", closeErr.Text)
	}
}

func TestResponsesWebSocketRegistryCloseWaitIsBounded(t *testing.T) {
	registry := &responsesWebSocketRegistry{}
	registrationCtx, cancelRegistration := context.WithCancel(context.Background())
	registration, ok := registry.register(cancelRegistration)
	if !ok {
		t.Fatal("register returned closing")
	}
	defer registry.unregister(registration)

	waitCtx, cancelWait := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancelWait()
	started := time.Now()
	err := registry.close(waitCtx)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("close error = %v, want context deadline exceeded", err)
	}
	if elapsed := time.Since(started); elapsed > time.Second {
		t.Fatalf("bounded close took %s", elapsed)
	}
	select {
	case <-registrationCtx.Done():
	default:
		t.Fatal("bounded close did not cancel registration")
	}
}

func TestHandlerCloseResponsesWebSocketsCancelsActiveUpstream(t *testing.T) {
	upstreamStarted := make(chan struct{})
	upstreamCanceled := make(chan struct{})
	releaseUpstream := make(chan struct{})
	defer close(releaseUpstream)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
		close(upstreamStarted)
		select {
		case <-r.Context().Done():
			close(upstreamCanceled)
		case <-releaseUpstream:
		}
	}))
	defer upstream.Close()

	handler := newResponsesWebSocketTestHandlerWithUpstreamType(upstream.URL, requestTypeResponses)
	handlerDone := make(chan struct{})
	gateway := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		handler.ResponsesWebSocket(w, r)
		close(handlerDone)
	}))
	defer gateway.Close()

	conn := mustDialResponsesWebSocket(t, gateway.URL)
	defer conn.Close()
	sendResponsesWebSocketJSON(t, conn, map[string]interface{}{
		"type":   "response.create",
		"model":  "mock-gpt",
		"input":  "wait for shutdown",
		"stream": true,
	})
	select {
	case <-upstreamStarted:
	case <-time.After(2 * time.Second):
		t.Fatal("upstream streaming request did not start")
	}

	shutdownCtx, cancelShutdown := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancelShutdown()
	if err := handler.CloseResponsesWebSockets(shutdownCtx); err != nil {
		t.Fatalf("close responses websockets: %v", err)
	}
	if err := gateway.Config.Shutdown(shutdownCtx); err != nil {
		t.Fatalf("shutdown HTTP server: %v", err)
	}

	select {
	case <-upstreamCanceled:
	case <-time.After(time.Second):
		t.Fatal("active upstream request context was not canceled")
	}
	select {
	case <-handlerDone:
	case <-time.After(time.Second):
		t.Fatal("upgraded websocket handler remained after HTTP shutdown")
	}

	_, _, readErr := conn.ReadMessage()
	var closeErr *websocket.CloseError
	if !errors.As(readErr, &closeErr) {
		t.Fatalf("client read error = %v, want websocket close error", readErr)
	}
	if closeErr.Code != websocket.CloseGoingAway {
		t.Fatalf("close code = %d, want %d", closeErr.Code, websocket.CloseGoingAway)
	}
}

func TestHandlerRejectsResponsesWebSocketUpgradeAfterShutdown(t *testing.T) {
	handler := &Handler{}
	shutdownCtx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := handler.CloseResponsesWebSockets(shutdownCtx); err != nil {
		t.Fatalf("close empty responses websocket registry: %v", err)
	}

	server := httptest.NewServer(http.HandlerFunc(handler.ResponsesWebSocket))
	defer server.Close()
	wsURL := "ws" + strings.TrimPrefix(server.URL, "http") + "/v1/responses"
	conn, response, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if conn != nil {
		_ = conn.Close()
		t.Fatal("websocket upgrade succeeded after shutdown")
	}
	if err == nil {
		t.Fatal("websocket dial unexpectedly succeeded")
	}
	if response == nil {
		t.Fatal("websocket dial returned no HTTP response")
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("upgrade status = %d, want %d", response.StatusCode, http.StatusServiceUnavailable)
	}
}
