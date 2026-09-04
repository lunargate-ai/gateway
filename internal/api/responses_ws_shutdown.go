package api

import (
	"context"
	"errors"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

const (
	responsesWebSocketShutdownTimeout = 5 * time.Second
	responsesWebSocketShutdownReason  = "server shutting down"
)

var errResponsesWebSocketShuttingDown = errors.New("responses websocket service is shutting down")

// responsesWebSocketRegistry owns every upgraded Responses connection. HTTP
// server shutdown does not wait for hijacked connections, so they need an
// explicit lifecycle tied to the API handler.
type responsesWebSocketRegistry struct {
	mu               sync.Mutex
	registrations    map[*responsesWebSocketRegistration]struct{}
	idle             chan struct{}
	closing          bool
	shutdownDeadline time.Time
}

type responsesWebSocketRegistration struct {
	cancel      context.CancelFunc
	conn        *websocket.Conn
	closing     bool
	closeIssued bool
}

func (r *responsesWebSocketRegistry) register(cancel context.CancelFunc) (*responsesWebSocketRegistration, bool) {
	if r == nil || cancel == nil {
		return nil, false
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closing {
		return nil, false
	}
	if len(r.registrations) == 0 {
		r.registrations = make(map[*responsesWebSocketRegistration]struct{})
		r.idle = make(chan struct{})
	}
	registration := &responsesWebSocketRegistration{cancel: cancel}
	r.registrations[registration] = struct{}{}
	return registration, true
}

func (r *responsesWebSocketRegistry) attach(
	registration *responsesWebSocketRegistration,
	conn *websocket.Conn,
) bool {
	if r == nil || registration == nil || conn == nil {
		return false
	}

	r.mu.Lock()
	if _, ok := r.registrations[registration]; !ok {
		r.mu.Unlock()
		notifyResponsesWebSocketGoingAway(conn, time.Now())
		_ = conn.Close()
		return false
	}
	registration.conn = conn
	closing := r.closing || registration.closing
	deadline := r.shutdownDeadline
	issueClose := closing && !registration.closeIssued
	if issueClose {
		registration.closing = true
		registration.closeIssued = true
	}
	r.mu.Unlock()

	if closing {
		if issueClose {
			notifyResponsesWebSocketGoingAway(conn, responsesWebSocketCloseDeadline(deadline))
		}
		registration.cancel()
		_ = conn.Close()
		return false
	}
	return true
}

func (r *responsesWebSocketRegistry) unregister(registration *responsesWebSocketRegistration) {
	if r == nil || registration == nil {
		return
	}
	registration.cancel()

	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.registrations[registration]; !ok {
		return
	}
	delete(r.registrations, registration)
	if len(r.registrations) == 0 && r.idle != nil {
		close(r.idle)
		r.idle = nil
	}
}

func (r *responsesWebSocketRegistry) close(ctx context.Context) error {
	if r == nil {
		return nil
	}
	if ctx == nil {
		ctx = context.Background()
	}
	waitCtx, cancel := context.WithTimeout(ctx, responsesWebSocketShutdownTimeout)
	defer cancel()
	deadline, _ := waitCtx.Deadline()

	r.mu.Lock()
	r.closing = true
	if r.shutdownDeadline.IsZero() || deadline.Before(r.shutdownDeadline) {
		r.shutdownDeadline = deadline
	}
	deadline = r.shutdownDeadline
	idle := r.idle
	cancels := make([]context.CancelFunc, 0, len(r.registrations))
	connections := make([]*websocket.Conn, 0, len(r.registrations))
	for registration := range r.registrations {
		registration.closing = true
		cancels = append(cancels, registration.cancel)
		if registration.conn != nil && !registration.closeIssued {
			registration.closeIssued = true
			connections = append(connections, registration.conn)
		}
	}
	r.mu.Unlock()

	// Send control frames before cancellation can make a WebSocket handler
	// return, but cap the entire notification phase to a shared short deadline.
	closeDeadline := responsesWebSocketCloseDeadline(deadline)
	for _, conn := range connections {
		notifyResponsesWebSocketGoingAway(conn, closeDeadline)
	}
	// Cancel every in-flight provider request before closing the transports so
	// provider work stops even when a peer does not process the close frame.
	for _, cancelRegistration := range cancels {
		cancelRegistration()
	}
	for _, conn := range connections {
		_ = conn.Close()
	}

	if idle == nil {
		return nil
	}
	select {
	case <-idle:
		return nil
	case <-waitCtx.Done():
		return waitCtx.Err()
	}
}

func responsesWebSocketCloseDeadline(shutdownDeadline time.Time) time.Time {
	deadline := time.Now().Add(time.Second)
	if !shutdownDeadline.IsZero() && shutdownDeadline.Before(deadline) {
		return shutdownDeadline
	}
	return deadline
}

func notifyResponsesWebSocketGoingAway(conn *websocket.Conn, deadline time.Time) {
	if conn == nil {
		return
	}
	if deadline.IsZero() {
		deadline = responsesWebSocketCloseDeadline(time.Time{})
	}
	_ = conn.WriteControl(
		websocket.CloseMessage,
		websocket.FormatCloseMessage(websocket.CloseGoingAway, responsesWebSocketShutdownReason),
		deadline,
	)
}

// CloseResponsesWebSockets gracefully terminates hijacked Responses
// WebSockets and waits, for a bounded period, until their handlers return.
func (h *Handler) CloseResponsesWebSockets(ctx context.Context) error {
	if h == nil {
		return nil
	}
	return h.responsesWebSocketRegistryRef().close(ctx)
}
