package remotecontrol

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"
)

func TestBoundedWebSocketWriterSetsDeadlineForEveryWrite(t *testing.T) {
	connection := &recordingWebSocketWriteConnection{}
	client := &Client{websocketWriteTimeout: time.Second}
	writeJSON := client.boundedWebSocketWriter(connection)

	before := time.Now()
	if err := writeJSON(map[string]string{"type": "first"}); err != nil {
		t.Fatalf("first write returned error: %v", err)
	}
	if err := writeJSON(map[string]string{"type": "second"}); err != nil {
		t.Fatalf("second write returned error: %v", err)
	}
	after := time.Now()

	connection.mu.Lock()
	deadlines := append([]time.Time(nil), connection.deadlines...)
	writes := connection.writes
	connection.mu.Unlock()
	if writes != 2 || len(deadlines) != 2 {
		t.Fatalf("writes = %d, deadlines = %d; want two of each", writes, len(deadlines))
	}
	for index, deadline := range deadlines {
		if deadline.Before(before.Add(time.Second)) || deadline.After(after.Add(time.Second)) {
			t.Fatalf("deadline[%d] = %s, want one second after its write", index, deadline)
		}
	}
}

func TestContextCancellationClosesConnectionWithoutWaitingForWriter(t *testing.T) {
	connection := newBlockingWebSocketWriteConnection()
	client := &Client{websocketWriteTimeout: time.Minute}
	writeJSON := client.boundedWebSocketWriter(connection)
	writeResult := make(chan error, 1)
	go func() {
		writeResult <- writeJSON(map[string]string{"type": "blocked"})
	}()

	select {
	case <-connection.writeStarted:
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for blocked WebSocket write")
	}

	ctx, cancel := context.WithCancel(context.Background())
	stop := make(chan struct{})
	closerDone := make(chan struct{})
	go func() {
		closeConnectionOnContext(ctx, connection, stop)
		close(closerDone)
	}()
	cancel()

	select {
	case <-closerDone:
	case <-time.After(time.Second):
		t.Fatal("context cancellation waited for the WebSocket writer")
	}
	select {
	case err := <-writeResult:
		if !errors.Is(err, errTestConnectionClosed) {
			t.Fatalf("blocked write error = %v, want closed connection", err)
		}
	case <-time.After(time.Second):
		t.Fatal("closing the connection did not unblock the WebSocket write")
	}
}

type recordingWebSocketWriteConnection struct {
	mu        sync.Mutex
	deadlines []time.Time
	writes    int
}

func (c *recordingWebSocketWriteConnection) SetWriteDeadline(deadline time.Time) error {
	c.mu.Lock()
	c.deadlines = append(c.deadlines, deadline)
	c.mu.Unlock()
	return nil
}

func (c *recordingWebSocketWriteConnection) WriteJSON(interface{}) error {
	c.mu.Lock()
	c.writes++
	c.mu.Unlock()
	return nil
}

var errTestConnectionClosed = errors.New("test connection closed")

type blockingWebSocketWriteConnection struct {
	writeStarted chan struct{}
	closed       chan struct{}
	startOnce    sync.Once
	closeOnce    sync.Once
}

func newBlockingWebSocketWriteConnection() *blockingWebSocketWriteConnection {
	return &blockingWebSocketWriteConnection{
		writeStarted: make(chan struct{}),
		closed:       make(chan struct{}),
	}
}

func (c *blockingWebSocketWriteConnection) SetWriteDeadline(time.Time) error {
	return nil
}

func (c *blockingWebSocketWriteConnection) WriteJSON(interface{}) error {
	c.startOnce.Do(func() { close(c.writeStarted) })
	<-c.closed
	return errTestConnectionClosed
}

func (c *blockingWebSocketWriteConnection) Close() error {
	c.closeOnce.Do(func() { close(c.closed) })
	return nil
}
