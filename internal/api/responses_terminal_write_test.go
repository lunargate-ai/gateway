package api

import (
	"bytes"
	"errors"
	"net/http"
	"strings"
	"testing"
)

var errResponsesTerminalWrite = errors.New("terminal write failed")

type responsesTerminalFailWriter struct {
	header http.Header
	body   bytes.Buffer
}

func (w *responsesTerminalFailWriter) Header() http.Header {
	return w.header
}

func (w *responsesTerminalFailWriter) WriteHeader(int) {}

func (w *responsesTerminalFailWriter) Write(payload []byte) (int, error) {
	if bytes.Contains(payload, []byte("event: response.completed")) {
		return 0, errResponsesTerminalWrite
	}
	return w.body.Write(payload)
}

func (w *responsesTerminalFailWriter) FlushError() error {
	return nil
}

func TestResponsesStreamProxyCommitsTerminalStateOnlyAfterSuccessfulWrite(t *testing.T) {
	writer := &responsesTerminalFailWriter{header: make(http.Header)}
	proxy := newResponsesStreamProxy(writer)
	chunk := "data: {\"id\":\"chatcmpl_terminal_write\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"answer\"}}]}\n\n"
	if _, err := proxy.Write([]byte(chunk)); err != nil {
		t.Fatalf("content write: %v", err)
	}
	_, terminalErr := proxy.Write([]byte("data: [DONE]\n\n"))
	if !errors.Is(terminalErr, errResponsesTerminalWrite) {
		t.Fatalf("terminal write error = %v, want writer failure", terminalErr)
	}
	if proxy.completed || proxy.terminalResponse != nil || proxy.completedResponse != nil {
		t.Fatalf("failed terminal write committed state: completed=%t terminal=%#v completed_response=%#v", proxy.completed, proxy.terminalResponse, proxy.completedResponse)
	}

	proxy.RecordStreamError(terminalErr)
	if err := proxy.finalize(); err != nil {
		t.Fatalf("finalize after terminal write failure: %v", err)
	}
	if !proxy.completed || proxy.terminalResponse == nil || proxy.terminalResponse["status"] != "failed" {
		t.Fatalf("recovery terminal state = completed:%t response:%#v", proxy.completed, proxy.terminalResponse)
	}
	if proxy.completedResponse != nil {
		t.Fatalf("failed response was retained as completed: %#v", proxy.completedResponse)
	}
	body := writer.body.String()
	if strings.Contains(body, "event: response.completed") || !strings.Contains(body, "event: response.failed") {
		t.Fatalf("terminal wire events after recovery: %q", body)
	}
}
