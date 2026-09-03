package api

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/lunargate-ai/gateway/internal/streaming"
)

func TestResponsesGeneratedSSERecordSizeMatchesTransportBoundary(t *testing.T) {
	eventType := "response.completed"
	payloadSize := streaming.MaxStreamRecordBytes - responsesGeneratedSSERecordSize(eventType, nil)
	payload := make([]byte, payloadSize)
	if got := responsesGeneratedSSERecordSize(eventType, payload); got != streaming.MaxStreamRecordBytes {
		t.Fatalf("record size = %d, want %d", got, streaming.MaxStreamRecordBytes)
	}
	if got := responsesGeneratedSSERecordSize(eventType, append(payload, 0)); got != streaming.MaxStreamRecordBytes+1 {
		t.Fatalf("oversized record = %d, want %d", got, streaming.MaxStreamRecordBytes+1)
	}
}

func TestResponsesWebSocketTranslatedStreamFailsClosedBeforeOversizedTerminal(t *testing.T) {
	const (
		chunkCount = 240
		chunkSize  = 18 << 10
	)
	chunk := strings.Repeat("x", chunkSize)
	encodedChunk, err := json.Marshal(chunk)
	if err != nil {
		t.Fatalf("encode chunk: %v", err)
	}

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for range chunkCount {
			_, _ = fmt.Fprintf(w, "data: {\"id\":\"chatcmpl_large\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"mock-gpt\",\"choices\":[{\"index\":0,\"delta\":{\"content\":%s}}]}\n\n", encodedChunk)
		}
		_, _ = io.WriteString(w, "data: {\"id\":\"chatcmpl_large\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"mock-gpt\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()

	handler := newResponsesWebSocketTestHandler(upstream.URL)
	defer handler.cache.Stop()
	server := httptest.NewServer(http.HandlerFunc(handler.ResponsesWebSocket))
	defer server.Close()
	conn := mustDialResponsesWebSocket(t, server.URL)
	defer conn.Close()

	sendResponsesWebSocketJSON(t, conn, map[string]interface{}{
		"type":  "response.create",
		"model": "lunargate/auto",
		"input": "large response",
	})
	receivedTextBytes := 0
	terminalType := ""
	for range chunkCount + 16 {
		event := readResponsesWebSocketEvent(t, conn)
		eventType, _ := event["type"].(string)
		switch eventType {
		case "response.output_text.delta":
			delta, _ := event["delta"].(string)
			receivedTextBytes += len(delta)
		case "response.completed", "response.incomplete", "response.failed", "error":
			terminalType = eventType
		}
		if terminalType != "" {
			break
		}
	}

	if receivedTextBytes != chunkCount*chunkSize {
		t.Fatalf("received text bytes = %d, want %d", receivedTextBytes, chunkCount*chunkSize)
	}
	if terminalType != "response.failed" {
		t.Fatalf("terminal event = %q, want response.failed", terminalType)
	}
}
