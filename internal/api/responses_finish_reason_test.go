package api

import (
	"errors"
	"fmt"
	"net/http/httptest"
	"testing"

	"github.com/lunargate-ai/gateway/internal/streaming"
	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestResponsesStreamProxy_MapsIncompleteFinishReasons(t *testing.T) {
	tests := []struct {
		name         string
		finishReason string
		wantReason   string
	}{
		{
			name:         "length",
			finishReason: "length",
			wantReason:   models.ResponsesIncompleteReasonMaxOutputTokens,
		},
		{
			name:         "content filter",
			finishReason: "content_filter",
			wantReason:   models.ResponsesIncompleteReasonContentFilter,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rec := httptest.NewRecorder()
			proxy := newResponsesStreamProxy(rec)

			partial := `data: {"id":"resp_incomplete","object":"chat.completion.chunk","created":123,"model":"mock-gpt","choices":[{"index":0,"delta":{"content":"partial answer"},"finish_reason":null}]}` + "\n\n"
			if _, err := proxy.Write([]byte(partial)); err != nil {
				t.Fatalf("write partial chunk: %v", err)
			}
			terminal := fmt.Sprintf(`data: {"id":"resp_incomplete","object":"chat.completion.chunk","created":123,"model":"mock-gpt","choices":[{"index":0,"finish_reason":%q}],"usage":{"prompt_tokens":7,"completion_tokens":5,"total_tokens":12}}`, tc.finishReason) + "\n\n"
			if _, err := proxy.Write([]byte(terminal)); err != nil {
				t.Fatalf("write terminal chunk: %v", err)
			}
			if _, err := proxy.Write([]byte("data: [DONE]\n\n")); err != nil {
				t.Fatalf("write done frame: %v", err)
			}
			if err := proxy.finalize(); err != nil {
				t.Fatalf("finalize: %v", err)
			}

			events := decodeSSEEvents(t, rec.Body.String())
			assertSequenceNumbersMonotonic(t, events)
			var incompleteResponse map[string]interface{}
			for _, event := range events {
				switch event["type"] {
				case "response.completed":
					t.Fatal("incomplete finish must not emit response.completed")
				case "response.failed":
					t.Fatal("valid incomplete finish must not emit response.failed")
				case "response.incomplete":
					incompleteResponse, _ = event["response"].(map[string]interface{})
				case "response.output_item.done":
					item, _ := event["item"].(map[string]interface{})
					if item != nil && item["status"] != "incomplete" {
						t.Errorf("terminal output item status = %#v, want incomplete", item["status"])
					}
				}
			}
			if incompleteResponse == nil {
				t.Fatal("expected response.incomplete event")
			}
			if incompleteResponse["status"] != "incomplete" {
				t.Fatalf("response status = %#v, want incomplete", incompleteResponse["status"])
			}
			details, _ := incompleteResponse["incomplete_details"].(map[string]interface{})
			if details == nil || details["reason"] != tc.wantReason {
				t.Fatalf("incomplete_details = %#v, want reason %q", details, tc.wantReason)
			}
			if incompleteResponse["output_text"] != "partial answer" {
				t.Fatalf("output_text = %#v, want partial answer", incompleteResponse["output_text"])
			}
			usage, _ := incompleteResponse["usage"].(map[string]interface{})
			if usage == nil || usage["input_tokens"] != float64(7) || usage["output_tokens"] != float64(5) || usage["total_tokens"] != float64(12) {
				t.Fatalf("usage = %#v, want 7/5/12", usage)
			}
			if proxy.completedResponse != nil {
				t.Fatalf("incomplete stream must not be cached as completed: %#v", proxy.completedResponse)
			}
		})
	}
}

func TestResponsesStreamProxy_IncompleteFinishDoesNotMaskTruncatedStream(t *testing.T) {
	tests := []struct {
		name        string
		streamError error
	}{
		{
			name: "missing done frame",
		},
		{
			name:        "reported transport error",
			streamError: errors.New("upstream socket reset"),
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rec := httptest.NewRecorder()
			proxy := newResponsesStreamProxy(rec)
			chunk := `data: {"id":"resp_truncated","object":"chat.completion.chunk","created":123,"model":"mock-gpt","choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":"length"}]}` + "\n\n"
			if _, err := proxy.Write([]byte(chunk)); err != nil {
				t.Fatalf("write partial chunk: %v", err)
			}
			if tc.streamError != nil {
				proxy.RecordStreamError(tc.streamError)
			}
			if err := proxy.finalize(); err != nil {
				t.Fatalf("finalize: %v", err)
			}

			events := decodeSSEEvents(t, rec.Body.String())
			var failedResponse map[string]interface{}
			for _, event := range events {
				switch event["type"] {
				case "response.completed", "response.incomplete":
					t.Fatalf("truncated stream emitted successful terminal event %q", event["type"])
				case "response.failed":
					failedResponse, _ = event["response"].(map[string]interface{})
				}
			}
			if failedResponse == nil {
				t.Fatal("expected response.failed event")
			}
			failure, _ := failedResponse["error"].(map[string]interface{})
			message, _ := failure["message"].(string)
			if message != streaming.ChatStreamErrorMessage {
				t.Fatalf("failure message = %q, want %q", message, streaming.ChatStreamErrorMessage)
			}
		})
	}
}
