package providers

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
)

func TestOpenAITranslator_ParseResponse_ResponsesTerminalStates(t *testing.T) {
	tests := []struct {
		name          string
		body          string
		wantFinish    string
		wantErrorType string
		wantMessage   string
	}{
		{
			name:       "completed",
			body:       `{"id":"resp_completed","object":"response","created_at":1,"status":"completed","model":"gpt-5.4","output":[{"type":"message","content":[{"type":"output_text","text":"done"}]}]}`,
			wantFinish: "stop",
		},
		{
			name:       "completed with function call",
			body:       `{"id":"resp_tool","object":"response","created_at":1,"status":"completed","model":"gpt-5.4","output":[{"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{}"}]}`,
			wantFinish: "tool_calls",
		},
		{
			name:       "incomplete max output tokens",
			body:       `{"id":"resp_incomplete","object":"response","created_at":1,"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"model":"gpt-5.4","output":[]}`,
			wantFinish: "length",
		},
		{
			name:       "incomplete content filter",
			body:       `{"id":"resp_filtered","object":"response","created_at":1,"status":"incomplete","incomplete_details":{"reason":"content_filter"},"model":"gpt-5.4","output":[]}`,
			wantFinish: "content_filter",
		},
		{
			name:          "failed",
			body:          `{"id":"resp_failed","object":"response","created_at":1,"status":"failed","model":"gpt-5.4","output":[],"error":{"code":"server_error","message":"generation failed"}}`,
			wantErrorType: "response_failed",
			wantMessage:   "generation failed",
		},
		{
			name:          "cancelled",
			body:          `{"id":"resp_cancelled","object":"response","created_at":1,"status":"cancelled","model":"gpt-5.4","output":[]}`,
			wantErrorType: "response_cancelled",
			wantMessage:   "OpenAI Responses request was cancelled",
		},
		{
			name:          "unsupported incomplete reason",
			body:          `{"id":"resp_incomplete","object":"response","created_at":1,"status":"incomplete","incomplete_details":{"reason":"future_reason"},"model":"gpt-5.4","output":[]}`,
			wantErrorType: "invalid_response_status",
			wantMessage:   `invalid OpenAI Responses terminal state: unsupported incomplete_details.reason "future_reason"`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			translator := NewOpenAITranslator(config.ProviderConfig{
				APIKey:  "dummy",
				BaseURL: "https://api.openai.com/v1",
			})
			response := &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(strings.NewReader(tt.body)),
			}

			unified, err := translator.ParseResponse(response)
			if tt.wantErrorType != "" {
				assertOpenAIResponsesProviderError(t, err, tt.wantErrorType, tt.wantMessage)
				if unified != nil {
					t.Fatalf("unified response = %#v, want nil", unified)
				}
				return
			}
			if err != nil {
				t.Fatalf("ParseResponse returned error: %v", err)
			}
			if unified == nil || len(unified.Choices) != 1 || unified.Choices[0].FinishReason == nil {
				t.Fatalf("expected one terminal choice, got %#v", unified)
			}
			if got := *unified.Choices[0].FinishReason; got != tt.wantFinish {
				t.Fatalf("finish_reason = %q, want %q", got, tt.wantFinish)
			}
		})
	}
}

func TestOpenAITranslator_ParseResponse_NativeResponsesFailurePreservesEnvelope(t *testing.T) {
	translator := NewOpenAITranslator(config.ProviderConfig{
		APIKey:  "dummy",
		BaseURL: "https://api.openai.com/v1",
	})
	body := `{"id":"resp_failed","object":"response","created_at":1,"status":"failed","model":"gpt-5.4","output":[],"error":{"code":"server_error","message":"generation failed"}}`
	ctx := WithSourceRequestType(
		WithUpstreamRequestType(context.Background(), "responses"),
		"responses",
	)
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.openai.com/v1/responses", nil)
	if err != nil {
		t.Fatalf("create request: %v", err)
	}
	response := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(body)),
		Request:    request,
	}

	unified, err := translator.ParseResponse(response)
	if err != nil {
		t.Fatalf("native Responses failure must remain an intact response envelope: %v", err)
	}
	if unified == nil || string(unified.RawJSON) != body {
		t.Fatalf("raw response = %s, want %s", unified.RawJSON, body)
	}
}

func TestOpenAIStreamTranslator_ResponsesTerminalStates(t *testing.T) {
	tests := []struct {
		name          string
		event         string
		wantFinish    string
		wantErrorType string
		wantMessage   string
	}{
		{
			name:       "completed",
			event:      `{"type":"response.completed","response":{"id":"resp_completed","created_at":1,"status":"completed","model":"gpt-5.4","output":[]}}`,
			wantFinish: "stop",
		},
		{
			name:       "incomplete",
			event:      `{"type":"response.incomplete","response":{"id":"resp_incomplete","created_at":1,"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"model":"gpt-5.4","output":[]}}`,
			wantFinish: "length",
		},
		{
			name:          "failed",
			event:         `{"type":"response.failed","response":{"id":"resp_failed","created_at":1,"status":"failed","model":"gpt-5.4","output":[],"error":{"code":"server_error","message":"generation failed"}}}`,
			wantErrorType: "response_failed",
			wantMessage:   "generation failed",
		},
		{
			name:          "cancelled",
			event:         `{"type":"response.cancelled","response":{"id":"resp_cancelled","created_at":1,"status":"cancelled","model":"gpt-5.4","output":[]}}`,
			wantErrorType: "response_cancelled",
			wantMessage:   "OpenAI Responses request was cancelled",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			translator := newOpenAIStreamTranslatorForTest()
			chunk, err := translator.ParseStreamChunk([]byte(tt.event))
			if tt.wantErrorType != "" {
				assertOpenAIResponsesProviderError(t, err, tt.wantErrorType, tt.wantMessage)
				if chunk != nil {
					t.Fatalf("chunk = %#v, want nil", chunk)
				}
				return
			}
			if !errors.Is(err, ErrStreamDone) {
				t.Fatalf("error = %v, want ErrStreamDone", err)
			}
			if chunk == nil || len(chunk.Choices) != 1 || chunk.Choices[0].FinishReason == nil {
				t.Fatalf("expected one terminal choice, got %#v", chunk)
			}
			if got := *chunk.Choices[0].FinishReason; got != tt.wantFinish {
				t.Fatalf("finish_reason = %q, want %q", got, tt.wantFinish)
			}
		})
	}
}

func assertOpenAIResponsesProviderError(t *testing.T, err error, wantType, wantMessage string) {
	t.Helper()
	var providerErr *ProviderError
	if !errors.As(err, &providerErr) {
		t.Fatalf("error = %v, want ProviderError", err)
	}
	if providerErr.StatusCode != http.StatusBadGateway || providerErr.Provider != "openai" {
		t.Fatalf("provider error metadata = %#v", providerErr)
	}
	if providerErr.Type != wantType {
		t.Fatalf("error type = %q, want %q", providerErr.Type, wantType)
	}
	if providerErr.Message != wantMessage {
		t.Fatalf("error message = %q, want %q", providerErr.Message, wantMessage)
	}
}
