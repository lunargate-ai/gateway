package api

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/internal/streaming"
)

func TestResponsesStreamProxyTranslatesChatProviderErrorToFailed(t *testing.T) {
	const upstreamSecret = "provider diagnostic secret"
	providerResp := &http.Response{
		StatusCode: http.StatusOK,
		Body: io.NopCloser(strings.NewReader(
			"data: {\"error\":{\"message\":\"" + upstreamSecret + "\",\"type\":\"server_error\"}}\n\n",
		)),
	}
	translator := providers.NewOpenAITranslator(config.ProviderConfig{APIKey: "dummy"})
	recorder := httptest.NewRecorder()
	proxy := newResponsesStreamProxy(recorder)

	streamErr := streaming.NewHandler().StreamResponse(context.Background(), proxy, providerResp, translator)
	var providerErr *providers.ProviderError
	if !errors.As(streamErr, &providerErr) {
		t.Fatalf("stream error = %v, want wrapped ProviderError", streamErr)
	}
	if providerErr.Message != upstreamSecret {
		t.Fatalf("observable provider error message = %q, want %q", providerErr.Message, upstreamSecret)
	}
	proxy.RecordStreamError(streamErr)
	if err := proxy.finalize(); err != nil {
		t.Fatalf("finalize: %v", err)
	}

	events := decodeSSEEvents(t, recorder.Body.String())
	foundFailed := false
	for _, event := range events {
		switch event["type"] {
		case "response.completed", "response.incomplete":
			t.Fatalf("provider error emitted successful terminal event %q", event["type"])
		case "response.failed":
			foundFailed = true
			response, _ := event["response"].(map[string]interface{})
			failure, _ := response["error"].(map[string]interface{})
			if failure["code"] != streaming.ChatStreamErrorCode {
				t.Fatalf("failure code = %#v, want %q", failure["code"], streaming.ChatStreamErrorCode)
			}
			if failure["message"] != streaming.ChatStreamErrorMessage {
				t.Fatalf("failure message = %#v, want %q", failure["message"], streaming.ChatStreamErrorMessage)
			}
		}
	}
	if !foundFailed {
		t.Fatal("expected response.failed event")
	}
	if strings.Contains(recorder.Body.String(), upstreamSecret) {
		t.Fatalf("provider diagnostic leaked to Responses client: %s", recorder.Body.String())
	}
}

func TestResponsesStreamProxyDoesNotTreatFalseErrorExtensionAsFailure(t *testing.T) {
	providerResp := &http.Response{
		StatusCode: http.StatusOK,
		Body: io.NopCloser(strings.NewReader(
			"data: {\"id\":\"chatcmpl_1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt\",\"choices\":[],\"error\":false}\n\n" +
				"data: [DONE]\n\n",
		)),
	}
	translator := providers.NewOpenAITranslator(config.ProviderConfig{APIKey: "dummy"})
	recorder := httptest.NewRecorder()
	proxy := newResponsesStreamProxy(recorder)

	if err := streaming.NewHandler().StreamResponse(context.Background(), proxy, providerResp, translator); err != nil {
		t.Fatalf("StreamResponse: %v", err)
	}
	if err := proxy.finalize(); err != nil {
		t.Fatalf("finalize: %v", err)
	}

	events := decodeSSEEvents(t, recorder.Body.String())
	foundCompleted := false
	for _, event := range events {
		switch event["type"] {
		case "response.failed":
			t.Fatalf("false error extension emitted response.failed: %#v", event)
		case "response.completed":
			foundCompleted = true
		}
	}
	if !foundCompleted {
		t.Fatalf("events = %#v, want response.completed", events)
	}
}

func TestChatStreamErrorPayloadRequiresStructuredError(t *testing.T) {
	tests := []struct {
		name    string
		payload string
		want    bool
	}{
		{name: "null", payload: `{"error":null}`, want: false},
		{name: "false", payload: `{"error":false}`, want: false},
		{name: "number", payload: `{"error":0}`, want: false},
		{name: "array", payload: `{"error":[]}`, want: false},
		{name: "object", payload: `{"error":{"message":"failed"}}`, want: true},
		{name: "string", payload: `{"error":"failed"}`, want: true},
		{name: "explicit type", payload: `{"type":"error","error":false}`, want: true},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := isChatStreamErrorPayload([]byte(test.payload)); got != test.want {
				t.Fatalf("isChatStreamErrorPayload() = %t, want %t", got, test.want)
			}
		})
	}
}
