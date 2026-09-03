package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestResponsesNativeNonStreamPreservesRawSuccessfulEnvelope(t *testing.T) {
	testCases := []struct {
		name       string
		statusCode int
		status     string
	}{
		{name: "queued", statusCode: http.StatusAccepted, status: "queued"},
		{name: "incomplete", statusCode: http.StatusPartialContent, status: "incomplete"},
		{name: "failed", statusCode: http.StatusNonAuthoritativeInfo, status: "failed"},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			body := fmt.Sprintf(`{
  "id": "resp_native_%s",
  "object": "response",
  "created_at": 1788372000,
  "status": %q,
  "model": "gpt-5.4",
  "output": [
    {
      "type": "hosted_tool_call",
      "id": "tool_123",
      "status": "completed",
      "server_label": "deep-research",
      "result": {"citations": [{"url": "https://example.test/source"}]},
      "future_item_field": {"kept": true}
    },
    {
      "type": "message",
      "id": "msg_123",
      "status": %q,
      "role": "assistant",
      "content": [{
        "type": "output_text",
        "text": "native response",
        "annotations": [{"type": "url_citation", "url": "https://example.test/source"}],
        "future_content_field": [1, 2, 3]
      }]
    }
  ],
  "output_text": "native response",
  "usage": {
    "input_tokens": 7,
    "output_tokens": 5,
    "total_tokens": 12,
    "input_tokens_details": {"cached_tokens": 3},
    "output_tokens_details": {"reasoning_tokens": 2},
    "future_usage_field": {"kept": true}
  },
  "future_top_level": {"nested": ["kept", 9007199254740993]}
}
`, testCase.name, testCase.status, testCase.status)

			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/v1/responses" {
					t.Fatalf("upstream path = %q, want /v1/responses", r.URL.Path)
				}
				w.Header().Set("Content-Type", "application/json; charset=utf-8")
				w.Header().Set("X-OpenAI-Request-ID", "upstream-request-id")
				w.Header().Set("X-RateLimit-Remaining-Requests", "42")
				w.Header().Set("Set-Cookie", "session=secret; HttpOnly")
				w.Header().Set("Set-Cookie2", "legacy=secret")
				w.Header().Set("Connection", "X-Upstream-Hop")
				w.Header().Set("X-Upstream-Hop", "secret")
				w.Header().Set("Proxy-Connection", "keep-alive")
				w.Header().Set("X-LunarGate-Provider", "spoofed")
				w.WriteHeader(testCase.statusCode)
				_, _ = w.Write([]byte(body))
			}))
			defer upstream.Close()

			handler, cache := newNativeContinuationTestHandler(t, upstream.URL+"/v1", requestTypeResponses)
			defer cache.Stop()
			recorder := httptest.NewRecorder()
			handler.Responses(recorder, httptest.NewRequest(
				http.MethodPost,
				"/v1/responses",
				strings.NewReader(`{"model":"gpt-5.4","input":"hello","store":false}`),
			))

			if recorder.Code != testCase.statusCode {
				t.Fatalf("status = %d, want %d; body=%s", recorder.Code, testCase.statusCode, recorder.Body.String())
			}
			if got := recorder.Body.String(); got != body {
				t.Fatalf("raw response changed\n got: %s\nwant: %s", got, body)
			}
			if got := recorder.Header().Get("X-OpenAI-Request-ID"); got != "upstream-request-id" {
				t.Fatalf("safe request ID header = %q", got)
			}
			if got := recorder.Header().Get("X-RateLimit-Remaining-Requests"); got != "42" {
				t.Fatalf("safe rate-limit header = %q", got)
			}
			if got := recorder.Header().Get("X-LunarGate-Provider"); got != "openai" {
				t.Fatalf("gateway provider header was overwritten: %q", got)
			}
			for _, key := range []string{
				"Set-Cookie",
				"Set-Cookie2",
				"Connection",
				"X-Upstream-Hop",
				"Proxy-Connection",
			} {
				if got := recorder.Header().Values(key); len(got) != 0 {
					t.Fatalf("unsafe header %s leaked: %q", key, got)
				}
			}
		})
	}
}

func TestChatCompletionsToResponsesStillReturnsChatEnvelope(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"resp_for_chat",
			"object":"response",
			"created_at":1788372000,
			"status":"completed",
			"model":"gpt-5.4",
			"output":[{"type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"chat response"}]}],
			"output_text":"chat response",
			"usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3},
			"future_top_level":{"native_only":true}
		}`))
	}))
	defer upstream.Close()

	handler, cache := newNativeContinuationTestHandler(t, upstream.URL+"/v1", requestTypeResponses)
	defer cache.Stop()
	recorder := httptest.NewRecorder()
	handler.ChatCompletions(recorder, httptest.NewRequest(
		http.MethodPost,
		"/v1/responses",
		strings.NewReader(`{"model":"gpt-5.4","messages":[{"role":"user","content":"hello"}]}`),
	))

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", recorder.Code, recorder.Body.String())
	}
	var response map[string]interface{}
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode chat response: %v", err)
	}
	if got := response["object"]; got != "chat.completion" {
		t.Fatalf("object = %#v, want chat.completion", got)
	}
	if _, leaked := response["future_top_level"]; leaked {
		t.Fatalf("native Responses envelope leaked into Chat Completions: %s", recorder.Body.String())
	}
}

func TestCopyHeadersBlocksConnectionNominatedAndCookieHeaders(t *testing.T) {
	src := http.Header{
		"Connection":       {"X-First-Hop, X-Second-Hop"},
		"X-First-Hop":      {"secret-one"},
		"X-Second-Hop":     {"secret-two"},
		"Proxy-Connection": {"keep-alive"},
		"Set-Cookie":       {"session=secret"},
		"Set-Cookie2":      {"legacy=secret"},
		"X-Safe":           {"one", "two"},
		"x-existing":       {"spoofed"},
	}
	dst := http.Header{"X-Existing": {"gateway"}}

	copyHeaders(dst, src)

	if got := dst.Values("X-Safe"); len(got) != 2 || got[0] != "one" || got[1] != "two" {
		t.Fatalf("safe header = %q, want [one two]", got)
	}
	if got := dst.Values("X-Existing"); len(got) != 1 || got[0] != "gateway" {
		t.Fatalf("existing destination header was overwritten: %q", got)
	}
	for _, key := range []string{
		"Connection",
		"X-First-Hop",
		"X-Second-Hop",
		"Proxy-Connection",
		"Set-Cookie",
		"Set-Cookie2",
	} {
		if got := dst.Values(key); len(got) != 0 {
			t.Fatalf("unsafe header %s copied: %q", key, got)
		}
	}
}
