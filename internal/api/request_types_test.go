package api

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/routing"
)

func TestChatAPIRequestTypesSeparatesClientAndUpstream(t *testing.T) {
	tests := []struct {
		name         string
		client       string
		target       routing.Target
		wantClient   string
		wantUpstream string
	}{
		{
			name:         "chat client through responses upstream",
			client:       "chat_completions",
			target:       routing.Target{UpstreamRequestType: "responses"},
			wantClient:   "chat_completions",
			wantUpstream: "responses",
		},
		{
			name:         "responses client through translated chat upstream",
			client:       "responses",
			target:       routing.Target{},
			wantClient:   "responses",
			wantUpstream: "chat_completions",
		},
		{
			name:         "legacy chat name is canonicalized",
			client:       " CHAT ",
			target:       routing.Target{},
			wantClient:   "chat_completions",
			wantUpstream: "chat_completions",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := chatAPIRequestTypes(tt.client, tt.target)
			if got.client != tt.wantClient || got.upstream != tt.wantUpstream {
				t.Fatalf("request types = %#v, want client=%q upstream=%q", got, tt.wantClient, tt.wantUpstream)
			}
			base := map[string]string{"x-lunargate-request-type": "stale", "custom": "kept"}
			tags := got.tags(base)
			if tags["x-lunargate-request-type"] != tt.wantClient {
				t.Fatalf("client request type tag = %q, want %q", tags["x-lunargate-request-type"], tt.wantClient)
			}
			if tags["x-lunargate-upstream-request-type"] != tt.wantUpstream {
				t.Fatalf("upstream request type tag = %q, want %q", tags["x-lunargate-upstream-request-type"], tt.wantUpstream)
			}
			if base["x-lunargate-request-type"] != "stale" {
				t.Fatalf("base tags were mutated: %#v", base)
			}
		})
	}
}

func TestCollectorSeparatesChatRequestTypesAcrossOutcomes(t *testing.T) {
	tests := []struct {
		name       string
		stream     bool
		statusCode int
		response   string
	}{
		{
			name:       "success",
			statusCode: http.StatusOK,
			response:   `{"id":"resp_success","object":"response","created_at":1,"status":"completed","model":"gpt-observed","output":[],"output_text":"ok","usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`,
		},
		{
			name:       "provider error",
			statusCode: http.StatusInternalServerError,
			response:   `{"error":{"message":"upstream failed","type":"server_error"}}`,
		},
		{
			name:       "stream success",
			stream:     true,
			statusCode: http.StatusOK,
			response: strings.Join([]string{
				`data: {"type":"response.created","response":{"id":"resp_stream","object":"response","created_at":1,"status":"in_progress","model":"gpt-observed","output":[]}}`,
				`data: {"type":"response.output_text.delta","response_id":"resp_stream","item_id":"msg_stream","output_index":0,"content_index":0,"delta":"ok"}`,
				`data: {"type":"response.completed","response":{"id":"resp_stream","object":"response","created_at":1,"status":"completed","model":"gpt-observed","output":[],"output_text":"ok","usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}`,
				"",
			}, "\n\n"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/v1/responses" {
					t.Errorf("upstream path = %q, want /v1/responses", r.URL.Path)
				}
				if tt.stream {
					w.Header().Set("Content-Type", "text/event-stream")
				} else {
					w.Header().Set("Content-Type", "application/json")
				}
				w.WriteHeader(tt.statusCode)
				_, _ = fmt.Fprint(w, tt.response)
			}))
			defer upstream.Close()

			capture := newCollectorCapture(t, true, true)
			handler, _ := newObservedOpenAIHandler(t, upstream.URL, config.TargetConfig{
				Provider:            "openai",
				Model:               "gpt-observed",
				Weight:              1,
				UpstreamRequestType: "responses",
			}, capture.client, config.CacheConfig{Enabled: false})
			payload := `{"model":"gpt-observed","messages":[{"role":"user","content":"hello"}]}`
			if tt.stream {
				payload = `{"model":"gpt-observed","stream":true,"messages":[{"role":"user","content":"hello"}]}`
			}
			recorder := httptest.NewRecorder()
			handler.ChatCompletions(recorder, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(payload)))

			trace, metric, requestLog := capture.waitForRequestEvents(t)
			assertCapturedTraceRequestTypes(t, trace, "chat_completions", "responses")
			assertCapturedRequestTypes(t, metric, "chat_completions", "responses")
			assertCapturedRequestTypes(t, requestLog, "chat_completions", "responses")
		})
	}
}

func TestCollectorSeparatesResponsesClientFromTranslatedChatUpstream(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("upstream path = %q, want /v1/chat/completions", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-observed","object":"chat.completion","created":1,"model":"gpt-observed","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	defer upstream.Close()

	capture := newCollectorCapture(t, true, true)
	handler, _ := newObservedOpenAIHandler(t, upstream.URL, config.TargetConfig{
		Provider: "openai",
		Model:    "gpt-observed",
		Weight:   1,
	}, capture.client, config.CacheConfig{Enabled: false})
	recorder := httptest.NewRecorder()
	handler.Responses(recorder, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(
		`{"model":"gpt-observed","input":"hello"}`,
	)))

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", recorder.Code, recorder.Body.String())
	}
	trace, metric, requestLog := capture.waitForRequestEvents(t)
	assertCapturedTraceRequestTypes(t, trace, "responses", "chat_completions")
	assertCapturedRequestTypes(t, metric, "responses", "chat_completions")
	assertCapturedRequestTypes(t, requestLog, "responses", "chat_completions")
}

func TestCollectorRecordsEmbeddingsRequestTypesOnSuccessAndError(t *testing.T) {
	for _, statusCode := range []int{http.StatusOK, http.StatusInternalServerError} {
		statusCode := statusCode
		t.Run(http.StatusText(statusCode), func(t *testing.T) {
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/v1/embeddings" {
					t.Errorf("upstream path = %q, want /v1/embeddings", r.URL.Path)
				}
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(statusCode)
				if statusCode == http.StatusOK {
					_, _ = w.Write([]byte(`{"object":"list","data":[{"object":"embedding","embedding":[0.1,0.2],"index":0}],"model":"text-embedding-observed","usage":{"prompt_tokens":1,"total_tokens":1}}`))
					return
				}
				_, _ = w.Write([]byte(`{"error":{"message":"upstream failed","type":"server_error"}}`))
			}))
			defer upstream.Close()

			capture := newCollectorCapture(t, true, true)
			handler, _ := newObservedOpenAIHandler(t, upstream.URL, config.TargetConfig{
				Provider: "openai",
				Model:    "text-embedding-observed",
				Weight:   1,
			}, capture.client, config.CacheConfig{Enabled: false})
			recorder := httptest.NewRecorder()
			handler.Embeddings(recorder, httptest.NewRequest(http.MethodPost, "/v1/embeddings", strings.NewReader(
				`{"model":"text-embedding-observed","input":"hello"}`,
			)))

			trace, metric, requestLog := capture.waitForRequestEvents(t)
			assertCapturedTraceRequestTypes(t, trace, "embeddings", "embeddings")
			assertCapturedRequestTypes(t, metric, "embeddings", "embeddings")
			assertCapturedRequestTypes(t, requestLog, "embeddings", "embeddings")
		})
	}
}
