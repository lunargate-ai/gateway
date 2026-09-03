package api

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestChatCompletionsFromResponsesPreservesRefusalHTTP(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/responses" {
			t.Errorf("upstream path = %q, want /v1/responses", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"resp_refusal","object":"response","created_at":123,"status":"completed","model":"gpt-5.4",
			"output":[{"type":"message","id":"msg_refusal","status":"completed","role":"assistant","content":[{"type":"refusal","refusal":"I can't help with that."}]}],
			"output_text":"","usage":{"input_tokens":3,"output_tokens":4,"total_tokens":7},
			"future_responses_field":{"must_not_leak_into_chat":true}
		}`)
	}))
	defer upstream.Close()

	handler, cache := newNativeContinuationTestHandler(t, upstream.URL+"/v1", requestTypeResponses)
	defer cache.Stop()
	recorder := httptest.NewRecorder()
	handler.ChatCompletions(recorder, httptest.NewRequest(
		http.MethodPost,
		"/v1/responses",
		strings.NewReader(`{"model":"gpt-5.4","messages":[{"role":"user","content":"unsafe request"}]}`),
	))

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, body=%s", recorder.Code, recorder.Body.String())
	}
	var response models.UnifiedResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode Chat Completions response: %v", err)
	}
	if len(response.Choices) != 1 || response.Choices[0].Message == nil {
		t.Fatalf("choices = %#v", response.Choices)
	}
	choice := response.Choices[0]
	if choice.Message.Refusal != "I can't help with that." {
		t.Fatalf("refusal = %q", choice.Message.Refusal)
	}
	if choice.Message.Content != nil {
		t.Fatalf("refusal was disguised as content: %#v", choice.Message.Content)
	}
	if choice.FinishReason == nil || *choice.FinishReason != "stop" {
		t.Fatalf("finish_reason = %#v, want stop", choice.FinishReason)
	}
	if strings.Contains(recorder.Body.String(), "future_responses_field") || strings.Contains(recorder.Body.String(), `"type":"refusal"`) {
		t.Fatalf("Responses envelope leaked into Chat response: %s", recorder.Body.String())
	}
}

func TestChatCompletionsFromResponsesPreservesRefusalSSE(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/responses" {
			t.Errorf("upstream path = %q, want /v1/responses", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		events := []string{
			`{"type":"response.created","response":{"id":"resp_refusal_stream","object":"response","created_at":123,"status":"in_progress","model":"gpt-5.4","output":[]}}`,
			`{"type":"response.output_item.added","output_index":0,"item":{"id":"msg_refusal","type":"message","status":"in_progress","role":"assistant","content":[]}}`,
			`{"type":"response.content_part.added","item_id":"msg_refusal","output_index":0,"content_index":0,"part":{"type":"refusal","refusal":""}}`,
			`{"type":"response.refusal.delta","item_id":"msg_refusal","output_index":0,"content_index":0,"delta":"I can't "}`,
			`{"type":"response.refusal.delta","item_id":"msg_refusal","output_index":0,"content_index":0,"delta":"help with that."}`,
			`{"type":"response.refusal.done","item_id":"msg_refusal","output_index":0,"content_index":0,"refusal":"I can't help with that."}`,
			`{"type":"response.content_part.done","item_id":"msg_refusal","output_index":0,"content_index":0,"part":{"type":"refusal","refusal":"I can't help with that."}}`,
			`{"type":"response.output_item.done","output_index":0,"item":{"id":"msg_refusal","type":"message","status":"completed","role":"assistant","content":[{"type":"refusal","refusal":"I can't help with that."}]}}`,
			`{"type":"response.completed","response":{"id":"resp_refusal_stream","object":"response","created_at":123,"status":"completed","model":"gpt-5.4","output":[{"id":"msg_refusal","type":"message","status":"completed","role":"assistant","content":[{"type":"refusal","refusal":"I can't help with that."}]}],"output_text":"","usage":{"input_tokens":3,"output_tokens":4,"total_tokens":7}}}`,
		}
		for _, event := range events {
			_, _ = io.WriteString(w, "data: "+event+"\n\n")
		}
	}))
	defer upstream.Close()

	handler, cache := newNativeContinuationTestHandler(t, upstream.URL+"/v1", requestTypeResponses)
	defer cache.Stop()
	recorder := httptest.NewRecorder()
	handler.ChatCompletions(recorder, httptest.NewRequest(
		http.MethodPost,
		"/v1/responses",
		strings.NewReader(`{"model":"gpt-5.4","messages":[{"role":"user","content":"unsafe request"}],"stream":true}`),
	))

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, body=%s", recorder.Code, recorder.Body.String())
	}
	body := recorder.Body.String()
	if strings.Contains(body, `"type":"response.refusal`) || strings.Contains(body, `event: response.`) {
		t.Fatalf("native Responses event leaked into Chat SSE: %s", body)
	}

	var refusal strings.Builder
	terminalChoices := 0
	doneMarkers := 0
	for _, frame := range strings.Split(body, "\n\n") {
		payload := strings.TrimPrefix(strings.TrimSpace(frame), "data: ")
		if payload == "" {
			continue
		}
		if payload == "[DONE]" {
			doneMarkers++
			continue
		}
		var chunk models.StreamChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			t.Fatalf("decode Chat SSE payload %q: %v", payload, err)
		}
		for _, choice := range chunk.Choices {
			if choice.Delta != nil {
				refusal.WriteString(choice.Delta.Refusal)
			}
			if choice.FinishReason != nil {
				terminalChoices++
				if *choice.FinishReason != "stop" {
					t.Fatalf("finish_reason = %q, want stop", *choice.FinishReason)
				}
			}
		}
	}

	if got := refusal.String(); got != "I can't help with that." {
		t.Fatalf("refusal deltas = %q", got)
	}
	if terminalChoices != 1 {
		t.Fatalf("terminal choice count = %d, want 1; body=%s", terminalChoices, body)
	}
	if doneMarkers != 1 || !strings.HasSuffix(body, "data: [DONE]\n\n") {
		t.Fatalf("DONE markers = %d or stream did not terminate exactly; body=%q", doneMarkers, body)
	}
}
