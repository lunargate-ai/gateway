package providers

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestOpenAITranslator_StreamingRequestIncludesUsage(t *testing.T) {
	translator := NewOpenAITranslator(config.ProviderConfig{
		APIKey:  "dummy",
		BaseURL: "https://api.openai.com/v1",
	})

	req, err := translator.TranslateRequest(context.Background(), &models.UnifiedRequest{
		Model:    "gpt-5.4",
		Stream:   true,
		Messages: []models.Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("TranslateRequest returned error: %v", err)
	}

	body, err := io.ReadAll(req.Body)
	if err != nil {
		t.Fatalf("failed to read request body: %v", err)
	}

	var payload models.UnifiedRequest
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("failed to unmarshal request payload: %v", err)
	}

	if !payload.Stream {
		t.Fatalf("expected stream=true in upstream payload")
	}
	if payload.StreamOptions == nil {
		t.Fatalf("expected stream_options to be present in upstream payload")
	}
	if !payload.StreamOptions.IncludeUsage {
		t.Fatalf("expected stream_options.include_usage=true in upstream payload")
	}
}

func TestOpenAITranslator_ResponsesUpstreamUsesResponsesEndpoint(t *testing.T) {
	translator := NewOpenAITranslator(config.ProviderConfig{
		APIKey:  "dummy",
		BaseURL: "https://api.openai.com/v1",
	})

	ctx := WithUpstreamRequestType(context.Background(), "responses")
	req, err := translator.TranslateRequest(ctx, &models.UnifiedRequest{
		Model:    "gpt-5.3-codex",
		Stream:   true,
		Messages: []models.Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("TranslateRequest returned error: %v", err)
	}
	if !strings.HasSuffix(req.URL.Path, "/responses") {
		t.Fatalf("expected upstream endpoint to be /responses, got %q", req.URL.String())
	}

	body, err := io.ReadAll(req.Body)
	if err != nil {
		t.Fatalf("failed to read request body: %v", err)
	}

	var payload map[string]interface{}
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("failed to unmarshal request payload: %v", err)
	}
	if _, ok := payload["input"]; !ok {
		t.Fatalf("expected responses payload to contain input")
	}
	if _, ok := payload["messages"]; ok {
		t.Fatalf("expected responses payload not to contain chat-completions messages field")
	}
}

func TestOpenAITranslator_ResponsesUpstreamPreservesPreviousResponseID(t *testing.T) {
	translator := NewOpenAITranslator(config.ProviderConfig{
		APIKey:  "dummy",
		BaseURL: "https://api.openai.com/v1",
	})

	ctx := WithUpstreamRequestType(context.Background(), "responses")
	req, err := translator.TranslateRequest(ctx, &models.UnifiedRequest{
		Model:              "gpt-5.2",
		PreviousResponseID: "resp_prev_123",
		Messages: []models.Message{
			{Role: "tool", ToolCallID: "call_123", Content: `{"ok":true}`},
		},
	})
	if err != nil {
		t.Fatalf("TranslateRequest returned error: %v", err)
	}

	body, err := io.ReadAll(req.Body)
	if err != nil {
		t.Fatalf("failed to read request body: %v", err)
	}

	var payload map[string]interface{}
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("failed to unmarshal request payload: %v", err)
	}
	if got, _ := payload["previous_response_id"].(string); got != "resp_prev_123" {
		t.Fatalf("expected previous_response_id to be preserved, got %q", got)
	}
}

func TestOpenAITranslator_ResponsesUpstreamMapsReasoningEffort(t *testing.T) {
	translator := NewOpenAITranslator(config.ProviderConfig{
		APIKey:  "dummy",
		BaseURL: "https://api.openai.com/v1",
	})

	ctx := WithUpstreamRequestType(context.Background(), "responses")
	req, err := translator.TranslateRequest(ctx, &models.UnifiedRequest{
		Model:           "gpt-5.2",
		ReasoningEffort: "high",
		Messages:        []models.Message{{Role: "user", Content: "hello"}},
	})
	if err != nil {
		t.Fatalf("TranslateRequest returned error: %v", err)
	}

	body, err := io.ReadAll(req.Body)
	if err != nil {
		t.Fatalf("failed to read request body: %v", err)
	}

	var payload map[string]interface{}
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("failed to unmarshal request payload: %v", err)
	}
	reasoning, ok := payload["reasoning"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected reasoning object in responses payload")
	}
	if got, _ := reasoning["effort"].(string); got != "high" {
		t.Fatalf("expected reasoning.effort=high, got %q", got)
	}
}

func TestOpenAITranslator_ResponsesUpstreamFunctionCallIDsAreFCAndCallIDIsPreserved(t *testing.T) {
	translator := NewOpenAITranslator(config.ProviderConfig{
		APIKey:  "dummy",
		BaseURL: "https://api.openai.com/v1",
	})

	ctx := WithUpstreamRequestType(context.Background(), "responses")
	req, err := translator.TranslateRequest(ctx, &models.UnifiedRequest{
		Model:  "gpt-5.3-codex",
		Stream: true,
		Messages: []models.Message{
			{Role: "user", Content: "run tool"},
			{
				Role:    "assistant",
				Content: "",
				ToolCalls: []models.ToolCall{{
					ID:   "call_YFO4VrNVBrC7VbiGUrdPfpqZ",
					Type: "function",
					Function: models.ToolCallFunction{
						Name:      "terminal",
						Arguments: `{"command":"pwd"}`,
					},
				}},
			},
		},
	})
	if err != nil {
		t.Fatalf("TranslateRequest returned error: %v", err)
	}

	body, err := io.ReadAll(req.Body)
	if err != nil {
		t.Fatalf("failed to read request body: %v", err)
	}

	var payload map[string]interface{}
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("failed to unmarshal request payload: %v", err)
	}
	input, ok := payload["input"].([]interface{})
	if !ok {
		t.Fatalf("expected input array in responses payload")
	}

	var fnItem map[string]interface{}
	for _, raw := range input {
		item, _ := raw.(map[string]interface{})
		if item == nil {
			continue
		}
		if itemType, _ := item["type"].(string); itemType == "function_call" {
			fnItem = item
			break
		}
	}
	if fnItem == nil {
		t.Fatalf("expected function_call item in responses input")
	}

	itemID, _ := fnItem["id"].(string)
	if !strings.HasPrefix(itemID, "fc") {
		t.Fatalf("expected function_call id to start with fc, got %q", itemID)
	}
	callID, _ := fnItem["call_id"].(string)
	if callID != "call_YFO4VrNVBrC7VbiGUrdPfpqZ" {
		t.Fatalf("expected call_id to be preserved, got %q", callID)
	}
}

func TestOpenAITranslator_ParseResponse_ResponsesObject(t *testing.T) {
	translator := NewOpenAITranslator(config.ProviderConfig{
		APIKey:  "dummy",
		BaseURL: "https://api.openai.com/v1",
	})

	respBody := `{"id":"resp_1","object":"response","created_at":1,"status":"completed","model":"gpt-5.3-codex","output":[{"type":"function_call","id":"call_1","call_id":"call_1","name":"terminal","arguments":"{\"command\":\"pwd\"}"}],"output_text":"done","usage":{"input_tokens":3,"output_tokens":2,"total_tokens":5}}`
	resp := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(respBody)),
	}

	unified, err := translator.ParseResponse(resp)
	if err != nil {
		t.Fatalf("ParseResponse returned error: %v", err)
	}
	if unified.Object != "chat.completion" {
		t.Fatalf("expected chat.completion object, got %q", unified.Object)
	}
	if len(unified.Choices) == 0 || unified.Choices[0].Message == nil {
		t.Fatalf("expected assistant message in unified response")
	}
	if len(unified.Choices[0].Message.ToolCalls) != 1 {
		t.Fatalf("expected one tool call, got %d", len(unified.Choices[0].Message.ToolCalls))
	}
}

func TestOpenAITranslator_ParseResponse_ResponsesObjectFallsBackToMessageContent(t *testing.T) {
	translator := NewOpenAITranslator(config.ProviderConfig{
		APIKey:  "dummy",
		BaseURL: "https://api.openai.com/v1",
	})

	respBody := `{"id":"resp_2","object":"response","created_at":1,"status":"completed","model":"gpt-5.2","output":[{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"final answer from message content"}]}],"output_text":"","usage":{"input_tokens":3,"output_tokens":2,"total_tokens":5}}`
	resp := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(respBody)),
	}

	unified, err := translator.ParseResponse(resp)
	if err != nil {
		t.Fatalf("ParseResponse returned error: %v", err)
	}
	if len(unified.Choices) == 0 || unified.Choices[0].Message == nil {
		t.Fatalf("expected assistant message in unified response")
	}
	if got, _ := unified.Choices[0].Message.Content.(string); got != "final answer from message content" {
		t.Fatalf("expected assistant content from message content fallback, got %q", got)
	}
}

func TestOpenAITranslator_ParseResponse_ResponsesObjectPreservesReasoningSummary(t *testing.T) {
	translator := NewOpenAITranslator(config.ProviderConfig{
		APIKey:  "dummy",
		BaseURL: "https://api.openai.com/v1",
	})

	respBody := `{"id":"resp_reason","object":"response","created_at":1,"status":"completed","model":"gpt-5.2","output":[{"type":"reasoning","id":"rs_1","status":"completed","summary":[{"type":"summary_text","text":"plan before final answer"}]},{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"done"}]}],"output_text":"done","usage":{"input_tokens":3,"output_tokens":2,"total_tokens":5}}`
	resp := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(respBody)),
	}

	unified, err := translator.ParseResponse(resp)
	if err != nil {
		t.Fatalf("ParseResponse returned error: %v", err)
	}
	if len(unified.Choices) == 0 || unified.Choices[0].Message == nil {
		t.Fatalf("expected assistant message in unified response")
	}
	if got := unified.Choices[0].Message.ReasoningContent; got != "plan before final answer" {
		t.Fatalf("expected reasoning summary to be preserved, got %q", got)
	}
}

func TestOpenAITranslator_ParseStreamChunk_ResponsesEvents(t *testing.T) {
	translator := NewOpenAITranslator(config.ProviderConfig{
		APIKey:  "dummy",
		BaseURL: "https://api.openai.com/v1",
	})

	chunk, err := translator.ParseStreamChunk([]byte(`{"type":"response.output_text.delta","response_id":"resp_1","delta":"hello"}`))
	if err != nil {
		t.Fatalf("ParseStreamChunk returned error: %v", err)
	}
	if chunk == nil || len(chunk.Choices) == 0 || chunk.Choices[0].Delta == nil {
		t.Fatalf("expected converted stream chunk")
	}
	if got, _ := chunk.Choices[0].Delta.Content.(string); got != "hello" {
		t.Fatalf("expected delta content hello, got %q", got)
	}

	doneChunk, doneErr := translator.ParseStreamChunk([]byte(`{"type":"response.completed","response":{"id":"resp_1"}}`))
	if doneErr != ErrStreamDone {
		t.Fatalf("expected ErrStreamDone, got chunk=%v err=%v", doneChunk, doneErr)
	}
}

func TestOpenAITranslator_ParseStreamChunk_ContentPartDoneEmitsText(t *testing.T) {
	translator := NewOpenAITranslator(config.ProviderConfig{
		APIKey:  "dummy",
		BaseURL: "https://api.openai.com/v1",
	})

	chunk, err := translator.ParseStreamChunk([]byte(`{"type":"response.content_part.done","response_id":"resp_1","item_id":"msg_1","output_index":0,"content_index":0,"part":{"type":"output_text","text":"final text"}}`))
	if err != nil {
		t.Fatalf("ParseStreamChunk returned error: %v", err)
	}
	if chunk == nil || len(chunk.Choices) == 0 || chunk.Choices[0].Delta == nil {
		t.Fatalf("expected converted stream chunk")
	}
	if got, _ := chunk.Choices[0].Delta.Content.(string); got != "final text" {
		t.Fatalf("expected content from content_part.done, got %q", got)
	}
}

func TestOpenAITranslator_ParseStreamChunk_OutputTextDoneEmitsText(t *testing.T) {
	translator := NewOpenAITranslator(config.ProviderConfig{
		APIKey:  "dummy",
		BaseURL: "https://api.openai.com/v1",
	})

	chunk, err := translator.ParseStreamChunk([]byte(`{"type":"response.output_text.done","response_id":"resp_1","item_id":"msg_1","output_index":0,"content_index":0,"text":"final text from done"}`))
	if err != nil {
		t.Fatalf("ParseStreamChunk returned error: %v", err)
	}
	if chunk == nil || len(chunk.Choices) == 0 || chunk.Choices[0].Delta == nil {
		t.Fatalf("expected converted stream chunk")
	}
	if got, _ := chunk.Choices[0].Delta.Content.(string); got != "final text from done" {
		t.Fatalf("expected content from output_text.done, got %q", got)
	}
}

func TestOpenAITranslator_ParseStreamChunk_OutputItemDoneMessageEmitsText(t *testing.T) {
	translator := NewOpenAITranslator(config.ProviderConfig{
		APIKey:  "dummy",
		BaseURL: "https://api.openai.com/v1",
	})

	chunk, err := translator.ParseStreamChunk([]byte(`{"type":"response.output_item.done","response_id":"resp_1","output_index":0,"item":{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"from item done"}]}}`))
	if err != nil {
		t.Fatalf("ParseStreamChunk returned error: %v", err)
	}
	if chunk == nil || len(chunk.Choices) == 0 || chunk.Choices[0].Delta == nil {
		t.Fatalf("expected converted stream chunk")
	}
	if got, _ := chunk.Choices[0].Delta.Content.(string); got != "from item done" {
		t.Fatalf("expected content from output_item.done message, got %q", got)
	}
}

func TestOpenAITranslator_ParseStreamChunk_ReasoningSummaryTextDeltaEmitsReasoning(t *testing.T) {
	translator := NewOpenAITranslator(config.ProviderConfig{
		APIKey:  "dummy",
		BaseURL: "https://api.openai.com/v1",
	})

	chunk, err := translator.ParseStreamChunk([]byte(`{"type":"response.reasoning_summary_text.delta","response_id":"resp_1","item_id":"rs_1","output_index":1,"summary_index":0,"delta":"thinking delta"}`))
	if err != nil {
		t.Fatalf("ParseStreamChunk returned error: %v", err)
	}
	if chunk == nil || len(chunk.Choices) == 0 || chunk.Choices[0].Delta == nil {
		t.Fatalf("expected converted stream chunk")
	}
	if chunk.Choices[0].Delta.ReasoningContent != "thinking delta" {
		t.Fatalf("expected reasoning content delta, got %q", chunk.Choices[0].Delta.ReasoningContent)
	}
}

func TestOpenAITranslator_ParseStreamChunk_ReasoningTextDeltaEmitsReasoning(t *testing.T) {
	translator := NewOpenAITranslator(config.ProviderConfig{
		APIKey:  "dummy",
		BaseURL: "https://api.openai.com/v1",
	})

	chunk, err := translator.ParseStreamChunk([]byte(`{"type":"response.reasoning_text.delta","response_id":"resp_1","item_id":"rs_1","output_index":1,"delta":"hidden thinking"}`))
	if err != nil {
		t.Fatalf("ParseStreamChunk returned error: %v", err)
	}
	if chunk == nil || len(chunk.Choices) == 0 || chunk.Choices[0].Delta == nil {
		t.Fatalf("expected converted stream chunk")
	}
	if chunk.Choices[0].Delta.ReasoningContent != "hidden thinking" {
		t.Fatalf("expected reasoning content delta, got %q", chunk.Choices[0].Delta.ReasoningContent)
	}
}

func TestOpenAITranslator_ParseStreamChunk_ReasoningSummaryPartDoneEmitsReasoning(t *testing.T) {
	translator := NewOpenAITranslator(config.ProviderConfig{
		APIKey:  "dummy",
		BaseURL: "https://api.openai.com/v1",
	})

	chunk, err := translator.ParseStreamChunk([]byte(`{"type":"response.reasoning_summary_part.done","response_id":"resp_1","item_id":"rs_1","output_index":1,"summary_index":0,"part":{"type":"summary_text","text":"thinking final"}}`))
	if err != nil {
		t.Fatalf("ParseStreamChunk returned error: %v", err)
	}
	if chunk == nil || len(chunk.Choices) == 0 || chunk.Choices[0].Delta == nil {
		t.Fatalf("expected converted stream chunk")
	}
	if chunk.Choices[0].Delta.ReasoningContent != "thinking final" {
		t.Fatalf("expected reasoning content from summary part, got %q", chunk.Choices[0].Delta.ReasoningContent)
	}
}

func TestOpenAITranslator_ParseStreamChunk_ResponseCreatedSetsResponseID(t *testing.T) {
	translator := NewOpenAITranslator(config.ProviderConfig{
		APIKey:  "dummy",
		BaseURL: "https://api.openai.com/v1",
	})

	chunk, err := translator.ParseStreamChunk([]byte(`{"type":"response.created","response":{"id":"resp_created_1","created_at":123,"model":"gpt-5.3-codex"}}`))
	if err != nil {
		t.Fatalf("ParseStreamChunk returned error: %v", err)
	}
	if chunk == nil {
		t.Fatalf("expected chunk for response.created")
	}
	if chunk.ID != "resp_created_1" {
		t.Fatalf("expected response id from response.created, got %q", chunk.ID)
	}
	if chunk.Model != "gpt-5.3-codex" {
		t.Fatalf("expected model from response.created, got %q", chunk.Model)
	}
	if chunk.Created != 123 {
		t.Fatalf("expected created_at from response.created, got %d", chunk.Created)
	}
}

func TestOpenAITranslator_ChatCompletionsRouteCanUseResponsesUpstream(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/responses" {
			t.Fatalf("expected upstream path /v1/responses, got %q", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"resp_1","object":"response","created_at":1,"status":"completed","model":"gpt-5.3-codex","output":[],"output_text":"ok","usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`))
	}))
	defer upstream.Close()

	translator := NewOpenAITranslator(config.ProviderConfig{
		APIKey:  "dummy",
		BaseURL: upstream.URL + "/v1",
	})

	ctx := WithUpstreamRequestType(context.Background(), "responses")
	httpReq, err := translator.TranslateRequest(ctx, &models.UnifiedRequest{
		Model:    "gpt-5.3-codex",
		Messages: []models.Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("TranslateRequest returned error: %v", err)
	}

	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		t.Fatalf("upstream call failed: %v", err)
	}

	unified, err := translator.ParseResponse(resp)
	if err != nil {
		t.Fatalf("ParseResponse returned error: %v", err)
	}
	if len(unified.Choices) == 0 || unified.Choices[0].Message == nil {
		t.Fatalf("expected assistant choice after responses parse")
	}
	if got, _ := unified.Choices[0].Message.Content.(string); got != "ok" {
		t.Fatalf("expected content ok, got %q", got)
	}
}
