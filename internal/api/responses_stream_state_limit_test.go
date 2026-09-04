package api

import (
	"errors"
	"fmt"
	"net/http/httptest"
	"testing"

	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestResponsesStreamProxy_StateBudgetBoundaryFailsClosed(t *testing.T) {
	recorder := httptest.NewRecorder()
	proxy := newResponsesStreamProxy(recorder)
	proxy.stateBytes = responsesStreamStateMaxBytes - 1

	chunk := func(content string) string {
		return fmt.Sprintf("data: {\"id\":\"chatcmpl-budget\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt\",\"choices\":[{\"index\":0,\"delta\":{\"content\":%q}}]}\n\n", content)
	}
	if _, err := proxy.Write([]byte(chunk("x"))); err != nil {
		t.Fatalf("exact boundary write: %v", err)
	}
	if proxy.stateBytes != responsesStreamStateMaxBytes {
		t.Fatalf("state bytes = %d, want exact limit %d", proxy.stateBytes, responsesStreamStateMaxBytes)
	}

	_, streamErr := proxy.Write([]byte(chunk("y")))
	if !errors.Is(streamErr, errResponsesStreamStateTooLarge) {
		t.Fatalf("over-limit write error = %v, want state limit", streamErr)
	}
	if got := proxy.text.String(); got != "x" {
		t.Fatalf("accumulated text = %q, want only accepted boundary delta", got)
	}

	proxy.RecordStreamError(streamErr)
	if err := proxy.finalize(); err != nil {
		t.Fatalf("finalize after state limit: %v", err)
	}
	events := decodeSSEEvents(t, recorder.Body.String())
	if !containsEventType(events, "response.failed") {
		t.Fatal("state limit must terminate translated Responses with response.failed")
	}
	if containsEventType(events, "response.completed") {
		t.Fatal("state limit must not emit response.completed")
	}
}

func TestResponsesStreamProxy_StateBudgetCoversAllAccumulators(t *testing.T) {
	tests := []struct {
		name   string
		append func(*responsesStreamProxy, string) error
	}{
		{
			name: "text",
			append: func(proxy *responsesStreamProxy, value string) error {
				_, err := proxy.mergeTextDelta(value)
				return err
			},
		},
		{
			name: "refusal",
			append: func(proxy *responsesStreamProxy, value string) error {
				_, err := proxy.mergeRefusalDelta(value)
				return err
			},
		},
		{
			name: "reasoning",
			append: func(proxy *responsesStreamProxy, value string) error {
				_, err := proxy.mergeReasoningDelta(value)
				return err
			},
		},
		{
			name: "tool arguments",
			append: func(proxy *responsesStreamProxy, value string) error {
				index := 0
				proxy.toolCalls["idx_0"] = &responsesToolCallState{
					ItemID:      "fc_budget",
					CallID:      "call_budget",
					Name:        "lookup",
					OutputIndex: 1,
					Added:       true,
				}
				proxy.toolCallOrder = []string{"idx_0"}
				return proxy.processToolCallDelta(models.ToolCall{
					Index: &index,
					Function: models.ToolCallFunction{
						Arguments: value,
					},
				})
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			proxy := newResponsesStreamProxy(httptest.NewRecorder())
			proxy.stateBytes = responsesStreamStateMaxBytes - 1
			if err := test.append(proxy, "x"); err != nil {
				t.Fatalf("exact boundary append: %v", err)
			}
			if err := test.append(proxy, "y"); !errors.Is(err, errResponsesStreamStateTooLarge) {
				t.Fatalf("over-limit append error = %v, want state limit", err)
			}
		})
	}
}

func TestResponsesStreamProxy_ToolCountBoundary(t *testing.T) {
	proxy := newResponsesStreamProxy(httptest.NewRecorder())
	proxy.responseID = "resp_tool_budget"
	proxy.model = "gpt"
	for i := 0; i < responsesStreamMaxToolCalls-1; i++ {
		key := fmt.Sprintf("idx_%d", i)
		proxy.toolCalls[key] = &responsesToolCallState{Added: true, Done: true}
		proxy.toolCallOrder = append(proxy.toolCallOrder, key)
	}

	boundaryIndex := responsesStreamMaxToolCalls - 1
	if err := proxy.processToolCallDelta(models.ToolCall{
		Index: &boundaryIndex,
		ID:    "call_boundary",
		Function: models.ToolCallFunction{
			Name: "lookup",
		},
	}); err != nil {
		t.Fatalf("tool at exact count boundary: %v", err)
	}
	if len(proxy.toolCalls) != responsesStreamMaxToolCalls {
		t.Fatalf("tool state count = %d, want %d", len(proxy.toolCalls), responsesStreamMaxToolCalls)
	}

	overIndex := responsesStreamMaxToolCalls
	err := proxy.processToolCallDelta(models.ToolCall{
		Index: &overIndex,
		ID:    "call_over",
		Function: models.ToolCallFunction{
			Name: "lookup",
		},
	})
	if !errors.Is(err, errResponsesStreamTooManyTools) {
		t.Fatalf("tool over count limit error = %v, want tool limit", err)
	}
	if len(proxy.toolCalls) != responsesStreamMaxToolCalls {
		t.Fatalf("tool state count grew after rejection: %d", len(proxy.toolCalls))
	}
}
