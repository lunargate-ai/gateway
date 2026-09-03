package api

import (
	"encoding/json"
	"net/http/httptest"
	"strconv"
	"testing"

	"github.com/lunargate-ai/gateway/internal/streaming"
)

func TestParseNativeResponsesStreamTerminalNormalizesObservedUsage(t *testing.T) {
	maximum := int(^uint(0) >> 1)
	tests := []struct {
		name            string
		usage           string
		wantInput       int
		wantOutput      int
		wantTotal       int
		wantCached      int
		wantCacheWrite  int
		wantRawInput    int64
		wantObservedRaw int
	}{
		{
			name:            "negative counters",
			usage:           `{"input_tokens":-5,"output_tokens":7,"total_tokens":-1,"input_tokens_details":{"cached_tokens":-2,"cache_write_tokens":-3}}`,
			wantOutput:      7,
			wantTotal:       7,
			wantRawInput:    -5,
			wantObservedRaw: 0,
		},
		{
			name: "overflow and overlapping cache details",
			usage: `{"input_tokens":` + strconv.Itoa(maximum) + `,"output_tokens":` + strconv.Itoa(maximum) +
				`,"total_tokens":1,"input_tokens_details":{"cached_tokens":` + strconv.Itoa(maximum) + `,"cache_write_tokens":` + strconv.Itoa(maximum) + `}}`,
			wantInput:       maximum,
			wantOutput:      maximum,
			wantTotal:       maximum,
			wantCacheWrite:  maximum,
			wantRawInput:    int64(maximum),
			wantObservedRaw: maximum,
		},
		{
			name:            "valid cache details",
			usage:           `{"input_tokens":100,"output_tokens":20,"total_tokens":120,"input_tokens_details":{"cached_tokens":40,"cache_write_tokens":30}}`,
			wantInput:       100,
			wantOutput:      20,
			wantTotal:       120,
			wantCached:      40,
			wantCacheWrite:  30,
			wantRawInput:    100,
			wantObservedRaw: 100,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			event := streaming.SSEEvent{Data: []byte(`{"type":"response.completed","response":{"id":"resp-cache","status":"completed","model":"gpt-4o","output":[],"usage":` + test.usage + `}}`)}
			terminal, ok := parseNativeResponsesStreamTerminal(event)
			if !ok {
				t.Fatal("terminal event was not parsed")
			}
			if terminal.tokensInput != test.wantInput || terminal.tokensOutput != test.wantOutput || terminal.tokensTotal != test.wantTotal {
				t.Fatalf("terminal totals = %d/%d/%d, want %d/%d/%d", terminal.tokensInput, terminal.tokensOutput, terminal.tokensTotal, test.wantInput, test.wantOutput, test.wantTotal)
			}
			if terminal.tokenUsage.CachedInputTokens != test.wantCached || terminal.tokenUsage.CacheWriteInputTokens != test.wantCacheWrite {
				t.Fatalf("terminal cache usage = %#v, want cached=%d write=%d", terminal.tokenUsage, test.wantCached, test.wantCacheWrite)
			}

			rawUsage := terminal.response["usage"].(map[string]interface{})
			if got := nativeResponsesInteger(rawUsage["input_tokens"]); got != test.wantRawInput {
				t.Fatalf("passthrough input_tokens = %d, want %d", got, test.wantRawInput)
			}
			observedUsage := terminal.collectorResponse["usage"].(map[string]interface{})
			if got := nativeResponsesTokenCount(observedUsage["input_tokens"]); got != test.wantObservedRaw {
				t.Fatalf("collector input_tokens = %d, want %d", got, test.wantObservedRaw)
			}
			if got := nativeResponsesTokenCount(observedUsage["total_tokens"]); got != test.wantTotal {
				t.Fatalf("collector total_tokens = %d, want %d", got, test.wantTotal)
			}
		})
	}
}

func TestResponsesStreamProxyPreservesCacheUsageInTerminal(t *testing.T) {
	recorder := httptest.NewRecorder()
	proxy := newResponsesStreamProxy(recorder)
	chunk := `data: {"id":"resp-cache","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[],"usage":{"prompt_tokens":100,"completion_tokens":20,"total_tokens":120,"prompt_tokens_details":{"cached_tokens":40,"cache_write_tokens":30}}}` + "\n\n"
	if _, err := proxy.Write([]byte(chunk)); err != nil {
		t.Fatalf("write usage chunk: %v", err)
	}
	if _, err := proxy.Write([]byte("data: [DONE]\n\n")); err != nil {
		t.Fatalf("write done chunk: %v", err)
	}
	if err := proxy.finalize(); err != nil {
		t.Fatalf("finalize: %v", err)
	}

	for _, event := range decodeSSEEvents(t, recorder.Body.String()) {
		if event["type"] != "response.completed" {
			continue
		}
		response := event["response"].(map[string]interface{})
		usage := response["usage"].(map[string]interface{})
		details := usage["input_tokens_details"].(map[string]interface{})
		if details["cached_tokens"] != float64(40) || details["cache_write_tokens"] != float64(30) {
			t.Fatalf("terminal cache details = %#v, want cached=40 write=30", details)
		}
		return
	}
	t.Fatal("response.completed event not found")
}

func TestNativeResponsesCollectorUsageRemainsJSONSerializable(t *testing.T) {
	event := streaming.SSEEvent{Data: []byte(`{"type":"response.completed","response":{"id":"resp-cache","status":"completed","usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}}}`)}
	terminal, ok := parseNativeResponsesStreamTerminal(event)
	if !ok {
		t.Fatal("terminal event was not parsed")
	}
	if _, err := json.Marshal(terminal.collectorResponse); err != nil {
		t.Fatalf("marshal collector response: %v", err)
	}
}
