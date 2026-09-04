package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
)

func TestMetricsOnlyCollectorDoesNotExportUpstreamErrorContent(t *testing.T) {
	const secret = "prompt-secret-from-provider"
	const attackerType = "attacker-controlled-error-type"
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"message":"` + secret + `","type":"` + attackerType + `"}}`))
	}))
	defer upstream.Close()

	capture := newCollectorCapture(t, false, false)
	handler, _ := newObservedOpenAIHandler(t, upstream.URL, config.TargetConfig{
		Provider: "openai",
		Model:    "gpt-test",
		Weight:   1,
	}, capture.client, config.CacheConfig{Enabled: false})

	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(
		`{"model":"gpt-test","messages":[{"role":"user","content":"hello"}]}`,
	))
	response := httptest.NewRecorder()
	handler.ChatCompletions(response, request)
	if response.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400; body=%s", response.Code, response.Body.String())
	}

	_, metric, requestLog := capture.waitForTraceAndMetric(t)
	if requestLog != nil {
		t.Fatalf("metrics-only collector emitted request log: %#v", requestLog)
	}
	if _, exists := metric["error_message"]; exists {
		t.Fatalf("metric contains error_message: %#v", metric)
	}
	if got := metric["error_code"]; got != "invalid_request" {
		t.Fatalf("metric error_code = %#v, want invalid_request", got)
	}
	encoded, err := json.Marshal(metric)
	if err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []string{secret, attackerType} {
		if strings.Contains(string(encoded), forbidden) {
			t.Fatalf("metric leaked upstream content %q: %s", forbidden, encoded)
		}
	}
}
