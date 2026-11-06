package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
)

func TestCachedServiceTiersRemainIndependentOfContentSharing(t *testing.T) {
	for _, sharing := range []bool{false, true} {
		for _, effective := range []string{"default", ""} {
			t.Run(fmt.Sprintf("sharing=%v/effective=%s", sharing, effective), func(t *testing.T) {
				var calls atomic.Int32
				upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					calls.Add(1)
					w.Header().Set("Content-Type", "application/json")
					tier := ""
					if effective != "" {
						tier = `,"service_tier":"` + effective + `"`
					}
					fmt.Fprint(w, `{"id":"chatcmpl_cache_tier","object":"chat.completion","model":"gpt-6.1-sol","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}`+tier+`}`)
				}))
				defer upstream.Close()
				capture := newCollectorCapture(t, sharing, sharing)
				handler, _ := newObservedOpenAIHandler(t, upstream.URL, config.TargetConfig{Provider: "openai", Model: "gpt-6.1-sol", Weight: 1}, capture.client, enabledTestCacheConfig())
				for i, tier := range []string{"fast", "fast", "default"} {
					payload := fmt.Sprintf(`{"model":"gpt-6.1-sol","service_tier":%q,"messages":[{"role":"user","content":"hi"}]}`, tier)
					w := httptest.NewRecorder()
					handler.ChatCompletions(w, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(payload)))
					if w.Code != http.StatusOK {
						t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
					}
					trace, metric, requestLog := capture.waitForEvents(t, sharing)
					if i != 1 {
						if w.Header().Get("X-LunarGate-Cache-Status") != "MISS" {
							t.Fatal("different requested tiers must use different cache entries")
						}
						continue
					}
					if w.Header().Get("X-LunarGate-Cache-Status") != "HIT" || calls.Load() != 1 {
						t.Fatal("identical request must hit cache")
					}
					if trace["requested_service_tier"] != "fast" {
						t.Fatalf("cached trace tier = %#v", trace["requested_service_tier"])
					}
					for _, event := range []map[string]interface{}{metric, requestLog} {
						if event == nil {
							continue
						}
						if event["requested_service_tier"] != "fast" || (effective == "" && event["effective_service_tier"] != nil) || (effective != "" && event["effective_service_tier"] != effective) {
							t.Fatalf("cached tiers lost: %#v", event)
						}
						assertCapturedCacheHit(t, event)
					}
					if metric["cost_usd"] != float64(0) {
						t.Fatalf("cache hit cost = %#v", metric["cost_usd"])
					}
					if !sharing && requestLog != nil {
						t.Fatal("content sharing disabled but request log present")
					}
				}
				if calls.Load() != 2 {
					t.Fatalf("upstream calls = %d, want 2", calls.Load())
				}
			})
		}
	}
}

func TestServiceTierForwardingAndMetricsWithoutContentSharing(t *testing.T) {
	for _, upstreamType := range []string{"chat_completions", "responses"} {
		for _, stream := range []bool{false, true} {
			for _, effective := range []string{"ultrafast", "default", ""} {
				t.Run(fmt.Sprintf("%s/stream=%v/effective=%s", upstreamType, stream, effective), func(t *testing.T) {
					upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						var request map[string]interface{}
						if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
							t.Error(err)
						}
						if request["service_tier"] != "ultrafast" {
							t.Errorf("tier lost: %#v", request)
						}
						tier := ""
						if effective != "" {
							tier = `,"service_tier":"` + effective + `"`
						}
						if upstreamType == "responses" {
							response := `{"id":"resp_tier","object":"response","created_at":1,"model":"gpt-6.1-sol","status":"completed","output":[{"type":"message","id":"msg_tier","role":"assistant","content":[{"type":"output_text","text":"ok","annotations":[]}]}],"usage":{"input_tokens":10,"output_tokens":2,"total_tokens":12}` + tier + `}`
							if stream {
								w.Header().Set("Content-Type", "text/event-stream")
								fmt.Fprintf(w, "data: {\"type\":\"response.completed\",\"sequence_number\":0,\"response\":%s}\n\n", response)
							} else {
								w.Header().Set("Content-Type", "application/json")
								fmt.Fprint(w, response)
							}
						} else if stream {
							w.Header().Set("Content-Type", "text/event-stream")
							fmt.Fprintf(w, "data: {\"id\":\"chatcmpl_tier\",\"model\":\"gpt-6.1-sol\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2,\"total_tokens\":12}%s}\n\ndata: [DONE]\n\n", tier)
						} else {
							w.Header().Set("Content-Type", "application/json")
							fmt.Fprint(w, `{"id":"chatcmpl_tier","object":"chat.completion","model":"gpt-6.1-sol","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}`+tier+`}`)
						}
					}))
					defer upstream.Close()
					capture := newCollectorCapture(t, false, false)
					handler, _ := newObservedOpenAIHandler(t, upstream.URL, config.TargetConfig{Provider: "openai", Model: "gpt-6.1-sol", Weight: 1, UpstreamRequestType: upstreamType}, capture.client, config.CacheConfig{Enabled: false})
					payload := fmt.Sprintf(`{"model":"gpt-6.1-sol","service_tier":"ultrafast","stream":%v,"stream_options":{"include_usage":false},"messages":[{"role":"user","content":"hi"}]}`, stream)
					w := httptest.NewRecorder()
					handler.ChatCompletions(w, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(payload)))
					if w.Code != 200 {
						t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
					}
					_, metric, _ := capture.waitForTraceAndMetric(t)
					if metric["requested_service_tier"] != "ultrafast" {
						t.Fatal(metric)
					}
					if effective == "" {
						if metric["effective_service_tier"] != nil {
							t.Fatal(metric)
						}
					} else if metric["effective_service_tier"] != effective {
						t.Fatal(metric)
					}
					if stream && strings.Contains(w.Body.String(), `"usage":{`) {
						t.Fatal("include_usage=false ignored")
					}
				})
			}
		}
	}
}
