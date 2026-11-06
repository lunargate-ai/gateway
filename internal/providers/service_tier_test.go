package providers

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestAnthropicServiceTierMapping(t *testing.T) {
	for _, tt := range []struct {
		tier, speed     string
		enabled, reject bool
	}{
		{"", "", false, false}, {"auto", "", false, false}, {"default", "standard", false, false},
		{"fast", "fast", true, false}, {"priority", "fast", true, false},
		{"fast", "", false, true}, {"ultrafast", "", true, true}, {"flex", "", true, true},
	} {
		t.Run(tt.tier+"/"+tt.speed, func(t *testing.T) {
			translator := NewAnthropicTranslator(config.ProviderConfig{Capabilities: config.ProviderCapabilities{FastMode: tt.enabled}})
			req := &models.UnifiedRequest{Model: "claude-opus-5", Messages: []models.Message{{Role: "user", Content: "hi"}}}
			if tt.tier != "" {
				req.ServiceTier = &tt.tier
			}
			ctx := WithUpstreamRequestHeaders(context.Background(), http.Header{"Anthropic-Beta": {"existing-beta,fast-mode-2026-02-01"}})
			request, err := translator.TranslateRequest(ctx, req)
			if tt.reject {
				var compat *models.CompatibilityError
				if !errors.As(err, &compat) || compat.Field != "service_tier" {
					t.Fatalf("error = %v", err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			defer request.Body.Close()
			var body map[string]interface{}
			if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
				t.Fatal(err)
			}
			if got, _ := body["speed"].(string); got != tt.speed {
				t.Fatalf("speed=%q want=%q", got, tt.speed)
			}
			if _, exists := body["service_tier"]; exists {
				t.Fatal("OpenAI field leaked into Anthropic request")
			}
			if strings.Count(strings.Join(request.Header.Values("Anthropic-Beta"), ","), "fast-mode-2026-02-01") != 1 {
				t.Fatal("duplicate beta")
			}
		})
	}
}

func TestAnthropicSpeedMetadataIndependentOfTokens(t *testing.T) {
	base := NewAnthropicTranslator(config.ProviderConfig{})
	for _, speed := range []string{"fast", "standard", ""} {
		response, err := base.ParseResponse(&http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(`{"id":"msg_1","model":"claude-opus-5-5","content":[],"usage":{"speed":"` + speed + `"}}`))})
		if err != nil {
			t.Fatal(err)
		}
		if speed == "" {
			if response.ServiceTier != nil {
				t.Fatal("inferred tier")
			}
			continue
		}
		want := speed
		if speed == "standard" {
			want = "default"
		}
		if response.ServiceTier == nil || *response.ServiceTier != want {
			t.Fatalf("tier=%v", response.ServiceTier)
		}
		stream := NewAnthropicStreamTranslator(base)
		chunk, err := stream.ParseStreamChunk([]byte(`{"type":"message_start","message":{"id":"msg_1","model":"claude-opus-5-5","usage":{"speed":"` + speed + `"}}}`))
		if err != nil || chunk.ServiceTier == nil || *chunk.ServiceTier != want {
			t.Fatalf("chunk=%+v err=%v", chunk, err)
		}
		chunk, err = stream.ParseStreamChunk([]byte(`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`))
		if err != nil || chunk.ServiceTier == nil || *chunk.ServiceTier != want {
			t.Fatalf("lost speed: %+v %v", chunk, err)
		}
	}
}

func TestAnthropicSpeedBetaModels(t *testing.T) {
	for _, model := range []string{"claude-opus-5", "claude-opus-4-8", "claude-opus-4-8-20260901", "claude-opus-4-6"} {
		if !anthropicSpeedNeedsBeta(model) {
			t.Fatalf("missing beta for %s", model)
		}
	}
	if anthropicSpeedNeedsBeta("claude-opus-5-5") {
		t.Fatal("GA model should not need beta")
	}
}
