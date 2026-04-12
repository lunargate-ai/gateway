package api

import (
	"encoding/json"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestResolveCollectorInferenceParameters_UsesProviderDefaultsAndRequestOverrides(t *testing.T) {
	temperature := 1.0
	topP := 0.95
	topK := 64

	providerCfgs := map[string]config.ProviderConfig{
		"openai-defaults": {
			Type:        "openai",
			Temperature: &temperature,
			TopP:        &topP,
			TopK:        &topK,
		},
		"ollama-defaults": {
			Type:        "ollama",
			Temperature: &temperature,
			TopP:        &topP,
			TopK:        &topK,
		},
	}

	h := &Handler{
		registry:        providers.NewRegistry(providerCfgs),
		providerClients: newProviderClientRegistry(providerCfgs),
	}

	gotOpenAI := h.resolveCollectorInferenceParameters("openai-defaults", &models.UnifiedRequest{})
	if gotOpenAI.Temperature == nil || *gotOpenAI.Temperature != 1.0 {
		t.Fatalf("expected openai temperature=1.0, got %#v", gotOpenAI.Temperature)
	}
	if gotOpenAI.TopP == nil || *gotOpenAI.TopP != 0.95 {
		t.Fatalf("expected openai top_p=0.95, got %#v", gotOpenAI.TopP)
	}
	if gotOpenAI.TopK != nil {
		t.Fatalf("expected openai top_k to be omitted, got %#v", gotOpenAI.TopK)
	}

	gotOllama := h.resolveCollectorInferenceParameters("ollama-defaults", &models.UnifiedRequest{})
	if gotOllama.TopK == nil || *gotOllama.TopK != 64 {
		t.Fatalf("expected ollama top_k=64, got %#v", gotOllama.TopK)
	}

	reqTemp := 0.3
	reqTopP := 0.7
	reqTopK := 12
	gotRequestOverrides := h.resolveCollectorInferenceParameters("ollama-defaults", &models.UnifiedRequest{
		Temperature: &reqTemp,
		TopP:        &reqTopP,
		TopK:        &reqTopK,
	})
	if gotRequestOverrides.Temperature == nil || *gotRequestOverrides.Temperature != 0.3 {
		t.Fatalf("expected request temperature override, got %#v", gotRequestOverrides.Temperature)
	}
	if gotRequestOverrides.TopP == nil || *gotRequestOverrides.TopP != 0.7 {
		t.Fatalf("expected request top_p override, got %#v", gotRequestOverrides.TopP)
	}
	if gotRequestOverrides.TopK == nil || *gotRequestOverrides.TopK != 12 {
		t.Fatalf("expected request top_k override, got %#v", gotRequestOverrides.TopK)
	}
}

func TestBuildCollectorRequestLogPayload_AddsInferenceMetadata(t *testing.T) {
	payload := []byte(`{"model":"test","messages":[{"role":"user","content":"hi"}]}`)
	temperature := 1.0
	topP := 0.95
	topK := 64

	reqAny := buildCollectorRequestLogPayload(payload, collectorInferenceParameters{
		Temperature: &temperature,
		TopP:        &topP,
		TopK:        &topK,
	})

	reqObj, ok := reqAny.(map[string]interface{})
	if !ok {
		t.Fatalf("expected request payload map, got %T", reqAny)
	}
	metaObj, ok := reqObj["_lunargate"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected _lunargate metadata map, got %#v", reqObj["_lunargate"])
	}
	inferenceObj, ok := metaObj["inference_parameters"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected inference_parameters map, got %#v", metaObj["inference_parameters"])
	}
	if got, ok := inferenceObj["temperature"].(float64); !ok || got != 1.0 {
		t.Fatalf("expected temperature=1.0, got %#v", inferenceObj["temperature"])
	}
	if got, ok := inferenceObj["top_p"].(float64); !ok || got != 0.95 {
		t.Fatalf("expected top_p=0.95, got %#v", inferenceObj["top_p"])
	}
	switch got := inferenceObj["top_k"].(type) {
	case int:
		if got != 64 {
			t.Fatalf("expected top_k=64, got %#v", inferenceObj["top_k"])
		}
	case float64:
		if got != 64 {
			t.Fatalf("expected top_k=64, got %#v", inferenceObj["top_k"])
		}
	default:
		t.Fatalf("expected top_k to be numeric, got %#v", inferenceObj["top_k"])
	}

	withoutInference := buildCollectorRequestLogPayload(payload, collectorInferenceParameters{})
	var baseline map[string]interface{}
	if err := json.Unmarshal(payload, &baseline); err != nil {
		t.Fatalf("failed to unmarshal baseline payload: %v", err)
	}
	if got, ok := withoutInference.(map[string]interface{}); !ok || len(got) != len(baseline) {
		t.Fatalf("expected unchanged payload without inference parameters, got %#v", withoutInference)
	}
}

func TestEnrichCollectorTagsWithInference_AppendsSamplingTags(t *testing.T) {
	h := &Handler{registry: providers.NewRegistry(map[string]config.ProviderConfig{})}
	headers := map[string]string{"x-team": "platform"}
	temperature := 1.0
	topP := 0.95
	topK := 64

	tags := h.enrichCollectorTagsWithInference(headers, "openai", "openai/gpt-4.1", true, collectorInferenceParameters{
		Temperature: &temperature,
		TopP:        &topP,
		TopK:        &topK,
	})

	if tags["x-team"] != "platform" {
		t.Fatalf("expected original header tags to be preserved, got %#v", tags["x-team"])
	}
	if tags["x-lunargate-inference-temperature"] != "1" {
		t.Fatalf("expected temperature tag to equal 1, got %q", tags["x-lunargate-inference-temperature"])
	}
	if tags["x-lunargate-inference-top-p"] != "0.95" {
		t.Fatalf("expected top_p tag to equal 0.95, got %q", tags["x-lunargate-inference-top-p"])
	}
	if tags["x-lunargate-inference-top-k"] != "64" {
		t.Fatalf("expected top_k tag to equal 64, got %q", tags["x-lunargate-inference-top-k"])
	}
}
