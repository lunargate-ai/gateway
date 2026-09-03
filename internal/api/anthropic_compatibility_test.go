package api

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/internal/routing"
	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestValidateChatCompatibilityUsesExactAnthropicTarget(t *testing.T) {
	handler := &Handler{registry: providers.NewRegistry(map[string]config.ProviderConfig{
		"anthropic-primary": {Type: "anthropic"},
	})}
	penalty := 0.25

	err := handler.validateChatCompatibility(
		routing.Target{Provider: "anthropic-primary"},
		&models.UnifiedRequest{PresencePenalty: &penalty},
	)
	var compatibilityErr *models.CompatibilityError
	if !errors.As(err, &compatibilityErr) {
		t.Fatalf("error = %v, want CompatibilityError", err)
	}
	if compatibilityErr.Field != "presence_penalty" || compatibilityErr.Provider != "anthropic-primary" {
		t.Fatalf("compatibility error = %#v", compatibilityErr)
	}
}

func TestCompatibleChatFallbacksFiltersAnthropicWithoutChangingSemantics(t *testing.T) {
	handler := &Handler{registry: providers.NewRegistry(map[string]config.ProviderConfig{
		"anthropic-backup": {Type: "anthropic"},
		"openai-backup":    {Type: "openai"},
	})}
	penalty := 0.25
	fallbacks := []routing.Target{
		{Provider: "anthropic-backup", Model: "claude-sonnet-4-5"},
		{Provider: "openai-backup", Model: "gpt-5.4"},
	}

	got := handler.compatibleChatFallbacks(fallbacks, &models.UnifiedRequest{PresencePenalty: &penalty})
	if len(got) != 1 || got[0].Provider != "openai-backup" {
		t.Fatalf("compatible fallbacks = %#v, want only openai-backup", got)
	}
}

func TestCompatibleChatFallbacksUsesAnthropicCapabilitiesPerTarget(t *testing.T) {
	handler := &Handler{registry: providers.NewRegistry(map[string]config.ProviderConfig{
		"anthropic-disabled": {Type: "anthropic"},
		"anthropic-enabled": {
			Type: "anthropic",
			Capabilities: config.ProviderCapabilities{
				StructuredOutputs: true,
			},
		},
	})}
	req := &models.UnifiedRequest{ResponseFormat: &models.ResponseFormat{
		Type: "json_schema",
		JSONSchema: &models.JSONSchemaResponseFormat{
			Schema: map[string]interface{}{"type": "object"},
		},
	}}
	fallbacks := []routing.Target{
		{Provider: "anthropic-disabled", Model: "claude-sonnet-4-5"},
		{Provider: "anthropic-enabled", Model: "claude-opus-5"},
	}

	got := handler.compatibleChatFallbacks(fallbacks, req)
	if len(got) != 1 || got[0].Provider != "anthropic-enabled" {
		t.Fatalf("compatible fallbacks = %#v, want only anthropic-enabled", got)
	}
}

func TestStrictChatDecodeAcceptsJSONSchemaResponseFormat(t *testing.T) {
	var req models.UnifiedRequest
	err := decodeJSONStrict(strings.NewReader(`{
		"model":"claude-opus-5",
		"messages":[],
		"response_format":{
			"type":"json_schema",
			"json_schema":{
				"name":"answer",
				"description":"structured answer",
				"strict":true,
				"schema":{"type":"object"}
			}
		}
	}`), &req)
	if err != nil {
		t.Fatalf("strict decode rejected standard json_schema payload: %v", err)
	}
	if req.ResponseFormat == nil || req.ResponseFormat.JSONSchema == nil || req.ResponseFormat.JSONSchema.Name != "answer" {
		t.Fatalf("decoded response_format = %#v", req.ResponseFormat)
	}
	schema, ok := req.ResponseFormat.JSONSchema.Schema.(map[string]interface{})
	if !ok || schema["type"] != "object" {
		encoded, _ := json.Marshal(req.ResponseFormat.JSONSchema.Schema)
		t.Fatalf("decoded schema = %s", encoded)
	}
}
