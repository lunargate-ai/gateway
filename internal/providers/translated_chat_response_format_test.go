package providers

import (
	"context"
	"errors"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestTranslatedChatTargetsAnnotateNativeJSONSchema(t *testing.T) {
	for _, strict := range []bool{false, true} {
		t.Run(map[bool]string{false: "strict_false", true: "strict_true"}[strict], func(t *testing.T) {
			original := map[string]interface{}{
				"type":       "object",
				"properties": map[string]interface{}{"answer": map[string]interface{}{"type": "string"}},
			}
			format := &models.ResponseFormat{
				Type: "json_schema",
				JSONSchema: &models.JSONSchemaResponseFormat{
					Name:        "answer",
					Description: "A structured answer",
					Schema:      original,
					Strict:      &strict,
				},
			}

			anthropic := NewAnthropicTranslator(config.ProviderConfig{
				APIKey:       "dummy",
				Capabilities: config.ProviderCapabilities{StructuredOutputs: true},
			})
			anthropicPayload := translateAnthropicRequest(t, anthropic, &models.UnifiedRequest{
				Model:          "claude",
				Messages:       []models.Message{{Role: "user", Content: "hi"}},
				ResponseFormat: format,
			})
			anthropicSchema := anthropicPayload.OutputConfig.Format.Schema.(map[string]interface{})
			assertTranslatedSchemaAnnotations(t, anthropicSchema)

			ollama := NewOllamaTranslator(config.ProviderConfig{})
			req, err := ollama.TranslateRequest(context.Background(), &models.UnifiedRequest{
				Model:          "qwen",
				Messages:       []models.Message{{Role: "user", Content: "hi"}},
				ResponseFormat: format,
			})
			if err != nil {
				t.Fatalf("Ollama TranslateRequest returned error: %v", err)
			}
			ollamaSchema := decodeOllamaRequestBody(t, req.Body)["format"].(map[string]interface{})
			assertTranslatedSchemaAnnotations(t, ollamaSchema)

			if _, mutated := original["title"]; mutated {
				t.Fatalf("input schema was mutated: %#v", original)
			}
			if _, mutated := original["description"]; mutated {
				t.Fatalf("input schema was mutated: %#v", original)
			}
		})
	}
}

func TestTranslatedChatTargetsRejectConflictingSchemaAnnotations(t *testing.T) {
	tests := []struct {
		name      string
		format    *models.JSONSchemaResponseFormat
		wantField string
	}{
		{
			name: "name conflict",
			format: &models.JSONSchemaResponseFormat{
				Name: "wrapper", Schema: map[string]interface{}{"type": "object", "title": "schema"},
			},
			wantField: "response_format.json_schema.name",
		},
		{
			name: "description type conflict",
			format: &models.JSONSchemaResponseFormat{
				Description: "wrapper", Schema: map[string]interface{}{"type": "object", "description": []interface{}{"schema"}},
			},
			wantField: "response_format.json_schema.description",
		},
	}
	validators := []struct {
		name     string
		validate func(string, *models.UnifiedRequest) error
	}{
		{
			name: "anthropic",
			validate: NewAnthropicTranslator(config.ProviderConfig{
				Capabilities: config.ProviderCapabilities{StructuredOutputs: true},
			}).ValidateRequestCompatibility,
		},
		{name: "ollama", validate: NewOllamaTranslator(config.ProviderConfig{}).ValidateRequestCompatibility},
	}

	for _, validator := range validators {
		for _, tt := range tests {
			t.Run(validator.name+"/"+tt.name, func(t *testing.T) {
				err := validator.validate(validator.name, &models.UnifiedRequest{
					ResponseFormat: &models.ResponseFormat{Type: "json_schema", JSONSchema: tt.format},
				})
				var compatibilityErr *models.CompatibilityError
				if !errors.As(err, &compatibilityErr) || compatibilityErr.Field != tt.wantField {
					t.Fatalf("error = %#v, want field %q", err, tt.wantField)
				}
			})
		}
	}
}

func TestTranslatedChatTargetsAcceptMatchingSchemaAnnotations(t *testing.T) {
	format := &models.JSONSchemaResponseFormat{
		Name:        "answer",
		Description: "A structured answer",
		Schema: map[string]interface{}{
			"type":        "object",
			"title":       "answer",
			"description": "A structured answer",
		},
	}
	schema, err := translatedChatAnnotatedJSONSchema("translated", format)
	if err != nil {
		t.Fatalf("matching annotations were rejected: %v", err)
	}
	assertTranslatedSchemaAnnotations(t, schema)
}

func assertTranslatedSchemaAnnotations(t *testing.T, schema map[string]interface{}) {
	t.Helper()
	if schema["title"] != "answer" || schema["description"] != "A structured answer" {
		t.Fatalf("schema annotations = %#v", schema)
	}
}
