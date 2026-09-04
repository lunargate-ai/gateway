package providers

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestTranslatedChatTargetsRejectLossyNestedControls(t *testing.T) {
	tests := []struct {
		name      string
		raw       string
		wantField string
	}{
		{
			name:      "stream obfuscation",
			raw:       `{"model":"model","messages":[],"stream_options":{"include_obfuscation":true}}`,
			wantField: "stream_options.include_obfuscation",
		},
		{
			name:      "reasoning summary",
			raw:       `{"model":"model","messages":[],"reasoning":{"effort":"high","summary":"auto"}}`,
			wantField: "reasoning.summary",
		},
		{
			name:      "conflicting reasoning effort",
			raw:       `{"model":"model","messages":[],"reasoning_effort":"low","reasoning":{"effort":"high"}}`,
			wantField: "reasoning.effort",
		},
	}

	validators := []struct {
		name       string
		providerID string
		validate   func(string, *models.UnifiedRequest) error
	}{
		{
			name:       "anthropic",
			providerID: "anthropic-backup",
			validate:   NewAnthropicTranslator(config.ProviderConfig{APIKey: "dummy"}).ValidateRequestCompatibility,
		},
		{
			name:       "ollama",
			providerID: "ollama-backup",
			validate:   NewOllamaTranslator(config.ProviderConfig{}).ValidateRequestCompatibility,
		},
	}

	for _, validator := range validators {
		for _, tt := range tests {
			t.Run(validator.name+"/"+tt.name, func(t *testing.T) {
				err := validator.validate(validator.providerID, &models.UnifiedRequest{
					SourceRequestType: "chat_completions",
					RawJSON:           json.RawMessage(tt.raw),
				})
				var compatibilityErr *models.CompatibilityError
				if !errors.As(err, &compatibilityErr) {
					t.Fatalf("error = %v, want CompatibilityError", err)
				}
				if compatibilityErr.Field != tt.wantField || compatibilityErr.Provider != validator.providerID {
					t.Fatalf("compatibility error = %#v, want field=%q provider=%q", compatibilityErr, tt.wantField, validator.providerID)
				}
			})
		}
	}
}

func TestTranslatedChatTargetsAllowMappedNestedControls(t *testing.T) {
	raw := json.RawMessage(`{
		"model":"model",
		"messages":[],
		"stream_options":{"include_usage":true},
		"reasoning_effort":"high",
		"reasoning":{"effort":"high"}
	}`)
	request := &models.UnifiedRequest{
		SourceRequestType: "chat_completions",
		RawJSON:           raw,
	}

	if err := validateTranslatedChatRawControls("translated", request); err != nil {
		t.Fatalf("mapped nested controls were rejected: %v", err)
	}
}

func TestTranslatedChatTargetsRejectUnknownMessageMembers(t *testing.T) {
	tests := []struct {
		name      string
		raw       string
		wantField string
	}{
		{
			name:      "assistant audio",
			raw:       `{"model":"model","messages":[{"role":"assistant","audio":{"id":"audio_1"}}]}`,
			wantField: "messages[0].audio",
		},
		{
			name:      "reasoning alias",
			raw:       `{"model":"model","messages":[{"role":"assistant","reasoning":"private"}]}`,
			wantField: "messages[0].reasoning",
		},
		{
			name:      "text cache breakpoint",
			raw:       `{"model":"model","messages":[{"role":"user","content":[{"type":"text","text":"hi","prompt_cache_breakpoint":{"type":"ephemeral"}}]}]}`,
			wantField: "messages[0].content[0].prompt_cache_breakpoint",
		},
		{
			name:      "image reference option",
			raw:       `{"model":"model","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,aQ==","resize":"fit"}}]}]}`,
			wantField: "messages[0].content[0].image_url.resize",
		},
		{
			name:      "tool call option",
			raw:       `{"model":"model","messages":[{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}","future":"x"}}]}]}`,
			wantField: "messages[0].tool_calls[0].function.future",
		},
		{
			name:      "legacy function call option",
			raw:       `{"model":"model","messages":[{"role":"assistant","function_call":{"name":"lookup","arguments":"{}","future":"x"}}]}`,
			wantField: "messages[0].function_call.future",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateTranslatedChatRawControls("translated", &models.UnifiedRequest{
				SourceRequestType: "chat_completions",
				RawJSON:           json.RawMessage(tt.raw),
			})
			var compatibilityErr *models.CompatibilityError
			if !errors.As(err, &compatibilityErr) {
				t.Fatalf("error = %v, want CompatibilityError", err)
			}
			if compatibilityErr.Field != tt.wantField {
				t.Fatalf("field = %q, want %q", compatibilityErr.Field, tt.wantField)
			}
		})
	}
}

func TestTranslatedChatTargetsRejectUnknownToolMembers(t *testing.T) {
	tests := []struct {
		name      string
		raw       string
		wantField string
	}{
		{
			name:      "tool wrapper",
			raw:       `{"model":"model","messages":[],"tools":[{"type":"function","function":{"name":"lookup"},"future":"x"}]}`,
			wantField: "tools[0].future",
		},
		{
			name:      "function definition",
			raw:       `{"model":"model","messages":[],"tools":[{"type":"function","function":{"name":"lookup","future":"x"}}]}`,
			wantField: "tools[0].function.future",
		},
		{
			name:      "legacy function definition",
			raw:       `{"model":"model","messages":[],"functions":[{"name":"lookup","future":"x"}]}`,
			wantField: "functions[0].future",
		},
		{
			name:      "tool choice wrapper",
			raw:       `{"model":"model","messages":[],"tool_choice":{"type":"function","function":{"name":"lookup"},"future":"x"}}`,
			wantField: "tool_choice.future",
		},
		{
			name:      "tool choice function",
			raw:       `{"model":"model","messages":[],"tool_choice":{"type":"function","function":{"name":"lookup","future":"x"}}}`,
			wantField: "tool_choice.function.future",
		},
		{
			name:      "legacy function choice",
			raw:       `{"model":"model","messages":[],"function_call":{"name":"lookup","future":"x"}}`,
			wantField: "function_call.future",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateTranslatedChatRawControls("translated", &models.UnifiedRequest{
				SourceRequestType: "chat_completions",
				RawJSON:           json.RawMessage(tt.raw),
			})
			var compatibilityErr *models.CompatibilityError
			if !errors.As(err, &compatibilityErr) || compatibilityErr.Field != tt.wantField {
				t.Fatalf("error = %#v, want field %q", err, tt.wantField)
			}
		})
	}
}

func TestTranslatedChatTargetsRejectLossyToolChoiceShapes(t *testing.T) {
	tests := []struct {
		name      string
		raw       string
		wantField string
	}{
		{
			name:      "function ignored by auto mode",
			raw:       `{"model":"model","messages":[],"tool_choice":{"type":"auto","function":{"name":"lookup"}}}`,
			wantField: "tool_choice.function",
		},
		{
			name:      "non-function object mode",
			raw:       `{"model":"model","messages":[],"tool_choice":{"type":"none"}}`,
			wantField: "tool_choice.type",
		},
		{
			name:      "missing object type",
			raw:       `{"model":"model","messages":[],"tool_choice":{"function":{"name":"lookup"}}}`,
			wantField: "tool_choice.type",
		},
		{
			name:      "missing function object",
			raw:       `{"model":"model","messages":[],"tool_choice":{"type":"function"}}`,
			wantField: "tool_choice.function",
		},
	}

	validators := []struct {
		name       string
		providerID string
		validate   func(string, *models.UnifiedRequest) error
	}{
		{
			name:       "anthropic",
			providerID: "anthropic-backup",
			validate:   NewAnthropicTranslator(config.ProviderConfig{APIKey: "dummy"}).ValidateRequestCompatibility,
		},
		{
			name:       "ollama",
			providerID: "ollama-backup",
			validate:   NewOllamaTranslator(config.ProviderConfig{}).ValidateRequestCompatibility,
		},
	}

	for _, validator := range validators {
		for _, tt := range tests {
			t.Run(validator.name+"/"+tt.name, func(t *testing.T) {
				err := validator.validate(validator.providerID, &models.UnifiedRequest{
					SourceRequestType: "chat_completions",
					RawJSON:           json.RawMessage(tt.raw),
				})
				var compatibilityErr *models.CompatibilityError
				if !errors.As(err, &compatibilityErr) {
					t.Fatalf("error = %v, want CompatibilityError", err)
				}
				if compatibilityErr.Field != tt.wantField || compatibilityErr.Provider != validator.providerID {
					t.Fatalf("compatibility error = %#v, want field=%q provider=%q", compatibilityErr, tt.wantField, validator.providerID)
				}
			})
		}
	}
}

func TestTranslatedChatTargetsRejectUnknownResponseFormatMembers(t *testing.T) {
	tests := []struct {
		name      string
		raw       string
		wantField string
	}{
		{
			name:      "format wrapper",
			raw:       `{"model":"model","messages":[],"response_format":{"type":"json_schema","json_schema":{"name":"answer","schema":{"type":"object"}},"future":"x"}}`,
			wantField: "response_format.future",
		},
		{
			name:      "json schema wrapper",
			raw:       `{"model":"model","messages":[],"response_format":{"type":"json_schema","json_schema":{"name":"answer","schema":{"type":"object"},"future":"x"}}}`,
			wantField: "response_format.json_schema.future",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateTranslatedChatRawControls("translated", &models.UnifiedRequest{
				SourceRequestType: "chat_completions",
				RawJSON:           json.RawMessage(tt.raw),
			})
			var compatibilityErr *models.CompatibilityError
			if !errors.As(err, &compatibilityErr) || compatibilityErr.Field != tt.wantField {
				t.Fatalf("error = %#v, want field %q", err, tt.wantField)
			}
		})
	}
}
