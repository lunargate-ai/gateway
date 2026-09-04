package api

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/pkg/models"
)

const precisionTestInteger = "9007199254740993"

type precisionRequestTranslator interface {
	TranslateRequest(context.Context, *models.UnifiedRequest) (*http.Request, error)
}

func TestDecodeJSONStrictPreservesLargeInteger(t *testing.T) {
	var payload struct {
		Value interface{} `json:"value"`
	}
	if err := decodeJSONStrict(strings.NewReader(`{"value":`+precisionTestInteger+`}`), &payload); err != nil {
		t.Fatalf("decodeJSONStrict returned error: %v", err)
	}
	assertPrecisionNumber(t, payload.Value)
}

func TestDecodeJSONStrictRejectsTrailingValue(t *testing.T) {
	var payload interface{}
	if err := decodeJSONStrict(strings.NewReader(`{"value":1} {}`), &payload); err == nil {
		t.Fatal("decodeJSONStrict accepted a trailing JSON value")
	}
}

func TestStrictChatDecodePreservesLargeIntegerThroughTranslatedProviders(t *testing.T) {
	raw := []byte(`{
		"model":"test-model",
		"messages":[{"role":"user","content":"hello"}],
		"tools":[{"type":"function","function":{
			"name":"lookup",
			"parameters":{"type":"object","properties":{"id":{"type":"integer","const":` + precisionTestInteger + `}}}
		}}]
	}`)

	tests := []struct {
		name       string
		translator precisionRequestTranslator
		extract    func(*testing.T, map[string]interface{}) interface{}
	}{
		{
			name: "anthropic",
			translator: providers.NewAnthropicTranslator(config.ProviderConfig{
				APIKey: "test-key",
			}),
			extract: func(t *testing.T, body map[string]interface{}) interface{} {
				tool := precisionObject(t, precisionArray(t, body["tools"])[0])
				schema := precisionObject(t, tool["input_schema"])
				property := precisionObject(t, precisionObject(t, schema["properties"])["id"])
				return property["const"]
			},
		},
		{
			name:       "ollama",
			translator: providers.NewOllamaTranslator(config.ProviderConfig{}),
			extract: func(t *testing.T, body map[string]interface{}) interface{} {
				tool := precisionObject(t, precisionArray(t, body["tools"])[0])
				function := precisionObject(t, tool["function"])
				schema := precisionObject(t, function["parameters"])
				property := precisionObject(t, precisionObject(t, schema["properties"])["id"])
				return property["const"]
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			request := decodePrecisionChatRequest(t, raw)
			body := translatePrecisionRequest(t, tt.translator, request)
			assertPrecisionNumber(t, tt.extract(t, body))
		})
	}
}

func TestStrictResponsesDecodePreservesLargeSchemaIntegerThroughTranslatedProviders(t *testing.T) {
	raw := []byte(`{
		"model":"test-model",
		"input":"hello",
		"text":{"format":{
			"type":"json_schema",
			"name":"payload",
			"schema":{"type":"object","properties":{"id":{"type":"integer","const":` + precisionTestInteger + `}}}
		}}
	}`)

	tests := []struct {
		name       string
		translator precisionRequestTranslator
		extract    func(*testing.T, map[string]interface{}) interface{}
	}{
		{
			name: "anthropic",
			translator: providers.NewAnthropicTranslator(config.ProviderConfig{
				APIKey: "test-key",
				Capabilities: config.ProviderCapabilities{
					StructuredOutputs: true,
				},
			}),
			extract: func(t *testing.T, body map[string]interface{}) interface{} {
				outputConfig := precisionObject(t, body["output_config"])
				format := precisionObject(t, outputConfig["format"])
				schema := precisionObject(t, format["schema"])
				property := precisionObject(t, precisionObject(t, schema["properties"])["id"])
				return property["const"]
			},
		},
		{
			name:       "ollama",
			translator: providers.NewOllamaTranslator(config.ProviderConfig{}),
			extract: func(t *testing.T, body map[string]interface{}) interface{} {
				schema := precisionObject(t, body["format"])
				property := precisionObject(t, precisionObject(t, schema["properties"])["id"])
				return property["const"]
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			request := decodePrecisionResponsesRequest(t, raw)
			body := translatePrecisionRequest(t, tt.translator, request)
			assertPrecisionNumber(t, tt.extract(t, body))
		})
	}
}

func TestStrictResponsesDecodePreservesLargeToolInputForAnthropic(t *testing.T) {
	raw := []byte(`{
		"model":"test-model",
		"input":[{
			"type":"function_call",
			"call_id":"call_1",
			"name":"lookup",
			"arguments":"{\"id\":` + precisionTestInteger + `}"
		}]
	}`)

	request := decodePrecisionResponsesRequest(t, raw)
	body := translatePrecisionRequest(t, providers.NewAnthropicTranslator(config.ProviderConfig{
		APIKey: "test-key",
	}), request)
	message := precisionObject(t, precisionArray(t, body["messages"])[0])
	block := precisionObject(t, precisionArray(t, message["content"])[0])
	input := precisionObject(t, block["input"])
	assertPrecisionNumber(t, input["id"])
}

func decodePrecisionChatRequest(t *testing.T, raw []byte) *models.UnifiedRequest {
	t.Helper()
	var request models.UnifiedRequest
	if err := decodeJSONStrict(strings.NewReader(string(raw)), &request); err != nil {
		t.Fatalf("decode chat request: %v", err)
	}
	request.RawJSON = append(json.RawMessage(nil), raw...)
	request.SourceRequestType = "chat_completions"
	if err := models.NormalizeUnifiedRequest(&request); err != nil {
		t.Fatalf("normalize chat request: %v", err)
	}
	return &request
}

func decodePrecisionResponsesRequest(t *testing.T, raw []byte) *models.UnifiedRequest {
	t.Helper()
	var request models.ResponsesRequest
	if err := decodeJSONStrict(strings.NewReader(string(raw)), &request); err != nil {
		t.Fatalf("decode Responses request: %v", err)
	}
	request.RawJSON = append(json.RawMessage(nil), raw...)
	unified, err := models.ResponsesToUnifiedRequest(&request)
	if err != nil {
		t.Fatalf("translate Responses request: %v", err)
	}
	if err := models.NormalizeUnifiedRequest(unified); err != nil {
		t.Fatalf("normalize Responses request: %v", err)
	}
	return unified
}

func translatePrecisionRequest(
	t *testing.T,
	translator precisionRequestTranslator,
	request *models.UnifiedRequest,
) map[string]interface{} {
	t.Helper()
	httpRequest, err := translator.TranslateRequest(context.Background(), request)
	if err != nil {
		t.Fatalf("TranslateRequest returned error: %v", err)
	}
	defer httpRequest.Body.Close()
	var body map[string]interface{}
	if err := decodeJSONStrict(httpRequest.Body, &body); err != nil {
		t.Fatalf("decode translated body: %v", err)
	}
	return body
}

func precisionObject(t *testing.T, value interface{}) map[string]interface{} {
	t.Helper()
	object, ok := value.(map[string]interface{})
	if !ok {
		t.Fatalf("value = %#v (%T), want JSON object", value, value)
	}
	return object
}

func precisionArray(t *testing.T, value interface{}) []interface{} {
	t.Helper()
	array, ok := value.([]interface{})
	if !ok || len(array) == 0 {
		t.Fatalf("value = %#v (%T), want non-empty JSON array", value, value)
	}
	return array
}

func assertPrecisionNumber(t *testing.T, value interface{}) {
	t.Helper()
	number, ok := value.(json.Number)
	if !ok || number.String() != precisionTestInteger {
		t.Fatalf("number = %#v (%T), want json.Number(%s)", value, value, precisionTestInteger)
	}
}
