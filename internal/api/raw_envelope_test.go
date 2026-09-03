package api

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestParseUnifiedRequestPreservesUnknownFields(t *testing.T) {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{
		"model":"gpt-5.4",
		"messages":[{"role":"user","content":"hello"}],
		"future_openai_field":{"enabled":true}
	}`))

	_, parsed, ok := parseUnifiedRequest(rec, req, false)
	if !ok {
		t.Fatalf("parse failed with status %d: %s", rec.Code, rec.Body.String())
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(parsed.RawJSON, &raw); err != nil {
		t.Fatalf("decode preserved payload: %v", err)
	}
	if _, ok := raw["future_openai_field"]; !ok {
		t.Fatal("unknown field was not preserved")
	}
}

func TestResponsesRawMapUsesPreservedEnvelope(t *testing.T) {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(`{
		"model":"gpt-5.4",
		"input":"hello",
		"include":["reasoning.encrypted_content"]
	}`))

	parsed, ok := parseResponsesRequest(rec, req)
	if !ok {
		t.Fatalf("parse failed with status %d: %s", rec.Code, rec.Body.String())
	}
	payload, err := responsesRequestToRawMap(parsed)
	if err != nil {
		t.Fatalf("responsesRequestToRawMap: %v", err)
	}
	if _, ok := payload["include"]; !ok {
		t.Fatal("include field was lost from the Responses envelope")
	}
}

func TestParseEmbeddingsRequestPreservesUnknownFields(t *testing.T) {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/v1/embeddings", strings.NewReader(`{
		"model":"text-embedding-3-small",
		"input":"hello",
		"future_openai_field":"kept"
	}`))

	_, parsed, ok := parseEmbeddingsRequest(rec, req, false)
	if !ok {
		t.Fatalf("parse failed with status %d: %s", rec.Code, rec.Body.String())
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(parsed.RawJSON, &raw); err != nil {
		t.Fatalf("decode preserved payload: %v", err)
	}
	if _, ok := raw["future_openai_field"]; !ok {
		t.Fatal("unknown field was not preserved")
	}
}
