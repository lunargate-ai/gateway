package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestWriteAPIJSONUsesPreservedNativeResponse(t *testing.T) {
	raw := json.RawMessage(`{"id":"chatcmpl_1","object":"chat.completion","created":1,"model":"gpt-5.4","choices":[],"service_tier":"priority"}`)
	rec := httptest.NewRecorder()
	writeAPIJSON(rec, http.StatusOK, &models.UnifiedResponse{
		RawJSON: raw,
		ID:      "chatcmpl_1",
		Object:  "chat.completion",
	})

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d", rec.Code)
	}
	var payload map[string]interface{}
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if payload["service_tier"] != "priority" {
		t.Fatalf("service_tier was not preserved: %#v", payload)
	}
}

func TestWriteAPIJSONFallsBackForGeneratedResponse(t *testing.T) {
	rec := httptest.NewRecorder()
	writeAPIJSON(rec, http.StatusOK, &models.UnifiedResponse{
		ID:      "chatcmpl_1",
		Object:  "chat.completion",
		Created: 1,
		Model:   "translated",
		Choices: []models.Choice{},
	})

	var response models.UnifiedResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode generated response: %v", err)
	}
	if response.Model != "translated" {
		t.Fatalf("model = %q", response.Model)
	}
}
