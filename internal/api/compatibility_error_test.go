package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestWriteCompatibilityErrorIncludesFieldAndProvider(t *testing.T) {
	rec := httptest.NewRecorder()
	writeCompatibilityError(rec, &models.CompatibilityError{
		Field:    "tools[0].type",
		Provider: "anthropic-primary",
		Reason:   "hosted tools require native Responses support",
	})

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusBadRequest)
	}
	var response models.ErrorResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if response.Error.Type != "invalid_request_error" {
		t.Fatalf("type = %q", response.Error.Type)
	}
	if response.Error.Param == nil || *response.Error.Param != "tools[0].type" {
		t.Fatalf("param = %#v", response.Error.Param)
	}
	if response.Error.Code == nil || *response.Error.Code != "unsupported_feature" {
		t.Fatalf("code = %#v", response.Error.Code)
	}
	if got := response.Error.Message; got != `field "tools[0].type" is not supported by provider "anthropic-primary": hosted tools require native Responses support` {
		t.Fatalf("message = %q", got)
	}
}
