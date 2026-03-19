package providers

import (
	"context"
	"encoding/json"
	"io"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestOpenAITranslator_StreamingRequestIncludesUsage(t *testing.T) {
	translator := NewOpenAITranslator(config.ProviderConfig{
		APIKey:  "dummy",
		BaseURL: "https://api.openai.com/v1",
	})

	req, err := translator.TranslateRequest(context.Background(), &models.UnifiedRequest{
		Model:    "gpt-5.4",
		Stream:   true,
		Messages: []models.Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("TranslateRequest returned error: %v", err)
	}

	body, err := io.ReadAll(req.Body)
	if err != nil {
		t.Fatalf("failed to read request body: %v", err)
	}

	var payload models.UnifiedRequest
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("failed to unmarshal request payload: %v", err)
	}

	if !payload.Stream {
		t.Fatalf("expected stream=true in upstream payload")
	}
	if payload.StreamOptions == nil {
		t.Fatalf("expected stream_options to be present in upstream payload")
	}
	if !payload.StreamOptions.IncludeUsage {
		t.Fatalf("expected stream_options.include_usage=true in upstream payload")
	}
}
