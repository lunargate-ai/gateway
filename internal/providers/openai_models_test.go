package providers

import (
	"slices"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
)

func TestOpenAITranslatorDefaultModel(t *testing.T) {
	t.Run("uses current fallback", func(t *testing.T) {
		translator := NewOpenAITranslator(config.ProviderConfig{})
		if got := translator.DefaultModel(); got != "gpt-5.6-terra" {
			t.Fatalf("DefaultModel() = %q, want %q", got, "gpt-5.6-terra")
		}
	})

	t.Run("preserves configured model", func(t *testing.T) {
		translator := NewOpenAITranslator(config.ProviderConfig{DefaultModel: "custom-model"})
		if got := translator.DefaultModel(); got != "custom-model" {
			t.Fatalf("DefaultModel() = %q, want %q", got, "custom-model")
		}
	})
}

func TestOpenAITranslatorStaticModelsMatchVerifiedCatalog(t *testing.T) {
	translator := NewOpenAITranslator(config.ProviderConfig{})
	models := translator.Models()
	modelIDs := make([]string, 0, len(models))
	for _, model := range models {
		modelIDs = append(modelIDs, model.ID)
		if model.Object != "model" {
			t.Errorf("model %q Object = %q, want %q", model.ID, model.Object, "model")
		}
		if model.OwnedBy != "openai" {
			t.Errorf("model %q OwnedBy = %q, want %q", model.ID, model.OwnedBy, "openai")
		}
	}

	want := []string{
		"gpt-5.6-terra",
		"gpt-5.6-sol",
		"gpt-5.6-luna",
		"gpt-5.5",
		"gpt-5.4",
		"gpt-5.4-mini",
		"gpt-5.2",
		"gpt-4o",
		"gpt-4o-mini",
		"text-embedding-3-small",
		"text-embedding-3-large",
	}
	if !slices.Equal(modelIDs, want) {
		t.Fatalf("static model IDs = %#v, want %#v", modelIDs, want)
	}
}
