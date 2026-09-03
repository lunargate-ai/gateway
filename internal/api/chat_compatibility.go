package api

import (
	"bytes"
	"encoding/json"
	"errors"
	"strings"

	"github.com/lunargate-ai/gateway/internal/routing"
	"github.com/lunargate-ai/gateway/pkg/models"
	"github.com/rs/zerolog/log"
)

func (h *Handler) validateChatCompatibility(providerID string, req *models.UnifiedRequest) error {
	if h == nil || h.registry == nil || req == nil {
		return nil
	}
	providerType, ok := h.registry.Type(providerID)
	if !ok {
		return nil
	}
	if strings.EqualFold(providerType, "openai") && rawJSONObjectHasField(req.RawJSON, "top_k") {
		return &models.CompatibilityError{
			Field:    "top_k",
			Provider: providerID,
			Reason:   "OpenAI Chat Completions does not define top_k",
		}
	}
	return nil
}

func (h *Handler) compatibleChatFallbacks(fallbacks []routing.Target, req *models.UnifiedRequest) []routing.Target {
	if len(fallbacks) == 0 {
		return nil
	}
	compatible := make([]routing.Target, 0, len(fallbacks))
	for _, target := range fallbacks {
		if err := h.validateChatCompatibility(target.Provider, req); err != nil {
			event := log.Warn().
				Err(err).
				Str("provider", target.Provider).
				Str("model", target.Model)
			var compatibilityErr *models.CompatibilityError
			if errors.As(err, &compatibilityErr) {
				event = event.Str("field", compatibilityErr.Field)
			}
			event.Msg("skipping incompatible fallback target")
			continue
		}
		compatible = append(compatible, target)
	}
	return compatible
}

func rawJSONObjectHasField(raw json.RawMessage, field string) bool {
	if len(bytes.TrimSpace(raw)) == 0 || strings.TrimSpace(field) == "" {
		return false
	}
	var payload map[string]json.RawMessage
	if err := json.Unmarshal(raw, &payload); err != nil {
		return false
	}
	_, ok := payload[field]
	return ok
}
