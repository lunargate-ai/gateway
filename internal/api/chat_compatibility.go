package api

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"

	"github.com/lunargate-ai/gateway/internal/routing"
	"github.com/lunargate-ai/gateway/pkg/models"
	"github.com/rs/zerolog/log"
)

func (h *Handler) validateChatCompatibility(target routing.Target, req *models.UnifiedRequest) error {
	if h == nil || h.registry == nil || req == nil {
		return nil
	}
	providerID := target.Provider
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
	if strings.EqualFold(strings.TrimSpace(req.SourceRequestType), "responses") &&
		strings.TrimSpace(req.PreviousResponseID) != "" &&
		!strings.EqualFold(strings.TrimSpace(target.UpstreamRequestType), "responses") {
		return &models.CompatibilityError{
			Field:    "previous_response_id",
			Provider: providerID,
			Reason:   "response state is not available locally and this target does not provide native Responses continuation",
		}
	}

	for index, toolType := range rawRequestToolTypes(req.RawJSON) {
		if toolType == "" || toolType == "function" {
			continue
		}
		if !strings.EqualFold(strings.TrimSpace(target.UpstreamRequestType), "responses") || !h.providerSupportsHostedTool(providerID, toolType) {
			return &models.CompatibilityError{
				Field:    "tools[" + strconv.Itoa(index) + "].type",
				Provider: providerID,
				Reason:   fmt.Sprintf("hosted tool %q is not enabled for this target", toolType),
			}
		}
	}
	return nil
}

func (h *Handler) providerSupportsHostedTool(providerID string, toolType string) bool {
	capabilities, ok := h.registry.Capabilities(providerID)
	if !ok {
		return false
	}
	wanted := strings.ToLower(strings.TrimSpace(toolType))
	for _, supported := range capabilities.HostedTools {
		if strings.ToLower(strings.TrimSpace(supported)) == wanted {
			return true
		}
	}
	return false
}

func (h *Handler) compatibleChatFallbacks(fallbacks []routing.Target, req *models.UnifiedRequest) []routing.Target {
	if len(fallbacks) == 0 {
		return nil
	}
	compatible := make([]routing.Target, 0, len(fallbacks))
	for _, target := range fallbacks {
		if err := h.validateChatCompatibility(target, req); err != nil {
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

func rawRequestToolTypes(raw json.RawMessage) []string {
	if len(bytes.TrimSpace(raw)) == 0 {
		return nil
	}
	var payload struct {
		Tools []struct {
			Type string `json:"type"`
		} `json:"tools"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil
	}
	types := make([]string, 0, len(payload.Tools))
	for _, tool := range payload.Tools {
		types = append(types, strings.ToLower(strings.TrimSpace(tool.Type)))
	}
	return types
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
