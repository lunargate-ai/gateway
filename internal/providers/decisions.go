package providers

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image/jpeg"
	"image/png"
	"net/http"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/safeurl"
	"github.com/lunargate-ai/gateway/pkg/models"
)

// DecisionsModelEnabled deliberately requires an explicit native capability and
// model allowlist; API compatibility or a model-list response is not sufficient.
func DecisionsModelEnabled(providerType string, capabilities config.ProviderCapabilities, model string) bool {
	if !capabilities.Decisions {
		return false
	}
	model = strings.TrimSpace(model)
	if providerType == "openai" && model != "gpt-6-luna" {
		return false
	}
	if providerType == "typesafe" && !strings.HasPrefix(model, "jev-") {
		return false
	}
	if providerType != "openai" && providerType != "typesafe" {
		return false
	}
	for _, allowed := range capabilities.DecisionModels {
		if model == allowed {
			return true
		}
	}
	return false
}

func DecisionModelCapabilities(cfg config.ProviderConfig, model string) *models.ModelCapabilities {
	providerType := strings.ToLower(strings.TrimSpace(cfg.Type))
	enabled := DecisionsModelEnabled(providerType, cfg.Capabilities, model)
	if !enabled && providerType != "typesafe" {
		return nil
	}
	return &models.ModelCapabilities{Decisions: enabled, DecisionsOnly: providerType == "typesafe", DecisionImages: enabled && (providerType == "openai" || cfg.Capabilities.DecisionImages)}
}

func validateDecisionModel(provider, providerType, model string, cfg config.ProviderConfig, req *models.DecisionsRequest) error {
	if !DecisionsModelEnabled(providerType, cfg.Capabilities, model) {
		return &models.CompatibilityError{Provider: provider, Field: "model", Reason: "native Decisions support is not enabled for this model"}
	}
	if err := req.Validate(); err != nil {
		if e, ok := err.(*models.CompatibilityError); ok {
			copied := *e
			copied.Provider = provider
			return &copied
		}
		return err
	}
	return nil
}

func (t *OpenAITranslator) DecisionsUpstreamType() string { return "decisions" }
func (t *OpenAITranslator) ValidateDecisionsRequest(provider, model string, req *models.DecisionsRequest) error {
	if err := validateDecisionModel(provider, "openai", model, t.cfg, req); err != nil {
		return err
	}
	// Luna rejects duplicate supplied names. Filter that known native limit
	// before weighted routing or cross-provider fallback; unnamed items remain
	// ordered and do not conflict with each other.
	seen := map[string]bool{}
	for i, question := range req.Questions {
		if question.Name == nil {
			continue
		}
		if seen[*question.Name] {
			return &models.CompatibilityError{Provider: provider, Field: fmt.Sprintf("questions[%d].name", i), Reason: "Luna requires unique question names"}
		}
		seen[*question.Name] = true
	}
	return nil
}

func (t *OpenAITranslator) TranslateDecisionsRequest(ctx context.Context, req *models.DecisionsRequest) (*http.Request, error) {
	if err := t.ValidateDecisionsRequest("openai", req.Model, req); err != nil {
		return nil, err
	}
	var payload map[string]json.RawMessage
	if len(req.RawJSON) != 0 {
		if err := json.Unmarshal(req.RawJSON, &payload); err != nil {
			return nil, fmt.Errorf("decode preserved Decisions request: %w", err)
		}
	} else {
		body, err := json.Marshal(req)
		if err != nil {
			return nil, err
		}
		if err := json.Unmarshal(body, &payload); err != nil {
			return nil, err
		}
	}
	payload["model"], _ = json.Marshal(req.Model)
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("encode Decisions request: %w", err)
	}
	endpoint, err := safeurl.JoinHTTPPath(t.BaseURL(), "decisions")
	if err != nil {
		return nil, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+t.cfg.APIKey)
	if t.cfg.Organization != "" {
		httpReq.Header.Set("OpenAI-Organization", t.cfg.Organization)
	}
	applyUpstreamRequestHeaders(ctx, httpReq, "Idempotency-Key", "OpenAI-Beta")
	return httpReq, nil
}

func (t *OpenAITranslator) ParseDecisionsResponse(resp *http.Response, req *models.DecisionsRequest) (*models.DecisionsResponse, error) {
	body, err := readUpstreamResponseBody(resp, "openai")
	if err != nil {
		return nil, err
	}
	var result models.DecisionsResponse
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("decode native Decisions response: %w", err)
	}
	var envelope map[string]json.RawMessage
	if err := json.Unmarshal(body, &envelope); err != nil || len(envelope["usage"]) == 0 {
		return nil, fmt.Errorf("native Decisions response has no usage")
	}
	if err := result.Validate(req); err != nil {
		return nil, err
	}
	return &result, nil
}

type TypeSafeTranslator struct{ cfg config.ProviderConfig }

func NewTypeSafeTranslator(cfg config.ProviderConfig) *TypeSafeTranslator {
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.typesafe.ai/v1"
	}
	if cfg.DefaultModel == "" {
		cfg.DefaultModel = "jev-latest"
	}
	cfg.Type = "typesafe"
	return &TypeSafeTranslator{cfg: cfg}
}
func (t *TypeSafeTranslator) Name() string         { return "typesafe" }
func (t *TypeSafeTranslator) DefaultModel() string { return t.cfg.DefaultModel }
func (t *TypeSafeTranslator) BaseURL() string {
	return strings.TrimRight(strings.TrimSpace(t.cfg.BaseURL), "/")
}
func (t *TypeSafeTranslator) SupportsStreaming() bool { return false }
func (t *TypeSafeTranslator) Models() []models.ModelInfo {
	ids := append([]string{t.DefaultModel()}, t.cfg.Capabilities.DecisionModels...)
	seen := map[string]bool{}
	result := make([]models.ModelInfo, 0, len(ids))
	for _, id := range ids {
		if seen[id] {
			continue
		}
		seen[id] = true
		result = append(result, models.ModelInfo{ID: id, Object: "model", OwnedBy: "typesafe", Created: time.Now().Unix(), Capabilities: DecisionModelCapabilities(t.cfg, id)})
	}
	return result
}
func typesafeGenerativeError() error {
	return &models.CompatibilityError{Provider: "typesafe", Field: "request_type", Reason: "TypeSafe is Decisions-only; use POST /v1/decisions"}
}
func (t *TypeSafeTranslator) ValidateRequestCompatibility(_ string, _ *models.UnifiedRequest) error {
	return typesafeGenerativeError()
}
func (t *TypeSafeTranslator) TranslateRequest(context.Context, *models.UnifiedRequest) (*http.Request, error) {
	return nil, typesafeGenerativeError()
}
func (t *TypeSafeTranslator) ParseResponse(*http.Response) (*models.UnifiedResponse, error) {
	return nil, typesafeGenerativeError()
}
func (t *TypeSafeTranslator) ParseStreamChunk([]byte) (*models.StreamChunk, error) {
	return nil, typesafeGenerativeError()
}
func (t *TypeSafeTranslator) DecisionsUpstreamType() string { return "systemone" }

func (t *TypeSafeTranslator) ValidateDecisionsRequest(provider, model string, req *models.DecisionsRequest) error {
	if err := validateDecisionModel(provider, "typesafe", model, t.cfg, req); err != nil {
		return err
	}
	for _, header := range []string{"Idempotency-Key", "OpenAI-Beta", "Anthropic-Beta"} {
		if req.ProviderHeaders.Get(header) != "" {
			return &models.CompatibilityError{Provider: provider, Field: "headers." + header, Reason: "no native TypeSafe control equivalent"}
		}
	}
	_, err := t.decisionPayload(provider, model, req)
	return err
}

func (t *TypeSafeTranslator) TranslateDecisionsRequest(ctx context.Context, req *models.DecisionsRequest) (*http.Request, error) {
	if err := t.ValidateDecisionsRequest("typesafe", req.Model, req); err != nil {
		return nil, err
	}
	payload, err := t.decisionPayload("typesafe", req.Model, req)
	if err != nil {
		return nil, err
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("encode TypeSafe request: %w", err)
	}
	endpoint, err := safeurl.JoinHTTPPath(t.BaseURL(), "systemone")
	if err != nil {
		return nil, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+t.cfg.APIKey)
	if _, ok := payload["assets"]; ok {
		httpReq.Header.Set("TypeSafe-Danger-Zone-Images", "true")
	}
	return httpReq, nil
}

type decisionAssetBuilder struct {
	provider string
	images   bool
	assets   map[string]interface{}
}

func (b *decisionAssetBuilder) bad(field, reason string) error {
	return &models.CompatibilityError{Provider: b.provider, Field: field, Reason: reason}
}

func (b *decisionAssetBuilder) content(raw json.RawMessage, path string) (interface{}, error) {
	var value interface{}
	if err := decodeJSONPreserveNumbers(raw, &value); err != nil {
		return nil, b.bad(path, "invalid content")
	}
	if _, ok := value.(string); ok {
		return value, nil
	}
	parts, ok := value.([]interface{})
	if !ok {
		return nil, b.bad(path, "content must be text or content parts")
	}
	for i, part := range parts {
		field := fmt.Sprintf("%s[%d]", path, i)
		object, ok := part.(map[string]interface{})
		if !ok {
			return nil, b.bad(field, "content parts must be objects")
		}
		kind, _ := object["type"].(string)
		switch kind {
		case "input_text", "output_text", "refusal":
			allowed := map[string]bool{"type": true, "text": true}
			if kind == "refusal" {
				allowed["refusal"] = true
			}
			for key := range object {
				if !allowed[key] {
					return nil, b.bad(field+"."+key, "no TypeSafe content equivalent")
				}
			}
			if kind != "refusal" {
				if _, ok := object["text"].(string); !ok {
					return nil, b.bad(field+".text", "text is required")
				}
			}
		case "input_json":
			for key := range object {
				if key != "type" && key != "data" {
					return nil, b.bad(field+"."+key, "no TypeSafe JSON content equivalent")
				}
			}
			if _, ok := object["data"]; !ok {
				return nil, b.bad(field+".data", "JSON data is required")
			}
		case "input_image":
			if !b.images {
				return nil, b.bad(field, "enable decision_images to use experimental Jev images")
			}
			for key := range object {
				if key != "type" && key != "image_url" && key != "detail" {
					return nil, b.bad(field+"."+key, "no TypeSafe image equivalent")
				}
			}
			if detail, exists := object["detail"]; exists && detail != nil && detail != "auto" {
				return nil, b.bad(field+".detail", "Jev has no image detail control")
			}
			imageURL, ok := object["image_url"].(string)
			if !ok {
				return nil, b.bad(field+".image_url", "inline PNG/JPEG data URL is required")
			}
			format, data, err := typesafeImageData(imageURL)
			if err != nil {
				return nil, b.bad(field+".image_url", err.Error())
			}
			ref := "lunargate-image-" + uuid.NewString()
			b.assets[ref] = map[string]interface{}{"type": "image", "format": format, "data_base64": data}
			object["image_url"] = ref
		default:
			return nil, b.bad(field+".type", "unsupported TypeSafe evidence content type")
		}
	}
	return parts, nil
}

func typesafeImageData(value string) (string, string, error) {
	format := ""
	prefix := ""
	for _, candidate := range []struct{ prefix, format string }{{"data:image/png;base64,", "png"}, {"data:image/jpeg;base64,", "jpg"}, {"data:image/jpg;base64,", "jpg"}} {
		if strings.HasPrefix(value, candidate.prefix) {
			prefix, format = candidate.prefix, candidate.format
			break
		}
	}
	if prefix == "" {
		return "", "", fmt.Errorf("Jev requires inline base64 PNG/JPEG; remote URLs are not fetched")
	}
	data := strings.TrimPrefix(value, prefix)
	decoded, err := base64.StdEncoding.DecodeString(data)
	if err != nil || len(decoded) == 0 || len(decoded) > 16<<20 {
		return "", "", fmt.Errorf("invalid or oversized base64 image")
	}
	if format == "png" {
		_, err = png.DecodeConfig(bytes.NewReader(decoded))
	} else {
		_, err = jpeg.DecodeConfig(bytes.NewReader(decoded))
	}
	if err != nil {
		return "", "", fmt.Errorf("image bytes do not match the declared PNG/JPEG format")
	}
	return format, base64.StdEncoding.EncodeToString(decoded), nil
}

func (b *decisionAssetBuilder) state(raw json.RawMessage) (interface{}, error) {
	var value interface{}
	if err := decodeJSONPreserveNumbers(raw, &value); err != nil {
		return nil, b.bad("input", "invalid input")
	}
	if _, ok := value.(string); ok {
		return value, nil
	}
	items, ok := value.([]interface{})
	if !ok {
		return nil, b.bad("input", "input must be text or ordered evidence")
	}
	calls := map[string]bool{}
	for i, item := range items {
		path := fmt.Sprintf("input[%d]", i)
		object, ok := item.(map[string]interface{})
		if !ok {
			return nil, b.bad(path, "input items must be messages or tool evidence")
		}
		kind, _ := object["type"].(string)
		switch kind {
		case "", "message":
			role, _ := object["role"].(string)
			switch role {
			case "user", "assistant", "system", "developer":
			default:
				return nil, b.bad(path+".role", "unsupported message role")
			}
			content, exists := object["content"]
			if !exists {
				return nil, b.bad(path+".content", "message content is required")
			}
			encoded, _ := json.Marshal(content)
			converted, err := b.content(encoded, path+".content")
			if err != nil {
				return nil, err
			}
			object["content"] = converted
		case "function_call", "custom_tool_call":
			id, _ := object["call_id"].(string)
			if id == "" || calls[id] {
				return nil, b.bad(path+".call_id", "tool call IDs must be non-empty and distinct")
			}
			calls[id] = true
		case "function_call_output", "custom_tool_call_output":
			id, _ := object["call_id"].(string)
			if id == "" || !calls[id] {
				return nil, b.bad(path+".call_id", "tool output requires a preceding matching call")
			}
			if output, exists := object["output"]; exists {
				encoded, _ := json.Marshal(output)
				converted, err := b.content(encoded, path+".output")
				if err != nil {
					return nil, err
				}
				object["output"] = converted
			}
		default:
			return nil, b.bad(path+".type", "unsupported TypeSafe evidence item type")
		}
	}
	return items, nil
}

func (t *TypeSafeTranslator) decisionPayload(provider, model string, req *models.DecisionsRequest) (map[string]interface{}, error) {
	b := &decisionAssetBuilder{provider: provider, images: t.cfg.Capabilities.DecisionImages, assets: map[string]interface{}{}}
	var raw map[string]json.RawMessage
	if len(req.RawJSON) > 0 {
		if err := json.Unmarshal(req.RawJSON, &raw); err != nil {
			return nil, b.bad("request", "invalid request")
		}
	}
	for field := range raw {
		if field != "model" && field != "input" && field != "questions" {
			return nil, b.bad(field, "no native TypeSafe equivalent")
		}
	}
	state, err := b.state(req.Input)
	if err != nil {
		return nil, err
	}
	var rawQuestions []map[string]json.RawMessage
	if len(raw["questions"]) > 0 {
		if err := json.Unmarshal(raw["questions"], &rawQuestions); err != nil {
			return nil, b.bad("questions", "invalid question definitions")
		}
	}
	questions := make(map[string]interface{}, len(req.Questions))
	for i, q := range req.Questions {
		path := fmt.Sprintf("questions[%d]", i)
		allowed := map[string]bool{"type": true, "name": true, "instructions": true}
		if q.Type == "choice" {
			allowed["choices"] = true
		}
		if q.Type == "score" {
			allowed["levels"] = true
		}
		if i < len(rawQuestions) {
			for field := range rawQuestions[i] {
				if !allowed[field] {
					return nil, b.bad(path+"."+field, "no native TypeSafe question equivalent")
				}
			}
		}
		instructions, err := b.content(q.Instructions, path+".instructions")
		if err != nil {
			return nil, err
		}
		question := map[string]interface{}{"type": q.Type, "instructions": instructions}
		switch q.Type {
		case "predicate":
			question["type"] = "noul"
		case "choice":
			if len(q.Choices) > 255 {
				return nil, b.bad(path+".choices", "Jev accepts at most 255 choices")
			}
			criteria := map[string]interface{}{}
			for j, c := range q.Choices {
				criteria[fmt.Sprintf("c%d", j)] = map[string]interface{}{"value": c.Value, "description": c.Description}
			}
			question["criteria"] = criteria
		case "score":
			criteria := make([]interface{}, len(q.Levels))
			for j, l := range q.Levels {
				criteria[j] = map[string]interface{}{"label": l.Label, "description": l.Description}
			}
			question["criteria"] = criteria
		}
		// Reject additive nested controls rather than erasing them in translation.
		if i < len(rawQuestions) {
			for _, field := range []string{"choices", "levels"} {
				var entries []map[string]json.RawMessage
				if len(rawQuestions[i][field]) == 0 {
					continue
				}
				if err := json.Unmarshal(rawQuestions[i][field], &entries); err != nil {
					return nil, b.bad(path+"."+field, "invalid criteria")
				}
				for j, entry := range entries {
					for key := range entry {
						if key != "description" && !(field == "choices" && key == "value") && !(field == "levels" && key == "label") {
							return nil, b.bad(fmt.Sprintf("%s.%s[%d].%s", path, field, j, key), "no native TypeSafe criterion equivalent")
						}
					}
				}
			}
		}
		questions[fmt.Sprintf("q%d", i)] = question
	}
	payload := map[string]interface{}{"model": model, "state": state, "questions": questions}
	if len(b.assets) > 0 {
		payload["assets"] = b.assets
	}
	return payload, nil
}

func (t *TypeSafeTranslator) ParseDecisionsResponse(resp *http.Response, req *models.DecisionsRequest) (*models.DecisionsResponse, error) {
	body, err := readUpstreamResponseBody(resp, "typesafe")
	if err != nil {
		return nil, err
	}
	var native struct {
		Model   string `json:"model"`
		Answers map[string]struct {
			Type          string             `json:"type"`
			Noul          *float64           `json:"noul"`
			Choice        string             `json:"choice"`
			Score         *float64           `json:"score"`
			Confidence    *float64           `json:"confidence"`
			Probabilities map[string]float64 `json:"probabilities"`
		} `json:"answers"`
		Usage *struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(body, &native); err != nil {
		return nil, fmt.Errorf("decode TypeSafe response: %w", err)
	}
	if native.Usage == nil || len(native.Answers) != len(req.Questions) {
		return nil, fmt.Errorf("TypeSafe returned incomplete answers or usage")
	}
	result := &models.DecisionsResponse{Model: native.Model, Answers: make([]models.DecisionAnswer, len(req.Questions)), Usage: models.DecisionUsage{InputTokens: native.Usage.InputTokens, OutputTokens: native.Usage.OutputTokens, TotalTokens: models.SaturatingTokenSum(native.Usage.InputTokens, native.Usage.OutputTokens)}}
	for i, q := range req.Questions {
		n, ok := native.Answers[fmt.Sprintf("q%d", i)]
		if !ok {
			return nil, fmt.Errorf("TypeSafe omitted question %d", i)
		}
		answer := models.DecisionAnswer{Type: q.Type, Name: q.Name}
		switch q.Type {
		case "predicate":
			if n.Type != "noul" {
				return nil, fmt.Errorf("TypeSafe returned an incorrect question type")
			}
			answer.Probability = n.Noul
		case "choice":
			if n.Type != "choice" || len(n.Probabilities) != len(q.Choices) {
				return nil, fmt.Errorf("TypeSafe returned an incorrect choice distribution")
			}
			answer.Confidence = n.Confidence
			for j, c := range q.Choices {
				id := fmt.Sprintf("c%d", j)
				p, ok := n.Probabilities[id]
				if !ok {
					return nil, fmt.Errorf("TypeSafe omitted a choice probability")
				}
				answer.Probabilities = append(answer.Probabilities, models.DecisionProbability{Value: c.Value, Probability: p})
				if n.Choice == id {
					answer.Choice = c.Value
				}
			}
		case "score":
			if n.Type != "score" || len(n.Probabilities) != len(q.Levels) {
				return nil, fmt.Errorf("TypeSafe returned an incorrect score distribution")
			}
			answer.Confidence = n.Confidence
			answer.Score = n.Score
			for j, l := range q.Levels {
				p, ok := n.Probabilities[fmt.Sprint(j)]
				if !ok {
					return nil, fmt.Errorf("TypeSafe omitted a score probability")
				}
				value, _ := json.Marshal(j)
				label := l.Label
				answer.Probabilities = append(answer.Probabilities, models.DecisionProbability{Value: value, Label: &label, Probability: p})
			}
		}
		result.Answers[i] = answer
	}
	if err := result.Validate(req); err != nil {
		return nil, err
	}
	return result, nil
}
