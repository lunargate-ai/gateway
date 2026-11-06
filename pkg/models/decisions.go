package models

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"strings"
)

// DecisionsTranslator is independent of the generative Chat contract. Only
// providers with a native decision model implement this optional interface.
type DecisionsTranslator interface {
	ValidateDecisionsRequest(provider, model string, req *DecisionsRequest) error
	TranslateDecisionsRequest(context.Context, *DecisionsRequest) (*http.Request, error)
	ParseDecisionsResponse(*http.Response, *DecisionsRequest) (*DecisionsResponse, error)
	DecisionsUpstreamType() string
}

type DecisionsRequest struct {
	RawJSON         json.RawMessage    `json:"-"`
	ProviderHeaders http.Header        `json:"-"`
	Model           string             `json:"model"`
	Input           json.RawMessage    `json:"input"`
	Questions       []DecisionQuestion `json:"questions"`
}

type DecisionQuestion struct {
	Type         string           `json:"type"`
	Name         *string          `json:"name,omitempty"`
	Instructions json.RawMessage  `json:"instructions"`
	Choices      []DecisionChoice `json:"choices,omitempty"`
	Levels       []DecisionLevel  `json:"levels,omitempty"`
}

type DecisionChoice struct {
	Value       json.RawMessage `json:"value"`
	Description *string         `json:"description,omitempty"`
}

type DecisionLevel struct {
	Label       string  `json:"label"`
	Description *string `json:"description,omitempty"`
}

type DecisionProbability struct {
	Value       json.RawMessage `json:"value"`
	Label       *string         `json:"label,omitempty"`
	Probability float64         `json:"probability"`
}

type DecisionAnswer struct {
	Type          string                `json:"type"`
	Name          *string               `json:"name"`
	Probability   *float64              `json:"probability,omitempty"`
	Choice        json.RawMessage       `json:"choice,omitempty"`
	Score         *float64              `json:"score,omitempty"`
	Probabilities []DecisionProbability `json:"probabilities,omitempty"`
	Confidence    *float64              `json:"confidence,omitempty"`
}

type DecisionUsage struct {
	InputTokens         int                      `json:"input_tokens"`
	InputTokensDetails  *InputTokensDetails      `json:"input_tokens_details,omitempty"`
	OutputTokens        int                      `json:"output_tokens"`
	OutputTokensDetails *CompletionTokensDetails `json:"output_tokens_details,omitempty"`
	TotalTokens         int                      `json:"total_tokens"`
}

func (u DecisionUsage) TokenUsage() TokenUsage {
	usage := TokenUsage{InputTokens: u.InputTokens, OutputTokens: u.OutputTokens}
	if u.InputTokensDetails != nil {
		usage.CachedInputTokens = u.InputTokensDetails.CachedTokens
		usage.CacheWriteInputTokens = u.InputTokensDetails.CacheWriteTokens
	}
	if u.OutputTokensDetails != nil {
		usage.ReasoningOutputTokens = u.OutputTokensDetails.ReasoningTokens
	}
	return usage.Normalized()
}

type DecisionsResponse struct {
	RawJSON json.RawMessage  `json:"-"`
	Answers []DecisionAnswer `json:"answers"`
	Model   string           `json:"model"`
	Usage   DecisionUsage    `json:"usage"`
}

// Preserve additive native fields through both HTTP output and exact-match cache.
func (r DecisionsResponse) MarshalJSON() ([]byte, error) {
	if len(r.RawJSON) != 0 {
		return append([]byte(nil), r.RawJSON...), nil
	}
	type wire DecisionsResponse
	return json.Marshal(wire(r))
}

func (r *DecisionsResponse) UnmarshalJSON(data []byte) error {
	type wire DecisionsResponse
	var value wire
	if err := json.Unmarshal(data, &value); err != nil {
		return err
	}
	*r = DecisionsResponse(value)
	r.RawJSON = append(json.RawMessage(nil), data...)
	return nil
}

func DecisionValueKey(value json.RawMessage) (string, error) {
	var v interface{}
	decoder := json.NewDecoder(bytes.NewReader(value))
	if err := decoder.Decode(&v); err != nil {
		return "", err
	}
	switch v.(type) {
	case string, bool:
	default:
		return "", fmt.Errorf("choice values must be strings or booleans")
	}
	encoded, err := json.Marshal(v)
	return string(encoded), err
}

func (r *DecisionsRequest) Validate() error {
	bad := func(field, reason string) error { return &CompatibilityError{Field: field, Reason: reason} }
	if strings.TrimSpace(r.Model) == "" {
		return bad("model", "model is required")
	}
	input := bytes.TrimSpace(r.Input)
	if len(input) == 0 || bytes.Equal(input, []byte("null")) || (input[0] != '"' && input[0] != '[') {
		return bad("input", "input must be a string or an ordered array of input items")
	}
	if len(r.Questions) < 1 || len(r.Questions) > 400 {
		return bad("questions", "between 1 and 400 questions are required")
	}
	for i, q := range r.Questions {
		path := fmt.Sprintf("questions[%d]", i)
		instructions := bytes.TrimSpace(q.Instructions)
		if len(instructions) == 0 || (instructions[0] != '"' && instructions[0] != '[') {
			return bad(path+".instructions", "instructions must be a string or content parts")
		}
		switch q.Type {
		case "predicate":
			if len(q.Choices) != 0 || len(q.Levels) != 0 {
				return bad(path, "predicate does not accept choices or levels")
			}
		case "choice":
			if len(q.Choices) < 2 || len(q.Levels) != 0 {
				return bad(path+".choices", "choice requires at least two choices and no score levels")
			}
			seen := map[string]bool{}
			for j, choice := range q.Choices {
				key, err := DecisionValueKey(choice.Value)
				if err != nil || seen[key] {
					return bad(fmt.Sprintf("%s.choices[%d].value", path, j), "choice values must be distinct strings or booleans")
				}
				seen[key] = true
			}
		case "score":
			if len(q.Levels) < 2 || len(q.Levels) > 10 || len(q.Choices) != 0 {
				return bad(path+".levels", "score requires between 2 and 10 levels and no choices")
			}
		default:
			return bad(path+".type", "supported question types are predicate, choice, and score")
		}
	}
	return nil
}

func DecisionProbabilityValid(p float64) bool {
	return !math.IsNaN(p) && !math.IsInf(p, 0) && p >= 0 && p <= 1
}

func (r *DecisionsResponse) Validate(req *DecisionsRequest) error {
	if r == nil || strings.TrimSpace(r.Model) == "" || len(r.Answers) != len(req.Questions) {
		return fmt.Errorf("decision response must contain the model and one answer per question")
	}
	if r.Usage.InputTokens < 0 || r.Usage.OutputTokens < 0 || r.Usage.TotalTokens < 0 {
		return fmt.Errorf("decision usage must be non-negative")
	}
	for i, answer := range r.Answers {
		q := req.Questions[i]
		if (answer.Name == nil) != (q.Name == nil) || (answer.Name != nil && *answer.Name != *q.Name) {
			return fmt.Errorf("answer %d has an incorrect question name", i)
		}
		if answer.Type == "refusal" {
			continue
		}
		if answer.Type != q.Type {
			return fmt.Errorf("answer %d has an incorrect type", i)
		}
		if q.Type == "predicate" {
			if answer.Probability == nil || !DecisionProbabilityValid(*answer.Probability) {
				return fmt.Errorf("answer %d has an invalid predicate probability", i)
			}
			continue
		}
		if answer.Confidence == nil || !DecisionProbabilityValid(*answer.Confidence) {
			return fmt.Errorf("answer %d has invalid confidence", i)
		}
		count := len(q.Choices)
		if q.Type == "score" {
			count = len(q.Levels)
		}
		if len(answer.Probabilities) != count {
			return fmt.Errorf("answer %d has an incomplete distribution", i)
		}
		expected := map[string]bool{}
		if q.Type == "choice" {
			for _, c := range q.Choices {
				key, _ := DecisionValueKey(c.Value)
				expected[key] = true
			}
		}
		seen := map[string]bool{}
		sum := 0.0
		maxProbability := -1.0
		selectedProbability := -1.0
		selectedKey, _ := DecisionValueKey(answer.Choice)
		for _, p := range answer.Probabilities {
			if !DecisionProbabilityValid(p.Probability) {
				return fmt.Errorf("answer %d has an invalid distribution", i)
			}
			key := ""
			if q.Type == "choice" {
				key, _ = DecisionValueKey(p.Value)
				if !expected[key] {
					return fmt.Errorf("answer %d contains an unrequested choice", i)
				}
				if key == selectedKey {
					selectedProbability = p.Probability
				}
			} else {
				var index int
				if err := json.Unmarshal(p.Value, &index); err != nil || index < 0 || index >= count || p.Label == nil || *p.Label != q.Levels[index].Label {
					return fmt.Errorf("answer %d has an invalid score level", i)
				}
				key = fmt.Sprint(index)
			}
			if seen[key] {
				return fmt.Errorf("answer %d has a duplicate distribution value", i)
			}
			seen[key] = true
			sum += p.Probability
			if p.Probability > maxProbability {
				maxProbability = p.Probability
			}
		}
		if math.Abs(sum-1) > 1e-6 {
			return fmt.Errorf("answer %d probabilities do not sum to one", i)
		}
		if q.Type == "choice" && (selectedProbability < 0 || selectedProbability+1e-6 < maxProbability) {
			return fmt.Errorf("answer %d choice does not match its distribution", i)
		}
		// Native providers can round the reported score and probabilities
		// independently. Preserve the native score; recomputing its mean from
		// the rounded distribution would change or reject valid answers.
		if q.Type == "score" && (answer.Score == nil || math.IsNaN(*answer.Score) || math.IsInf(*answer.Score, 0) || *answer.Score < 0 || *answer.Score > float64(count-1)) {
			return fmt.Errorf("answer %d has an invalid score", i)
		}
	}
	return nil
}
