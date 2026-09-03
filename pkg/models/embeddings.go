package models

import (
	"bytes"
	"encoding/json"
	"fmt"
)

type EmbeddingsRequest struct {
	RawJSON        json.RawMessage `json:"-"`
	Model          string          `json:"model"`
	Input          interface{}     `json:"input"`
	EncodingFormat string          `json:"encoding_format,omitempty"`
	Dimensions     *int            `json:"dimensions,omitempty"`
	User           string          `json:"user,omitempty"`
}

type EmbeddingsResponse struct {
	RawJSON json.RawMessage `json:"-"`
	Object  string          `json:"object"`
	Data    []EmbeddingData `json:"data"`
	Model   string          `json:"model"`
	Usage   *EmbeddingUsage `json:"usage,omitempty"`
}

type EmbeddingData struct {
	Object    string         `json:"object"`
	Embedding EmbeddingValue `json:"embedding"`
	Index     int            `json:"index"`
}

// EmbeddingValue is the OpenAI-compatible union of a float vector and its
// base64 representation. Keeping the encoded JSON also avoids precision loss
// while proxying provider responses.
type EmbeddingValue json.RawMessage

func (v *EmbeddingValue) UnmarshalJSON(data []byte) error {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 {
		return fmt.Errorf("embedding is required")
	}

	if trimmed[0] == '"' {
		var encoded string
		if err := json.Unmarshal(trimmed, &encoded); err != nil {
			return fmt.Errorf("invalid base64 embedding: %w", err)
		}
		*v = append((*v)[:0], trimmed...)
		return nil
	}

	var vector []float64
	if err := json.Unmarshal(trimmed, &vector); err != nil || vector == nil {
		if err == nil {
			err = fmt.Errorf("expected a float vector or base64 string")
		}
		return fmt.Errorf("invalid embedding: %w", err)
	}
	*v = append((*v)[:0], trimmed...)
	return nil
}

func (v EmbeddingValue) MarshalJSON() ([]byte, error) {
	if len(bytes.TrimSpace(v)) == 0 {
		return nil, fmt.Errorf("embedding is required")
	}
	return append([]byte(nil), v...), nil
}

func NewFloatEmbeddingValue(vector []float64) EmbeddingValue {
	raw, _ := json.Marshal(vector)
	return EmbeddingValue(raw)
}

type EmbeddingUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

func CloneEmbeddingsResponse(resp *EmbeddingsResponse) *EmbeddingsResponse {
	if resp == nil {
		return nil
	}

	data, err := json.Marshal(resp)
	if err != nil {
		return resp
	}

	var out EmbeddingsResponse
	if err := json.Unmarshal(data, &out); err != nil {
		return resp
	}
	out.RawJSON = append(json.RawMessage(nil), resp.RawJSON...)
	return &out
}
