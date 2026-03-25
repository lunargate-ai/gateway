package models

import "encoding/json"

type EmbeddingsRequest struct {
	Model          string      `json:"model"`
	Input          interface{} `json:"input"`
	EncodingFormat string      `json:"encoding_format,omitempty"`
	Dimensions     *int        `json:"dimensions,omitempty"`
	User           string      `json:"user,omitempty"`
}

type EmbeddingsResponse struct {
	Object string          `json:"object"`
	Data   []EmbeddingData `json:"data"`
	Model  string          `json:"model"`
	Usage  *EmbeddingUsage `json:"usage,omitempty"`
}

type EmbeddingData struct {
	Object    string    `json:"object"`
	Embedding []float64 `json:"embedding"`
	Index     int       `json:"index"`
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
	return &out
}
