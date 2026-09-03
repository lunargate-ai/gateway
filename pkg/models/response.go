package models

import "encoding/json"

// UnifiedResponse represents an OpenAI-compatible chat completion response.
// All provider translators convert their native responses TO this format.
type UnifiedResponse struct {
	RawJSON           json.RawMessage `json:"-"`
	ID                string          `json:"id"`
	Object            string          `json:"object"`
	Created           int64           `json:"created"`
	Model             string          `json:"model"`
	Choices           []Choice        `json:"choices"`
	Usage             *Usage          `json:"usage,omitempty"`
	SystemFingerprint string          `json:"system_fingerprint,omitempty"`
}

type Choice struct {
	Index        int       `json:"index"`
	Message      *Message  `json:"message,omitempty"`
	Delta        *Message  `json:"delta,omitempty"`
	FinishReason *string   `json:"finish_reason"`
	Logprobs     *Logprobs `json:"logprobs,omitempty"`
}

type Usage struct {
	PromptTokens        int                 `json:"prompt_tokens"`
	CompletionTokens    int                 `json:"completion_tokens"`
	TotalTokens         int                 `json:"total_tokens"`
	PromptTokensDetails *InputTokensDetails `json:"prompt_tokens_details,omitempty"`
}

// InputTokensDetails is the provider-neutral prompt-cache decomposition used
// by OpenAI-compatible Chat and Responses usage envelopes. The TTL fields are
// internal because OpenAI does not expose them in its public response schema.
type InputTokensDetails struct {
	CachedTokens       int `json:"cached_tokens,omitempty"`
	CacheWriteTokens   int `json:"cache_write_tokens,omitempty"`
	CacheWriteTokens5m int `json:"-"`
	CacheWriteTokens1h int `json:"-"`
}

type Logprobs struct {
	Content []LogprobContent `json:"content,omitempty"`
}

type LogprobContent struct {
	Token   string  `json:"token"`
	Logprob float64 `json:"logprob"`
	Bytes   []int   `json:"bytes,omitempty"`
}

// StreamChunk represents a single chunk in an SSE stream.
type StreamChunk struct {
	// RawJSON preserves additive fields from an OpenAI-compatible Chat SSE
	// chunk. The streaming layer overlays normalized typed fields onto this
	// document before sending it to the client.
	RawJSON           json.RawMessage `json:"-"`
	ID                string          `json:"id"`
	Object            string          `json:"object"`
	Created           int64           `json:"created"`
	Model             string          `json:"model"`
	Choices           []Choice        `json:"choices"`
	SystemFingerprint string          `json:"system_fingerprint,omitempty"`
	Usage             *Usage          `json:"usage,omitempty"`
}

// ErrorResponse represents an OpenAI-compatible error response.
type ErrorResponse struct {
	Error ErrorDetail `json:"error"`
}

type ErrorDetail struct {
	Message string  `json:"message"`
	Type    string  `json:"type"`
	Param   *string `json:"param"`
	Code    *string `json:"code"`
}

// ModelList represents the response for GET /v1/models.
type ModelList struct {
	Object string      `json:"object"`
	Data   []ModelInfo `json:"data"`
}

type ModelInfo struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}
