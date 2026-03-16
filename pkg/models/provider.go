package models

import (
	"context"
	"net/http"
)

// ProviderTranslator is the core interface that every LLM provider must implement.
// It translates between the unified OpenAI-compatible format and the provider's native format.
type ProviderTranslator interface {
	// Name returns the provider identifier (e.g., "openai", "anthropic").
	Name() string

	// DefaultModel returns the provider's default model name.
	DefaultModel() string

	BaseURL() string

	// TranslateRequest converts a UnifiedRequest into a provider-specific HTTP request.
	TranslateRequest(ctx context.Context, req *UnifiedRequest) (*http.Request, error)

	// ParseResponse converts a provider's HTTP response into a UnifiedResponse.
	ParseResponse(resp *http.Response) (*UnifiedResponse, error)

	// ParseStreamChunk converts a raw SSE data line into a StreamChunk.
	// Returns nil, nil if the chunk should be skipped (e.g., empty keep-alive).
	// Returns nil, ErrStreamDone if the stream is complete.
	ParseStreamChunk(data []byte) (*StreamChunk, error)

	// SupportsStreaming returns whether this provider supports SSE streaming.
	SupportsStreaming() bool

	// Models returns the list of models available from this provider.
	Models() []ModelInfo
}

// RequestMetadata carries routing and observability info through the request lifecycle.
type RequestMetadata struct {
	RequestID      string
	TraceID        string
	Provider       string
	Model          string
	RouteUsed      string
	TargetIndex    int
	FallbackUsed   bool
	RetryCount     int
	CacheHit       bool
	Tags           map[string]string
}
