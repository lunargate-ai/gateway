package providers

import (
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
)

type generatedResponseBody struct {
	prefix string
	total  int64
	read   int64
	closed bool
}

func (b *generatedResponseBody) Read(p []byte) (int, error) {
	if b.read >= b.total {
		return 0, io.EOF
	}

	n := len(p)
	if remaining := b.total - b.read; int64(n) > remaining {
		n = int(remaining)
	}
	for i := 0; i < n; i++ {
		p[i] = 'x'
	}
	if b.read < int64(len(b.prefix)) {
		start := int(b.read)
		end := start + n
		if end > len(b.prefix) {
			end = len(b.prefix)
		}
		copy(p, b.prefix[start:end])
	}
	b.read += int64(n)
	return n, nil
}

func (b *generatedResponseBody) Close() error {
	b.closed = true
	return nil
}

func TestReadUpstreamResponseBodyAcceptsExactLimit(t *testing.T) {
	body := &generatedResponseBody{total: maxUpstreamResponseBodyBytes}
	response := &http.Response{StatusCode: http.StatusOK, Body: body}

	got, err := readUpstreamResponseBody(response, "openai")
	if err != nil {
		t.Fatalf("readUpstreamResponseBody returned error: %v", err)
	}
	if int64(len(got)) != maxUpstreamResponseBodyBytes {
		t.Fatalf("body length = %d, want %d", len(got), maxUpstreamResponseBodyBytes)
	}
	if body.read != maxUpstreamResponseBodyBytes {
		t.Fatalf("upstream bytes read = %d, want %d", body.read, maxUpstreamResponseBodyBytes)
	}
	if !body.closed {
		t.Fatal("upstream response body was not closed")
	}
}

func TestProviderParsersRejectOversizedResponseBodies(t *testing.T) {
	const secret = "secret-upstream-diagnostic"

	tests := []struct {
		name     string
		provider string
		status   int
		parse    func(*http.Response) error
	}{
		{
			name:     "openai chat success",
			provider: "openai",
			status:   http.StatusOK,
			parse: func(response *http.Response) error {
				_, err := NewOpenAITranslator(config.ProviderConfig{}).ParseResponse(response)
				return err
			},
		},
		{
			name:     "openai embeddings error",
			provider: "openai",
			status:   http.StatusInternalServerError,
			parse: func(response *http.Response) error {
				_, err := NewOpenAITranslator(config.ProviderConfig{}).ParseEmbeddingsResponse(response)
				return err
			},
		},
		{
			name:     "anthropic chat success",
			provider: "anthropic",
			status:   http.StatusOK,
			parse: func(response *http.Response) error {
				_, err := NewAnthropicTranslator(config.ProviderConfig{}).ParseResponse(response)
				return err
			},
		},
		{
			name:     "ollama chat error",
			provider: "ollama",
			status:   http.StatusInternalServerError,
			parse: func(response *http.Response) error {
				_, err := NewOllamaTranslator(config.ProviderConfig{}).ParseResponse(response)
				return err
			},
		},
		{
			name:     "ollama embeddings success",
			provider: "ollama",
			status:   http.StatusOK,
			parse: func(response *http.Response) error {
				_, err := NewOllamaTranslator(config.ProviderConfig{}).ParseEmbeddingsResponse(response)
				return err
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			body := &generatedResponseBody{
				prefix: secret,
				total:  maxUpstreamResponseBodyBytes + 1,
			}
			err := test.parse(&http.Response{StatusCode: test.status, Body: body})

			var providerErr *ProviderError
			if !errors.As(err, &providerErr) {
				t.Fatalf("error = %v, want ProviderError", err)
			}
			if providerErr.StatusCode != http.StatusBadGateway || providerErr.Type != "upstream_response_too_large" || providerErr.Provider != test.provider {
				t.Fatalf("provider error = %#v", providerErr)
			}
			if strings.Contains(err.Error(), secret) || strings.Contains(providerErr.Message, secret) {
				t.Fatalf("oversized response content leaked in error: %v", err)
			}
			if body.read != maxUpstreamResponseBodyBytes+1 {
				t.Fatalf("upstream bytes read = %d, want %d", body.read, maxUpstreamResponseBodyBytes+1)
			}
			if !body.closed {
				t.Fatal("upstream response body was not closed")
			}
		})
	}
}
