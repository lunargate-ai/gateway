package providers

import (
	"context"
	"net/http"
	"strings"
)

type upstreamRequestTypeKey struct{}
type sourceRequestTypeKey struct{}
type upstreamRequestHeadersKey struct{}

var allowedUpstreamRequestHeaders = []string{
	"Anthropic-Beta",
	"Idempotency-Key",
	"OpenAI-Beta",
}

func WithUpstreamRequestType(ctx context.Context, requestType string) context.Context {
	return context.WithValue(ctx, upstreamRequestTypeKey{}, requestType)
}

func UpstreamRequestTypeFromContext(ctx context.Context) string {
	if ctx == nil {
		return ""
	}
	v, _ := ctx.Value(upstreamRequestTypeKey{}).(string)
	return v
}

func WithSourceRequestType(ctx context.Context, requestType string) context.Context {
	return context.WithValue(ctx, sourceRequestTypeKey{}, requestType)
}

func SourceRequestTypeFromContext(ctx context.Context) string {
	if ctx == nil {
		return ""
	}
	v, _ := ctx.Value(sourceRequestTypeKey{}).(string)
	return v
}

// WithUpstreamRequestHeaders retains only the explicitly supported provider
// control headers. Authentication, cookies, organization selection, and all
// other client headers must come from gateway configuration or stay local.
func WithUpstreamRequestHeaders(ctx context.Context, headers http.Header) context.Context {
	forwarded := make(http.Header)
	for headerName, values := range headers {
		for _, allowedName := range allowedUpstreamRequestHeaders {
			if !strings.EqualFold(headerName, allowedName) {
				continue
			}
			for _, value := range values {
				if strings.TrimSpace(value) != "" {
					forwarded.Add(allowedName, value)
				}
			}
			break
		}
	}
	return context.WithValue(ctx, upstreamRequestHeadersKey{}, forwarded)
}

func applyUpstreamRequestHeaders(ctx context.Context, request *http.Request, names ...string) {
	if ctx == nil || request == nil {
		return
	}
	forwarded, _ := ctx.Value(upstreamRequestHeadersKey{}).(http.Header)
	for _, name := range names {
		for _, value := range forwarded.Values(name) {
			request.Header.Add(name, value)
		}
	}
}
