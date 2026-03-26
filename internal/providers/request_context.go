package providers

import "context"

type upstreamRequestTypeKey struct{}

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
