package providers

import (
	"fmt"
	"io"
	"net/http"
)

const maxUpstreamResponseBodyBytes int64 = 16 << 20

func readUpstreamResponseBody(resp *http.Response, provider string) ([]byte, error) {
	return readUpstreamResponseBodyWithLimit(resp, provider, maxUpstreamResponseBodyBytes)
}

func readUpstreamResponseBodyWithLimit(resp *http.Response, provider string, limit int64) ([]byte, error) {
	if resp == nil {
		return nil, fmt.Errorf("upstream response is nil")
	}
	if resp.Body == nil {
		return nil, fmt.Errorf("upstream response body is nil")
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(body)) > limit {
		return nil, &ProviderError{
			StatusCode: http.StatusBadGateway,
			Message:    "upstream response exceeds the 16 MiB limit",
			Type:       "upstream_response_too_large",
			Provider:   provider,
		}
	}
	return body, nil
}
