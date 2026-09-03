package observability

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"
)

func TestMetricErrorClassIsFiniteAndContentFree(t *testing.T) {
	tests := []struct {
		status int
		failed bool
		want   string
	}{
		{status: http.StatusOK, failed: false, want: ""},
		{status: http.StatusOK, failed: true, want: "upstream_error"},
		{status: http.StatusBadRequest, failed: true, want: "invalid_request"},
		{status: http.StatusUnauthorized, failed: true, want: "authentication"},
		{status: http.StatusForbidden, failed: true, want: "permission"},
		{status: http.StatusNotFound, failed: true, want: "not_found"},
		{status: http.StatusConflict, failed: true, want: "conflict"},
		{status: http.StatusRequestEntityTooLarge, failed: true, want: "request_too_large"},
		{status: http.StatusTooManyRequests, failed: true, want: "rate_limited"},
		{status: http.StatusGatewayTimeout, failed: true, want: "timeout"},
		{status: http.StatusBadGateway, failed: true, want: "upstream_error"},
		{status: 499, failed: true, want: "client_cancelled"},
	}

	for _, test := range tests {
		got := MetricErrorClass(test.status, test.failed)
		if test.want == "" {
			if got != nil {
				t.Fatalf("status %d class = %q, want nil", test.status, *got)
			}
			continue
		}
		if got == nil || *got != test.want {
			t.Fatalf("status %d class = %#v, want %q", test.status, got, test.want)
		}
	}
}

func TestMetricEventCannotSerializeErrorContent(t *testing.T) {
	class := "upstream_error"
	payload, err := json.Marshal(MetricEventData{
		StatusCode: http.StatusBadGateway,
		ErrorCode:  &class,
	})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(payload), "error_message") {
		t.Fatalf("metric payload exposes an error message field: %s", payload)
	}
}
