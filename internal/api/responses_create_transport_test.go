package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestNativeResponsesCreateRequiresHTTP200(t *testing.T) {
	tests := []struct {
		name   string
		status int
		stream bool
	}{
		{name: "no content", status: http.StatusNoContent},
		{name: "partial content", status: http.StatusPartialContent},
		{name: "redirect", status: http.StatusTemporaryRedirect},
		{name: "stream accepted", status: http.StatusAccepted, stream: true},
		{name: "stream partial content", status: http.StatusPartialContent, stream: true},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				if test.status >= 300 && test.status < 400 {
					w.Header().Set("Location", "/must-not-follow")
				}
				w.WriteHeader(test.status)
			}))
			defer upstream.Close()

			h := newResponsesWebSocketTestHandlerWithUpstreamType(upstream.URL, requestTypeResponses)
			body, err := json.Marshal(map[string]interface{}{
				"model":  "lunargate/auto",
				"input":  "hello",
				"stream": test.stream,
			})
			if err != nil {
				t.Fatalf("marshal request: %v", err)
			}
			req := httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body))
			recorder := httptest.NewRecorder()

			h.Responses(recorder, req)

			if recorder.Code != http.StatusBadGateway {
				t.Fatalf("status = %d, want %d; body=%s", recorder.Code, http.StatusBadGateway, recorder.Body.String())
			}
			var response models.ErrorResponse
			if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
				t.Fatalf("decode error response: %v; body=%s", err, recorder.Body.String())
			}
			if response.Error.Type != "invalid_response_status" {
				t.Fatalf("error type = %q, want invalid_response_status", response.Error.Type)
			}
		})
	}
}
