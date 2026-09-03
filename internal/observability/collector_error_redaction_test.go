package observability

import (
	"bytes"
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

func TestCollectorStatusErrorsDoNotExposeResponseBodies(t *testing.T) {
	tests := []struct {
		name   string
		status int
	}{
		{name: "authentication rejection", status: http.StatusUnauthorized},
		{name: "backend failure", status: http.StatusBadGateway},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			secret := "collector-response-secret-" + strconv.Itoa(test.status)
			backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(test.status)
				_, _ = w.Write([]byte(`{"detail":"` + secret + `"}`))
			}))
			defer backend.Close()

			client := &CollectorClient{
				httpClient: backend.Client(),
				cfg: normalizeCollectorConfig(
					config.GeneralConfig{BackendURL: backend.URL, APIKey: "gateway-key"},
					config.DataSharingConfig{Enabled: true},
				),
			}
			err := client.send(context.Background(), collectorItem{
				requestID: "request-redaction",
				payload:   []byte(`{"events":[]}`),
				identity: collectorIdentity{
					backendURL: backend.URL,
					apiKey:     "gateway-key",
				},
			})
			var statusErr *httpStatusError
			if !errors.As(err, &statusErr) || statusErr.statusCode != test.status {
				t.Fatalf("send error = %v, want status %d", err, test.status)
			}
			if strings.Contains(err.Error(), secret) {
				t.Fatalf("collector error exposed backend response: %v", err)
			}

			var output bytes.Buffer
			previousLogger := log.Logger
			log.Logger = zerolog.New(&output)
			t.Cleanup(func() { log.Logger = previousLogger })
			client.logSendError("request-redaction", err)

			logged := output.String()
			if strings.Contains(logged, secret) || strings.Contains(logged, "detail") {
				t.Fatalf("collector log exposed backend response: %s", logged)
			}
			if !strings.Contains(logged, strconv.Itoa(test.status)) {
				t.Fatalf("collector log lost status %d: %s", test.status, logged)
			}
		})
	}
}
