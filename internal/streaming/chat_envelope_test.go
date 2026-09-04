package streaming

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/pkg/models"
)

func TestStreamResponseCompletesMissingChatChunkEnvelope(t *testing.T) {
	translator := providers.NewOpenAITranslator(config.ProviderConfig{
		APIKey:       "dummy",
		DefaultModel: "fallback-model",
	})
	providerResp := &http.Response{
		StatusCode: http.StatusOK,
		Body: io.NopCloser(strings.NewReader(
			"data: {\"created\":1788382926,\"model\":\"resolved-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"O\"},\"finish_reason\":null}]}\n\n" +
				"data: {\"created\":1788382926,\"model\":\"resolved-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"K\"},\"finish_reason\":\"stop\"}]}\n\n" +
				"data: [DONE]\n\n",
		)),
	}
	recorder := httptest.NewRecorder()

	if err := NewHandler().StreamResponse(context.Background(), recorder, providerResp, translator); err != nil {
		t.Fatalf("StreamResponse returned error: %v", err)
	}

	frames := strings.Split(strings.TrimSpace(recorder.Body.String()), "\n\n")
	if len(frames) != 3 || frames[2] != "data: [DONE]" {
		t.Fatalf("unexpected stream frames: %q", recorder.Body.String())
	}
	var chunks [2]models.StreamChunk
	for index := range chunks {
		if err := json.Unmarshal([]byte(strings.TrimPrefix(frames[index], "data: ")), &chunks[index]); err != nil {
			t.Fatalf("decode chunk %d: %v", index, err)
		}
		if !strings.HasPrefix(chunks[index].ID, "chatcmpl-") {
			t.Fatalf("chunk %d id = %q", index, chunks[index].ID)
		}
		if chunks[index].Object != "chat.completion.chunk" {
			t.Fatalf("chunk %d object = %q", index, chunks[index].Object)
		}
		if chunks[index].Created != 1788382926 || chunks[index].Model != "resolved-model" {
			t.Fatalf("chunk %d envelope = %#v", index, chunks[index])
		}
	}
	if chunks[0].ID != chunks[1].ID {
		t.Fatalf("chunk ids differ: %q != %q", chunks[0].ID, chunks[1].ID)
	}
}
