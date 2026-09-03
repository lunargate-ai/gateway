package streaming

import (
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/lunargate-ai/gateway/pkg/models"
)

type chatStreamEnvelopeNormalizer struct {
	id           string
	created      int64
	defaultModel string
}

func newChatStreamEnvelopeNormalizer(defaultModel string) *chatStreamEnvelopeNormalizer {
	return &chatStreamEnvelopeNormalizer{defaultModel: strings.TrimSpace(defaultModel)}
}

func (n *chatStreamEnvelopeNormalizer) normalize(chunk *models.StreamChunk) *models.StreamChunk {
	if chunk == nil {
		return nil
	}
	if n == nil {
		return chunk
	}

	if n.id == "" {
		n.id = strings.TrimSpace(chunk.ID)
		if n.id == "" {
			n.id = "chatcmpl-" + strings.ReplaceAll(uuid.NewString(), "-", "")
		}
	}
	chunk.ID = n.id

	if n.created == 0 {
		n.created = chunk.Created
		if n.created == 0 {
			n.created = time.Now().Unix()
		}
	}
	chunk.Created = n.created

	if strings.TrimSpace(chunk.Object) == "" {
		chunk.Object = "chat.completion.chunk"
	}
	if strings.TrimSpace(chunk.Model) == "" {
		chunk.Model = n.defaultModel
	}

	return chunk
}
