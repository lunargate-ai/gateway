package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/lunargate-ai/gateway/pkg/models"
)

type anthropicStreamTranslator struct {
	base *AnthropicTranslator

	id    string
	model string

	toolCallIndexByContentBlockIndex map[int]int
	nextToolCallIndex               int
}

func NewAnthropicStreamTranslator(base *AnthropicTranslator) models.ProviderTranslator {
	return &anthropicStreamTranslator{
		base:                            base,
		toolCallIndexByContentBlockIndex: make(map[int]int, 8),
	}
}

func (t *anthropicStreamTranslator) Name() string {
	return t.base.Name()
}

func (t *anthropicStreamTranslator) DefaultModel() string {
	return t.base.DefaultModel()
}

func (t *anthropicStreamTranslator) BaseURL() string {
	return t.base.BaseURL()
}

func (t *anthropicStreamTranslator) TranslateRequest(ctx context.Context, req *models.UnifiedRequest) (*http.Request, error) {
	return t.base.TranslateRequest(ctx, req)
}

func (t *anthropicStreamTranslator) ParseResponse(resp *http.Response) (*models.UnifiedResponse, error) {
	return t.base.ParseResponse(resp)
}

type anthropicStreamEvent struct {
	Type         string                 `json:"type"`
	Index        int                    `json:"index,omitempty"`
	ContentBlock *anthropicContentBlock `json:"content_block,omitempty"`
	Delta        *anthropicStreamDelta  `json:"delta,omitempty"`
	Message      *anthropicResponse     `json:"message,omitempty"`
	Usage        *anthropicUsage        `json:"usage,omitempty"`

	Error *struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

type anthropicStreamDelta struct {
	Type        string  `json:"type,omitempty"`
	Text        string  `json:"text,omitempty"`
	PartialJSON string  `json:"partial_json,omitempty"`
	StopReason  *string `json:"stop_reason,omitempty"`
}

func (t *anthropicStreamTranslator) ParseStreamChunk(data []byte) (*models.StreamChunk, error) {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 {
		return nil, nil
	}

	var event anthropicStreamEvent
	if err := json.Unmarshal(trimmed, &event); err != nil {
		return nil, fmt.Errorf("failed to unmarshal anthropic stream event: %w", err)
	}

	if event.Type == "error" {
		msg := "anthropic stream error"
		if event.Error != nil && event.Error.Message != "" {
			msg = event.Error.Message
		}
		return nil, fmt.Errorf("%s", msg)
	}

	switch event.Type {
	case "ping":
		return nil, nil

	case "message_stop":
		return nil, ErrStreamDone

	case "message_start":
		if event.Message == nil {
			return nil, nil
		}
		t.id = event.Message.ID
		t.model = event.Message.Model

		var usage *models.Usage
		if event.Message.Usage.InputTokens > 0 {
			usage = &models.Usage{
				PromptTokens:     event.Message.Usage.InputTokens,
				CompletionTokens: 0,
				TotalTokens:      event.Message.Usage.InputTokens,
			}
		}

		return &models.StreamChunk{
			ID:      t.id,
			Object:  "chat.completion.chunk",
			Created: time.Now().Unix(),
			Model:   t.model,
			Choices: []models.Choice{{
				Index: 0,
				Delta: &models.Message{Role: "assistant"},
				FinishReason: nil,
			}},
			Usage: usage,
		}, nil

	case "content_block_start":
		if event.ContentBlock == nil {
			return nil, nil
		}

		switch event.ContentBlock.Type {
		case "text":
			if event.ContentBlock.Text == "" {
				return nil, nil
			}
			return &models.StreamChunk{
				ID:      t.id,
				Object:  "chat.completion.chunk",
				Created: time.Now().Unix(),
				Model:   t.model,
				Choices: []models.Choice{{
					Index: 0,
					Delta: &models.Message{Content: event.ContentBlock.Text},
					FinishReason: nil,
				}},
			}, nil

		case "tool_use":
			toolCallIdx, ok := t.toolCallIndexByContentBlockIndex[event.Index]
			if !ok {
				toolCallIdx = t.nextToolCallIndex
				t.toolCallIndexByContentBlockIndex[event.Index] = toolCallIdx
				t.nextToolCallIndex++
			}

			idx := toolCallIdx
			return &models.StreamChunk{
				ID:      t.id,
				Object:  "chat.completion.chunk",
				Created: time.Now().Unix(),
				Model:   t.model,
				Choices: []models.Choice{{
					Index: 0,
					Delta: &models.Message{
						ToolCalls: []models.ToolCall{{
							Index: &idx,
							ID:    event.ContentBlock.ID,
							Type:  "function",
							Function: models.ToolCallFunction{
								Name: event.ContentBlock.Name,
							},
						}},
					},
					FinishReason: nil,
				}},
			}, nil

		default:
			return nil, nil
		}

	case "content_block_delta":
		if event.Delta == nil {
			return nil, nil
		}
		switch event.Delta.Type {
		case "text_delta":
			if event.Delta.Text == "" {
				return nil, nil
			}
			return &models.StreamChunk{
				ID:      t.id,
				Object:  "chat.completion.chunk",
				Created: time.Now().Unix(),
				Model:   t.model,
				Choices: []models.Choice{{
					Index: 0,
					Delta: &models.Message{Content: event.Delta.Text},
					FinishReason: nil,
				}},
			}, nil

		case "input_json_delta":
			if event.Delta.PartialJSON == "" {
				return nil, nil
			}
			toolCallIdx, ok := t.toolCallIndexByContentBlockIndex[event.Index]
			if !ok {
				return nil, nil
			}
			idx := toolCallIdx
			return &models.StreamChunk{
				ID:      t.id,
				Object:  "chat.completion.chunk",
				Created: time.Now().Unix(),
				Model:   t.model,
				Choices: []models.Choice{{
					Index: 0,
					Delta: &models.Message{
						ToolCalls: []models.ToolCall{{
							Index: &idx,
							Type:  "function",
							Function: models.ToolCallFunction{
								Arguments: event.Delta.PartialJSON,
							},
						}},
					},
					FinishReason: nil,
				}},
			}, nil

		default:
			return nil, nil
		}

	case "message_delta":
		if event.Delta == nil {
			return nil, nil
		}

		fr := mapAnthropicStopReason(event.Delta.StopReason)
		var usage *models.Usage
		if event.Usage != nil {
			usage = &models.Usage{CompletionTokens: event.Usage.OutputTokens}
		}

		return &models.StreamChunk{
			ID:      t.id,
			Object:  "chat.completion.chunk",
			Created: time.Now().Unix(),
			Model:   t.model,
			Choices: []models.Choice{{
				Index:        0,
				Delta:        &models.Message{},
				FinishReason: fr,
			}},
			Usage: usage,
		}, nil

	default:
		return nil, nil
	}
}

func (t *anthropicStreamTranslator) SupportsStreaming() bool {
	return t.base.SupportsStreaming()
}

func (t *anthropicStreamTranslator) Models() []models.ModelInfo {
	return t.base.Models()
}
