package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/lunargate-ai/gateway/pkg/models"
)

type ollamaStreamTranslator struct {
	base *OllamaTranslator

	id      string
	created int64
	model   string
}

func NewOllamaStreamTranslator(base *OllamaTranslator) models.ProviderTranslator {
	return &ollamaStreamTranslator{
		base:    base,
		id:      fmt.Sprintf("chatcmpl_ollama_%d", time.Now().UnixNano()),
		created: time.Now().Unix(),
	}
}

func (t *ollamaStreamTranslator) Name() string {
	return t.base.Name()
}

func (t *ollamaStreamTranslator) DefaultModel() string {
	return t.base.DefaultModel()
}

func (t *ollamaStreamTranslator) BaseURL() string {
	return t.base.BaseURL()
}

func (t *ollamaStreamTranslator) TranslateRequest(ctx context.Context, req *models.UnifiedRequest) (*http.Request, error) {
	return t.base.TranslateRequest(ctx, req)
}

func (t *ollamaStreamTranslator) ParseResponse(resp *http.Response) (*models.UnifiedResponse, error) {
	return t.base.ParseResponse(resp)
}

func (t *ollamaStreamTranslator) ParseStreamChunk(data []byte) (*models.StreamChunk, error) {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 {
		return nil, nil
	}

	var ev ollamaChatResponse
	if err := json.Unmarshal(trimmed, &ev); err != nil {
		return nil, fmt.Errorf("failed to unmarshal ollama stream chunk: %w", err)
	}
	if strings.TrimSpace(ev.Error) != "" {
		return nil, &ProviderError{StatusCode: http.StatusBadGateway, Provider: "ollama", Type: "upstream_error", Message: strings.TrimSpace(ev.Error)}
	}

	if t.model == "" {
		t.model = strings.TrimSpace(ev.Model)
	}

	finishReason := (*string)(nil)
	var usage *models.Usage
	if ev.Done {
		finishReason = mapOllamaDoneReason(ev.DoneReason)
		if finishReason == nil {
			fr := "stop"
			finishReason = &fr
		}
		if ev.PromptEvalCount > 0 || ev.EvalCount > 0 {
			usage = &models.Usage{
				PromptTokens:     ev.PromptEvalCount,
				CompletionTokens: ev.EvalCount,
				TotalTokens:      ev.PromptEvalCount + ev.EvalCount,
			}
		}
	}

	msg := &models.Message{Role: "assistant"}
	needChunk := false

	if ev.Message.Content != "" {
		msg.Content = ev.Message.Content
		needChunk = true
	}

	reasoning := ""
	if ev.Message.Reasoning != "" {
		reasoning = ev.Message.Reasoning
	} else if ev.Message.Thinking != "" {
		reasoning = ev.Message.Thinking
	}
	if reasoning != "" {
		msg.ReasoningContent = reasoning
		needChunk = true
	}

	if len(ev.Message.ToolCalls) > 0 {
		needChunk = true
		msg.ToolCalls = make([]models.ToolCall, 0, len(ev.Message.ToolCalls))
		for i := range ev.Message.ToolCalls {
			idx := i
			id := fmt.Sprintf("call_%s_%d", t.id, i)
			args := "{}"
			if len(ev.Message.ToolCalls[i].Function.Arguments) > 0 {
				args = string(ev.Message.ToolCalls[i].Function.Arguments)
			}
			msg.ToolCalls = append(msg.ToolCalls, models.ToolCall{
				Index: &idx,
				ID:    id,
				Type:  "function",
				Function: models.ToolCallFunction{
					Name:      ev.Message.ToolCalls[i].Function.Name,
					Arguments: args,
				},
			})
		}
	}

	if !needChunk {
		if finishReason == nil {
			return nil, nil
		}
		msg = nil
	}

	return &models.StreamChunk{
		ID:      t.id,
		Object:  "chat.completion.chunk",
		Created: t.created,
		Model:   t.model,
		Choices: []models.Choice{{
			Index:        0,
			Delta:        msg,
			FinishReason: finishReason,
		}},
		Usage: usage,
	}, nil
}

func (t *ollamaStreamTranslator) SupportsStreaming() bool {
	return true
}

func (t *ollamaStreamTranslator) Models() []models.ModelInfo {
	return t.base.Models()
}
