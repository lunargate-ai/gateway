package api

import (
	"strings"

	"github.com/lunargate-ai/gateway/internal/routing"
)

const (
	requestTypeChatCompletions = "chat_completions"
	requestTypeResponses       = "responses"
	requestTypeEmbeddings      = "embeddings"
)

type apiRequestTypes struct {
	client   string
	upstream string
}

func chatAPIRequestTypes(client string, target routing.Target) apiRequestTypes {
	client = canonicalAPIRequestType(client)
	if client == "" {
		client = requestTypeChatCompletions
	}

	upstream := requestTypeChatCompletions
	if strings.EqualFold(strings.TrimSpace(target.UpstreamRequestType), requestTypeResponses) {
		upstream = requestTypeResponses
	}
	return apiRequestTypes{client: client, upstream: upstream}
}

func embeddingsAPIRequestTypes() apiRequestTypes {
	return apiRequestTypes{client: requestTypeEmbeddings, upstream: requestTypeEmbeddings}
}

func canonicalAPIRequestType(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	if value == "chat" {
		return requestTypeChatCompletions
	}
	return value
}

func (t apiRequestTypes) tags(base map[string]string) map[string]string {
	tags := make(map[string]string, len(base)+2)
	for key, value := range base {
		tags[key] = value
	}
	tags["x-lunargate-request-type"] = t.client
	tags["x-lunargate-upstream-request-type"] = t.upstream
	return tags
}
