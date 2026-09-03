package api

import (
	"context"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/lunargate-ai/gateway/internal/observability"
	"github.com/lunargate-ai/gateway/internal/routing"
)

const dynamicModelMetricLabel = "_dynamic"

func boundedModelMetricLabel(target routing.Target, resolvedModel string) string {
	if strings.TrimSpace(target.Model) == "" {
		return dynamicModelMetricLabel
	}
	if model := strings.TrimSpace(resolvedModel); model != "" {
		return model
	}
	return dynamicModelMetricLabel
}

func boundedProviderErrorMetricType(status int, candidate string) string {
	switch strings.ToLower(strings.TrimSpace(candidate)) {
	case "parse_error", "upstream_timeout", "invalid_response_status", "provider_error":
		return strings.ToLower(strings.TrimSpace(candidate))
	}
	if class := observability.MetricErrorClass(status, true); class != nil && strings.TrimSpace(*class) != "" {
		return strings.TrimSpace(*class)
	}
	return "upstream_error"
}

type cacheHitObservation struct {
	requestID    string
	startTime    time.Time
	requestTypes apiRequestTypes
	provider     string
	model        string
	metricsModel string
	route        string
	targetIndex  int
	user         *string
	sessionID    *string
	tags         map[string]string
	tokensInput  int
	tokensOutput int
	request      interface{}
	response     interface{}
}

func (h *Handler) recordCacheHit(ctx context.Context, observation cacheHitObservation) time.Duration {
	duration := time.Since(observation.startTime)
	h.metrics.RequestsTotal.WithLabelValues(
		observation.provider,
		observation.metricsModel,
		strconv.Itoa(http.StatusOK),
		observation.route,
	).Inc()
	h.metrics.RequestDuration.WithLabelValues(observation.provider, observation.metricsModel).Observe(duration.Seconds())

	if h.collector == nil {
		return duration
	}

	h.collector.Enqueue(ctx, observation.requestID, []observability.Event{{
		Type: "trace",
		Data: observability.TraceEventData{
			RequestID: observation.requestID,
			Timestamp: observation.startTime.UTC(),
			Phase:     "request_start",
			Tags:      observation.tags,
		},
	}})

	routeUsed := observation.route
	targetIndex := observation.targetIndex
	events := []observability.Event{{
		Type: "metric",
		Data: observability.MetricEventData{
			RequestID:           observation.requestID,
			Timestamp:           observation.startTime.UTC(),
			RequestType:         observation.requestTypes.client,
			UpstreamRequestType: observation.requestTypes.upstream,
			DurationMS:          duration.Milliseconds(),
			Provider:            observation.provider,
			Model:               observation.model,
			User:                observation.user,
			SessionID:           observation.sessionID,
			TokensInput:         observation.tokensInput,
			TokensOutput:        observation.tokensOutput,
			CostUSD:             0,
			StatusCode:          http.StatusOK,
			CacheHit:            true,
			RouteUsed:           &routeUsed,
			TargetIndex:         &targetIndex,
			FallbackUsed:        false,
			RetryCount:          0,
			Tags:                observation.tags,
		},
	}}

	if h.collector.SharePrompts() || h.collector.ShareResponses() {
		var request interface{}
		if h.collector.SharePrompts() {
			request = observation.request
		}
		var response interface{}
		if h.collector.ShareResponses() {
			response = observation.response
		}
		events = append(events, observability.Event{
			Type: "request_log",
			Data: observability.RequestLogEventData{
				RequestID:           observation.requestID,
				Timestamp:           observation.startTime.UTC(),
				RequestType:         observation.requestTypes.client,
				UpstreamRequestType: observation.requestTypes.upstream,
				User:                observation.user,
				SessionID:           observation.sessionID,
				Provider:            observation.provider,
				Model:               observation.model,
				StatusCode:          http.StatusOK,
				DurationMS:          duration.Milliseconds(),
				RouteUsed:           &routeUsed,
				CacheHit:            true,
				FallbackUsed:        false,
				RetryCount:          0,
				Tags:                observation.tags,
				Request:             request,
				Response:            response,
			},
		})
	}

	h.collector.Enqueue(ctx, observation.requestID, events)
	return duration
}
