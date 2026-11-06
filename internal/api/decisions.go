package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/middleware"
	"github.com/lunargate-ai/gateway/internal/modelid"
	"github.com/lunargate-ai/gateway/internal/observability"
	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/internal/resilience"
	"github.com/lunargate-ai/gateway/internal/routing"
	"github.com/lunargate-ai/gateway/pkg/models"
)

type parsedDecisionsKey struct{}

var pinnedJevModel = regexp.MustCompile(`^jev-[0-9]+\.[0-9]+\.[0-9]+$`)

func (h *Handler) effectiveDecisionModel(target routing.Target, requested string) string {
	model := h.effectiveTargetModel(target, requested)
	if model == "" || model == "auto" {
		if snapshot, ok := h.registry.Snapshot(target.Provider); ok {
			return snapshot.Translator.DefaultModel()
		}
	}
	return model
}

func (h *Handler) validateDecisionsTarget(target routing.Target, request *models.DecisionsRequest) error {
	snapshot, ok := h.registry.Snapshot(target.Provider)
	if !ok {
		return &models.CompatibilityError{Provider: target.Provider, Field: "model", Reason: "provider is not configured"}
	}
	translator, ok := snapshot.Translator.(models.DecisionsTranslator)
	if !ok {
		return &models.CompatibilityError{Provider: target.Provider, Field: "model", Reason: "provider does not implement native Decisions"}
	}
	model := h.effectiveDecisionModel(target, request.Model)
	if model == "" || model == "auto" {
		model = snapshot.Translator.DefaultModel()
	}
	return translator.ValidateDecisionsRequest(target.Provider, model, request)
}

func (h *Handler) decisionRequestTypes(target routing.Target) apiRequestTypes {
	upstream := "decisions"
	if snapshot, ok := h.registry.Snapshot(target.Provider); ok && snapshot.ProviderType == "typesafe" {
		upstream = "systemone"
	}
	return apiRequestTypes{client: "decisions", upstream: upstream}
}

func (h *Handler) callDecisionsProvider(ctx context.Context, target routing.Target, req *models.DecisionsRequest, before func()) (*http.Response, error) {
	snapshot, ok := circuitBreakerTargetSnapshotFromContext(ctx, target)
	if !ok {
		snapshot, ok = h.registry.Snapshot(target.Provider)
	}
	if !ok {
		return nil, resilience.NewRequestError(fmt.Errorf("provider is not configured"))
	}
	translator, ok := snapshot.Translator.(models.DecisionsTranslator)
	if !ok {
		return nil, resilience.NewRequestError(&models.CompatibilityError{Provider: target.Provider, Field: "model", Reason: "native Decisions is unavailable"})
	}
	copyReq := *req
	copyReq.Model = h.effectiveDecisionModel(target, req.Model)
	if copyReq.Model == "" || copyReq.Model == "auto" {
		copyReq.Model = snapshot.Translator.DefaultModel()
	}
	if err := translator.ValidateDecisionsRequest(target.Provider, copyReq.Model, &copyReq); err != nil {
		return nil, resilience.NewRequestError(err)
	}
	ctx = providers.WithUpstreamRequestHeaders(ctx, requestHeadersFromDecisionContext(ctx))
	ctx = withProviderRequestSnapshot(ctx, target.Provider, snapshot)
	httpReq, err := translator.TranslateDecisionsRequest(ctx, &copyReq)
	if err != nil {
		return nil, resilience.NewRequestError(err)
	}
	before()
	client := providerClientConfig{client: newProviderHTTPClient(), timeout: defaultUpstreamTimeout, mode: upstreamTimeoutModeTTFT}
	if h.providerClients != nil {
		if configured, ok := h.providerClients.Get(target.Provider); ok {
			client = configured
		}
	}
	resp, err := doProviderRequest(httpReq, client, target.Provider, "failed to call Decisions provider")
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		if resp.StatusCode < http.StatusBadRequest {
			_ = resp.Body.Close()
			return nil, invalidProviderResponseStatus(target.Provider, resp.StatusCode)
		}
		return resp, nil
	}
	parsed, err := translator.ParseDecisionsResponse(resp, &copyReq)
	if err != nil {
		return nil, &providerResponseParseError{cause: err}
	}
	resp.Request = resp.Request.WithContext(context.WithValue(resp.Request.Context(), parsedDecisionsKey{}, parsed))
	return resp, nil
}

type decisionHeadersKey struct{}

func requestHeadersFromDecisionContext(ctx context.Context) http.Header {
	headers, _ := ctx.Value(decisionHeadersKey{}).(http.Header)
	return headers
}

func (h *Handler) decisionCacheAllowed(target routing.Target, model string, r *http.Request) bool {
	if h.cache == nil || !h.cache.Enabled() || r.Header.Get("X-LunarGate-No-Cache") == "true" {
		return false
	}
	snapshot, ok := h.registry.Snapshot(target.Provider)
	return ok && (snapshot.ProviderType != "typesafe" || pinnedJevModel.MatchString(model))
}

// Decisions implements the native decision protocol; it never calls a
// generative endpoint or synthesizes probabilities from generated text.
func (h *Handler) Decisions(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	requestID := uuid.NewString()
	w.Header().Set("X-LunarGate-Request-ID", requestID)
	h.metrics.ActiveRequests.Inc()
	defer h.metrics.ActiveRequests.Dec()
	limitRequestBody(w, r)
	defer r.Body.Close()
	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeRequestReadError(w, err)
		return
	}
	var req models.DecisionsRequest
	if err := json.Unmarshal(body, &req); err != nil {
		writeRequestDecodeError(w, err)
		return
	}
	req.RawJSON = append(json.RawMessage(nil), body...)
	req.ProviderHeaders = make(http.Header)
	for _, header := range []string{"Idempotency-Key", "OpenAI-Beta", "Anthropic-Beta"} {
		for _, value := range r.Header.Values(header) {
			req.ProviderHeaders.Add(header, value)
		}
	}
	if err := req.Validate(); err != nil {
		var ce *models.CompatibilityError
		if errors.As(err, &ce) {
			writeCompatibilityError(w, ce)
		} else {
			writeError(w, 400, err.Error(), "invalid_request_error")
		}
		return
	}
	var document map[string]json.RawMessage
	_ = json.Unmarshal(body, &document)
	for _, field := range []string{"stream", "stream_options", "service_tier"} {
		if _, exists := document[field]; exists {
			writeCompatibilityError(w, &models.CompatibilityError{Field: field, Reason: "Decisions does not expose this control"})
			return
		}
	}
	if value := strings.TrimSpace(r.Header.Get("X-LunarGate-Model")); value != "" {
		req.Model = value
	}
	provider := strings.TrimSpace(r.Header.Get("X-LunarGate-Provider"))
	if p, model, ok := modelid.SplitCanonical(req.Model); ok {
		if p == "lunargate" && model == "auto" {
			req.Model = "lunargate/auto"
		} else {
			provider = p
		}
	} else if provider != "" {
		req.Model = modelid.BuildCanonical(provider, req.Model)
	}
	headers := extractHeaders(r)
	// The trusted API handler owns protocol and model constraints. A client
	// cannot smuggle an upstream or model-selection header around capability gates.
	delete(headers, "x-lunargate-provider")
	delete(headers, "x-lunargate-model")
	if req.Model != "lunargate/auto" {
		headers["x-lunargate-model"] = req.Model
	}
	if provider != "" {
		headers["x-lunargate-provider"] = provider
	}
	if h.collector != nil {
		if v := h.collector.GatewayLat(); v != "" {
			headers["x-lunargate-gateway-lat"] = v
		}
		if v := h.collector.GatewayLon(); v != "" {
			headers["x-lunargate-gateway-lon"] = v
		}
	}
	headers["x-lunargate-request-type"] = "decisions"
	var compatibilityErr error
	eligible := func(cfg config.TargetConfig) bool {
		target := routing.Target{Provider: cfg.Provider, Model: cfg.Model, Weight: cfg.Weight}
		if err := h.validateDecisionsTarget(target, &req); err != nil {
			if compatibilityErr == nil {
				compatibilityErr = err
			}
			return false
		}
		return true
	}
	resolved, err := h.router.ResolveEligible(r.Context(), r.URL.Path, routingHeadersForRequest(r, h.router.MatchHeaderNames(), headers), eligible)
	if err != nil && errors.Is(err, routing.ErrNoRouteMatched) && provider != "" && r.Header.Get("X-LunarGate-Route") == "" {
		model := modelid.ModelName(req.Model)
		direct := routing.Target{Provider: provider, Model: model, Weight: 100}
		if e := h.validateDecisionsTarget(direct, &req); e == nil {
			resolved = &routing.ResolvedRoute{RouteName: "decisions-direct", Target: direct}
			err = nil
		} else {
			compatibilityErr = e
		}
	}
	if err != nil {
		var ce *models.CompatibilityError
		if errors.As(compatibilityErr, &ce) {
			writeCompatibilityError(w, ce)
		} else {
			var unavailable *routing.RequestedTargetUnavailableError
			if errors.As(err, &unavailable) {
				writeRequestedTargetUnavailable(w, unavailable)
			} else {
				writeError(w, 502, "no native Decisions route is available", "routing_error")
			}
		}
		return
	}
	w.Header().Set("X-LunarGate-Route", resolved.RouteName)
	model := h.effectiveDecisionModel(resolved.Target, req.Model)
	types := h.decisionRequestTypes(resolved.Target)
	cacheKey := h.runtimeCacheKey(middleware.GenerateDecisionsKey(&req, resolved.Target.Provider, model, types.upstream, r.Header), resolved.Target.Provider)
	cacheAllowed := h.decisionCacheAllowed(resolved.Target, model, r)
	if cacheAllowed {
		if cached, ok := h.cache.Get(cacheKey).(*models.DecisionsResponse); ok && cached != nil {
			h.metrics.CacheHits.WithLabelValues("hit").Inc()
			w.Header().Set("X-LunarGate-Cache-Status", "HIT")
			h.writeDecisionsSuccess(w, r, start, requestID, &req, resolved, resolved.Target, cached, true, false, 0, "", -1)
			return
		}
		h.metrics.CacheHits.WithLabelValues("miss").Inc()
		w.Header().Set("X-LunarGate-Cache-Status", "MISS")
	}
	if h.collector != nil {
		tags := h.enrichCollectorTags(types.tags(headers), resolved.Target.Provider, modelid.BuildCanonical(resolved.Target.Provider, model), false)
		h.collector.Enqueue(r.Context(), requestID, []observability.Event{{Type: "trace", Data: observability.TraceEventData{RequestID: requestID, Timestamp: start.UTC(), Phase: "request_start", Tags: tags}}})
	}
	upstreamMS := int64(-1)
	before := func() {
		if upstreamMS < 0 {
			upstreamMS = time.Since(start).Milliseconds()
		}
	}
	ctx := context.WithValue(requestContextWithRetryPolicy(r), decisionHeadersKey{}, r.Header.Clone())
	ctx = h.withCircuitBreakerTargetSnapshots(ctx, resolved)
	resp, used, fallbackUsed, retryCount, breaker, err := h.fallback.Execute(ctx, resolved.Target, resolved.Fallbacks, func(ctx context.Context, target routing.Target) (*http.Response, error) {
		return h.callDecisionsProvider(ctx, target, &req, before)
	})
	h.observeCircuitBreakerState(used.Provider, breaker)
	var upstreamError *upstreamHTTPError
	providerType, _ := h.registry.Type(used.Provider)
	if err != nil {
		upstreamError, _ = upstreamHTTPErrorFromRetry(err, providerType)
	} else if resp.StatusCode != http.StatusOK {
		upstreamError = readUpstreamHTTPError(resp, providerType)
	}
	if err != nil || upstreamError != nil {
		status, code, message := http.StatusBadGateway, "provider_error", "Decisions provider failed"
		if upstreamError != nil {
			status, code, message = upstreamError.status, upstreamError.errType, upstreamError.message
		} else if isClientRequestTermination(r.Context(), err) {
			status, code, message = 499, "client_cancelled", "client disconnected"
		} else if s, c, m, ok := requestFailureFromError(err); ok {
			status, code, message = s, c, m
		} else if s, c, m, ok := classifyProviderResponseError(err); ok {
			status, code, message = s, c, m
		}
		w.Header().Set("X-LunarGate-Provider", used.Provider)
		w.Header().Set("X-LunarGate-Model", modelid.BuildCanonical(used.Provider, h.effectiveDecisionModel(used, req.Model)))
		w.Header().Set("X-LunarGate-Upstream-Request-Type", h.decisionRequestTypes(used).upstream)
		setTimingHeaders(w, time.Since(start).Milliseconds(), upstreamMS)
		if upstreamError != nil {
			upstreamError.write(w)
		} else {
			writeError(w, status, message, code)
		}
		var errorBody json.RawMessage
		if upstreamError != nil {
			errorBody = upstreamError.body
		} else {
			errorBody, _ = json.Marshal(models.ErrorResponse{Error: models.ErrorDetail{Type: code, Message: message}})
		}
		h.recordDecisions(r, start, requestID, &req, resolved, used, nil, errorBody, status, false, fallbackUsed, retryCount, breaker, upstreamMS, code, message)
		return
	}
	parsed, ok := resp.Request.Context().Value(parsedDecisionsKey{}).(*models.DecisionsResponse)
	if !ok || parsed == nil {
		writeError(w, 502, "parsed Decisions response is unavailable", "provider_error")
		h.recordDecisions(r, start, requestID, &req, resolved, used, nil, nil, 502, false, fallbackUsed, retryCount, breaker, upstreamMS, "provider_error", "parsed Decisions response is unavailable")
		return
	}
	copyHeaders(w.Header(), resp.Header)
	if fallbackUsed {
		h.metrics.FallbacksUsed.Inc()
	}
	// Never store a fallback response under the initially selected target.
	if !fallbackUsed && cacheAllowed {
		h.cache.Set(cacheKey, parsed)
	}
	h.writeDecisionsSuccess(w, r, start, requestID, &req, resolved, used, parsed, false, fallbackUsed, retryCount, breaker, upstreamMS)
}

func (h *Handler) writeDecisionsSuccess(w http.ResponseWriter, r *http.Request, start time.Time, id string, req *models.DecisionsRequest, route *routing.ResolvedRoute, target routing.Target, result *models.DecisionsResponse, cached, fallback bool, retries int, breaker string, upstreamMS int64) {
	w.Header().Set("X-LunarGate-Provider", target.Provider)
	w.Header().Set("X-LunarGate-Model", modelid.BuildCanonical(target.Provider, result.Model))
	w.Header().Set("X-LunarGate-Upstream-Request-Type", h.decisionRequestTypes(target).upstream)
	setTimingHeaders(w, time.Since(start).Milliseconds(), upstreamMS)
	if cached && h.collector != nil {
		tags := h.enrichCollectorTags(h.decisionRequestTypes(target).tags(extractHeaders(r)), target.Provider, modelid.BuildCanonical(target.Provider, result.Model), false)
		h.collector.Enqueue(r.Context(), id, []observability.Event{{Type: "trace", Data: observability.TraceEventData{RequestID: id, Timestamp: start.UTC(), Phase: "request_start", Tags: tags}}})
	}
	h.recordDecisions(r, start, id, req, route, target, result, nil, 200, cached, fallback, retries, breaker, upstreamMS, "", "")
	writeJSON(w, 200, result)
}

func (h *Handler) recordDecisions(r *http.Request, start time.Time, id string, req *models.DecisionsRequest, route *routing.ResolvedRoute, target routing.Target, result *models.DecisionsResponse, errorBody json.RawMessage, status int, cached, fallback bool, retries int, breaker string, upstreamMS int64, errorCode, errorMessage string) {
	duration := time.Since(start)
	metricModel := boundedModelMetricLabel(target, "")
	h.metrics.RequestsTotal.WithLabelValues(target.Provider, metricModel, strconv.Itoa(status), route.RouteName).Inc()
	h.metrics.RequestDuration.WithLabelValues(target.Provider, metricModel).Observe(duration.Seconds())
	usage := models.TokenUsage{}
	model := h.effectiveDecisionModel(target, req.Model)
	if result != nil {
		usage = result.Usage.TokenUsage()
		model = result.Model
		if !cached {
			h.metrics.ObserveTokenUsage(target.Provider, metricModel, usage)
		}
	}
	if status >= 400 && status != 499 {
		h.metrics.ProviderErrors.WithLabelValues(target.Provider, boundedProviderErrorMetricType(status, errorCode)).Inc()
	}
	if h.collector == nil {
		return
	}
	model = modelid.BuildCanonical(target.Provider, model)
	headers := extractHeaders(r)
	types := h.decisionRequestTypes(target)
	if v := h.collector.GatewayLat(); v != "" {
		headers["x-lunargate-gateway-lat"] = v
	}
	if v := h.collector.GatewayLon(); v != "" {
		headers["x-lunargate-gateway-lon"] = v
	}
	tags := h.enrichCollectorTags(types.tags(headers), target.Provider, model, false)
	user, session := extractUserAndSession(headers)
	routeName, index := route.RouteName, route.Index
	var before *int64
	if upstreamMS >= 0 {
		before = &upstreamMS
	}
	metric := observability.MetricEventData{RequestID: id, Timestamp: start.UTC(), RequestType: types.client, UpstreamRequestType: types.upstream, DurationMS: duration.Milliseconds(), GatewayPreUpstreamMS: before, Provider: target.Provider, Model: model, User: user, SessionID: session, TokensInput: usage.InputTokens, TokensOutput: usage.OutputTokens, TokensInputCached: usage.CachedInputTokens, TokensInputCacheWrite: usage.CacheWriteInputTokens, StatusCode: status, ErrorCode: observability.MetricErrorClass(status, status >= 400), CacheHit: cached, RouteUsed: &routeName, TargetIndex: &index, FallbackUsed: fallback, RetryCount: retries, CircuitBreakerState: &breaker, Tags: tags}
	events := []observability.Event{{Type: "metric", Data: metric}}
	if h.collector.SharePrompts() || h.collector.ShareResponses() {
		var requestPayload, responsePayload interface{}
		if h.collector.SharePrompts() {
			requestPayload = json.RawMessage(req.RawJSON)
		}
		if h.collector.ShareResponses() && result != nil {
			responsePayload = result
		} else if h.collector.ShareResponses() && len(errorBody) != 0 {
			responsePayload = errorBody
		}
		entry := observability.RequestLogEventData{RequestID: id, Timestamp: start.UTC(), RequestType: types.client, UpstreamRequestType: types.upstream, Provider: target.Provider, Model: model, User: user, SessionID: session, StatusCode: status, DurationMS: duration.Milliseconds(), RouteUsed: &routeName, CacheHit: cached, FallbackUsed: fallback, RetryCount: retries, Tags: tags, Request: requestPayload, Response: responsePayload}
		if errorCode != "" {
			entry.ErrorCode = &errorCode
			entry.ErrorMessage = &errorMessage
		}
		events = append(events, observability.Event{Type: "request_log", Data: entry})
	}
	h.collector.Enqueue(r.Context(), id, events)
}
