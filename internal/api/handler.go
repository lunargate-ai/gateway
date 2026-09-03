package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/google/uuid"
	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/internal/middleware"
	"github.com/lunargate-ai/gateway/internal/modelid"
	"github.com/lunargate-ai/gateway/internal/modelselect"
	"github.com/lunargate-ai/gateway/internal/modelstore"
	"github.com/lunargate-ai/gateway/internal/observability"
	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/internal/resilience"
	"github.com/lunargate-ai/gateway/internal/routing"
	"github.com/lunargate-ai/gateway/internal/streaming"
	"github.com/lunargate-ai/gateway/pkg/models"
	"github.com/rs/zerolog/log"
)

// Handler is the main API handler that orchestrates the request lifecycle.
type Handler struct {
	registry               *providers.Registry
	router                 *routing.Engine
	fallback               *resilience.FallbackExecutor
	cache                  *middleware.Cache
	streamer               *streaming.Handler
	metrics                *observability.Metrics
	collector              *observability.CollectorClient
	selector               *modelselect.Engine
	store                  *modelstore.Store
	providerClients        *providerClientRegistry
	responsesState         *responsesStateStore
	responseBindings       *responseBindingStore
	chatCompletionBindings *chatCompletionBindingStore
	conversationBindings   *conversationBindingStore
	conversationsState     *conversationStateStore
}

type trackedResponseWriter struct {
	http.ResponseWriter
	wroteHeader bool
}

type trackedFlusherResponseWriter struct {
	*trackedResponseWriter
	flusher http.Flusher
}

type capturedResponseWriter struct {
	headers    http.Header
	statusCode int
	body       bytes.Buffer
}

func (w *trackedResponseWriter) WriteHeader(statusCode int) {
	w.wroteHeader = true
	w.ResponseWriter.WriteHeader(statusCode)
}

func (w *trackedResponseWriter) Write(p []byte) (int, error) {
	if !w.wroteHeader {
		w.wroteHeader = true
	}
	return w.ResponseWriter.Write(p)
}

func (w *trackedFlusherResponseWriter) Flush() {
	_ = w.FlushError()
}

func (w *trackedFlusherResponseWriter) FlushError() error {
	if flusher, ok := w.ResponseWriter.(interface{ FlushError() error }); ok {
		return flusher.FlushError()
	}
	w.flusher.Flush()
	return nil
}

func newCapturedResponseWriter() *capturedResponseWriter {
	return &capturedResponseWriter{
		headers: make(http.Header),
	}
}

func (w *capturedResponseWriter) Header() http.Header {
	return w.headers
}

func (w *capturedResponseWriter) WriteHeader(statusCode int) {
	w.statusCode = statusCode
}

func (w *capturedResponseWriter) Write(p []byte) (int, error) {
	if w.statusCode == 0 {
		w.statusCode = http.StatusOK
	}
	return w.body.Write(p)
}

func newProviderHTTPClient(timeout time.Duration) *http.Client {
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.MaxIdleConns = 2048
	transport.MaxIdleConnsPerHost = 1024
	transport.IdleConnTimeout = 90 * time.Second
	transport.ResponseHeaderTimeout = timeout
	return &http.Client{
		Transport: transport,
		CheckRedirect: func(_ *http.Request, _ []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}
}

func writeRequestReadError(w http.ResponseWriter, err error) {
	var mbe *http.MaxBytesError
	if errors.As(err, &mbe) {
		writeError(w, http.StatusRequestEntityTooLarge, "request body too large", "invalid_request_error")
		return
	}
	writeError(w, http.StatusBadRequest, "failed to read request body", "invalid_request_error")
}

func writeRequestDecodeError(w http.ResponseWriter, err error) {
	var mbe *http.MaxBytesError
	if errors.As(err, &mbe) {
		writeError(w, http.StatusRequestEntityTooLarge, "request body too large", "invalid_request_error")
		return
	}
	writeError(w, http.StatusBadRequest, "invalid JSON in request body", "invalid_request_error")
}

func parseUnifiedRequest(w http.ResponseWriter, r *http.Request, captureBody bool) ([]byte, *models.UnifiedRequest, bool) {
	limitRequestBody(w, r)
	defer r.Body.Close()

	var req models.UnifiedRequest
	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeRequestReadError(w, err)
		return nil, nil, false
	}
	if err := decodeJSONStrict(bytes.NewReader(body), &req); err != nil {
		writeRequestDecodeError(w, err)
		return nil, nil, false
	}
	req.RawJSON = append(json.RawMessage(nil), body...)
	req.SourceRequestType = "chat_completions"
	if preserved, ok := preservedUnifiedRequestFromContext(r.Context()); ok {
		req.RawJSON = append(json.RawMessage(nil), preserved.rawJSON...)
		req.SourceRequestType = preserved.sourceRequestType
		body = append([]byte(nil), preserved.rawJSON...)
	}

	if err := models.NormalizeUnifiedRequest(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid tool/function calling payload", "invalid_request_error")
		return nil, nil, false
	}

	if !captureBody {
		return nil, &req, true
	}
	return body, &req, true
}

type collectorInferenceParameters struct {
	Temperature *float64
	TopP        *float64
	TopK        *int
}

func (p collectorInferenceParameters) hasAny() bool {
	return p.Temperature != nil || p.TopP != nil || p.TopK != nil
}

func buildCollectorRequestLogPayload(body []byte, params collectorInferenceParameters) interface{} {
	var reqAny interface{}
	if len(body) > 0 {
		_ = json.Unmarshal(body, &reqAny)
	}
	if !params.hasAny() {
		return reqAny
	}

	var requestObj map[string]interface{}
	if existing, ok := reqAny.(map[string]interface{}); ok && existing != nil {
		requestObj = existing
	} else {
		requestObj = map[string]interface{}{}
		if reqAny != nil {
			requestObj["request_raw"] = reqAny
		}
	}

	meta := map[string]interface{}{}
	if existingMeta, ok := requestObj["_lunargate"].(map[string]interface{}); ok && existingMeta != nil {
		meta = existingMeta
	}

	inference := map[string]interface{}{}
	if params.Temperature != nil {
		inference["temperature"] = *params.Temperature
	}
	if params.TopP != nil {
		inference["top_p"] = *params.TopP
	}
	if params.TopK != nil {
		inference["top_k"] = *params.TopK
	}
	meta["inference_parameters"] = inference
	requestObj["_lunargate"] = meta

	return requestObj
}

func setTimingHeaders(w http.ResponseWriter, totalMS int64, overheadMS int64) {
	if totalMS >= 0 {
		w.Header().Set("X-LunarGate-Latency-Ms", strconv.FormatInt(totalMS, 10))
	}
	if overheadMS < 0 {
		overheadMS = totalMS
	}
	if overheadMS >= 0 {
		w.Header().Set("X-LunarGate-Overhead-Duration-Ms", strconv.FormatInt(overheadMS, 10))
	}
}

func requestContextWithRetryPolicy(r *http.Request) context.Context {
	ctx := r.Context()
	if strings.EqualFold(strings.TrimSpace(r.Header.Get("X-LunarGate-No-Retry")), "true") {
		ctx = resilience.WithRetryDisabled(ctx)
	}
	if strings.EqualFold(strings.TrimSpace(r.Header.Get("X-LunarGate-No-Fallback")), "true") {
		ctx = resilience.WithFallbackDisabled(ctx)
	}
	return ctx
}

func (h *Handler) observeCircuitBreakerState(provider string, state string) {
	if h == nil || h.metrics == nil {
		return
	}
	if provider == "" || state == "" {
		return
	}

	value := 0.0
	switch strings.ToLower(strings.TrimSpace(state)) {
	case "half-open":
		value = 1
	case "open":
		value = 2
	}
	h.metrics.CircuitBreakerState.WithLabelValues(provider).Set(value)
}

// NewHandler creates a new API handler with all dependencies.
func NewHandler(
	registry *providers.Registry,
	router *routing.Engine,
	fallback *resilience.FallbackExecutor,
	cache *middleware.Cache,
	streamer *streaming.Handler,
	metrics *observability.Metrics,
	collector *observability.CollectorClient,
	selector *modelselect.Engine,
	store *modelstore.Store,
) *Handler {
	return &Handler{
		registry:               registry,
		router:                 router,
		fallback:               fallback,
		cache:                  cache,
		streamer:               streamer,
		metrics:                metrics,
		collector:              collector,
		selector:               selector,
		store:                  store,
		providerClients:        newProviderClientRegistry(nil),
		responsesState:         newResponsesStateStore(30 * time.Minute),
		responseBindings:       newResponseBindingStore(30 * time.Minute),
		chatCompletionBindings: newChatCompletionBindingStore(30 * time.Minute),
		conversationBindings:   newConversationBindingStore(30 * time.Minute),
		conversationsState:     newConversationStateStore(30 * time.Minute),
	}
}

func (h *Handler) UpdateProviderConfigs(providerConfigs map[string]config.ProviderConfig) {
	if h == nil || h.providerClients == nil {
		return
	}
	h.providerClients.Update(providerConfigs)
}

// ChatCompletions handles POST /v1/chat/completions.
func (h *Handler) ChatCompletions(w http.ResponseWriter, r *http.Request) {
	startTime := time.Now()
	requestID := uuid.New().String()

	cacheHit := false

	h.metrics.ActiveRequests.Inc()
	defer h.metrics.ActiveRequests.Dec()

	// Set request ID header early
	w.Header().Set("X-LunarGate-Request-ID", requestID)

	captureBody := h.collector != nil && h.collector.SharePrompts()
	body, parsedReq, ok := parseUnifiedRequest(w, r, captureBody)
	if !ok {
		return
	}
	req := *parsedReq

	explicitProvider := strings.TrimSpace(r.Header.Get("X-LunarGate-Provider"))
	explicitModel := strings.TrimSpace(r.Header.Get("X-LunarGate-Model"))
	autoSelection := strings.EqualFold(strings.TrimSpace(req.Model), "lunargate/auto") ||
		strings.EqualFold(explicitModel, "lunargate/auto")
	if explicitModel != "" {
		if p, m, ok := modelid.SplitCanonical(explicitModel); ok {
			explicitProvider = p
			req.Model = modelid.BuildCanonical(p, m)
		} else if explicitProvider != "" {
			req.Model = modelid.BuildCanonical(explicitProvider, explicitModel)
		} else {
			req.Model = explicitModel
		}
	}
	if explicitProvider != "" {
		if req.Model != "" {
			if p, m, ok := modelid.SplitCanonical(req.Model); ok {
				if strings.TrimSpace(p) == "" {
					req.Model = modelid.BuildCanonical(explicitProvider, m)
				}
			} else {
				req.Model = modelid.BuildCanonical(explicitProvider, req.Model)
			}
		}
	}

	if autoSelection {
		req.Model = ""
		if strings.EqualFold(strings.TrimSpace(explicitProvider), "lunargate") {
			explicitProvider = ""
		}
	}

	userSpecifiedModel := strings.TrimSpace(req.Model) != ""

	headers := extractHeaders(r)
	if autoSelection {
		delete(headers, "x-lunargate-model")
		if explicitProvider == "" {
			delete(headers, "x-lunargate-provider")
		}
	}
	requestType := canonicalAPIRequestType(req.SourceRequestType)
	if requestType == "" {
		requestType = canonicalAPIRequestType(headers["x-lunargate-request-type"])
	}
	if requestType == "" {
		requestType = requestTypeChatCompletions
	}
	headers["x-lunargate-request-type"] = requestType
	if req.Model != "" {
		headers["x-lunargate-model"] = strings.TrimSpace(req.Model)
		if p, _, ok := modelid.SplitCanonical(req.Model); ok {
			headers["x-lunargate-provider"] = strings.TrimSpace(p)
		}
	}
	if explicitProvider != "" {
		headers["x-lunargate-provider"] = explicitProvider
		if req.Model != "" {
			if _, _, ok := modelid.SplitCanonical(req.Model); !ok {
				req.Model = modelid.BuildCanonical(explicitProvider, req.Model)
				headers["x-lunargate-model"] = req.Model
			}
		}
	}

	if h.collector != nil {
		if v := strings.TrimSpace(h.collector.GatewayLat()); v != "" {
			headers["x-lunargate-gateway-lat"] = v
		}
		if v := strings.TrimSpace(h.collector.GatewayLon()); v != "" {
			headers["x-lunargate-gateway-lon"] = v
		}
	}

	if h.selector != nil && h.selector.Enabled() {
		cfg := h.selector.Config()
		if cfg.OverrideUserModel || !userSpecifiedModel {
			h.selector.EnrichHeaders(&req, headers)
		}
	}

	resolvePath := r.URL.Path
	originalPath := strings.TrimSpace(r.Header.Get("X-LunarGate-Original-Path"))
	if strings.EqualFold(requestType, "responses") && originalPath != "" {
		resolvePath = originalPath
	}
	resolved, err := h.router.Resolve(r.Context(), resolvePath, headers)
	if err != nil && strings.EqualFold(requestType, "responses") && originalPath != "" && originalPath != r.URL.Path {
		resolvePath = r.URL.Path
		resolved, err = h.router.Resolve(r.Context(), resolvePath, headers)
	}
	if err != nil {
		var unavailable *routing.RequestedTargetUnavailableError
		if errors.As(err, &unavailable) {
			writeRequestedTargetUnavailable(w, unavailable)
			return
		}
		log.Error().Err(err).Str("request_id", requestID).Msg("failed to resolve route")
		writeError(w, http.StatusBadGateway, "no route matched for this request", "routing_error")
		return
	}
	if err := h.validateChatCompatibility(resolved.Target, &req); err != nil {
		var compatibilityErr *models.CompatibilityError
		if errors.As(err, &compatibilityErr) {
			writeCompatibilityError(w, compatibilityErr)
			return
		}
		writeError(w, http.StatusBadRequest, err.Error(), "invalid_request_error")
		return
	}
	resolved.Fallbacks = h.compatibleChatFallbacks(resolved.Fallbacks, &req)
	w.Header().Set("X-LunarGate-Route", resolved.RouteName)

	requestedProvider := ""
	requestedModelRaw := ""
	if p, m, ok := modelid.SplitCanonical(req.Model); ok {
		requestedProvider = strings.TrimSpace(p)
		requestedModelRaw = strings.TrimSpace(m)
	} else {
		requestedModelRaw = strings.TrimSpace(req.Model)
	}

	overrideUserModel := false
	if h.selector != nil {
		overrideUserModel = h.selector.Config().OverrideUserModel
	}

	if !overrideUserModel && requestedProvider != "" {
		if strings.TrimSpace(resolved.Target.Provider) != requestedProvider {
			writeError(w, http.StatusBadRequest, "requested provider is not available for this route", "invalid_request_error")
			return
		}
		if requestedModelRaw != "" && strings.TrimSpace(resolved.Target.Model) != "" {
			if strings.TrimSpace(resolved.Target.Model) != requestedModelRaw {
				writeError(w, http.StatusBadRequest, "requested model is not available for this route", "invalid_request_error")
				return
			}
		}
	}

	if overrideUserModel || strings.TrimSpace(req.Model) == "" {
		model := strings.TrimSpace(resolved.Target.Model)
		if model == "" {
			if tr, ok := h.registry.Get(resolved.Target.Provider); ok {
				model = strings.TrimSpace(tr.DefaultModel())
			}
		}
		if model != "" {
			req.Model = modelid.BuildCanonical(resolved.Target.Provider, model)
			headers["x-lunargate-model"] = req.Model
		}
	} else {
		if _, _, ok := modelid.SplitCanonical(req.Model); !ok {
			if p := strings.TrimSpace(resolved.Target.Provider); p != "" {
				req.Model = modelid.BuildCanonical(p, req.Model)
				headers["x-lunargate-model"] = req.Model
			}
		}
	}
	resolvedSampling := h.resolveCollectorInferenceParameters(resolved.Target.Provider, &req)
	resolvedRequestTypes := chatAPIRequestTypes(requestType, resolved.Target)
	resolvedCollectorHeaders := resolvedRequestTypes.tags(headers)

	// An explicit storage policy is part of the upstream side effect. Replaying
	// a cached body for store:true would skip creation of the stored upstream
	// object, while caching store:false would violate the caller's policy.
	noCache := r.Header.Get("X-LunarGate-No-Cache") == "true" || req.Store != nil
	if !req.Stream && !noCache && h.cache.Enabled() {
		cacheKey := middleware.GenerateKeyForTarget(&req, resolved.Target.Provider, resolved.Target.UpstreamRequestType)
		if cached := h.cache.Get(cacheKey); cached != nil {
			h.metrics.CacheHits.WithLabelValues("hit").Inc()
			cacheHit = true
			cachedModelRaw := strings.TrimSpace(resolved.Target.Model)
			if cachedModelRaw == "" {
				cachedModelRaw = modelid.ModelName(req.Model)
				if cachedModelRaw == "" {
					if translator, ok := h.registry.Get(resolved.Target.Provider); ok {
						cachedModelRaw = strings.TrimSpace(translator.DefaultModel())
					}
				}
			}
			cachedModelCanonical := modelid.BuildCanonical(resolved.Target.Provider, cachedModelRaw)
			userPtr, sessionIDPtr := extractUserAndSession(headers)
			var tokensIn, tokensOut int
			switch response := cached.(type) {
			case *models.UnifiedResponse:
				if response != nil && response.Usage != nil {
					tokensIn = response.Usage.PromptTokens
					tokensOut = response.Usage.CompletionTokens
				}
			case models.UnifiedResponse:
				if response.Usage != nil {
					tokensIn = response.Usage.PromptTokens
					tokensOut = response.Usage.CompletionTokens
				}
			}
			var requestPayload interface{}
			if h.collector != nil && h.collector.SharePrompts() {
				requestPayload = buildCollectorRequestLogPayload(body, resolvedSampling)
			}
			duration := h.recordCacheHit(r.Context(), cacheHitObservation{
				requestID:    requestID,
				startTime:    startTime,
				requestTypes: resolvedRequestTypes,
				provider:     resolved.Target.Provider,
				model:        cachedModelCanonical,
				metricsModel: boundedModelMetricLabel(resolved.Target, cachedModelRaw),
				route:        resolved.RouteName,
				targetIndex:  resolved.Index,
				user:         userPtr,
				sessionID:    sessionIDPtr,
				tags: h.enrichCollectorTagsWithInference(
					resolvedCollectorHeaders,
					resolved.Target.Provider,
					cachedModelCanonical,
					false,
					resolvedSampling,
				),
				tokensInput:  tokensIn,
				tokensOutput: tokensOut,
				request:      requestPayload,
				response:     cached,
			})
			w.Header().Set("X-LunarGate-Cache-Status", "HIT")
			w.Header().Set("X-LunarGate-Provider", resolved.Target.Provider)
			w.Header().Set("X-LunarGate-Model", cachedModelCanonical)
			setTimingHeaders(w, duration.Milliseconds(), duration.Milliseconds())
			writeAPIJSON(w, http.StatusOK, cached)
			return
		}
		h.metrics.CacheHits.WithLabelValues("miss").Inc()
		w.Header().Set("X-LunarGate-Cache-Status", "MISS")
	}

	var userPtr *string
	if v, ok := headers["x-lunargate-user"]; ok {
		vv := v
		userPtr = &vv
	}
	var sessionIDPtr *string
	if v, ok := headers["x-lunargate-sessionid"]; ok {
		vv := v
		sessionIDPtr = &vv
	}
	log.Info().
		Str("request_id", requestID).
		Str("route", resolved.RouteName).
		Str("provider", resolved.Target.Provider).
		Str("model", req.Model).
		Bool("stream", req.Stream).
		Msg("routing request")

	if h.collector != nil {
		traceTags := h.enrichCollectorTagsWithInference(resolvedCollectorHeaders, resolved.Target.Provider, req.Model, req.Stream, resolvedSampling)
		startEvt := []observability.Event{{
			Type: "trace",
			Data: observability.TraceEventData{
				RequestID: requestID,
				Timestamp: startTime.UTC(),
				Phase:     "request_start",
				Tags:      traceTags,
			},
		}}
		h.collector.Enqueue(r.Context(), requestID, startEvt)
	}

	// Execute with fallback chain
	upstreamStartMS := int64(-1)
	markUpstreamStart := func() {
		if upstreamStartMS >= 0 {
			return
		}
		upstreamStartMS = time.Since(startTime).Milliseconds()
	}
	executeFunc := func(ctx context.Context, target routing.Target) (*http.Response, error) {
		return h.callProvider(ctx, target, &req, markUpstreamStart)
	}

	requestCtx := requestContextWithRetryPolicy(r)
	// A stored Chat Completion is a stateful upstream operation. If the
	// provider persisted the completion before returning a retryable failure,
	// replaying it here could create duplicate stored objects. Keep ordinary
	// stateless Chat requests retryable, but make store:true single-attempt and
	// single-target just like Responses create.
	if req.Store != nil && *req.Store {
		requestCtx = resilience.WithRetryDisabled(requestCtx)
		requestCtx = resilience.WithFallbackDisabled(requestCtx)
	}
	resp, usedTarget, fallbackUsed, retryCount, cbState, err := h.fallback.Execute(requestCtx, resolved.Target, resolved.Fallbacks, executeFunc)
	h.observeCircuitBreakerState(usedTarget.Provider, cbState)
	usedSampling := h.resolveCollectorInferenceParameters(usedTarget.Provider, &req)
	usedRequestTypes := chatAPIRequestTypes(requestType, usedTarget)
	usedCollectorHeaders := usedRequestTypes.tags(headers)
	if err != nil {
		duration := time.Since(startTime)
		status := http.StatusBadGateway
		errCode := "provider_error"
		errMsg := "all LLM providers unavailable"
		requestFailure := false
		var upstreamFailure *upstreamHTTPError
		if errors.Is(err, context.Canceled) {
			status = 499
			errCode = "client_cancelled"
			errMsg = "client disconnected"
			log.Info().
				Str("request_id", requestID).
				Dur("duration", duration).
				Msg("request cancelled")
		} else if requestStatus, requestErrCode, requestErrMsg, ok := requestFailureFromError(err); ok {
			status = requestStatus
			errCode = requestErrCode
			errMsg = requestErrMsg
			requestFailure = true
			log.Warn().
				Err(err).
				Str("request_id", requestID).
				Int("status_code", status).
				Dur("duration", duration).
				Msg("provider request rejected before upstream call")
		} else {
			providerType, _ := h.registry.Type(usedTarget.Provider)
			if failure, ok := upstreamHTTPErrorFromRetry(err, providerType); ok {
				upstreamFailure = failure
				status = failure.status
				errCode = failure.errType
				errMsg = failure.message
				log.Warn().
					Err(err).
					Str("request_id", requestID).
					Int("status_code", status).
					Dur("duration", duration).
					Msg("upstream provider failure after retries")
			} else {
				log.Error().Err(err).
					Str("request_id", requestID).
					Dur("duration", duration).
					Msg("all providers failed")
			}
		}
		if !errors.Is(err, context.Canceled) && !requestFailure {
			h.metrics.ProviderErrors.WithLabelValues(resolved.Target.Provider, "all_failed").Inc()
		}
		setTimingHeaders(w, duration.Milliseconds(), upstreamStartMS)
		if upstreamFailure != nil {
			upstreamFailure.write(w)
		} else {
			writeError(w, status, errMsg, errCode)
		}
		if h.collector != nil {
			errCodeForCollector := errCode
			errMsgForCollector := err.Error()
			if errors.Is(err, context.Canceled) {
				errMsgForCollector = "client disconnected"
			} else if upstreamFailure != nil {
				errMsgForCollector = upstreamFailure.message
			}
			routeUsed := resolved.RouteName
			targetIndex := resolved.Index
			failedModelRaw := strings.TrimSpace(usedTarget.Model)
			if failedModelRaw == "" {
				failedModelRaw = modelid.ModelName(req.Model)
				if failedModelRaw == "" {
					if translator, ok := h.registry.Get(usedTarget.Provider); ok {
						failedModelRaw = strings.TrimSpace(translator.DefaultModel())
					}
				}
			}
			failedModelCanonical := modelid.BuildCanonical(usedTarget.Provider, failedModelRaw)
			var upstreamPtr *int64
			if upstreamStartMS >= 0 {
				v := upstreamStartMS
				upstreamPtr = &v
			}
			events := []observability.Event{{
				Type: "metric",
				Data: observability.MetricEventData{
					RequestID:            requestID,
					Timestamp:            startTime.UTC(),
					RequestType:          usedRequestTypes.client,
					UpstreamRequestType:  usedRequestTypes.upstream,
					DurationMS:           duration.Milliseconds(),
					GatewayPreUpstreamMS: upstreamPtr,
					Provider:             usedTarget.Provider,
					Model:                failedModelCanonical,
					User:                 userPtr,
					SessionID:            sessionIDPtr,
					TokensInput:          0,
					TokensOutput:         0,
					CostUSD:              0,
					StatusCode:           status,
					ErrorCode:            observability.MetricErrorClass(status, true),
					CacheHit:             cacheHit,
					RouteUsed:            &routeUsed,
					TargetIndex:          &targetIndex,
					FallbackUsed:         fallbackUsed,
					RetryCount:           retryCount,
					CircuitBreakerState:  &cbState,
					Tags:                 h.enrichCollectorTagsWithInference(usedCollectorHeaders, usedTarget.Provider, failedModelCanonical, req.Stream, usedSampling),
				},
			}}

			if h.collector.SharePrompts() {
				reqAny := buildCollectorRequestLogPayload(body, usedSampling)
				events = append(events, observability.Event{
					Type: "request_log",
					Data: observability.RequestLogEventData{
						RequestID:           requestID,
						Timestamp:           startTime.UTC(),
						RequestType:         usedRequestTypes.client,
						UpstreamRequestType: usedRequestTypes.upstream,
						User:                userPtr,
						SessionID:           sessionIDPtr,
						Provider:            usedTarget.Provider,
						Model:               failedModelCanonical,
						StatusCode:          status,
						DurationMS:          duration.Milliseconds(),
						RouteUsed:           &routeUsed,
						CacheHit:            cacheHit,
						FallbackUsed:        fallbackUsed,
						RetryCount:          retryCount,
						ErrorCode:           &errCodeForCollector,
						ErrorMessage:        &errMsgForCollector,
						Tags:                h.enrichCollectorTagsWithInference(usedCollectorHeaders, usedTarget.Provider, failedModelCanonical, req.Stream, usedSampling),
						Request:             reqAny,
					},
				})
			}
			h.collector.Enqueue(r.Context(), requestID, events)
		}
		return
	}

	if fallbackUsed {
		h.metrics.FallbacksUsed.Inc()
	}

	// Set response headers
	w.Header().Set("X-LunarGate-Provider", usedTarget.Provider)
	providerSnapshot, ok := providerRequestSnapshotFromResponse(resp, usedTarget.Provider)
	if !ok {
		resp.Body.Close()
		writeError(w, http.StatusInternalServerError, "provider request snapshot not found", "internal_error")
		return
	}

	usedModelRaw := strings.TrimSpace(usedTarget.Model)
	if usedModelRaw == "" {
		usedModelRaw = modelid.ModelName(req.Model)
		if usedModelRaw == "" {
			usedModelRaw = strings.TrimSpace(providerSnapshot.Translator.DefaultModel())
		}
	}
	usedModelCanonical := modelid.BuildCanonical(usedTarget.Provider, usedModelRaw)
	metricsModelRaw := boundedModelMetricLabel(usedTarget, usedModelRaw)
	req.Model = usedModelCanonical
	w.Header().Set("X-LunarGate-Model", usedModelCanonical)

	usedProviderType := providerSnapshot.ProviderType
	setTimingHeaders(w, -1, upstreamStartMS)

	// Handle streaming response
	if req.Stream {
		trw := &trackedResponseWriter{ResponseWriter: w}
		var tw http.ResponseWriter = trw
		if f, ok := w.(http.Flusher); ok {
			tw = &trackedFlusherResponseWriter{trackedResponseWriter: trw, flusher: f}
		}
		translator := providerSnapshot.Translator
		if usedProviderType == "anthropic" {
			if a, ok := translator.(*providers.AnthropicTranslator); ok {
				translator = providers.NewAnthropicStreamTranslator(a)
			}
		}
		if usedProviderType == "ollama" {
			if o, ok := translator.(*providers.OllamaTranslator); ok {
				translator = providers.NewOllamaStreamTranslator(o)
			}
		}
		if usedProviderType == "openai" && strings.EqualFold(strings.TrimSpace(usedTarget.UpstreamRequestType), "responses") {
			if o, ok := translator.(*providers.OpenAITranslator); ok {
				translator = providers.NewOpenAIStreamTranslator(o)
			}
		}

		var streamedText strings.Builder
		var streamedReasoning strings.Builder
		toolCallByKey := make(map[string]*models.ToolCall, 8)
		toolCallOrder := make([]string, 0, 8)
		tokensIn := 0
		tokensOut := 0
		streamTokenUsage := models.TokenUsage{}
		var finishReason *string
		var ttftMS int64 = -1
		var ttltMS int64 = -1
		var nativeTerminal *nativeResponsesStreamTerminal
		var chatCompletionCandidate chatCompletionStreamBindingCandidate
		streamObserver := func(chunk *models.StreamChunk) {
			if chunk == nil {
				return
			}
			chatCompletionCandidate.observe(chunk)
			mergeObservedTokenUsage(&streamTokenUsage, chunk.Usage)
			tokensIn = streamTokenUsage.InputTokens
			tokensOut = streamTokenUsage.OutputTokens

			hasContent := false
			for _, c := range chunk.Choices {
				if c.FinishReason != nil {
					finishReason = c.FinishReason
				}
				if c.Delta == nil {
					continue
				}

				if len(c.Delta.ToolCalls) > 0 {
					hasContent = true
					for _, tc := range c.Delta.ToolCalls {
						key := ""
						if tc.Index != nil {
							key = fmt.Sprintf("idx:%d", *tc.Index)
						} else if tc.ID != "" {
							key = tc.ID
						} else if tc.Function.Name != "" {
							key = tc.Function.Name
						}
						if key == "" {
							continue
						}

						existing := toolCallByKey[key]
						if existing == nil {
							copyTC := models.ToolCall{
								Index: tc.Index,
								ID:    tc.ID,
								Type:  tc.Type,
								Function: models.ToolCallFunction{
									Name: tc.Function.Name,
								},
							}
							toolCallByKey[key] = &copyTC
							toolCallOrder = append(toolCallOrder, key)
							existing = &copyTC
						}

						if existing.ID == "" {
							existing.ID = tc.ID
						}
						if existing.Type == "" {
							existing.Type = tc.Type
						}
						if existing.Index == nil {
							existing.Index = tc.Index
						}
						if existing.Function.Name == "" {
							existing.Function.Name = tc.Function.Name
						}
						if tc.Function.Arguments != "" {
							existing.Function.Arguments += tc.Function.Arguments
						}
					}
				}

				if content, ok := c.Delta.Content.(string); ok && content != "" {
					streamedText.WriteString(content)
					hasContent = true
				}
				if c.Delta.ReasoningContent != "" {
					streamedReasoning.WriteString(c.Delta.ReasoningContent)
					hasContent = true
				}
			}

			if hasContent {
				now := time.Since(startTime).Milliseconds()
				if ttftMS < 0 {
					ttftMS = now
				}
				ttltMS = now
			}
		}

		var streamErr error
		nativeResponsesRequest := strings.EqualFold(strings.TrimSpace(requestType), requestTypeResponses) &&
			strings.EqualFold(strings.TrimSpace(usedRequestTypes.upstream), requestTypeResponses)
		nativeResponsesStream := nativeResponsesRequest && resp.StatusCode == http.StatusOK
		nativeResponseStatus := http.StatusOK
		if nativeResponsesRequest && resp.StatusCode != http.StatusOK && resp.StatusCode < http.StatusBadRequest {
			resp.Body.Close()
			streamErr = invalidNativeResponsesCreateStatus(usedTarget.Provider, resp.StatusCode)
		} else if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
			streamErr = readUpstreamHTTPError(resp, usedProviderType)
		} else if nativeResponsesStream {
			nativeResponseStatus = resp.StatusCode
			copyHeaders(w.Header(), resp.Header)
			nativeSink, _ := w.(nativeResponsesStreamSink)
			var nativeEventTransformer streaming.SSEEventDataTransformer
			if nativeSink != nil {
				nativeSink.enableNativePassthrough()
				nativeEventTransformer = nativeSink.transformNativeEventData
			}
			streamErr = h.streamer.ProxySSEWithDataTransformer(
				r.Context(),
				tw,
				resp,
				usedTarget.Provider,
				func(event streaming.SSEEvent) bool {
					now := time.Since(startTime).Milliseconds()
					if ttftMS < 0 && len(event.Data) > 0 {
						ttftMS = now
					}
					terminal, ok := parseNativeResponsesStreamTerminal(event)
					if !ok {
						return false
					}
					nativeTerminal = terminal
					tokensIn = terminal.tokensInput
					tokensOut = terminal.tokensOutput
					streamTokenUsage = terminal.tokenUsage
					ttltMS = now
					if nativeSink != nil {
						nativeSink.recordNativeTerminal(terminal)
					}
					return true
				},
				nativeEventTransformer,
			)
		} else if usedProviderType == "anthropic" {
			streamErr = h.streamer.StreamAnthropicResponseWithObserverAndUsage(
				r.Context(),
				tw,
				resp,
				translator,
				streamObserver,
				includeClientStreamUsage(requestType, &req),
			)
		} else if usedProviderType == "ollama" {
			streamErr = h.streamer.StreamNDJSONResponseWithObserverAndUsage(
				r.Context(),
				tw,
				resp,
				translator,
				streamObserver,
				includeClientStreamUsage(requestType, &req),
			)
		} else {
			streamErr = h.streamer.StreamResponseWithObserverAndUsage(
				r.Context(),
				tw,
				resp,
				translator,
				streamObserver,
				includeClientStreamUsage(requestType, &req),
			)
		}

		duration := time.Since(startTime)
		status := nativeResponseStatus
		var errCodePtr *string
		var errMsgPtr *string

		if streamErr != nil {
			if recorder, ok := w.(interface{ RecordStreamError(error) }); ok {
				recorder.RecordStreamError(streamErr)
			}
			status = http.StatusBadGateway
			errCode := "streaming_error"
			errMsg := streamErr.Error()
			var upstreamFailure *upstreamHTTPError

			if errors.Is(streamErr, context.Canceled) {
				status = 499
				errCode = "client_cancelled"
				errMsg = "client disconnected"
				log.Info().
					Str("request_id", requestID).
					Dur("duration", duration).
					Msg("streaming cancelled")
			} else if errors.As(streamErr, &upstreamFailure) && upstreamFailure != nil {
				status = upstreamFailure.status
				errCode = upstreamFailure.errType
				errMsg = upstreamFailure.message
				log.Warn().Err(streamErr).
					Str("request_id", requestID).
					Dur("duration", duration).
					Msg("upstream rejected streaming request")
			} else if pe := (*providers.ProviderError)(nil); errors.As(streamErr, &pe) && pe != nil {
				if pe.StatusCode != 0 {
					status = pe.StatusCode
				}
				if strings.TrimSpace(pe.Type) != "" {
					errCode = pe.Type
				}
				if strings.TrimSpace(pe.Message) != "" {
					errMsg = pe.Message
				}
				log.Error().Err(streamErr).
					Str("request_id", requestID).
					Dur("duration", duration).
					Msg("streaming error")
			} else if isUpstreamTTFTTimeout(streamErr) {
				errCode = "upstream_timeout"
				errMsg = "provider timed out waiting for first byte"
				log.Error().Err(streamErr).
					Str("request_id", requestID).
					Dur("duration", duration).
					Msg("streaming first-byte timeout")
			} else if isUpstreamTotalTimeout(streamErr) {
				errCode = "upstream_timeout"
				errMsg = "provider timed out before full response completed"
				log.Error().Err(streamErr).
					Str("request_id", requestID).
					Dur("duration", duration).
					Msg("streaming total timeout")
			} else {
				log.Error().Err(streamErr).
					Str("request_id", requestID).
					Dur("duration", duration).
					Msg("streaming error")
			}

			if !trw.wroteHeader && !errors.Is(streamErr, context.Canceled) {
				if upstreamFailure != nil {
					upstreamFailure.write(tw)
				} else {
					writeError(tw, status, errMsg, errCode)
				}
			}

			errCodePtr = &errCode
			errMsgPtr = &errMsg
		} else {
			if nativeTerminal != nil && nativeTerminal.status != "completed" {
				errCode := "response_" + strings.ReplaceAll(nativeTerminal.status, " ", "_")
				errMsg := "upstream response ended with status " + nativeTerminal.status
				errCodePtr = &errCode
				errMsgPtr = &errMsg
			}
			ev := log.Info().
				Str("request_id", requestID).
				Str("provider", usedTarget.Provider).
				Str("model", usedModelCanonical).
				Int64("duration_ms", duration.Milliseconds()).
				Int("tokens_in", tokensIn).
				Int("tokens_out", tokensOut)
			if ttftMS >= 0 {
				ev = ev.Int64("ttft_ms", ttftMS)
			}
			if ttltMS >= 0 {
				ev = ev.Int64("ttlt_ms", ttltMS)
			}
			if nativeTerminal != nil {
				ev = ev.Str("response_status", nativeTerminal.status)
				ev.Msg("stream terminated")
			} else {
				ev.Msg("stream completed")
			}
		}
		if streamErr == nil && req.Store != nil && *req.Store {
			h.retainNativeChatCompletionBinding(
				chatCompletionCandidate.completionID(),
				requestType,
				usedRequestTypes.upstream,
				w.Header(),
			)
		}

		h.metrics.RequestsTotal.WithLabelValues(usedTarget.Provider, metricsModelRaw, strconv.Itoa(status), resolved.RouteName).Inc()
		h.metrics.RequestDuration.WithLabelValues(usedTarget.Provider, metricsModelRaw).Observe(duration.Seconds())
		h.metrics.ObserveTokenUsage(usedTarget.Provider, metricsModelRaw, streamTokenUsage)

		if h.collector != nil {
			routeUsed := resolved.RouteName
			targetIndex := resolved.Index
			costUSD := observability.EstimateTokenUsageCostUSD(usedTarget.Provider, usedProviderType, usedModelRaw, streamTokenUsage)

			// Build optional timing pointers
			var upstreamPtr *int64
			if upstreamStartMS >= 0 {
				v := upstreamStartMS
				upstreamPtr = &v
			}
			var ttftPtr, ttltPtr *int64
			if ttftMS >= 0 {
				v := ttftMS
				ttftPtr = &v
			}
			if ttltMS >= 0 {
				v := ttltMS
				ttltPtr = &v
			}

			events := []observability.Event{{
				Type: "metric",
				Data: observability.MetricEventData{
					RequestID:               requestID,
					Timestamp:               startTime.UTC(),
					RequestType:             usedRequestTypes.client,
					UpstreamRequestType:     usedRequestTypes.upstream,
					DurationMS:              duration.Milliseconds(),
					GatewayPreUpstreamMS:    upstreamPtr,
					TtftMS:                  ttftPtr,
					TtltMS:                  ttltPtr,
					Provider:                usedTarget.Provider,
					Model:                   usedModelCanonical,
					User:                    userPtr,
					SessionID:               sessionIDPtr,
					TokensInput:             tokensIn,
					TokensOutput:            tokensOut,
					TokensInputCached:       streamTokenUsage.CachedInputTokens,
					TokensInputCacheWrite:   streamTokenUsage.CacheWriteInputTokens,
					TokensInputCacheWrite5m: streamTokenUsage.CacheWriteInputTokens5m,
					TokensInputCacheWrite1h: streamTokenUsage.CacheWriteInputTokens1h,
					CostUSD:                 costUSD,
					StatusCode:              status,
					ErrorCode:               observability.MetricErrorClass(status, errCodePtr != nil),
					CacheHit:                cacheHit,
					RouteUsed:               &routeUsed,
					TargetIndex:             &targetIndex,
					FallbackUsed:            fallbackUsed,
					RetryCount:              retryCount,
					CircuitBreakerState:     &cbState,
					Tags:                    h.enrichCollectorTagsWithInference(usedCollectorHeaders, usedTarget.Provider, usedModelCanonical, req.Stream, usedSampling),
				},
			}}

			if h.collector.SharePrompts() || h.collector.ShareResponses() {
				var reqObj interface{}
				if h.collector.SharePrompts() {
					reqObj = buildCollectorRequestLogPayload(body, usedSampling)
				}

				var respObj interface{}
				if h.collector.ShareResponses() && nativeTerminal != nil {
					respObj = nativeTerminal.collectorResponse
				} else if h.collector.ShareResponses() {
					usage := &models.Usage{
						PromptTokens:        tokensIn,
						CompletionTokens:    tokensOut,
						TotalTokens:         models.SaturatingTokenSum(tokensIn, tokensOut),
						PromptTokensDetails: inputTokenDetailsFromTokenUsage(streamTokenUsage),
					}
					if tokensIn == 0 && tokensOut == 0 {
						usage = nil
					}
					unifiedResp := models.UnifiedResponse{
						ID:      requestID,
						Object:  "chat.completion",
						Created: time.Now().Unix(),
						Model:   usedModelCanonical,
						Choices: []models.Choice{{
							Index: 0,
							Message: &models.Message{
								Role:             "assistant",
								Content:          streamedText.String(),
								ReasoningContent: streamedReasoning.String(),
								ToolCalls: func() []models.ToolCall {
									if len(toolCallOrder) == 0 {
										return nil
									}
									out := make([]models.ToolCall, 0, len(toolCallOrder))
									for _, k := range toolCallOrder {
										if tc := toolCallByKey[k]; tc != nil {
											out = append(out, *tc)
										}
									}
									return out
								}(),
							},
							FinishReason: finishReason,
						}},
						Usage: usage,
					}
					respBytes, _ := json.Marshal(unifiedResp)
					_ = json.Unmarshal(respBytes, &respObj)
				}

				events = append(events, observability.Event{
					Type: "request_log",
					Data: observability.RequestLogEventData{
						RequestID:           requestID,
						Timestamp:           startTime.UTC(),
						RequestType:         usedRequestTypes.client,
						UpstreamRequestType: usedRequestTypes.upstream,
						User:                userPtr,
						SessionID:           sessionIDPtr,
						Provider:            usedTarget.Provider,
						Model:               usedModelCanonical,
						StatusCode:          status,
						DurationMS:          duration.Milliseconds(),
						RouteUsed:           &routeUsed,
						CacheHit:            cacheHit,
						FallbackUsed:        fallbackUsed,
						RetryCount:          retryCount,
						ErrorCode:           errCodePtr,
						ErrorMessage:        errMsgPtr,
						Tags:                h.enrichCollectorTagsWithInference(usedCollectorHeaders, usedTarget.Provider, usedModelCanonical, req.Stream, usedSampling),
						Request:             reqObj,
						Response:            respObj,
					},
				})
			}
			h.collector.Enqueue(r.Context(), requestID, events)
		}
		return
	}

	// A native Responses endpoint may use any successful HTTP status. Keep its
	// transport envelope intact while still parsing the minimal unified snapshot
	// needed by metrics and local state handling.
	responseStatus := http.StatusOK
	var responseHeaders http.Header
	nativeResponsesRequest := strings.EqualFold(strings.TrimSpace(requestType), requestTypeResponses) &&
		strings.EqualFold(strings.TrimSpace(usedRequestTypes.upstream), requestTypeResponses)
	nativeResponsesEnvelope := nativeResponsesRequest && resp.StatusCode == http.StatusOK
	if nativeResponsesEnvelope {
		responseStatus = resp.StatusCode
		responseHeaders = resp.Header.Clone()
	}

	// Handle non-streaming response. HTTP failures are consumed here so a
	// native OpenAI error document is not reduced by a typed translator.
	var unified *models.UnifiedResponse
	var upstreamFailure *upstreamHTTPError
	if nativeResponsesRequest && resp.StatusCode != http.StatusOK && resp.StatusCode < http.StatusBadRequest {
		resp.Body.Close()
		err = invalidNativeResponsesCreateStatus(usedTarget.Provider, resp.StatusCode)
	} else if resp.StatusCode >= http.StatusBadRequest {
		upstreamFailure = readUpstreamHTTPError(resp, usedProviderType)
		err = upstreamFailure
	} else {
		unified, err = providerSnapshot.Translator.ParseResponse(resp)
		if err == nil {
			normalizeUnifiedResponseUsage(unified)
		}
	}
	if err != nil {
		duration := time.Since(startTime)
		status := http.StatusBadGateway
		respErrType := "provider_error"
		collectorErrCode := "provider_parse_error"
		errMsg := "failed to parse provider response: " + err.Error()
		metricErrType := "parse_error"
		var pe *providers.ProviderError
		if upstreamFailure != nil {
			status = upstreamFailure.status
			respErrType = upstreamFailure.errType
			collectorErrCode = upstreamFailure.errType
			errMsg = upstreamFailure.message
			metricErrType = upstreamFailure.errType
		} else if errors.As(err, &pe) {
			if pe.StatusCode != 0 {
				status = pe.StatusCode
			}
			if v := strings.TrimSpace(pe.Type); v != "" {
				respErrType = v
			} else {
				respErrType = "upstream_error"
			}
			collectorErrCode = respErrType
			if v := strings.TrimSpace(pe.Message); v != "" {
				errMsg = v
			} else {
				errMsg = err.Error()
			}
			metricErrType = respErrType
		} else if isUpstreamTTFTTimeout(err) {
			respErrType = "upstream_timeout"
			collectorErrCode = respErrType
			errMsg = "provider timed out waiting for first byte"
			metricErrType = respErrType
		} else if isUpstreamTotalTimeout(err) {
			respErrType = "upstream_timeout"
			collectorErrCode = respErrType
			errMsg = "provider timed out before full response completed"
			metricErrType = respErrType
		}

		log.Error().Err(err).
			Str("request_id", requestID).
			Str("provider", usedTarget.Provider).
			Dur("duration", duration).
			Msg("failed to parse provider response")
		h.metrics.ProviderErrors.WithLabelValues(usedTarget.Provider, boundedProviderErrorMetricType(status, metricErrType)).Inc()
		setTimingHeaders(w, duration.Milliseconds(), upstreamStartMS)
		if upstreamFailure != nil {
			upstreamFailure.write(w)
		} else {
			writeError(w, status, errMsg, respErrType)
		}
		if h.collector != nil {
			errCode := collectorErrCode
			routeUsed := resolved.RouteName
			targetIndex := resolved.Index
			var upstreamPtr *int64
			if upstreamStartMS >= 0 {
				v := upstreamStartMS
				upstreamPtr = &v
			}
			events := []observability.Event{{
				Type: "metric",
				Data: observability.MetricEventData{
					RequestID:            requestID,
					Timestamp:            startTime.UTC(),
					RequestType:          usedRequestTypes.client,
					UpstreamRequestType:  usedRequestTypes.upstream,
					DurationMS:           duration.Milliseconds(),
					GatewayPreUpstreamMS: upstreamPtr,
					Provider:             usedTarget.Provider,
					Model:                usedModelCanonical,
					User:                 userPtr,
					SessionID:            sessionIDPtr,
					TokensInput:          0,
					TokensOutput:         0,
					CostUSD:              0,
					StatusCode:           status,
					ErrorCode:            observability.MetricErrorClass(status, true),
					CacheHit:             cacheHit,
					RouteUsed:            &routeUsed,
					TargetIndex:          &targetIndex,
					FallbackUsed:         fallbackUsed,
					RetryCount:           retryCount,
					CircuitBreakerState:  &cbState,
					Tags:                 h.enrichCollectorTagsWithInference(usedCollectorHeaders, usedTarget.Provider, usedModelCanonical, req.Stream, usedSampling),
				},
			}}

			if h.collector.SharePrompts() {
				reqAny := buildCollectorRequestLogPayload(body, usedSampling)
				events = append(events, observability.Event{
					Type: "request_log",
					Data: observability.RequestLogEventData{
						RequestID:           requestID,
						Timestamp:           startTime.UTC(),
						RequestType:         usedRequestTypes.client,
						UpstreamRequestType: usedRequestTypes.upstream,
						User:                userPtr,
						SessionID:           sessionIDPtr,
						Provider:            usedTarget.Provider,
						Model:               usedModelCanonical,
						StatusCode:          status,
						DurationMS:          duration.Milliseconds(),
						RouteUsed:           &routeUsed,
						CacheHit:            cacheHit,
						FallbackUsed:        fallbackUsed,
						RetryCount:          retryCount,
						ErrorCode:           &errCode,
						ErrorMessage:        &errMsg,
						Tags:                h.enrichCollectorTagsWithInference(usedCollectorHeaders, usedTarget.Provider, usedModelCanonical, req.Stream, usedSampling),
						Request:             reqAny,
					},
				})
			}
			h.collector.Enqueue(r.Context(), requestID, events)
		}
		return
	}
	if nativeResponsesEnvelope {
		copyHeaders(w.Header(), responseHeaders)
	}
	if req.Store != nil && *req.Store {
		h.retainNativeChatCompletionBinding(
			unified.ID,
			requestType,
			usedRequestTypes.upstream,
			w.Header(),
		)
	}

	// Cache the response
	if !noCache && h.cache.Enabled() {
		cacheKey := middleware.GenerateKeyForTarget(&req, usedTarget.Provider, usedTarget.UpstreamRequestType)
		h.cache.Set(cacheKey, unified)
	}

	// Record metrics
	duration := time.Since(startTime)
	statusCode := strconv.Itoa(responseStatus)
	h.metrics.RequestsTotal.WithLabelValues(usedTarget.Provider, metricsModelRaw, statusCode, resolved.RouteName).Inc()
	h.metrics.RequestDuration.WithLabelValues(usedTarget.Provider, metricsModelRaw).Observe(duration.Seconds())

	tokenUsage := models.TokenUsageFromUsage(unified.Usage)
	h.metrics.ObserveTokenUsage(usedTarget.Provider, metricsModelRaw, tokenUsage)

	setTimingHeaders(w, duration.Milliseconds(), upstreamStartMS)

	log.Info().
		Str("request_id", requestID).
		Str("provider", usedTarget.Provider).
		Str("model", usedModelCanonical).
		Dur("duration", duration).
		Bool("fallback", fallbackUsed).
		Msg("request completed")

	if h.collector != nil {
		routeUsed := resolved.RouteName
		targetIndex := resolved.Index
		status := responseStatus
		tokensIn := tokenUsage.InputTokens
		tokensOut := tokenUsage.OutputTokens
		costUSD := observability.EstimateTokenUsageCostUSD(usedTarget.Provider, usedProviderType, usedModelRaw, tokenUsage)
		var upstreamPtr *int64
		if upstreamStartMS >= 0 {
			v := upstreamStartMS
			upstreamPtr = &v
		}

		events := []observability.Event{{
			Type: "metric",
			Data: observability.MetricEventData{
				RequestID:               requestID,
				Timestamp:               startTime.UTC(),
				RequestType:             usedRequestTypes.client,
				UpstreamRequestType:     usedRequestTypes.upstream,
				DurationMS:              duration.Milliseconds(),
				GatewayPreUpstreamMS:    upstreamPtr,
				Provider:                usedTarget.Provider,
				Model:                   usedModelCanonical,
				User:                    userPtr,
				SessionID:               sessionIDPtr,
				TokensInput:             tokensIn,
				TokensOutput:            tokensOut,
				TokensInputCached:       tokenUsage.CachedInputTokens,
				TokensInputCacheWrite:   tokenUsage.CacheWriteInputTokens,
				TokensInputCacheWrite5m: tokenUsage.CacheWriteInputTokens5m,
				TokensInputCacheWrite1h: tokenUsage.CacheWriteInputTokens1h,
				CostUSD:                 costUSD,
				StatusCode:              status,
				CacheHit:                cacheHit,
				RouteUsed:               &routeUsed,
				TargetIndex:             &targetIndex,
				FallbackUsed:            fallbackUsed,
				RetryCount:              retryCount,
				CircuitBreakerState:     &cbState,
				Tags:                    h.enrichCollectorTagsWithInference(usedCollectorHeaders, usedTarget.Provider, usedModelCanonical, req.Stream, usedSampling),
			},
		}}

		if h.collector.SharePrompts() || h.collector.ShareResponses() {
			var reqObj interface{}
			var respObj interface{}
			if h.collector.SharePrompts() {
				reqObj = buildCollectorRequestLogPayload(body, usedSampling)
			}
			if h.collector.ShareResponses() {
				respBytes := []byte(nil)
				if nativeResponsesEnvelope && json.Valid(unified.RawJSON) {
					respBytes = unified.RawJSON
				} else {
					respBytes, _ = json.Marshal(unified)
				}
				_ = json.Unmarshal(respBytes, &respObj)
			}
			events = append(events, observability.Event{
				Type: "request_log",
				Data: observability.RequestLogEventData{
					RequestID:           requestID,
					Timestamp:           startTime.UTC(),
					RequestType:         usedRequestTypes.client,
					UpstreamRequestType: usedRequestTypes.upstream,
					User:                userPtr,
					SessionID:           sessionIDPtr,
					Provider:            usedTarget.Provider,
					Model:               req.Model,
					StatusCode:          status,
					DurationMS:          duration.Milliseconds(),
					RouteUsed:           &routeUsed,
					CacheHit:            cacheHit,
					FallbackUsed:        fallbackUsed,
					RetryCount:          retryCount,
					Tags:                h.enrichCollectorTagsWithInference(usedCollectorHeaders, usedTarget.Provider, req.Model, req.Stream, usedSampling),
					Request:             reqObj,
					Response:            respObj,
				},
			})
		}

		h.collector.Enqueue(r.Context(), requestID, events)
	}

	writeAPIJSON(w, responseStatus, unified)
	return
}

func includeClientStreamUsage(requestType string, req *models.UnifiedRequest) bool {
	// The Responses stream contract requires terminal usage for its lifecycle
	// events. Chat Completions only exposes it when the client opts in.
	if strings.EqualFold(strings.TrimSpace(requestType), requestTypeResponses) {
		return true
	}
	return req != nil && req.StreamOptions != nil && req.StreamOptions.IncludeUsage
}

// ListModels handles GET /v1/models.
func (h *Handler) ListModels(w http.ResponseWriter, r *http.Request) {
	allModels := h.availableModels(r.Context())
	resp := models.ModelList{
		Object: "list",
		Data:   allModels,
	}
	writeJSON(w, http.StatusOK, resp)
}

// GetModel handles GET /v1/models/{model}.
func (h *Handler) GetModel(w http.ResponseWriter, r *http.Request) {
	modelID, err := url.PathUnescape(strings.TrimSpace(chi.URLParam(r, "*")))
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid model ID", "invalid_request_error")
		return
	}
	for _, model := range h.availableModels(r.Context()) {
		if model.ID == modelID {
			writeJSON(w, http.StatusOK, model)
			return
		}
	}
	writeError(w, http.StatusNotFound, "model not found", "invalid_request_error")
}

func (h *Handler) availableModels(ctx context.Context) []models.ModelInfo {
	var allModels []models.ModelInfo
	if h != nil && h.store != nil {
		allModels = h.store.AllModels(ctx)
	} else if h != nil && h.registry != nil {
		allModels = h.registry.AllModels()
	}
	auto := models.ModelInfo{ID: "lunargate/auto", Object: "model", Created: time.Now().Unix(), OwnedBy: "lunargate"}
	return append(allModels, auto)
}

// callProvider makes the actual HTTP request to the LLM provider.
func (h *Handler) callProvider(ctx context.Context, target routing.Target, req *models.UnifiedRequest, beforeUpstream func()) (*http.Response, error) {
	providerSnapshot, ok := h.registry.Snapshot(target.Provider)
	if !ok {
		return nil, fmt.Errorf("unknown provider: %s", target.Provider)
	}
	ctx = withProviderRequestSnapshot(ctx, target.Provider, providerSnapshot)
	if upstreamRequestType := strings.TrimSpace(target.UpstreamRequestType); upstreamRequestType != "" {
		ctx = providers.WithUpstreamRequestType(ctx, upstreamRequestType)
	}
	if sourceRequestType := strings.TrimSpace(req.SourceRequestType); sourceRequestType != "" {
		ctx = providers.WithSourceRequestType(ctx, sourceRequestType)
	}

	// Each route target owns the concrete upstream model. This is especially
	// important for fallbacks, which may use a different provider and model than
	// the primary target selected for the original request.
	reqCopy := *req
	if strings.TrimSpace(target.Model) != "" {
		reqCopy.Model = strings.TrimSpace(target.Model)
	}
	reqCopy.Model = modelid.ModelName(reqCopy.Model)

	httpReq, err := providerSnapshot.Translator.TranslateRequest(ctx, &reqCopy)
	if err != nil {
		return nil, resilience.NewRequestError(fmt.Errorf("failed to translate request for %s: %w", target.Provider, err))
	}
	if beforeUpstream != nil {
		beforeUpstream()
	}

	clientCfg := providerClientConfig{
		client:  newProviderHTTPClient(defaultUpstreamTimeout),
		timeout: defaultUpstreamTimeout,
		mode:    upstreamTimeoutModeTTFT,
	}
	if h.providerClients != nil {
		if configuredClient, ok := h.providerClients.Get(target.Provider); ok {
			clientCfg = configuredClient
		}
	}

	startedAt := time.Now()
	resp, err := clientCfg.client.Do(httpReq)
	if err != nil {
		if isHTTPTimeoutError(err) {
			if clientCfg.mode == upstreamTimeoutModeTotal {
				return nil, fmt.Errorf("%w: provider %s", errUpstreamTotalTimeout, target.Provider)
			}
			return nil, fmt.Errorf("%w: provider %s", errUpstreamTTFTTimeout, target.Provider)
		}
		return nil, fmt.Errorf("failed to call provider %s: %w", target.Provider, err)
	}
	if resp.Request == nil {
		resp.Request = httpReq
	}

	remaining := clientCfg.timeout - time.Since(startedAt)
	if transport, ok := clientCfg.client.Transport.(*http.Transport); ok && transport.ResponseHeaderTimeout > 0 {
		remaining = transport.ResponseHeaderTimeout - time.Since(startedAt)
	}
	if remaining <= 0 {
		resp.Body.Close()
		if clientCfg.mode == upstreamTimeoutModeTotal {
			return nil, fmt.Errorf("%w: provider %s", errUpstreamTotalTimeout, target.Provider)
		}
		return nil, fmt.Errorf("%w: provider %s", errUpstreamTTFTTimeout, target.Provider)
	}
	if clientCfg.mode == upstreamTimeoutModeTotal {
		resp.Body = wrapBodyWithTotalTimeout(resp.Body, remaining)
	} else {
		resp.Body = wrapBodyWithTTFTTimeout(resp.Body, remaining)
	}

	// For non-streaming, check status code. For streaming, let the streamer handle it.
	if !reqCopy.Stream && resp.StatusCode != http.StatusOK {
		return resp, nil // Let the retry logic check the status
	}

	return resp, nil
}

func invalidNativeResponsesCreateStatus(provider string, status int) *providers.ProviderError {
	return &providers.ProviderError{
		StatusCode: http.StatusBadGateway,
		Provider:   strings.TrimSpace(provider),
		Type:       "invalid_response_status",
		Message:    fmt.Sprintf("native Responses upstream returned status %d; expected 200", status),
	}
}

// extractHeaders pulls relevant headers into a map for route matching.
func extractHeaders(r *http.Request) map[string]string {
	headers := make(map[string]string)
	for _, key := range []string{
		"x-environment",
		"x-lunargate-request-type",
		"x-lunargate-provider",
		"x-lunargate-model",
		"x-lunargate-route",
		"x-lunargate-complexity",
		"x-lunargate-complexity-score",
		"x-lunargate-skill",
		"x-team",
		"x-app",
		"x-lunargate-user",
		"x-lunargate-sessionid",
		"x-lunargate-client-lat",
		"x-lunargate-client-lon",
	} {
		if val := r.Header.Get(key); val != "" {
			headers[key] = val
		}
	}
	return headers
}

func (h *Handler) enrichCollectorTags(headers map[string]string, provider string, model string, stream bool) map[string]string {
	tags := make(map[string]string, len(headers)+4)
	for k, v := range headers {
		tags[k] = v
	}
	if provider != "" {
		tags["x-lunargate-resolved-provider"] = provider
	}
	if model != "" {
		tags["x-lunargate-resolved-model"] = model
	}
	if stream {
		tags["x-lunargate-request-stream"] = "true"
	} else {
		tags["x-lunargate-request-stream"] = "false"
	}
	if tr, ok := h.registry.Get(provider); ok {
		if baseURL := strings.TrimSpace(tr.BaseURL()); baseURL != "" {
			tags["x-lunargate-upstream-base-url"] = baseURL
		}
	}
	return tags
}

func (h *Handler) enrichCollectorTagsWithInference(headers map[string]string, provider string, model string, stream bool, params collectorInferenceParameters) map[string]string {
	tags := h.enrichCollectorTags(headers, provider, model, stream)
	if params.Temperature != nil {
		tags["x-lunargate-inference-temperature"] = strconv.FormatFloat(*params.Temperature, 'f', -1, 64)
	}
	if params.TopP != nil {
		tags["x-lunargate-inference-top-p"] = strconv.FormatFloat(*params.TopP, 'f', -1, 64)
	}
	if params.TopK != nil {
		tags["x-lunargate-inference-top-k"] = strconv.Itoa(*params.TopK)
	}
	return tags
}

func (h *Handler) resolveCollectorInferenceParameters(provider string, req *models.UnifiedRequest) collectorInferenceParameters {
	params := collectorInferenceParameters{}
	if req != nil {
		params.Temperature = req.Temperature
		params.TopP = req.TopP
		params.TopK = req.TopK
	}

	if h == nil || h.providerClients == nil {
		return params
	}
	providerCfg, ok := h.providerClients.Config(provider)
	if !ok {
		return params
	}
	if params.Temperature == nil && providerCfg.Temperature != nil {
		v := *providerCfg.Temperature
		params.Temperature = &v
	}
	if params.TopP == nil && providerCfg.TopP != nil {
		v := *providerCfg.TopP
		params.TopP = &v
	}
	if params.TopK == nil && providerCfg.TopK != nil {
		if providerType, typeOK := h.registry.Type(provider); typeOK && strings.EqualFold(providerType, "ollama") {
			v := *providerCfg.TopK
			params.TopK = &v
		}
	}
	return params
}

func (h *Handler) executeChatCompletionsUnified(r *http.Request) (int, http.Header, *models.UnifiedResponse, []byte, error) {
	rec := newCapturedResponseWriter()
	h.ChatCompletions(rec, r)

	status := rec.statusCode
	if status == 0 {
		status = http.StatusOK
	}
	headers := rec.Header().Clone()
	body := rec.body.Bytes()

	if status >= 400 {
		return status, headers, nil, body, nil
	}
	var envelope struct {
		Object string `json:"object"`
	}
	if err := json.Unmarshal(body, &envelope); err == nil && strings.EqualFold(strings.TrimSpace(envelope.Object), "response") {
		return status, headers, &models.UnifiedResponse{
			RawJSON: append(json.RawMessage(nil), body...),
		}, nil, nil
	}

	var unified models.UnifiedResponse
	if err := json.Unmarshal(body, &unified); err != nil {
		return status, headers, nil, nil, err
	}
	return status, headers, &unified, nil, nil
}

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		log.Error().Err(err).Msg("failed to encode JSON response")
	}
}

func writeAPIJSON(w http.ResponseWriter, status int, v interface{}) {
	var raw json.RawMessage
	switch response := v.(type) {
	case *models.UnifiedResponse:
		if response != nil {
			raw = response.RawJSON
		}
	case models.UnifiedResponse:
		raw = response.RawJSON
	}
	if len(bytes.TrimSpace(raw)) == 0 || !json.Valid(raw) {
		writeJSON(w, status, v)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if _, err := w.Write(raw); err != nil {
		log.Error().Err(err).Msg("failed to write raw JSON response")
	}
}

func writeError(w http.ResponseWriter, status int, message string, errType string) {
	writeErrorDetail(w, status, message, errType, nil, nil)
}

func writeErrorDetail(w http.ResponseWriter, status int, message string, errType string, param *string, code *string) {
	resp := models.ErrorResponse{
		Error: models.ErrorDetail{
			Message: message,
			Type:    errType,
			Param:   param,
			Code:    code,
		},
	}
	writeJSON(w, status, resp)
}

func writeCompatibilityError(w http.ResponseWriter, compatibilityErr *models.CompatibilityError) {
	if compatibilityErr == nil {
		writeError(w, http.StatusBadRequest, "unsupported provider feature", "invalid_request_error")
		return
	}
	param := strings.TrimSpace(compatibilityErr.Field)
	code := "unsupported_feature"
	writeErrorDetail(
		w,
		http.StatusBadRequest,
		compatibilityErr.Error(),
		"invalid_request_error",
		&param,
		&code,
	)
}

func writeRequestedTargetUnavailable(w http.ResponseWriter, unavailable *routing.RequestedTargetUnavailableError) {
	param := "provider"
	code := "provider_not_found"
	message := "requested provider is not available for this route"
	if unavailable != nil && strings.TrimSpace(unavailable.Model) != "" {
		param = "model"
		code = "model_not_found"
		message = unavailable.Error()
	} else if unavailable != nil {
		message = unavailable.Error()
	}
	writeErrorDetail(w, http.StatusBadRequest, message, "invalid_request_error", &param, &code)
}
