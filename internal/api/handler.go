package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"
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
	registry  *providers.Registry
	router    *routing.Engine
	fallback  *resilience.FallbackExecutor
	cache     *middleware.Cache
	streamer  *streaming.Handler
	metrics   *observability.Metrics
	collector *observability.CollectorClient
	selector  *modelselect.Engine
	store     *modelstore.Store
	client    *http.Client
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
	w.flusher.Flush()
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

func newProviderHTTPClient() *http.Client {
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.MaxIdleConns = 2048
	transport.MaxIdleConnsPerHost = 1024
	transport.IdleConnTimeout = 90 * time.Second
	return &http.Client{
		Timeout:   120 * time.Second,
		Transport: transport,
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
	var body []byte

	if captureBody {
		var err error
		body, err = io.ReadAll(r.Body)
		if err != nil {
			writeRequestReadError(w, err)
			return nil, nil, false
		}
		if err := decodeJSONStrict(bytes.NewReader(body), &req); err != nil {
			writeRequestDecodeError(w, err)
			return nil, nil, false
		}
	} else {
		if err := decodeJSONStrict(r.Body, &req); err != nil {
			writeRequestDecodeError(w, err)
			return nil, nil, false
		}
	}

	if err := models.NormalizeUnifiedRequest(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid tool/function calling payload", "invalid_request_error")
		return nil, nil, false
	}

	return body, &req, true
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
		return resilience.WithRetryDisabled(ctx)
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
		registry:  registry,
		router:    router,
		fallback:  fallback,
		cache:     cache,
		streamer:  streamer,
		metrics:   metrics,
		collector: collector,
		selector:  selector,
		store:     store,
		client:    newProviderHTTPClient(),
	}
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

	if strings.EqualFold(strings.TrimSpace(req.Model), "lunargate/auto") {
		req.Model = ""
		if strings.EqualFold(strings.TrimSpace(explicitProvider), "lunargate") {
			explicitProvider = ""
		}
	}

	userSpecifiedModel := strings.TrimSpace(req.Model) != ""

	headers := extractHeaders(r)
	requestType := strings.TrimSpace(headers["x-lunargate-request-type"])
	if requestType == "" {
		requestType = "chat_completions"
		headers["x-lunargate-request-type"] = requestType
	}
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
		log.Error().Err(err).Str("request_id", requestID).Msg("failed to resolve route")
		writeError(w, http.StatusBadGateway, "no route matched for this request", "routing_error")
		return
	}
	w.Header().Set("X-LunarGate-Route", resolved.RouteName)
	if upstreamRequestType := strings.TrimSpace(resolved.Target.UpstreamRequestType); upstreamRequestType != "" {
		headers["x-lunargate-request-type"] = upstreamRequestType
	}

	requestedProvider := ""
	requestedModelRaw := ""
	if p, m, ok := modelid.SplitCanonical(req.Model); ok {
		requestedProvider = strings.TrimSpace(p)
		requestedModelRaw = strings.TrimSpace(m)
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

	noCache := r.Header.Get("X-LunarGate-No-Cache") == "true"
	if !req.Stream && !noCache && h.cache.Enabled() {
		cacheKey := middleware.GenerateKey(&req)
		if cached := h.cache.Get(cacheKey); cached != nil {
			h.metrics.CacheHits.WithLabelValues("hit").Inc()
			cacheHit = true
			durationMS := time.Since(startTime).Milliseconds()
			w.Header().Set("X-LunarGate-Cache-Status", "HIT")
			w.Header().Set("X-LunarGate-Provider", resolved.Target.Provider)
			w.Header().Set("X-LunarGate-Model", req.Model)
			setTimingHeaders(w, durationMS, durationMS)
			writeJSON(w, http.StatusOK, cached)
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
		traceTags := h.enrichCollectorTags(headers, resolved.Target.Provider, req.Model, req.Stream)
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
	resp, usedTarget, fallbackUsed, retryCount, cbState, err := h.fallback.Execute(requestCtx, resolved.Target, resolved.Fallbacks, executeFunc)
	h.observeCircuitBreakerState(usedTarget.Provider, cbState)
	if err != nil {
		duration := time.Since(startTime)
		status := http.StatusBadGateway
		errCode := "provider_error"
		errMsg := "all LLM providers unavailable"
		if errors.Is(err, context.Canceled) {
			status = 499
			errCode = "client_cancelled"
			errMsg = "client disconnected"
			log.Info().
				Str("request_id", requestID).
				Dur("duration", duration).
				Msg("request cancelled")
		} else {
			log.Error().Err(err).
				Str("request_id", requestID).
				Dur("duration", duration).
				Msg("all providers failed")
		}
		if !errors.Is(err, context.Canceled) {
			h.metrics.ProviderErrors.WithLabelValues(resolved.Target.Provider, "all_failed").Inc()
		}
		setTimingHeaders(w, duration.Milliseconds(), upstreamStartMS)
		writeError(w, status, errMsg, errCode)
		if h.collector != nil {
			errCodeForCollector := errCode
			errMsgForCollector := err.Error()
			if errors.Is(err, context.Canceled) {
				errMsgForCollector = "client disconnected"
			}
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
					RequestType:          requestType,
					DurationMS:           duration.Milliseconds(),
					GatewayPreUpstreamMS: upstreamPtr,
					Provider:             usedTarget.Provider,
					Model:                req.Model,
					User:                 userPtr,
					SessionID:            sessionIDPtr,
					TokensInput:          0,
					TokensOutput:         0,
					CostUSD:              0,
					StatusCode:           status,
					ErrorCode:            &errCodeForCollector,
					ErrorMessage:         &errMsgForCollector,
					CacheHit:             cacheHit,
					RouteUsed:            &routeUsed,
					TargetIndex:          &targetIndex,
					FallbackUsed:         fallbackUsed,
					RetryCount:           retryCount,
					CircuitBreakerState:  &cbState,
					Tags:                 h.enrichCollectorTags(headers, usedTarget.Provider, req.Model, req.Stream),
				},
			}}

			if h.collector.SharePrompts() {
				var reqAny interface{}
				_ = json.Unmarshal(body, &reqAny)
				events = append(events, observability.Event{
					Type: "request_log",
					Data: observability.RequestLogEventData{
						RequestID:    requestID,
						Timestamp:    startTime.UTC(),
						GatewayID:    h.collector.GatewayID(),
						RequestType:  requestType,
						User:         userPtr,
						SessionID:    sessionIDPtr,
						Provider:     usedTarget.Provider,
						Model:        req.Model,
						StatusCode:   status,
						DurationMS:   duration.Milliseconds(),
						RouteUsed:    &routeUsed,
						CacheHit:     cacheHit,
						FallbackUsed: fallbackUsed,
						RetryCount:   retryCount,
						ErrorCode:    &errCodeForCollector,
						ErrorMessage: &errMsgForCollector,
						Tags:         h.enrichCollectorTags(headers, usedTarget.Provider, req.Model, req.Stream),
						Request:      reqAny,
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
	usedModelRaw := strings.TrimSpace(usedTarget.Model)
	if usedModelRaw == "" {
		usedModelRaw = modelid.ModelName(req.Model)
		if usedModelRaw == "" {
			if tr, ok := h.registry.Get(usedTarget.Provider); ok {
				usedModelRaw = strings.TrimSpace(tr.DefaultModel())
			}
		}
	}
	usedModelCanonical := modelid.BuildCanonical(usedTarget.Provider, usedModelRaw)
	req.Model = usedModelCanonical
	w.Header().Set("X-LunarGate-Model", usedModelCanonical)

	usedProviderType, ok := h.registry.Type(usedTarget.Provider)
	if !ok {
		writeError(w, http.StatusInternalServerError, "provider type not found", "internal_error")
		return
	}
	setTimingHeaders(w, -1, upstreamStartMS)

	// Handle streaming response
	if req.Stream {
		trw := &trackedResponseWriter{ResponseWriter: w}
		var tw http.ResponseWriter = trw
		if f, ok := w.(http.Flusher); ok {
			tw = &trackedFlusherResponseWriter{trackedResponseWriter: trw, flusher: f}
		}
		translator, ok := h.registry.Get(usedTarget.Provider)
		if !ok {
			writeError(w, http.StatusInternalServerError, "provider translator not found", "internal_error")
			return
		}
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

		var streamedText strings.Builder
		var streamedReasoning strings.Builder
		toolCallByKey := make(map[string]*models.ToolCall, 8)
		toolCallOrder := make([]string, 0, 8)
		tokensIn := 0
		tokensOut := 0
		var finishReason *string
		var ttftMS int64 = -1
		var ttltMS int64 = -1
		streamObserver := func(chunk *models.StreamChunk) {
			if chunk == nil {
				return
			}
			if chunk.Usage != nil {
				if chunk.Usage.PromptTokens > tokensIn {
					tokensIn = chunk.Usage.PromptTokens
				}
				if chunk.Usage.CompletionTokens > tokensOut {
					tokensOut = chunk.Usage.CompletionTokens
				}
			}

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
		if usedProviderType == "anthropic" {
			streamErr = h.streamer.StreamAnthropicResponseWithObserver(r.Context(), tw, resp, translator, streamObserver)
		} else if usedProviderType == "ollama" {
			streamErr = h.streamer.StreamNDJSONResponseWithObserver(r.Context(), tw, resp, translator, streamObserver)
		} else {
			streamErr = h.streamer.StreamResponseWithObserver(r.Context(), tw, resp, translator, streamObserver)
		}

		duration := time.Since(startTime)
		status := http.StatusOK
		var errCodePtr *string
		var errMsgPtr *string

		if streamErr != nil {
			status = http.StatusBadGateway
			errCode := "streaming_error"
			errMsg := streamErr.Error()

			if errors.Is(streamErr, context.Canceled) {
				status = 499
				errCode = "client_cancelled"
				errMsg = "client disconnected"
				log.Info().
					Str("request_id", requestID).
					Dur("duration", duration).
					Msg("streaming cancelled")
			} else if pe, ok := streamErr.(*providers.ProviderError); ok {
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
			} else {
				log.Error().Err(streamErr).
					Str("request_id", requestID).
					Dur("duration", duration).
					Msg("streaming error")
			}

			if !trw.wroteHeader && !errors.Is(streamErr, context.Canceled) {
				writeError(tw, status, errMsg, errCode)
			}

			errCodePtr = &errCode
			errMsgPtr = &errMsg
		} else {
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
			ev.Msg("stream completed")
		}

		h.metrics.RequestsTotal.WithLabelValues(usedTarget.Provider, usedModelRaw, strconv.Itoa(status), resolved.RouteName).Inc()
		h.metrics.RequestDuration.WithLabelValues(usedTarget.Provider, usedModelRaw).Observe(duration.Seconds())
		if tokensIn > 0 {
			h.metrics.TokensTotal.WithLabelValues(usedTarget.Provider, usedModelRaw, "input").Add(float64(tokensIn))
		}
		if tokensOut > 0 {
			h.metrics.TokensTotal.WithLabelValues(usedTarget.Provider, usedModelRaw, "output").Add(float64(tokensOut))
		}

		if h.collector != nil {
			routeUsed := resolved.RouteName
			targetIndex := resolved.Index
			costUSD := observability.EstimateCostUSD(usedProviderType, usedModelRaw, tokensIn, tokensOut)
			costUSD = observability.EstimateCostUSD(usedProviderType, usedModelRaw, tokensIn, tokensOut)

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
					RequestID:            requestID,
					Timestamp:            startTime.UTC(),
					RequestType:          requestType,
					DurationMS:           duration.Milliseconds(),
					GatewayPreUpstreamMS: upstreamPtr,
					TtftMS:               ttftPtr,
					TtltMS:               ttltPtr,
					Provider:             usedTarget.Provider,
					Model:                usedModelCanonical,
					User:                 userPtr,
					SessionID:            sessionIDPtr,
					TokensInput:          tokensIn,
					TokensOutput:         tokensOut,
					CostUSD:              costUSD,
					StatusCode:           status,
					ErrorCode:            errCodePtr,
					ErrorMessage:         errMsgPtr,
					CacheHit:             cacheHit,
					RouteUsed:            &routeUsed,
					TargetIndex:          &targetIndex,
					FallbackUsed:         fallbackUsed,
					RetryCount:           retryCount,
					CircuitBreakerState:  &cbState,
					Tags:                 h.enrichCollectorTags(headers, usedTarget.Provider, usedModelCanonical, req.Stream),
				},
			}}

			if h.collector.SharePrompts() || h.collector.ShareResponses() {
				var reqObj interface{}
				if h.collector.SharePrompts() {
					_ = json.Unmarshal(body, &reqObj)
				}

				var respObj interface{}
				if h.collector.ShareResponses() {
					usage := &models.Usage{
						PromptTokens:     tokensIn,
						CompletionTokens: tokensOut,
						TotalTokens:      tokensIn + tokensOut,
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
						RequestID:    requestID,
						Timestamp:    startTime.UTC(),
						GatewayID:    h.collector.GatewayID(),
						RequestType:  requestType,
						User:         userPtr,
						SessionID:    sessionIDPtr,
						Provider:     usedTarget.Provider,
						Model:        usedModelCanonical,
						StatusCode:   status,
						DurationMS:   duration.Milliseconds(),
						RouteUsed:    &routeUsed,
						CacheHit:     cacheHit,
						FallbackUsed: fallbackUsed,
						RetryCount:   retryCount,
						ErrorCode:    errCodePtr,
						ErrorMessage: errMsgPtr,
						Tags:         h.enrichCollectorTags(headers, usedTarget.Provider, usedModelCanonical, req.Stream),
						Request:      reqObj,
						Response:     respObj,
					},
				})
			}
			h.collector.Enqueue(r.Context(), requestID, events)
		}
		return
	}

	// Handle non-streaming response
	translator, ok := h.registry.Get(usedTarget.Provider)
	if !ok {
		resp.Body.Close()
		writeError(w, http.StatusInternalServerError, "provider translator not found", "internal_error")
		return
	}

	unified, err := translator.ParseResponse(resp)
	if err != nil {
		duration := time.Since(startTime)
		status := http.StatusBadGateway
		respErrType := "provider_error"
		collectorErrCode := "provider_parse_error"
		errMsg := "failed to parse provider response: " + err.Error()
		metricErrType := "parse_error"
		if pe, ok := err.(*providers.ProviderError); ok {
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
		}

		log.Error().Err(err).
			Str("request_id", requestID).
			Str("provider", usedTarget.Provider).
			Dur("duration", duration).
			Msg("failed to parse provider response")
		h.metrics.ProviderErrors.WithLabelValues(usedTarget.Provider, metricErrType).Inc()
		setTimingHeaders(w, duration.Milliseconds(), upstreamStartMS)
		writeError(w, status, errMsg, respErrType)
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
					RequestType:          requestType,
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
					ErrorCode:            &errCode,
					ErrorMessage:         &errMsg,
					CacheHit:             cacheHit,
					RouteUsed:            &routeUsed,
					TargetIndex:          &targetIndex,
					FallbackUsed:         fallbackUsed,
					RetryCount:           retryCount,
					CircuitBreakerState:  &cbState,
					Tags:                 h.enrichCollectorTags(headers, usedTarget.Provider, usedModelCanonical, req.Stream),
				},
			}}

			if h.collector.SharePrompts() {
				var reqAny interface{}
				_ = json.Unmarshal(body, &reqAny)
				events = append(events, observability.Event{
					Type: "request_log",
					Data: observability.RequestLogEventData{
						RequestID:    requestID,
						Timestamp:    startTime.UTC(),
						GatewayID:    h.collector.GatewayID(),
						RequestType:  requestType,
						User:         userPtr,
						SessionID:    sessionIDPtr,
						Provider:     usedTarget.Provider,
						Model:        usedModelCanonical,
						StatusCode:   http.StatusBadGateway,
						DurationMS:   duration.Milliseconds(),
						RouteUsed:    &routeUsed,
						CacheHit:     cacheHit,
						FallbackUsed: fallbackUsed,
						RetryCount:   retryCount,
						ErrorCode:    &errCode,
						ErrorMessage: &errMsg,
						Tags:         h.enrichCollectorTags(headers, usedTarget.Provider, usedModelCanonical, req.Stream),
						Request:      reqAny,
					},
				})
			}
			h.collector.Enqueue(r.Context(), requestID, events)
		}
		return
	}

	// Cache the response
	if !noCache && h.cache.Enabled() {
		cacheKey := middleware.GenerateKey(&req)
		h.cache.Set(cacheKey, unified)
	}

	// Record metrics
	duration := time.Since(startTime)
	statusCode := "200"
	h.metrics.RequestsTotal.WithLabelValues(usedTarget.Provider, usedModelRaw, statusCode, resolved.RouteName).Inc()
	h.metrics.RequestDuration.WithLabelValues(usedTarget.Provider, usedModelRaw).Observe(duration.Seconds())

	if unified.Usage != nil {
		h.metrics.TokensTotal.WithLabelValues(usedTarget.Provider, usedModelRaw, "input").Add(float64(unified.Usage.PromptTokens))
		h.metrics.TokensTotal.WithLabelValues(usedTarget.Provider, usedModelRaw, "output").Add(float64(unified.Usage.CompletionTokens))
	}

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
		status := http.StatusOK
		var tokensIn, tokensOut int
		if unified.Usage != nil {
			tokensIn = unified.Usage.PromptTokens
			tokensOut = unified.Usage.CompletionTokens
		}
		costUSD := observability.EstimateCostUSD(usedProviderType, req.Model, tokensIn, tokensOut)
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
				RequestType:          requestType,
				DurationMS:           duration.Milliseconds(),
				GatewayPreUpstreamMS: upstreamPtr,
				Provider:             usedTarget.Provider,
				Model:                usedModelCanonical,
				User:                 userPtr,
				SessionID:            sessionIDPtr,
				TokensInput:          tokensIn,
				TokensOutput:         tokensOut,
				CostUSD:              costUSD,
				StatusCode:           status,
				CacheHit:             cacheHit,
				RouteUsed:            &routeUsed,
				TargetIndex:          &targetIndex,
				FallbackUsed:         fallbackUsed,
				RetryCount:           retryCount,
				CircuitBreakerState:  &cbState,
				Tags:                 h.enrichCollectorTags(headers, usedTarget.Provider, usedModelCanonical, req.Stream),
			},
		}}

		if h.collector.SharePrompts() || h.collector.ShareResponses() {
			var reqObj interface{}
			var respObj interface{}
			if h.collector.SharePrompts() {
				_ = json.Unmarshal(body, &reqObj)
			}
			if h.collector.ShareResponses() {
				respBytes, _ := json.Marshal(unified)
				_ = json.Unmarshal(respBytes, &respObj)
			}
			events = append(events, observability.Event{
				Type: "request_log",
				Data: observability.RequestLogEventData{
					RequestID:    requestID,
					Timestamp:    startTime.UTC(),
					GatewayID:    h.collector.GatewayID(),
					RequestType:  requestType,
					User:         userPtr,
					SessionID:    sessionIDPtr,
					Provider:     usedTarget.Provider,
					Model:        req.Model,
					StatusCode:   status,
					DurationMS:   duration.Milliseconds(),
					RouteUsed:    &routeUsed,
					CacheHit:     cacheHit,
					FallbackUsed: fallbackUsed,
					RetryCount:   retryCount,
					Tags:         h.enrichCollectorTags(headers, usedTarget.Provider, req.Model, req.Stream),
					Request:      reqObj,
					Response:     respObj,
				},
			})
		}

		h.collector.Enqueue(r.Context(), requestID, events)
	}

	writeJSON(w, http.StatusOK, unified)
	return
}

// ListModels handles GET /v1/models.
func (h *Handler) ListModels(w http.ResponseWriter, r *http.Request) {
	var allModels []models.ModelInfo
	if h.store != nil {
		allModels = h.store.AllModels(r.Context())
	} else {
		allModels = h.registry.AllModels()
	}
	auto := models.ModelInfo{ID: "lunargate/auto", Object: "model", Created: time.Now().Unix(), OwnedBy: "lunargate"}
	allModels = append(allModels, auto)
	resp := models.ModelList{
		Object: "list",
		Data:   allModels,
	}
	writeJSON(w, http.StatusOK, resp)
}

// GetModel handles GET /v1/models/{model}.
func (h *Handler) GetModel(w http.ResponseWriter, r *http.Request) {
	// For now, return a simple model info
	writeError(w, http.StatusNotFound, "model not found", "invalid_request_error")
}

// callProvider makes the actual HTTP request to the LLM provider.
func (h *Handler) callProvider(ctx context.Context, target routing.Target, req *models.UnifiedRequest, beforeUpstream func()) (*http.Response, error) {
	translator, ok := h.registry.Get(target.Provider)
	if !ok {
		return nil, fmt.Errorf("unknown provider: %s", target.Provider)
	}
	if upstreamRequestType := strings.TrimSpace(target.UpstreamRequestType); upstreamRequestType != "" {
		ctx = providers.WithUpstreamRequestType(ctx, upstreamRequestType)
	}

	// Override model with the target's model if the request model matches a generic name
	reqCopy := *req
	if reqCopy.Model == "" && target.Model != "" {
		reqCopy.Model = target.Model
	}
	reqCopy.Model = modelid.ModelName(reqCopy.Model)

	httpReq, err := translator.TranslateRequest(ctx, &reqCopy)
	if err != nil {
		return nil, fmt.Errorf("failed to translate request for %s: %w", target.Provider, err)
	}
	if beforeUpstream != nil {
		beforeUpstream()
	}

	resp, err := h.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to call provider %s: %w", target.Provider, err)
	}

	// For non-streaming, check status code. For streaming, let the streamer handle it.
	if !reqCopy.Stream && resp.StatusCode != http.StatusOK {
		return resp, nil // Let the retry logic check the status
	}

	return resp, nil
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

func writeError(w http.ResponseWriter, status int, message string, errType string) {
	resp := models.ErrorResponse{
		Error: models.ErrorDetail{
			Message: message,
			Type:    errType,
		},
	}
	writeJSON(w, status, resp)
}
