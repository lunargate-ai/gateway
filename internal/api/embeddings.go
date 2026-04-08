package api

import (
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
	"github.com/lunargate-ai/gateway/internal/observability"
	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/internal/routing"
	"github.com/lunargate-ai/gateway/pkg/models"
	"github.com/rs/zerolog/log"
)

type embeddingsTranslator interface {
	TranslateEmbeddingsRequest(ctx context.Context, req *models.EmbeddingsRequest) (*http.Request, error)
	ParseEmbeddingsResponse(resp *http.Response) (*models.EmbeddingsResponse, error)
}

func parseEmbeddingsRequest(w http.ResponseWriter, r *http.Request, captureBody bool) ([]byte, *models.EmbeddingsRequest, bool) {
	const maxRequestBodyBytes int64 = 10 << 20
	r.Body = http.MaxBytesReader(w, r.Body, maxRequestBodyBytes)
	defer r.Body.Close()

	var req models.EmbeddingsRequest
	var body []byte

	if captureBody {
		var err error
		body, err = io.ReadAll(r.Body)
		if err != nil {
			writeRequestReadError(w, err)
			return nil, nil, false
		}
		if err := json.Unmarshal(body, &req); err != nil {
			writeRequestDecodeError(w, err)
			return nil, nil, false
		}
	} else {
		decoder := json.NewDecoder(r.Body)
		if err := decoder.Decode(&req); err != nil {
			writeRequestDecodeError(w, err)
			return nil, nil, false
		}
		var extra json.RawMessage
		if err := decoder.Decode(&extra); err != io.EOF {
			writeRequestDecodeError(w, err)
			return nil, nil, false
		}
	}

	if strings.TrimSpace(req.Model) == "" {
		writeError(w, http.StatusBadRequest, "model is required", "invalid_request_error")
		return nil, nil, false
	}
	if req.Input == nil {
		writeError(w, http.StatusBadRequest, "input is required", "invalid_request_error")
		return nil, nil, false
	}

	return body, &req, true
}

func cloneTagsWithRequestType(tags map[string]string, requestType string) map[string]string {
	out := make(map[string]string, len(tags)+1)
	for k, v := range tags {
		out[k] = v
	}
	out["x-lunargate-request-type"] = requestType
	return out
}

func extractUserAndSession(headers map[string]string) (*string, *string) {
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
	return userPtr, sessionIDPtr
}

func (h *Handler) resolveEmbeddingsRoute(ctx context.Context, path string, headers map[string]string, requestedProvider string) (*routing.ResolvedRoute, error) {
	resolved, err := h.router.Resolve(ctx, path, headers)
	if err == nil {
		return resolved, nil
	}
	if strings.TrimSpace(requestedProvider) == "" {
		return nil, err
	}
	translator, ok := h.registry.Get(requestedProvider)
	if !ok {
		return nil, fmt.Errorf("unknown provider: %s", requestedProvider)
	}
	if _, ok := translator.(embeddingsTranslator); !ok {
		return nil, fmt.Errorf("provider %s does not support embeddings", requestedProvider)
	}

	directModel := modelid.ModelName(strings.TrimSpace(headers["x-lunargate-model"]))
	if directModel == "" {
		directModel = strings.TrimSpace(translator.DefaultModel())
	}

	return &routing.ResolvedRoute{
		RouteName: "embeddings-direct",
		Target: routing.Target{
			Provider: requestedProvider,
			Model:    directModel,
			Weight:   100,
		},
		Fallbacks: nil,
		Index:     0,
	}, nil
}

func (h *Handler) callEmbeddingsProvider(ctx context.Context, target routing.Target, req *models.EmbeddingsRequest, beforeUpstream func()) (*http.Response, error) {
	translator, ok := h.registry.Get(target.Provider)
	if !ok {
		return nil, fmt.Errorf("unknown provider: %s", target.Provider)
	}
	embeddingTranslator, ok := translator.(embeddingsTranslator)
	if !ok {
		return nil, fmt.Errorf("provider %s does not support embeddings", target.Provider)
	}

	reqCopy := *req
	if strings.TrimSpace(reqCopy.Model) == "" && strings.TrimSpace(target.Model) != "" {
		reqCopy.Model = strings.TrimSpace(target.Model)
	}
	reqCopy.Model = modelid.ModelName(reqCopy.Model)

	httpReq, err := embeddingTranslator.TranslateEmbeddingsRequest(ctx, &reqCopy)
	if err != nil {
		return nil, fmt.Errorf("failed to translate embeddings request for %s: %w", target.Provider, err)
	}
	if beforeUpstream != nil {
		beforeUpstream()
	}

	resp, err := h.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to call provider %s: %w", target.Provider, err)
	}
	if resp.StatusCode != http.StatusOK {
		return resp, nil
	}
	return resp, nil
}

func (h *Handler) Embeddings(w http.ResponseWriter, r *http.Request) {
	startTime := time.Now()
	requestID := uuid.New().String()
	cacheHit := false

	h.metrics.ActiveRequests.Inc()
	defer h.metrics.ActiveRequests.Dec()

	w.Header().Set("X-LunarGate-Request-ID", requestID)

	captureBody := h.collector != nil && h.collector.SharePrompts()
	body, parsedReq, ok := parseEmbeddingsRequest(w, r, captureBody)
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
		if p, m, ok := modelid.SplitCanonical(req.Model); ok {
			if strings.TrimSpace(p) == "" {
				req.Model = modelid.BuildCanonical(explicitProvider, m)
			}
		} else {
			req.Model = modelid.BuildCanonical(explicitProvider, req.Model)
		}
	}

	requestedProvider := ""
	requestedModelRaw := ""
	if p, m, ok := modelid.SplitCanonical(req.Model); ok {
		requestedProvider = strings.TrimSpace(p)
		requestedModelRaw = strings.TrimSpace(m)
	}
	if strings.TrimSpace(requestedProvider) == "" {
		requestedProvider = strings.TrimSpace(explicitProvider)
	}

	headers := extractHeaders(r)
	if req.Model != "" {
		headers["x-lunargate-model"] = strings.TrimSpace(req.Model)
		if p, _, ok := modelid.SplitCanonical(req.Model); ok {
			headers["x-lunargate-provider"] = strings.TrimSpace(p)
		}
	}
	if explicitProvider != "" {
		headers["x-lunargate-provider"] = explicitProvider
	}
	if h.collector != nil {
		if v := strings.TrimSpace(h.collector.GatewayLat()); v != "" {
			headers["x-lunargate-gateway-lat"] = v
		}
		if v := strings.TrimSpace(h.collector.GatewayLon()); v != "" {
			headers["x-lunargate-gateway-lon"] = v
		}
	}

	resolved, err := h.resolveEmbeddingsRoute(r.Context(), r.URL.Path, headers, requestedProvider)
	if err != nil {
		log.Error().Err(err).Str("request_id", requestID).Msg("failed to resolve embeddings route")
		writeError(w, http.StatusBadGateway, "no route matched for this request", "routing_error")
		return
	}
	w.Header().Set("X-LunarGate-Route", resolved.RouteName)

	if requestedProvider != "" && strings.TrimSpace(resolved.Target.Provider) != requestedProvider {
		writeError(w, http.StatusBadRequest, "requested provider is not available for this route", "invalid_request_error")
		return
	}
	if requestedModelRaw != "" && strings.TrimSpace(resolved.Target.Model) != "" && strings.TrimSpace(resolved.Target.Model) != requestedModelRaw {
		writeError(w, http.StatusBadRequest, "requested model is not available for this route", "invalid_request_error")
		return
	}

	if strings.TrimSpace(req.Model) == "" {
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
	} else if _, _, ok := modelid.SplitCanonical(req.Model); !ok {
		req.Model = modelid.BuildCanonical(resolved.Target.Provider, req.Model)
		headers["x-lunargate-model"] = req.Model
	}

	noCache := r.Header.Get("X-LunarGate-No-Cache") == "true"
	if !noCache && h.cache.Enabled() {
		cacheKey := middleware.GenerateEmbeddingsKey(&req)
		if cached := h.cache.Get(cacheKey); cached != nil {
			if cachedResp, ok := cached.(*models.EmbeddingsResponse); ok {
				h.metrics.CacheHits.WithLabelValues("hit").Inc()
				cacheHit = true
				durationMS := time.Since(startTime).Milliseconds()
				w.Header().Set("X-LunarGate-Cache-Status", "HIT")
				w.Header().Set("X-LunarGate-Provider", resolved.Target.Provider)
				w.Header().Set("X-LunarGate-Model", req.Model)
				setTimingHeaders(w, durationMS, durationMS)
				writeJSON(w, http.StatusOK, models.CloneEmbeddingsResponse(cachedResp))
				return
			}
		}
		h.metrics.CacheHits.WithLabelValues("miss").Inc()
		w.Header().Set("X-LunarGate-Cache-Status", "MISS")
	}

	userPtr, sessionIDPtr := extractUserAndSession(headers)

	log.Info().
		Str("request_id", requestID).
		Str("route", resolved.RouteName).
		Str("provider", resolved.Target.Provider).
		Str("model", req.Model).
		Msg("routing embeddings request")

	traceTags := cloneTagsWithRequestType(h.enrichCollectorTags(headers, resolved.Target.Provider, req.Model, false), "embeddings")
	if h.collector != nil {
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

	upstreamStartMS := int64(-1)
	markUpstreamStart := func() {
		if upstreamStartMS >= 0 {
			return
		}
		upstreamStartMS = time.Since(startTime).Milliseconds()
	}
	executeFunc := func(ctx context.Context, target routing.Target) (*http.Response, error) {
		return h.callEmbeddingsProvider(ctx, target, &req, markUpstreamStart)
	}

	requestCtx := requestContextWithRetryPolicy(r)
	resp, usedTarget, fallbackUsed, retryCount, cbState, err := h.fallback.Execute(requestCtx, resolved.Target, resolved.Fallbacks, executeFunc)
	h.observeCircuitBreakerState(usedTarget.Provider, cbState)
	if err != nil {
		duration := time.Since(startTime)
		status := http.StatusBadGateway
		errCode := "provider_error"
		errMsg := "all embedding providers unavailable"
		if errors.Is(err, context.Canceled) {
			status = 499
			errCode = "client_cancelled"
			errMsg = "client disconnected"
			log.Info().
				Str("request_id", requestID).
				Dur("duration", duration).
				Msg("embeddings request cancelled")
		} else {
			log.Error().Err(err).
				Str("request_id", requestID).
				Dur("duration", duration).
				Msg("all embeddings providers failed")
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
					RequestType:          "embeddings",
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
					Tags:                 traceTags,
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
						RequestType:  "embeddings",
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
						Tags:         traceTags,
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
		resp.Body.Close()
		writeError(w, http.StatusInternalServerError, "provider type not found", "internal_error")
		return
	}
	setTimingHeaders(w, -1, upstreamStartMS)

	translator, ok := h.registry.Get(usedTarget.Provider)
	if !ok {
		resp.Body.Close()
		writeError(w, http.StatusInternalServerError, "provider translator not found", "internal_error")
		return
	}
	embeddingsTranslator, ok := translator.(embeddingsTranslator)
	if !ok {
		resp.Body.Close()
		writeError(w, http.StatusBadGateway, "provider does not support embeddings", "provider_error")
		return
	}

	embeddingsResp, err := embeddingsTranslator.ParseEmbeddingsResponse(resp)
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
			Msg("failed to parse embeddings provider response")
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
					RequestType:          "embeddings",
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
					Tags:                 traceTags,
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
						RequestType:  "embeddings",
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
						ErrorCode:    &errCode,
						ErrorMessage: &errMsg,
						Tags:         traceTags,
						Request:      reqAny,
					},
				})
			}
			h.collector.Enqueue(r.Context(), requestID, events)
		}
		return
	}

	if !noCache && h.cache.Enabled() {
		cacheKey := middleware.GenerateEmbeddingsKey(&req)
		h.cache.Set(cacheKey, models.CloneEmbeddingsResponse(embeddingsResp))
	}

	duration := time.Since(startTime)
	h.metrics.RequestsTotal.WithLabelValues(usedTarget.Provider, usedModelRaw, strconv.Itoa(http.StatusOK), resolved.RouteName).Inc()
	h.metrics.RequestDuration.WithLabelValues(usedTarget.Provider, usedModelRaw).Observe(duration.Seconds())

	var tokensIn int
	if embeddingsResp.Usage != nil {
		tokensIn = embeddingsResp.Usage.PromptTokens
		if tokensIn == 0 {
			tokensIn = embeddingsResp.Usage.TotalTokens
		}
	}
	if tokensIn > 0 {
		h.metrics.TokensTotal.WithLabelValues(usedTarget.Provider, usedModelRaw, "input").Add(float64(tokensIn))
	}

	setTimingHeaders(w, duration.Milliseconds(), upstreamStartMS)

	log.Info().
		Str("request_id", requestID).
		Str("provider", usedTarget.Provider).
		Str("model", usedModelCanonical).
		Dur("duration", duration).
		Bool("fallback", fallbackUsed).
		Int("tokens_in", tokensIn).
		Msg("embeddings request completed")

	if h.collector != nil {
		routeUsed := resolved.RouteName
		targetIndex := resolved.Index
		costUSD := observability.EstimateCostUSD(usedProviderType, usedModelRaw, tokensIn, 0)
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
				RequestType:          "embeddings",
				DurationMS:           duration.Milliseconds(),
				GatewayPreUpstreamMS: upstreamPtr,
				Provider:             usedTarget.Provider,
				Model:                usedModelCanonical,
				User:                 userPtr,
				SessionID:            sessionIDPtr,
				TokensInput:          tokensIn,
				TokensOutput:         0,
				CostUSD:              costUSD,
				StatusCode:           http.StatusOK,
				CacheHit:             cacheHit,
				RouteUsed:            &routeUsed,
				TargetIndex:          &targetIndex,
				FallbackUsed:         fallbackUsed,
				RetryCount:           retryCount,
				CircuitBreakerState:  &cbState,
				Tags:                 traceTags,
			},
		}}
		if h.collector.SharePrompts() || h.collector.ShareResponses() {
			var reqObj interface{}
			var respObj interface{}
			if h.collector.SharePrompts() {
				_ = json.Unmarshal(body, &reqObj)
			}
			if h.collector.ShareResponses() {
				respBytes, _ := json.Marshal(embeddingsResp)
				_ = json.Unmarshal(respBytes, &respObj)
			}
			events = append(events, observability.Event{
				Type: "request_log",
				Data: observability.RequestLogEventData{
					RequestID:    requestID,
					Timestamp:    startTime.UTC(),
					GatewayID:    h.collector.GatewayID(),
					RequestType:  "embeddings",
					User:         userPtr,
					SessionID:    sessionIDPtr,
					Provider:     usedTarget.Provider,
					Model:        req.Model,
					StatusCode:   http.StatusOK,
					DurationMS:   duration.Milliseconds(),
					RouteUsed:    &routeUsed,
					CacheHit:     cacheHit,
					FallbackUsed: fallbackUsed,
					RetryCount:   retryCount,
					Tags:         traceTags,
					Request:      reqObj,
					Response:     respObj,
				},
			})
		}
		h.collector.Enqueue(r.Context(), requestID, events)
	}

	writeJSON(w, http.StatusOK, embeddingsResp)
}
