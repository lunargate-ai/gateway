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
	"github.com/lunargate-ai/gateway/internal/observability"
	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/internal/resilience"
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

	if strings.TrimSpace(req.Model) == "" {
		writeError(w, http.StatusBadRequest, "model is required", "invalid_request_error")
		return nil, nil, false
	}
	if req.Input == nil {
		writeError(w, http.StatusBadRequest, "input is required", "invalid_request_error")
		return nil, nil, false
	}

	if !captureBody {
		return nil, &req, true
	}
	return body, &req, true
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
	var unavailable *routing.RequestedTargetUnavailableError
	if errors.As(err, &unavailable) {
		return nil, err
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

func (h *Handler) validateEmbeddingsCompatibility(target routing.Target, req *models.EmbeddingsRequest) error {
	if h == nil || h.registry == nil || req == nil {
		return nil
	}
	providerType, typeOK := h.registry.Type(target.Provider)
	format := strings.ToLower(strings.TrimSpace(req.EncodingFormat))
	switch format {
	case "", "float":
	case "base64":
		capabilities, ok := h.registry.Capabilities(target.Provider)
		if ok && typeOK && capabilities.EmbeddingsBase64 && strings.EqualFold(providerType, "openai") {
			return nil
		}
		return &models.CompatibilityError{
			Field:    "encoding_format",
			Provider: target.Provider,
			Reason:   "base64 embeddings are not enabled for this provider",
		}
	default:
		return &models.CompatibilityError{
			Field:    "encoding_format",
			Provider: target.Provider,
			Reason:   fmt.Sprintf("unsupported value %q", req.EncodingFormat),
		}
	}
	if typeOK && strings.EqualFold(providerType, "ollama") {
		if req.Dimensions != nil {
			return &models.CompatibilityError{
				Field:    "dimensions",
				Provider: target.Provider,
				Reason:   "Ollama's embed API does not expose output dimension selection",
			}
		}
		if strings.TrimSpace(req.User) != "" {
			return &models.CompatibilityError{
				Field:    "user",
				Provider: target.Provider,
				Reason:   "Ollama's embed API has no equivalent end-user identifier field",
			}
		}
		if !ollamaEmbeddingInputCompatible(req.Input) {
			return &models.CompatibilityError{
				Field:    "input",
				Provider: target.Provider,
				Reason:   "Ollama's embed API accepts only a string or an array of strings",
			}
		}
	}
	return nil
}

func ollamaEmbeddingInputCompatible(input interface{}) bool {
	switch value := input.(type) {
	case string:
		return true
	case []string:
		return true
	case []interface{}:
		for _, item := range value {
			if _, ok := item.(string); !ok {
				return false
			}
		}
		return true
	default:
		return false
	}
}

func (h *Handler) compatibleEmbeddingsFallbacks(fallbacks []routing.Target, req *models.EmbeddingsRequest) []routing.Target {
	if len(fallbacks) == 0 {
		return nil
	}
	compatible := make([]routing.Target, 0, len(fallbacks))
	for _, target := range fallbacks {
		if err := h.validateEmbeddingsCompatibility(target, req); err != nil {
			log.Warn().
				Err(err).
				Str("provider", target.Provider).
				Str("model", target.Model).
				Msg("skipping incompatible embeddings fallback target")
			continue
		}
		compatible = append(compatible, target)
	}
	return compatible
}

func (h *Handler) callEmbeddingsProvider(ctx context.Context, target routing.Target, req *models.EmbeddingsRequest, beforeUpstream func()) (*http.Response, error) {
	providerSnapshot, ok := h.registry.Snapshot(target.Provider)
	if !ok {
		return nil, fmt.Errorf("unknown provider: %s", target.Provider)
	}
	embeddingTranslator, ok := providerSnapshot.Translator.(embeddingsTranslator)
	if !ok {
		return nil, fmt.Errorf("provider %s does not support embeddings", target.Provider)
	}
	ctx = withProviderRequestSnapshot(ctx, target.Provider, providerSnapshot)

	reqCopy := *req
	if strings.TrimSpace(target.Model) != "" {
		reqCopy.Model = strings.TrimSpace(target.Model)
	}
	reqCopy.Model = modelid.ModelName(reqCopy.Model)

	httpReq, err := embeddingTranslator.TranslateEmbeddingsRequest(ctx, &reqCopy)
	if err != nil {
		return nil, resilience.NewRequestError(fmt.Errorf("failed to translate embeddings request for %s: %w", target.Provider, err))
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
	} else {
		requestedModelRaw = strings.TrimSpace(req.Model)
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
		var unavailable *routing.RequestedTargetUnavailableError
		if errors.As(err, &unavailable) {
			writeRequestedTargetUnavailable(w, unavailable)
			return
		}
		log.Error().Err(err).Str("request_id", requestID).Msg("failed to resolve embeddings route")
		writeError(w, http.StatusBadGateway, "no route matched for this request", "routing_error")
		return
	}
	if err := h.validateEmbeddingsCompatibility(resolved.Target, &req); err != nil {
		var compatibilityErr *models.CompatibilityError
		if errors.As(err, &compatibilityErr) {
			writeCompatibilityError(w, compatibilityErr)
			return
		}
		writeError(w, http.StatusBadRequest, err.Error(), "invalid_request_error")
		return
	}
	resolved.Fallbacks = h.compatibleEmbeddingsFallbacks(resolved.Fallbacks, &req)
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
	requestTypes := embeddingsAPIRequestTypes()
	collectorHeaders := requestTypes.tags(headers)
	traceTags := h.enrichCollectorTags(collectorHeaders, resolved.Target.Provider, req.Model, false)

	noCache := r.Header.Get("X-LunarGate-No-Cache") == "true"
	if !noCache && h.cache.Enabled() {
		cacheKey := middleware.GenerateEmbeddingsKeyForTarget(&req, resolved.Target.Provider, resolved.Target.UpstreamRequestType)
		if cached := h.cache.Get(cacheKey); cached != nil {
			if cachedResp, ok := cached.(*models.EmbeddingsResponse); ok {
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
				tokensIn := 0
				if cachedResp.Usage != nil {
					tokensIn = cachedResp.Usage.PromptTokens
					if tokensIn == 0 {
						tokensIn = cachedResp.Usage.TotalTokens
					}
				}
				var requestPayload interface{}
				if h.collector != nil && h.collector.SharePrompts() {
					_ = json.Unmarshal(body, &requestPayload)
				}
				duration := h.recordCacheHit(r.Context(), cacheHitObservation{
					requestID:    requestID,
					startTime:    startTime,
					requestTypes: requestTypes,
					provider:     resolved.Target.Provider,
					model:        cachedModelCanonical,
					metricsModel: cachedModelRaw,
					route:        resolved.RouteName,
					targetIndex:  resolved.Index,
					user:         userPtr,
					sessionID:    sessionIDPtr,
					tags:         h.enrichCollectorTags(collectorHeaders, resolved.Target.Provider, cachedModelCanonical, false),
					tokensInput:  tokensIn,
					request:      requestPayload,
					response:     embeddingsResponseForCollector(cachedResp),
				})
				w.Header().Set("X-LunarGate-Cache-Status", "HIT")
				w.Header().Set("X-LunarGate-Provider", resolved.Target.Provider)
				w.Header().Set("X-LunarGate-Model", cachedModelCanonical)
				setTimingHeaders(w, duration.Milliseconds(), duration.Milliseconds())
				writeEmbeddingsJSON(w, http.StatusOK, models.CloneEmbeddingsResponse(cachedResp))
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
		requestFailure := false
		var upstreamFailure *upstreamHTTPError
		if errors.Is(err, context.Canceled) {
			status = 499
			errCode = "client_cancelled"
			errMsg = "client disconnected"
			log.Info().
				Str("request_id", requestID).
				Dur("duration", duration).
				Msg("embeddings request cancelled")
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
				Msg("embeddings request rejected before upstream call")
		} else if isUpstreamTTFTTimeout(err) {
			errCode = "upstream_timeout"
			errMsg = "provider timed out waiting for first byte"
			log.Error().Err(err).
				Str("request_id", requestID).
				Dur("duration", duration).
				Msg("embeddings first-byte timeout")
		} else if isUpstreamTotalTimeout(err) {
			errCode = "upstream_timeout"
			errMsg = "provider timed out before full response completed"
			log.Error().Err(err).
				Str("request_id", requestID).
				Dur("duration", duration).
				Msg("embeddings total timeout")
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
					Msg("upstream embeddings provider failure after retries")
			} else {
				log.Error().Err(err).
					Str("request_id", requestID).
					Dur("duration", duration).
					Msg("all embeddings providers failed")
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
			} else if isUpstreamTTFTTimeout(err) {
				errCodeForCollector = "upstream_timeout"
				errMsgForCollector = "provider timed out waiting for first byte"
			} else if isUpstreamTotalTimeout(err) {
				errCodeForCollector = "upstream_timeout"
				errMsgForCollector = "provider timed out before full response completed"
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
					RequestType:          requestTypes.client,
					UpstreamRequestType:  requestTypes.upstream,
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
						RequestID:           requestID,
						Timestamp:           startTime.UTC(),
						RequestType:         requestTypes.client,
						UpstreamRequestType: requestTypes.upstream,
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
						ErrorCode:           &errCodeForCollector,
						ErrorMessage:        &errMsgForCollector,
						Tags:                traceTags,
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
	req.Model = usedModelCanonical
	w.Header().Set("X-LunarGate-Model", usedModelCanonical)

	usedProviderType := providerSnapshot.ProviderType
	setTimingHeaders(w, -1, upstreamStartMS)

	embeddingsTranslator, ok := providerSnapshot.Translator.(embeddingsTranslator)
	if !ok {
		resp.Body.Close()
		writeError(w, http.StatusBadGateway, "provider does not support embeddings", "provider_error")
		return
	}
	responseHeaders := resp.Header.Clone()

	var embeddingsResp *models.EmbeddingsResponse
	var upstreamFailure *upstreamHTTPError
	if resp.StatusCode >= http.StatusBadRequest {
		upstreamFailure = readUpstreamHTTPError(resp, usedProviderType)
		err = upstreamFailure
	} else {
		embeddingsResp, err = embeddingsTranslator.ParseEmbeddingsResponse(resp)
	}
	copyHeaders(w.Header(), responseHeaders)
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
			Msg("failed to parse embeddings provider response")
		h.metrics.ProviderErrors.WithLabelValues(usedTarget.Provider, metricErrType).Inc()
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
					RequestType:          requestTypes.client,
					UpstreamRequestType:  requestTypes.upstream,
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
						RequestID:           requestID,
						Timestamp:           startTime.UTC(),
						RequestType:         requestTypes.client,
						UpstreamRequestType: requestTypes.upstream,
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
						Tags:                traceTags,
						Request:             reqAny,
					},
				})
			}
			h.collector.Enqueue(r.Context(), requestID, events)
		}
		return
	}

	if !noCache && h.cache.Enabled() {
		cacheKey := middleware.GenerateEmbeddingsKeyForTarget(&req, usedTarget.Provider, usedTarget.UpstreamRequestType)
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
				RequestType:          requestTypes.client,
				UpstreamRequestType:  requestTypes.upstream,
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
				respObj = embeddingsResponseForCollector(embeddingsResp)
			}
			events = append(events, observability.Event{
				Type: "request_log",
				Data: observability.RequestLogEventData{
					RequestID:           requestID,
					Timestamp:           startTime.UTC(),
					RequestType:         requestTypes.client,
					UpstreamRequestType: requestTypes.upstream,
					User:                userPtr,
					SessionID:           sessionIDPtr,
					Provider:            usedTarget.Provider,
					Model:               req.Model,
					StatusCode:          http.StatusOK,
					DurationMS:          duration.Milliseconds(),
					RouteUsed:           &routeUsed,
					CacheHit:            cacheHit,
					FallbackUsed:        fallbackUsed,
					RetryCount:          retryCount,
					Tags:                traceTags,
					Request:             reqObj,
					Response:            respObj,
				},
			})
		}
		h.collector.Enqueue(r.Context(), requestID, events)
	}

	writeEmbeddingsJSON(w, http.StatusOK, embeddingsResp)
}

func embeddingsResponseJSON(resp *models.EmbeddingsResponse) []byte {
	if resp != nil && json.Valid(bytes.TrimSpace(resp.RawJSON)) {
		return append([]byte(nil), resp.RawJSON...)
	}
	body, _ := json.Marshal(resp)
	return body
}

func embeddingsResponseForCollector(resp *models.EmbeddingsResponse) interface{} {
	var payload interface{}
	if err := json.Unmarshal(embeddingsResponseJSON(resp), &payload); err != nil {
		return resp
	}
	return payload
}

func writeEmbeddingsJSON(w http.ResponseWriter, status int, resp *models.EmbeddingsResponse) {
	raw := embeddingsResponseJSON(resp)
	if len(bytes.TrimSpace(raw)) == 0 || !json.Valid(raw) {
		writeJSON(w, status, resp)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if _, err := w.Write(raw); err != nil {
		log.Error().Err(err).Msg("failed to write embeddings JSON response")
	}
}
