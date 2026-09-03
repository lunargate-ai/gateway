package api

import (
	"net/http"

	"github.com/go-chi/chi/v5"
	chimw "github.com/go-chi/chi/v5/middleware"
	"github.com/lunargate-ai/gateway/internal/health"
	"github.com/lunargate-ai/gateway/internal/middleware"
	"github.com/lunargate-ai/gateway/internal/security"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// NewRouter creates and configures the chi router with all routes and middleware.
func NewRouter(handler *Handler, authManager *security.Manager, rateLimiter *middleware.RateLimiter, healthChecker *health.Checker) *chi.Mux {
	r := chi.NewRouter()

	// Global middleware
	r.Use(middleware.CapturePeerAddress)
	r.Use(chimw.RealIP)
	r.Use(chimw.Recoverer)
	r.Use(chimw.RequestID)

	// Health & operational endpoints (no rate limiting)
	r.Get("/health", healthChecker.HealthHandler())
	r.Get("/ready", healthChecker.ReadyHandler())
	r.Get("/metrics", promhttp.Handler().ServeHTTP)

	// OpenAI-compatible API routes
	r.Route("/v1", func(r chi.Router) {
		if authManager != nil {
			r.Use(authManager.Middleware)
		}
		if rateLimiter != nil {
			r.Use(rateLimiter.Middleware)
		}

		messagePolicy := newResponsesWebSocketMessagePolicy(authManager, rateLimiter)
		r.Get("/chat/completions", handler.withRuntime((*Handler).ListStoredChatCompletions))
		r.Post("/chat/completions", handler.withRuntime((*Handler).ChatCompletions))
		r.Get("/chat/completions/{completion_id}", handler.withRuntime((*Handler).RetrieveStoredChatCompletion))
		r.Post("/chat/completions/{completion_id}", handler.withRuntime((*Handler).UpdateStoredChatCompletion))
		r.Delete("/chat/completions/{completion_id}", handler.withRuntime((*Handler).DeleteStoredChatCompletion))
		r.Get("/chat/completions/{completion_id}/messages", handler.withRuntime((*Handler).ListStoredChatCompletionMessages))
		r.Post("/responses", handler.withRuntime((*Handler).Responses))
		r.Get("/responses", handler.withRuntime(func(bound *Handler, w http.ResponseWriter, request *http.Request) {
			bound.responsesWebSocket(w, request, messagePolicy)
		}))
		r.Post("/responses/compact", handler.withRuntime((*Handler).CompactResponses))
		r.Post("/responses/input_tokens", handler.withRuntime((*Handler).CountResponseInputTokens))
		r.Get("/responses/{response_id}", handler.withRuntime((*Handler).RetrieveResponse))
		r.Delete("/responses/{response_id}", handler.withRuntime((*Handler).DeleteResponse))
		r.Post("/responses/{response_id}/cancel", handler.withRuntime((*Handler).CancelResponse))
		r.Get("/responses/{response_id}/input_items", handler.withRuntime((*Handler).ListResponseInputItems))
		r.Post("/conversations", handler.withRuntime((*Handler).CreateConversation))
		r.Get("/conversations/{conversation_id}", handler.withRuntime((*Handler).GetConversation))
		r.Post("/conversations/{conversation_id}", handler.withRuntime((*Handler).UpdateConversation))
		r.Delete("/conversations/{conversation_id}", handler.withRuntime((*Handler).DeleteConversation))
		r.Post("/conversations/{conversation_id}/items", handler.withRuntime((*Handler).CreateConversationItems))
		r.Get("/conversations/{conversation_id}/items", handler.withRuntime((*Handler).ListConversationItems))
		r.Get("/conversations/{conversation_id}/items/{item_id}", handler.withRuntime((*Handler).GetConversationItem))
		r.Delete("/conversations/{conversation_id}/items/{item_id}", handler.withRuntime((*Handler).DeleteConversationItem))
		r.Post("/embeddings", handler.withRuntime((*Handler).Embeddings))
		r.Get("/models", handler.withRuntime((*Handler).ListModels))
		// Canonical model IDs may contain provider and vendor path segments.
		r.Get("/models/*", handler.withRuntime((*Handler).GetModel))
	})

	return r
}
