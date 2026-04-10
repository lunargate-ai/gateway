package api

import (
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
		r.Use(rateLimiter.Middleware)
		r.Post("/chat/completions", handler.ChatCompletions)
		r.Post("/responses", handler.Responses)
		r.Get("/responses", handler.ResponsesWebSocket)
		r.Post("/embeddings", handler.Embeddings)
		r.Get("/models", handler.ListModels)
		r.Get("/models/{model}", handler.GetModel)
	})

	return r
}
