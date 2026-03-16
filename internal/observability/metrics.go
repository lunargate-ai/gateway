package observability

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// Metrics holds all Prometheus metrics for the gateway.
type Metrics struct {
	RequestsTotal    *prometheus.CounterVec
	RequestDuration  *prometheus.HistogramVec
	TokensTotal      *prometheus.CounterVec
	CacheHits        *prometheus.CounterVec
	ProviderErrors   *prometheus.CounterVec
	FallbacksUsed    prometheus.Counter
	CircuitBreakerState *prometheus.GaugeVec
	ActiveRequests   prometheus.Gauge
}

// NewMetrics registers and returns all gateway metrics.
func NewMetrics() *Metrics {
	return NewMetricsWithRegisterer(prometheus.DefaultRegisterer)
}

func NewMetricsWithRegisterer(reg prometheus.Registerer) *Metrics {
	factory := promauto.With(reg)
	return &Metrics{
		RequestsTotal: factory.NewCounterVec(prometheus.CounterOpts{
			Namespace: "ai_router",
			Name:      "requests_total",
			Help:      "Total number of requests processed",
		}, []string{"provider", "model", "status_code", "route"}),

		RequestDuration: factory.NewHistogramVec(prometheus.HistogramOpts{
			Namespace: "ai_router",
			Name:      "request_duration_seconds",
			Help:      "Request duration in seconds",
			Buckets:   []float64{0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120},
		}, []string{"provider", "model"}),

		TokensTotal: factory.NewCounterVec(prometheus.CounterOpts{
			Namespace: "ai_router",
			Name:      "tokens_total",
			Help:      "Total number of tokens processed",
		}, []string{"provider", "model", "direction"}),

		CacheHits: factory.NewCounterVec(prometheus.CounterOpts{
			Namespace: "ai_router",
			Name:      "cache_hits_total",
			Help:      "Total number of cache hits/misses",
		}, []string{"status"}),

		ProviderErrors: factory.NewCounterVec(prometheus.CounterOpts{
			Namespace: "ai_router",
			Name:      "provider_errors_total",
			Help:      "Total number of provider errors",
		}, []string{"provider", "error_type"}),

		FallbacksUsed: factory.NewCounter(prometheus.CounterOpts{
			Namespace: "ai_router",
			Name:      "fallbacks_used_total",
			Help:      "Total number of fallback invocations",
		}),

		CircuitBreakerState: factory.NewGaugeVec(prometheus.GaugeOpts{
			Namespace: "ai_router",
			Name:      "circuit_breaker_state",
			Help:      "Circuit breaker state (0=closed, 1=half-open, 2=open)",
		}, []string{"provider"}),

		ActiveRequests: factory.NewGauge(prometheus.GaugeOpts{
			Namespace: "ai_router",
			Name:      "active_requests",
			Help:      "Number of currently active requests",
		}),
	}
}
