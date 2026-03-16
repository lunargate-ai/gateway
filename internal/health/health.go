package health

import (
	"encoding/json"
	"net/http"
	"sync/atomic"
	"time"
)

// Checker provides health and readiness endpoints.
type Checker struct {
	startTime time.Time
	ready     atomic.Bool
	version   string
}

// NewChecker creates a new health checker.
func NewChecker(version string) *Checker {
	c := &Checker{
		startTime: time.Now(),
		version:   version,
	}
	c.ready.Store(true)
	return c
}

// SetReady sets the readiness state.
func (c *Checker) SetReady(ready bool) {
	c.ready.Store(ready)
}

type healthResponse struct {
	Status  string `json:"status"`
	Version string `json:"version"`
	Uptime  string `json:"uptime"`
}

type readyResponse struct {
	Status string `json:"status"`
}

// HealthHandler returns an HTTP handler for the health endpoint.
func (c *Checker) HealthHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(healthResponse{
			Status:  "healthy",
			Version: c.version,
			Uptime:  time.Since(c.startTime).Round(time.Second).String(),
		})
	}
}

// ReadyHandler returns an HTTP handler for the readiness endpoint.
func (c *Checker) ReadyHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if c.ready.Load() {
			w.WriteHeader(http.StatusOK)
			json.NewEncoder(w).Encode(readyResponse{Status: "ready"})
		} else {
			w.WriteHeader(http.StatusServiceUnavailable)
			json.NewEncoder(w).Encode(readyResponse{Status: "not ready"})
		}
	}
}
