package api

import (
	"context"

	"github.com/lunargate-ai/gateway/internal/providers"
	"github.com/lunargate-ai/gateway/internal/routing"
)

type circuitBreakerTargetSnapshotsContextKey struct{}

type circuitBreakerTargetSnapshot struct {
	provider string
	snapshot providers.ProviderSnapshot
}

// withCircuitBreakerTargetSnapshots binds each resolved target to the provider
// generation whose identity selected its circuit breaker. Keeping the snapshot
// in the request context prevents a concurrent reload from using the old
// breaker key with a new account or endpoint.
func (h *Handler) withCircuitBreakerTargetSnapshots(ctx context.Context, resolved *routing.ResolvedRoute) context.Context {
	if h == nil || h.registry == nil || resolved == nil {
		return ctx
	}

	snapshots := make(map[string]circuitBreakerTargetSnapshot, 1+len(resolved.Fallbacks))
	bind := func(target routing.Target) routing.Target {
		snapshot, ok := h.registry.Snapshot(target.Provider)
		if !ok || snapshot.Translator == nil || snapshot.CircuitBreakerKey() == "" {
			return target
		}
		target = target.WithCircuitBreakerKey(snapshot.CircuitBreakerKey())
		snapshots[circuitBreakerTargetSnapshotMapKey(target)] = circuitBreakerTargetSnapshot{
			provider: target.Provider,
			snapshot: snapshot,
		}
		return target
	}

	resolved.Target = bind(resolved.Target)
	for i := range resolved.Fallbacks {
		resolved.Fallbacks[i] = bind(resolved.Fallbacks[i])
	}
	if len(snapshots) == 0 {
		return ctx
	}
	return context.WithValue(ctx, circuitBreakerTargetSnapshotsContextKey{}, snapshots)
}

func circuitBreakerTargetSnapshotFromContext(ctx context.Context, target routing.Target) (providers.ProviderSnapshot, bool) {
	if ctx == nil {
		return providers.ProviderSnapshot{}, false
	}
	snapshots, ok := ctx.Value(circuitBreakerTargetSnapshotsContextKey{}).(map[string]circuitBreakerTargetSnapshot)
	if !ok {
		return providers.ProviderSnapshot{}, false
	}
	bound, ok := snapshots[circuitBreakerTargetSnapshotMapKey(target)]
	if !ok || bound.provider != target.Provider || bound.snapshot.Translator == nil {
		return providers.ProviderSnapshot{}, false
	}
	return bound.snapshot, true
}

func circuitBreakerTargetSnapshotMapKey(target routing.Target) string {
	return target.Provider + "\x00" + target.CircuitBreakerKey()
}
