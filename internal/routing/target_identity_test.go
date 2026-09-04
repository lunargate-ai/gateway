package routing

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

func TestTargetCircuitBreakerKeyIsInternalAndCopySafe(t *testing.T) {
	original := Target{Provider: "openai", Model: "gpt-test", Weight: 1}
	if got := original.CircuitBreakerKey(); got != "openai" {
		t.Fatalf("legacy breaker key = %q, want provider alias", got)
	}

	const opaqueKey = "opaque-account-fingerprint"
	bound := original.WithCircuitBreakerKey(opaqueKey)
	if got := bound.CircuitBreakerKey(); got != opaqueKey {
		t.Fatalf("bound breaker key = %q, want %q", got, opaqueKey)
	}
	if got := original.CircuitBreakerKey(); got != "openai" {
		t.Fatalf("binding mutated original target: %q", got)
	}

	encoded, err := json.Marshal(bound)
	if err != nil {
		t.Fatalf("marshal target: %v", err)
	}
	if strings.Contains(string(encoded), opaqueKey) || strings.Contains(string(encoded), "circuitBreaker") {
		t.Fatal("serialized target exposes breaker identity")
	}
	if diagnostic := fmt.Sprintf("%#v", bound); strings.Contains(diagnostic, opaqueKey) {
		t.Fatal("diagnostic target formatting exposes breaker identity")
	}
}
