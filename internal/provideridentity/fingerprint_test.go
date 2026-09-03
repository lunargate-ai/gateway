package provideridentity

import (
	"strings"
	"testing"
)

func TestAccountFingerprintUsesCanonicalIdentity(t *testing.T) {
	first := AccountFingerprint(" OpenAI ", " HTTPS://API.EXAMPLE.COM/v1/ ", " org ", "secret")
	second := AccountFingerprint("openai", "https://api.example.com/v1", "org", "secret")
	if first != second {
		t.Fatal("equivalent provider accounts have different fingerprints")
	}

	for name, changed := range map[string]string{
		"type":         AccountFingerprint("anthropic", "https://api.example.com/v1", "org", "secret"),
		"endpoint":     AccountFingerprint("openai", "https://other.example.com/v1", "org", "secret"),
		"organization": AccountFingerprint("openai", "https://api.example.com/v1", "other", "secret"),
		"credential":   AccountFingerprint("openai", "https://api.example.com/v1", "org", "other-secret"),
	} {
		if changed == second {
			t.Fatalf("%s change did not change account fingerprint", name)
		}
	}
	if strings.Contains(second, "secret") || strings.Contains(second, "api.example.com") {
		t.Fatal("fingerprint exposes account material")
	}
}

func TestCircuitBreakerKeyScopesAliasesAndWireVersions(t *testing.T) {
	account := AccountFingerprint("anthropic", "https://api.anthropic.com", "", "secret")
	base := CircuitBreakerKey("primary", account, "2023-06-01")
	if base == CircuitBreakerKey("fallback", account, "2023-06-01") {
		t.Fatal("different provider aliases share a circuit-breaker key")
	}
	if base == CircuitBreakerKey("primary", account, "2024-01-01") {
		t.Fatal("different API versions share a circuit-breaker key")
	}
	if strings.Contains(base, account) || strings.Contains(base, "primary") {
		t.Fatal("circuit-breaker key exposes identity input")
	}
}
