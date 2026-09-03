package provideridentity

import (
	"crypto/sha256"
	"encoding/hex"
	"net/url"
	"strconv"
	"strings"
)

// AccountFingerprint returns a stable, one-way identifier for the effective
// provider account. Callers must treat the returned digest as sensitive
// internal metadata and must not log or expose it.
func AccountFingerprint(providerType, baseURL, organization, apiKey string) string {
	return digest(
		strings.ToLower(strings.TrimSpace(providerType)),
		normalizeBaseURL(baseURL),
		strings.TrimSpace(organization),
		apiKey,
	)
}

// CircuitBreakerKey derives an opaque lookup key from the provider alias,
// account fingerprint, and any API version that changes the wire contract.
// It is intentionally unsuitable for logs or API responses.
func CircuitBreakerKey(providerID, accountFingerprint, apiVersion string) string {
	return digest(
		providerID,
		accountFingerprint,
		strings.TrimSpace(apiVersion),
	)
}

func digest(values ...string) string {
	hash := sha256.New()
	for _, value := range values {
		_, _ = hash.Write([]byte(strconv.Itoa(len(value))))
		_, _ = hash.Write([]byte{':'})
		_, _ = hash.Write([]byte(value))
	}
	return hex.EncodeToString(hash.Sum(nil))
}

func normalizeBaseURL(raw string) string {
	raw = strings.TrimSpace(raw)
	parsed, err := url.Parse(raw)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return strings.TrimRight(raw, "/")
	}
	parsed.Scheme = strings.ToLower(parsed.Scheme)
	parsed.Host = strings.ToLower(parsed.Host)
	parsed.Path = strings.TrimRight(parsed.Path, "/")
	parsed.Fragment = ""
	return parsed.String()
}
