package modelid

import "strings"

func SplitCanonical(modelID string) (providerID string, model string, ok bool) {
	m := strings.TrimSpace(modelID)
	if m == "" {
		return "", "", false
	}
	idx := strings.IndexByte(m, '/')
	if idx <= 0 || idx >= len(m)-1 {
		return "", m, false
	}
	p := strings.TrimSpace(m[:idx])
	r := strings.TrimSpace(m[idx+1:])
	if p == "" || r == "" {
		return "", m, false
	}
	return p, r, true
}

func BuildCanonical(providerID string, model string) string {
	p := strings.TrimSpace(providerID)
	m := strings.TrimSpace(model)
	if p == "" {
		return m
	}
	if m == "" {
		return p + "/"
	}
	return p + "/" + m
}

func ModelName(modelID string) string {
	_, m, ok := SplitCanonical(modelID)
	if ok {
		return m
	}
	return strings.TrimSpace(modelID)
}
