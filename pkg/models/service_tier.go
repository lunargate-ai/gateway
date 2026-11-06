package models

import "encoding/json"

// KnownServiceTier bounds metadata to public, content-free API values. Native
// forwarding still preserves the original JSON and future provider fields.
func KnownServiceTier(value *string) *string {
	if value == nil {
		return nil
	}
	switch *value {
	case "auto", "default", "fast", "priority", "ultrafast", "flex", "scale":
		copy := *value
		return &copy
	}
	return nil
}

func RequestServiceTier(req *UnifiedRequest) *string {
	if req == nil {
		return nil
	}
	if req.ServiceTier != nil {
		return req.ServiceTier
	}
	var payload struct {
		ServiceTier *string `json:"service_tier"`
	}
	if json.Unmarshal(req.RawJSON, &payload) == nil {
		return payload.ServiceTier
	}
	return nil
}
