package streaming

import (
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/lunargate-ai/gateway/pkg/models"
)

func marshalStreamChunk(chunk *models.StreamChunk) ([]byte, error) {
	typed, err := json.Marshal(chunk)
	if err != nil {
		return nil, err
	}
	if chunk == nil || len(bytes.TrimSpace(chunk.RawJSON)) == 0 {
		return typed, nil
	}

	merged, err := overlayStreamJSON(chunk.RawJSON, typed)
	if err != nil {
		return nil, fmt.Errorf("merge preserved stream chunk: %w", err)
	}
	return merged, nil
}

// overlayStreamJSON recursively overlays normalized typed JSON on the raw
// upstream document. Object fields unknown to the gateway remain untouched;
// matching array entries are merged by position so additive fields nested in
// choices, deltas, tool calls, and logprobs survive parse and marshal.
func overlayStreamJSON(base, overlay json.RawMessage) (json.RawMessage, error) {
	base = bytes.TrimSpace(base)
	overlay = bytes.TrimSpace(overlay)
	if len(base) == 0 || len(overlay) == 0 {
		return append(json.RawMessage(nil), overlay...), nil
	}

	if base[0] == '{' && overlay[0] == '{' {
		var baseFields map[string]json.RawMessage
		var overlayFields map[string]json.RawMessage
		if err := json.Unmarshal(base, &baseFields); err != nil {
			return nil, err
		}
		if err := json.Unmarshal(overlay, &overlayFields); err != nil {
			return nil, err
		}
		for key, overlayValue := range overlayFields {
			if baseValue, ok := baseFields[key]; ok {
				merged, err := overlayStreamJSON(baseValue, overlayValue)
				if err != nil {
					return nil, fmt.Errorf("field %q: %w", key, err)
				}
				baseFields[key] = merged
				continue
			}
			baseFields[key] = append(json.RawMessage(nil), overlayValue...)
		}
		return json.Marshal(baseFields)
	}

	if base[0] == '[' && overlay[0] == '[' {
		var baseItems []json.RawMessage
		var overlayItems []json.RawMessage
		if err := json.Unmarshal(base, &baseItems); err != nil {
			return nil, err
		}
		if err := json.Unmarshal(overlay, &overlayItems); err != nil {
			return nil, err
		}
		if len(overlayItems) == 0 {
			return append(json.RawMessage(nil), overlay...), nil
		}
		for index, overlayValue := range overlayItems {
			if index < len(baseItems) {
				merged, err := overlayStreamJSON(baseItems[index], overlayValue)
				if err != nil {
					return nil, fmt.Errorf("item %d: %w", index, err)
				}
				baseItems[index] = merged
				continue
			}
			baseItems = append(baseItems, append(json.RawMessage(nil), overlayValue...))
		}
		return json.Marshal(baseItems)
	}

	return append(json.RawMessage(nil), overlay...), nil
}

func removeRawStreamField(raw json.RawMessage, field string) (json.RawMessage, bool) {
	if len(bytes.TrimSpace(raw)) == 0 {
		return raw, false
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil || fields == nil {
		return raw, false
	}
	if _, ok := fields[field]; !ok {
		return raw, false
	}
	delete(fields, field)
	filtered, err := json.Marshal(fields)
	if err != nil {
		return raw, false
	}
	return filtered, true
}
