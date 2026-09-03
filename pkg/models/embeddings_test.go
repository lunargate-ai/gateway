package models

import (
	"encoding/json"
	"testing"
)

func TestEmbeddingValueAcceptsFloatAndBase64Representations(t *testing.T) {
	for _, embedding := range []string{`[0.1,-0.25,3]`, `"AQIDBA=="`} {
		body := []byte(`{"object":"embedding","embedding":` + embedding + `,"index":0}`)
		var data EmbeddingData
		if err := json.Unmarshal(body, &data); err != nil {
			t.Fatalf("unmarshal %s: %v", embedding, err)
		}
		roundTrip, err := json.Marshal(data)
		if err != nil {
			t.Fatalf("marshal %s: %v", embedding, err)
		}
		var got map[string]json.RawMessage
		if err := json.Unmarshal(roundTrip, &got); err != nil {
			t.Fatalf("decode round trip %s: %v", embedding, err)
		}
		if string(got["embedding"]) != embedding {
			t.Fatalf("embedding round trip = %s, want %s", got["embedding"], embedding)
		}
	}
}

func TestEmbeddingValueRejectsInvalidRepresentation(t *testing.T) {
	for _, embedding := range []string{`null`, `{"value":1}`, `[1,"bad"]`} {
		body := []byte(`{"object":"embedding","embedding":` + embedding + `,"index":0}`)
		var data EmbeddingData
		if err := json.Unmarshal(body, &data); err == nil {
			t.Fatalf("expected %s to be rejected", embedding)
		}
	}
}
