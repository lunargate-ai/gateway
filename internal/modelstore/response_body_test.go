package modelstore

import (
	"strings"
	"testing"
)

func TestDecodeModelsResponseBodyLimit(t *testing.T) {
	const document = `{"object":"list","data":[{"id":"model-ok"}]}`
	const padding = "          "
	limit := int64(len(document) + len(padding))

	var exact openAIModelsList
	if err := decodeModelsResponseWithLimit(strings.NewReader(document+padding), &exact, limit); err != nil {
		t.Fatalf("exact-limit response returned error: %v", err)
	}
	if len(exact.Data) != 1 || exact.Data[0].ID != "model-ok" {
		t.Fatalf("decoded models = %#v", exact.Data)
	}

	const secret = "secret-models-response-tail"
	var oversized openAIModelsList
	err := decodeModelsResponseWithLimit(strings.NewReader(document+padding+secret), &oversized, limit)
	if err == nil {
		t.Fatal("oversized response returned no error")
	}
	if !strings.Contains(err.Error(), "exceeds") {
		t.Fatalf("error = %v, want size-limit error", err)
	}
	if strings.Contains(err.Error(), secret) {
		t.Fatalf("oversized response content leaked in error: %v", err)
	}
}

func TestDecodeModelsResponseRejectsTrailingDocument(t *testing.T) {
	var response openAIModelsList
	err := decodeModelsResponseWithLimit(
		strings.NewReader(`{"object":"list","data":[]} {"object":"list","data":[]}`),
		&response,
		1024,
	)
	if err == nil {
		t.Fatal("multiple JSON documents returned no error")
	}
}
