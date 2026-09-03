package modelstore

import (
	"encoding/json"
	"fmt"
	"io"
)

const maxModelsResponseBodyBytes int64 = 16 << 20

func decodeModelsResponse(body io.Reader, target interface{}) error {
	return decodeModelsResponseWithLimit(body, target, maxModelsResponseBodyBytes)
}

func decodeModelsResponseWithLimit(body io.Reader, target interface{}, limit int64) error {
	if body == nil {
		return fmt.Errorf("models response body is nil")
	}

	payload, err := io.ReadAll(io.LimitReader(body, limit+1))
	if err != nil {
		return fmt.Errorf("failed to read models response: %w", err)
	}
	if int64(len(payload)) > limit {
		return fmt.Errorf("models response exceeds %d byte limit", limit)
	}
	if err := json.Unmarshal(payload, target); err != nil {
		return fmt.Errorf("invalid models response JSON: %w", err)
	}
	return nil
}
