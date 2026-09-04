package providers

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
)

// decodeJSONPreserveNumbers decodes a single JSON value without coercing
// arbitrary numbers through float64. Reject trailing values like json.Unmarshal.
func decodeJSONPreserveNumbers(data []byte, dst interface{}) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	if err := decoder.Decode(dst); err != nil {
		return err
	}
	var extra json.RawMessage
	if err := decoder.Decode(&extra); err != io.EOF {
		if err == nil {
			return errors.New("input must contain a single JSON value")
		}
		return err
	}
	return nil
}
