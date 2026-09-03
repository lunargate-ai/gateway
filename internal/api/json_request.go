package api

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
)

const maxRequestBodyBytes int64 = 10 << 20

func limitRequestBody(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, maxRequestBodyBytes)
}

func decodeJSONStrict(reader io.Reader, dst interface{}) error {
	decoder := json.NewDecoder(reader)
	decoder.UseNumber()
	if err := decoder.Decode(dst); err != nil {
		return err
	}
	var extra json.RawMessage
	if err := decoder.Decode(&extra); err != io.EOF {
		if err == nil {
			return errors.New("request body must contain a single JSON value")
		}
		return err
	}
	return nil
}
