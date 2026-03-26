package api

import (
	"encoding/json"
	"io"
	"net/http"
)

const maxRequestBodyBytes int64 = 10 << 20

func limitRequestBody(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, maxRequestBodyBytes)
}

func decodeJSONStrict(reader io.Reader, dst interface{}) error {
	decoder := json.NewDecoder(reader)
	if err := decoder.Decode(dst); err != nil {
		return err
	}
	var extra json.RawMessage
	if err := decoder.Decode(&extra); err != io.EOF {
		return err
	}
	return nil
}
