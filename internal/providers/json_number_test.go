package providers

import (
	"encoding/json"
	"testing"
)

func TestDecodeJSONPreserveNumbersRejectsTrailingValue(t *testing.T) {
	var value interface{}
	if err := decodeJSONPreserveNumbers([]byte(`{"value":9007199254740993} {}`), &value); err == nil {
		t.Fatal("decodeJSONPreserveNumbers accepted a trailing JSON value")
	}
}

func TestDecodeJSONPreserveNumbersKeepsIntegerLexeme(t *testing.T) {
	var value map[string]interface{}
	if err := decodeJSONPreserveNumbers([]byte(`{"value":9007199254740993}`), &value); err != nil {
		t.Fatalf("decodeJSONPreserveNumbers returned error: %v", err)
	}
	number, ok := value["value"].(json.Number)
	if !ok || number.String() != "9007199254740993" {
		t.Fatalf("value = %#v (%T), want exact json.Number", value["value"], value["value"])
	}
}
