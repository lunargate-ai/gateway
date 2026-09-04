package models

import "testing"

func TestCompatibilityErrorFormatsStableMessage(t *testing.T) {
	err := (&CompatibilityError{Field: "background", Provider: "ollama"}).Error()
	want := `field "background" is not supported by provider "ollama"`
	if err != want {
		t.Fatalf("error = %q, want %q", err, want)
	}
}
