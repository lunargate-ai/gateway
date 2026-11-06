package providers

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/lunargate-ai/gateway/internal/config"
	"github.com/lunargate-ai/gateway/pkg/models"
)

func decisionTestRequest(t *testing.T, body string) *models.DecisionsRequest {
	t.Helper()
	var req models.DecisionsRequest
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		t.Fatal(err)
	}
	req.RawJSON = json.RawMessage(body)
	if err := req.Validate(); err != nil {
		t.Fatal(err)
	}
	return &req
}
func decisionTestTypeSafe(images bool) *TypeSafeTranslator {
	return NewTypeSafeTranslator(config.ProviderConfig{Capabilities: config.ProviderCapabilities{Decisions: true, DecisionModels: []string{"jev-latest", "jev-1.14.0"}, DecisionImages: images}})
}
func decisionTestResponse(body string) *http.Response {
	return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(body))}
}

const decisionMixedRequest = `{"model":"jev-latest","input":[{"role":"user","content":[{"type":"input_json","data":{"sequence":9007199254740993}}]}],"questions":[{"type":"predicate","name":"same","instructions":"Is this billing?"},{"type":"choice","name":"same","instructions":"Select the typed value","choices":[{"value":true,"description":"boolean"},{"value":"true","description":"string"}]},{"type":"score","instructions":"Severity","levels":[{"label":"low"},{"label":"high","description":"blocked"}]}]}`
const decisionMixedResponse = `{"model":"jev-1.14.0","answers":{"q2":{"type":"score","score":0.75,"confidence":0.5,"probabilities":{"0":0.25,"1":0.75}},"q0":{"type":"noul","noul":0.9},"q1":{"type":"choice","choice":"c1","confidence":0.6,"probabilities":{"c1":0.8,"c0":0.2}}},"usage":{"input_tokens":42,"output_tokens":5}}`

func TestTypeSafeDecisionsPreserveIdentityOrderAndPrecision(t *testing.T) {
	translator := decisionTestTypeSafe(false)
	req := decisionTestRequest(t, decisionMixedRequest)
	httpReq, err := translator.TranslateDecisionsRequest(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(httpReq.Body)
	if httpReq.URL.Path != "/v1/systemone" || !bytes.Contains(body, []byte("9007199254740993")) {
		t.Fatalf("wrong transport or precision: %s %s", httpReq.URL.Path, body)
	}
	var payload map[string]json.RawMessage
	_ = json.Unmarshal(body, &payload)
	var questions map[string]map[string]json.RawMessage
	_ = json.Unmarshal(payload["questions"], &questions)
	if len(questions) != 3 || !bytes.Contains(questions["q1"]["criteria"], []byte(`"value":true`)) || !bytes.Contains(questions["q1"]["criteria"], []byte(`"value":"true"`)) {
		t.Fatalf("typed values were lost: %s", body)
	}
	result, err := translator.ParseDecisionsResponse(decisionTestResponse(decisionMixedResponse), req)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Answers) != 3 || result.Answers[0].Type != "predicate" || string(result.Answers[1].Choice) != `"true"` || result.Answers[2].Name != nil || result.Answers[0].Name == nil || *result.Answers[1].Name != "same" {
		t.Fatalf("mapping: %#v", result.Answers)
	}
	if result.Usage.TotalTokens != 47 || result.Model != "jev-1.14.0" || *result.Answers[2].Score != 0.75 {
		t.Fatalf("usage/score/model: %#v", result)
	}
}

func TestTypeSafeDecisionsRejectUnsupportedControlsAndNativeCorruption(t *testing.T) {
	translator := decisionTestTypeSafe(false)
	for _, field := range []string{"prompt_cache_key", "safety_identifier", "temperature"} {
		req := decisionTestRequest(t, strings.Replace(decisionMixedRequest, `"model":"jev-latest"`, `"model":"jev-latest","`+field+`":"x"`, 1))
		if _, err := translator.TranslateDecisionsRequest(context.Background(), req); err == nil || !strings.Contains(err.Error(), field) {
			t.Fatalf("%s should fail explicitly: %v", field, err)
		}
	}
	req := decisionTestRequest(t, decisionMixedRequest)
	for _, body := range []string{
		strings.Replace(decisionMixedResponse, `"q0"`, `"missing"`, 1),
		strings.Replace(decisionMixedResponse, `"q0":{"type":"noul","noul":0.9}`, `"q0":{"type":"refusal"}`, 1),
		strings.Replace(decisionMixedResponse, `"c0":0.2`, `"c0":0.1`, 1),
		strings.Replace(decisionMixedResponse, `"score":0.75`, `"score":1.75`, 1),
		strings.Replace(decisionMixedResponse, `"confidence":0.6`, `"confidence":1.6`, 1),
		strings.Replace(decisionMixedResponse, `"input_tokens":42`, `"input_tokens":-1`, 1),
	} {
		if _, err := translator.ParseDecisionsResponse(decisionTestResponse(body), req); err == nil {
			t.Fatalf("accepted corrupt native response: %s", body)
		}
	}
}

func TestTypeSafeDecisionsImagesOptInAndNoRemoteFetch(t *testing.T) {
	var pngData bytes.Buffer
	if err := png.Encode(&pngData, image.NewRGBA(image.Rect(0, 0, 2, 2))); err != nil {
		t.Fatal(err)
	}
	url := "data:image/png;base64," + base64.StdEncoding.EncodeToString(pngData.Bytes())
	body := `{"model":"jev-latest","input":"compare","questions":[{"type":"predicate","instructions":[{"type":"input_text","text":"Is this an image?"},{"type":"input_image","image_url":"` + url + `"}]}]}`
	req := decisionTestRequest(t, body)
	if _, err := decisionTestTypeSafe(false).TranslateDecisionsRequest(context.Background(), req); err == nil {
		t.Fatal("images enabled without opt-in")
	}
	httpReq, err := decisionTestTypeSafe(true).TranslateDecisionsRequest(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	encoded, _ := io.ReadAll(httpReq.Body)
	if httpReq.Header.Get("TypeSafe-Danger-Zone-Images") != "true" || !bytes.Contains(encoded, []byte(`"assets"`)) || bytes.Contains(encoded, []byte("data:image")) {
		t.Fatalf("incorrect image translation: %s", encoded)
	}
	var jpegData bytes.Buffer
	if err := jpeg.Encode(&jpegData, image.NewRGBA(image.Rect(0, 0, 2, 2)), nil); err != nil {
		t.Fatal(err)
	}
	jpegBody := strings.Replace(body, url, "data:image/jpeg;base64,"+base64.StdEncoding.EncodeToString(jpegData.Bytes()), 1)
	if _, err := decisionTestTypeSafe(true).TranslateDecisionsRequest(context.Background(), decisionTestRequest(t, jpegBody)); err != nil {
		t.Fatalf("valid JPEG rejected: %v", err)
	}
	for _, url := range []string{"https://example.com/a.png", "data:image/png;base64,broken", "data:image/gif;base64,R0lGODlh"} {
		req := decisionTestRequest(t, strings.Replace(body, "data:image/png;base64,"+base64.StdEncoding.EncodeToString(pngData.Bytes()), url, 1))
		if _, err := decisionTestTypeSafe(true).TranslateDecisionsRequest(context.Background(), req); err == nil {
			t.Fatalf("accepted unsupported image %s", url)
		}
	}
}

func TestTypeSafeDecisionsToolEvidenceRemainsData(t *testing.T) {
	body := `{"model":"jev-latest","input":[{"type":"function_call","call_id":"call1","name":"fetch","arguments":"{}"},{"type":"function_call_output","call_id":"call1","output":"result"}],"questions":[{"type":"predicate","instructions":"Did fetch return a result?"}]}`
	req := decisionTestRequest(t, body)
	httpReq, err := decisionTestTypeSafe(false).TranslateDecisionsRequest(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	encoded, _ := io.ReadAll(httpReq.Body)
	if !bytes.Contains(encoded, []byte(`"call_id":"call1"`)) || !bytes.Contains(encoded, []byte(`"output":"result"`)) {
		t.Fatalf("tool evidence lost: %s", encoded)
	}
	req = decisionTestRequest(t, strings.Replace(body, `"call_id":"call1","output"`, `"call_id":"orphan","output"`, 1))
	if _, err := decisionTestTypeSafe(false).TranslateDecisionsRequest(context.Background(), req); err == nil {
		t.Fatal("accepted orphan tool result")
	}
}

func TestTypeSafeDecisionsPreserveIndependentlyRoundedScore(t *testing.T) {
	levels := make([]models.DecisionLevel, 10)
	for i := range levels {
		levels[i].Label = fmt.Sprint(i)
	}
	req := &models.DecisionsRequest{Model: "jev-latest", Input: json.RawMessage(`"synthetic evidence"`), Questions: []models.DecisionQuestion{{Type: "score", Instructions: json.RawMessage(`"Certainty?"`), Levels: levels}}}
	// Real Jev reporting precision: weighted rounded probabilities total 7.68,
	// while its independently reported native score is 7.73. Preserve both.
	body := `{"model":"jev-1.14.0","answers":{"q0":{"type":"score","score":7.73,"confidence":0.6,"probabilities":{"0":0.02,"1":0.02,"2":0.01,"3":0.02,"4":0.02,"5":0.07,"6":0.03,"7":0.08,"8":0.16,"9":0.57}}},"usage":{"input_tokens":42,"output_tokens":5}}`
	result, err := decisionTestTypeSafe(false).ParseDecisionsResponse(decisionTestResponse(body), req)
	if err != nil {
		t.Fatal(err)
	}
	if *result.Answers[0].Score != 7.73 || result.Answers[0].Probabilities[0].Probability != 0.02 || *result.Answers[0].Confidence != 0.6 {
		t.Fatalf("native score, distribution or confidence changed: %#v", result.Answers[0])
	}
}

func TestOpenAIDecisionsPreserveCompleteEnvelopesAndRefusals(t *testing.T) {
	translator := NewOpenAITranslator(config.ProviderConfig{APIKey: "test", Capabilities: config.ProviderCapabilities{Decisions: true, DecisionModels: []string{"gpt-6-luna"}}})
	req := decisionTestRequest(t, `{"model":"gpt-6-luna","input":"test","questions":[{"type":"predicate","instructions":"A?"},{"type":"predicate","name":"b","instructions":"B?"}],"future":{"large":9007199254740993},"prompt_cache_key":"partition"}`)
	httpReq, err := translator.TranslateDecisionsRequest(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(httpReq.Body)
	if !bytes.Contains(body, []byte(`9007199254740993`)) || !bytes.Contains(body, []byte(`"prompt_cache_key":"partition"`)) {
		t.Fatalf("lost additive fields: %s", body)
	}
	native := `{"model":"gpt-6-luna","answers":[{"type":"refusal","name":null},{"type":"predicate","name":"b","probability":0.9,"future_answer":true}],"usage":{"input_tokens":1200,"output_tokens":0,"total_tokens":1200,"input_tokens_details":{"cached_tokens":1024}},"future":true}`
	result, err := translator.ParseDecisionsResponse(decisionTestResponse(native), req)
	if err != nil {
		t.Fatal(err)
	}
	encoded, _ := json.Marshal(result)
	if !bytes.Contains(encoded, []byte(`"future_answer":true`)) || result.Usage.TokenUsage().CachedInputTokens != 1024 {
		t.Fatalf("lost native fields: %s", encoded)
	}
	if err := translator.ValidateDecisionsRequest("openai", "gpt-4o", req); err == nil {
		t.Fatal("generic model accepted as native Decisions")
	}
	duplicate := decisionTestRequest(t, strings.Replace(decisionMixedRequest, `"model":"jev-latest"`, `"model":"gpt-6-luna"`, 1))
	if err := translator.ValidateDecisionsRequest("openai", "gpt-6-luna", duplicate); err == nil {
		t.Fatal("Luna accepted duplicate supplied question names")
	}
}
