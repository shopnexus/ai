package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/shopnexus/ai/embed"
)

// stubEmbedder answers vector [i, 0, 0, ...] for input i, or err.
type stubEmbedder struct {
	err   error
	got   *[]embed.Input
	block chan struct{} // when set, Embed waits on it or on ctx
}

func (s stubEmbedder) HasModel(name string) bool { return name == "siglip2-so400m-patch16-384" }

func (s stubEmbedder) Embed(ctx context.Context, _ string, in []embed.Input) (embed.Result, error) {
	if s.got != nil {
		*s.got = in
	}
	if s.block != nil {
		select {
		case <-s.block:
		case <-ctx.Done():
			return embed.Result{}, ctx.Err()
		}
	}
	if s.err != nil {
		return embed.Result{}, s.err
	}
	res := embed.Result{Vectors: make([][]float32, len(in))}
	for i := range in {
		res.Vectors[i] = []float32{float32(i), 0, 0}
		if in[i].Image != nil {
			res.Tokens += embed.ImageTokens
		} else {
			res.Tokens += 3
		}
	}
	return res, nil
}

func postEmbeddings(t *testing.T, e embedder, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	newHandler(&fake{}, e, limits{maxUpload: 1 << 20, maxInputs: 3, maxImageBytes: 1 << 10}, slog.New(slog.DiscardHandler)).ServeHTTP(rec, req)
	return rec
}

func errCode(t *testing.T, rec *httptest.ResponseRecorder) string {
	t.Helper()
	var r struct {
		Error struct{ Code, Message string } `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &r); err != nil {
		t.Fatalf("body %q: %v", rec.Body, err)
	}
	return r.Error.Code
}

const pngURL = "data:image/png;base64,iVBORw0KGgo="

func TestEmbeddingsMixesTextAndImages(t *testing.T) {
	var got []embed.Input
	rec := postEmbeddings(t, stubEmbedder{got: &got},
		`{"model":"siglip2-so400m-patch16-384","input":["a blue bicycle","`+pngURL+`"]}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	if got[0].Text != "a blue bicycle" || got[0].Image != nil {
		t.Errorf("input 0 = %+v", got[0])
	}
	if !bytes.Equal(got[1].Image, []byte("\x89PNG\r\n\x1a\n")) {
		t.Errorf("input 1 image = %q", got[1].Image)
	}
	var r struct {
		Object string `json:"object"`
		Model  string `json:"model"`
		Data   []struct {
			Object    string    `json:"object"`
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
		Usage struct {
			PromptTokens int `json:"prompt_tokens"`
			TotalTokens  int `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &r); err != nil {
		t.Fatal(err)
	}
	if r.Object != "list" || r.Model != "siglip2-so400m-patch16-384" || len(r.Data) != 2 {
		t.Fatalf("reply = %+v", r)
	}
	if r.Data[1].Index != 1 || r.Data[1].Object != "embedding" || r.Data[1].Embedding[0] != 1 {
		t.Errorf("data[1] = %+v", r.Data[1])
	}
	if r.Usage.PromptTokens != 3+embed.ImageTokens || r.Usage.TotalTokens != r.Usage.PromptTokens {
		t.Errorf("usage = %+v", r.Usage)
	}
}

func TestEmbeddingsTakesOneString(t *testing.T) {
	rec := postEmbeddings(t, stubEmbedder{}, `{"model":"siglip2-so400m-patch16-384","input":"a cat"}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
}

func TestEmbeddingsBase64Encoding(t *testing.T) {
	rec := postEmbeddings(t, stubEmbedder{}, `{"model":"siglip2-so400m-patch16-384","input":["a","b"],"encoding_format":"base64"}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	var r struct {
		Data []struct {
			Embedding string `json:"embedding"`
		} `json:"data"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &r); err != nil {
		t.Fatal(err)
	}
	raw, err := base64.StdEncoding.DecodeString(r.Data[1].Embedding)
	if err != nil || len(raw) != 12 {
		t.Fatalf("raw = %v, %v", raw, err)
	}
	if v := math.Float32frombits(binary.LittleEndian.Uint32(raw)); v != 1 {
		t.Errorf("first float = %v, want 1", v)
	}
}

func TestEmbeddingsRefuses(t *testing.T) {
	big := "data:image/png;base64," + base64.StdEncoding.EncodeToString(make([]byte, 2<<10))
	cases := []struct {
		name, body string
		status     int
		code       string
	}{
		{"no model", `{"input":["a"]}`, 400, "invalid_request"},
		{"unknown model", `{"model":"x","input":["a"]}`, 404, "model_not_found"},
		{"no input", `{"model":"siglip2-so400m-patch16-384","input":[]}`, 400, "invalid_request"},
		{"input not strings", `{"model":"siglip2-so400m-patch16-384","input":[[1,2]]}`, 400, "invalid_request"},
		{"too many", `{"model":"siglip2-so400m-patch16-384","input":["a","b","c","d"]}`, 400, "invalid_request"},
		{"bad base64", `{"model":"siglip2-so400m-patch16-384","input":["data:image/png;base64,***"]}`, 400, "invalid_request"},
		{"gif", `{"model":"siglip2-so400m-patch16-384","input":["data:image/gif;base64,R0lG"]}`, 400, "invalid_request"},
		{"image too large", `{"model":"siglip2-so400m-patch16-384","input":["` + big + `"]}`, 413, "too_large"},
		{"bad format", `{"model":"siglip2-so400m-patch16-384","input":["a"],"encoding_format":"int8"}`, 400, "invalid_request"},
		{"not json", `{`, 400, "invalid_request"},
	}
	for _, c := range cases {
		rec := postEmbeddings(t, stubEmbedder{}, c.body)
		if rec.Code != c.status || errCode(t, rec) != c.code {
			t.Errorf("%s: %d %s, want %d %s (%s)", c.name, rec.Code, errCode(t, rec), c.status, c.code, rec.Body)
		}
	}
}

func TestEmbeddingsBadImageNamesTheInput(t *testing.T) {
	rec := postEmbeddings(t, stubEmbedder{err: &embed.InputError{Index: 1, Err: fmt.Errorf("%w: not a JPEG, PNG or WebP", embed.ErrBadImage)}},
		`{"model":"siglip2-so400m-patch16-384","input":["a","`+pngURL+`"]}`)
	if rec.Code != 400 || errCode(t, rec) != "invalid_request" || !strings.Contains(rec.Body.String(), "input 1") {
		t.Errorf("%d %s", rec.Code, rec.Body)
	}
}

func TestEmbeddingsModelFailureIs500(t *testing.T) {
	rec := postEmbeddings(t, stubEmbedder{err: fmt.Errorf("vision tower: boom")}, `{"model":"siglip2-so400m-patch16-384","input":["a"]}`)
	if rec.Code != 500 || errCode(t, rec) != "internal_error" || strings.Contains(rec.Body.String(), "boom") {
		t.Errorf("%d %s", rec.Code, rec.Body)
	}
}

func TestEmbeddingsClientGoneWhileWaiting(t *testing.T) {
	block := make(chan struct{})
	defer close(block)
	ctx, cancel := context.WithCancel(context.Background())
	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings",
		strings.NewReader(`{"model":"siglip2-so400m-patch16-384","input":["a"]}`)).WithContext(ctx)
	var logs bytes.Buffer
	h := newHandler(&fake{}, stubEmbedder{block: block}, limits{maxUpload: 1 << 20, maxInputs: 3, maxImageBytes: 1 << 10},
		slog.New(slog.NewJSONHandler(&logs, nil)))
	done := make(chan struct{})
	go func() { h.ServeHTTP(httptest.NewRecorder(), req); close(done) }()
	time.Sleep(20 * time.Millisecond)
	cancel()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("the handler kept waiting after the client left")
	}
	if !strings.Contains(logs.String(), `"status":499`) {
		t.Errorf("log = %s", logs.String())
	}
}

func TestEmbeddingsLogHoldsNoInput(t *testing.T) {
	var logs bytes.Buffer
	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings",
		strings.NewReader(`{"model":"siglip2-so400m-patch16-384","input":["secret words","`+pngURL+`"]}`))
	newHandler(&fake{}, stubEmbedder{}, limits{maxUpload: 1 << 20, maxInputs: 3, maxImageBytes: 1 << 10},
		slog.New(slog.NewJSONHandler(&logs, nil))).ServeHTTP(httptest.NewRecorder(), req)
	if strings.Contains(logs.String(), "secret") || strings.Contains(logs.String(), "iVBOR") {
		t.Errorf("log holds input: %s", logs.String())
	}
	for _, want := range []string{`"images":1`, `"texts":1`, `"status":200`} {
		if !strings.Contains(logs.String(), want) {
			t.Errorf("log lacks %s: %s", want, logs.String())
		}
	}
}
