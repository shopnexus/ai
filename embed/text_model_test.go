//go:build model || measure

// Model tests read the shipped SigLIP 2 from $AI_TEST_MODELS, an absolute
// path filled by scripts/fetch-assets.sh models (or siglip-int8 / siglip-fp32
// while measuring):
//
//	AI_TEST_MODELS=$PWD/tmp/models go test -race -tags model ./embed/
package embed

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

const modelName = "siglip2-so400m-patch16-384"

type reference struct {
	Images []struct {
		File      string    `json:"file"`
		Embedding []float32 `json:"embedding"`
	} `json:"images"`
	Texts []struct {
		Text      string    `json:"text"`
		InputIDs  []int64   `json:"input_ids"`
		Tokens    int       `json:"tokens"`
		Embedding []float32 `json:"embedding"`
	} `json:"texts"`
}

func loadReference(t *testing.T) reference {
	t.Helper()
	b, err := os.ReadFile("testdata/reference.json")
	if err != nil {
		t.Fatal(err)
	}
	var r reference
	if err := json.Unmarshal(b, &r); err != nil {
		t.Fatal(err)
	}
	return r
}

func modelDir(t *testing.T) string {
	t.Helper()
	v := os.Getenv("AI_TEST_MODELS")
	if v == "" || !filepath.IsAbs(v) {
		t.Fatalf("set AI_TEST_MODELS to an absolute path (see the comment at the top of this file)")
	}
	return filepath.Join(v, modelName)
}

func TestTokenizerMatchesTransformers(t *testing.T) {
	tok, err := newTokenizer(filepath.Join(modelDir(t), "tokenizer.model"))
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range loadReference(t).Texts {
		ids, n := tok.ids(want.Text)
		if !slices.Equal(ids, want.InputIDs) {
			t.Errorf("%q: ids = %v\nwant %v", want.Text, ids, want.InputIDs)
		}
		if n != want.Tokens {
			t.Errorf("%q: tokens = %d, want %d", want.Text, n, want.Tokens)
		}
	}
}
