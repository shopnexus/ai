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
	"runtime"
	"slices"
	"strings"
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

// A huge text is cut before the tokenizer sees it: the ids are the first
// 63 pieces plus <eos>, and encoding does not grow with the input.
func TestTokenizerCutsAHugeText(t *testing.T) {
	tok, err := newTokenizer(filepath.Join(modelDir(t), "tokenizer.model"))
	if err != nil {
		t.Fatal(err)
	}
	huge := strings.Repeat("xe đạp trẻ em ", 4<<20) // about 70 MB
	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)
	ids, n := tok.ids(huge)
	runtime.ReadMemStats(&after)
	if n != maxTextTokens || len(ids) != maxTextTokens {
		t.Fatalf("tokens = %d, ids = %d", n, len(ids))
	}
	if grew := after.TotalAlloc - before.TotalAlloc; grew > 64<<20 {
		t.Errorf("encoding a huge text allocated %d MB", grew>>20)
	}
}
