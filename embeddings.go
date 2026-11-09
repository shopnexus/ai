package main

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"net/http"
	"strings"
	"time"

	"github.com/shopnexus/ai/embed"
)

type embedder interface {
	HasModel(name string) bool
	Embed(ctx context.Context, model string, in []embed.Input) (embed.Result, error)
}

// imageTypes are the data URL prefixes the service takes.
var imageTypes = []string{"data:image/jpeg;base64,", "data:image/png;base64,", "data:image/webp;base64,"}

type embeddingsRequest struct {
	Model          string          `json:"model"`
	Input          json.RawMessage `json:"input"`
	EncodingFormat string          `json:"encoding_format"`
}

// embeddings serves OpenAI's POST /v1/embeddings. An input that starts with
// data:image/ is an image; anything else is text for the text tower.
func (h *handler) embeddings(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	var images, texts int
	var req embeddingsRequest
	attrs := func() []any { return []any{"model", req.Model, "images", images, "texts", texts} }
	fail := func(e apiError, cause error) {
		writeError(w, e)
		level := slog.LevelInfo
		if e.status >= 500 {
			level = slog.LevelError
		}
		h.log.Log(r.Context(), level, "embedding", append(attrs(),
			"status", e.status, "code", e.code, "err", errString(cause), "ms", time.Since(start).Milliseconds())...)
	}

	// A base64 image is 4/3 of its bytes; 64 KiB covers the JSON around them.
	r.Body = http.MaxBytesReader(w, r.Body, int64(h.lim.maxInputs)*int64(h.lim.maxImageBytes)*4/3+64<<10)
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		var tooBig *http.MaxBytesError
		if errors.As(err, &tooBig) {
			fail(apiError{http.StatusRequestEntityTooLarge, "too_large", "the body is too large"}, err)
			return
		}
		fail(apiError{http.StatusBadRequest, "invalid_request", "the body must be a JSON object"}, err)
		return
	}
	var inputs []string
	var one string
	if json.Unmarshal(req.Input, &one) == nil {
		inputs = []string{one}
	} else if err := json.Unmarshal(req.Input, &inputs); err != nil {
		fail(apiError{http.StatusBadRequest, "invalid_request", "input must be a string or an array of strings"}, err)
		return
	}
	switch {
	case req.Model == "":
		fail(apiError{http.StatusBadRequest, "invalid_request", "model is required"}, nil)
		return
	case len(inputs) == 0:
		fail(apiError{http.StatusBadRequest, "invalid_request", "input is required"}, nil)
		return
	case len(inputs) > h.lim.maxInputs:
		fail(apiError{http.StatusBadRequest, "invalid_request", fmt.Sprintf("send at most %d inputs", h.lim.maxInputs)}, nil)
		return
	case req.EncodingFormat != "" && req.EncodingFormat != "float" && req.EncodingFormat != "base64":
		fail(apiError{http.StatusBadRequest, "invalid_request", "encoding_format must be float or base64"}, nil)
		return
	case !h.e.HasModel(req.Model):
		fail(apiError{http.StatusNotFound, "model_not_found", fmt.Sprintf("the model %q does not exist", req.Model)}, nil)
		return
	}

	in := make([]embed.Input, len(inputs))
	for i, s := range inputs {
		if !strings.HasPrefix(s, "data:image/") {
			in[i] = embed.Input{Text: s}
			texts++
			continue
		}
		images++
		payload, ok := "", false
		for _, p := range imageTypes {
			if rest, found := strings.CutPrefix(s, p); found {
				payload, ok = rest, true
				break
			}
		}
		if !ok {
			fail(apiError{http.StatusBadRequest, "invalid_request", fmt.Sprintf("input %d: images must be JPEG, PNG or WebP data URLs in base64", i)}, nil)
			return
		}
		if base64.StdEncoding.DecodedLen(len(payload)) > h.lim.maxImageBytes+2 {
			fail(apiError{http.StatusRequestEntityTooLarge, "too_large", fmt.Sprintf("input %d: the image is larger than %d MB", i, h.lim.maxImageBytes>>20)}, nil)
			return
		}
		b, err := base64.StdEncoding.DecodeString(payload)
		if err != nil {
			fail(apiError{http.StatusBadRequest, "invalid_request", fmt.Sprintf("input %d: the base64 does not decode", i)}, err)
			return
		}
		if len(b) > h.lim.maxImageBytes {
			fail(apiError{http.StatusRequestEntityTooLarge, "too_large", fmt.Sprintf("input %d: the image is larger than %d MB", i, h.lim.maxImageBytes>>20)}, nil)
			return
		}
		in[i] = embed.Input{Image: b}
	}

	res, err := h.e.Embed(r.Context(), req.Model, in)
	if err != nil {
		var bad *embed.InputError
		switch {
		case r.Context().Err() != nil:
			h.log.Info("embedding", append(attrs(), "status", 499, "code", "client_gone", "ms", time.Since(start).Milliseconds())...)
		case errors.As(err, &bad) && errors.Is(err, embed.ErrBadImage):
			fail(apiError{http.StatusBadRequest, "invalid_request", bad.Error()}, err)
		case errors.Is(err, embed.ErrUnknownModel):
			fail(apiError{http.StatusNotFound, "model_not_found", fmt.Sprintf("the model %q does not exist", req.Model)}, err)
		default:
			fail(apiError{http.StatusInternalServerError, "internal_error", "embedding failed"}, err)
		}
		return
	}

	data := make([]map[string]any, len(res.Vectors))
	for i, v := range res.Vectors {
		var emb any = v
		if req.EncodingFormat == "base64" {
			raw := make([]byte, 4*len(v))
			for k, x := range v {
				binary.LittleEndian.PutUint32(raw[4*k:], math.Float32bits(x))
			}
			emb = base64.StdEncoding.EncodeToString(raw)
		}
		data[i] = map[string]any{"object": "embedding", "index": i, "embedding": emb}
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"object": "list", "model": req.Model, "data": data,
		// LiteLLM prices the call from usage, not from the base64 length.
		"usage": map[string]int{"prompt_tokens": res.Tokens, "total_tokens": res.Tokens},
	})
	// The inputs stay out of the log: a photo or a query is the person's.
	h.log.Info("embedding", append(attrs(), "tokens", res.Tokens, "status", http.StatusOK,
		"ms", time.Since(start).Milliseconds())...)
}
