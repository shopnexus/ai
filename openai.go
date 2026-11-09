package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"mime/multipart"
	"net/http"
	"os"
	"time"

	"github.com/shopnexus/ai/speech"
)

type transcriber interface {
	HasModel(name string) bool
	Transcribe(ctx context.Context, model, path string) (speech.Result, error)
}

type handler struct {
	t         transcriber
	maxUpload int64
	log       *slog.Logger
}

// newHandler serves OpenAI's POST /v1/audio/transcriptions and the probes.
func newHandler(t transcriber, maxUploadBytes int64, log *slog.Logger) http.Handler {
	h := &handler{t: t, maxUpload: maxUploadBytes, log: log}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/audio/transcriptions", h.transcriptions)
	ok := func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) }
	mux.HandleFunc("GET /healthz", ok) // the process runs
	mux.HandleFunc("GET /readyz", ok)  // the server listens only after every model has loaded
	return mux
}

type form struct {
	file           string // temp file holding the upload; "" when absent
	bytes          int64
	model          string
	language       string
	responseFormat string
}

// apiError is a reply in OpenAI's error shape.
type apiError struct {
	status  int
	code    string
	message string
}

func (h *handler) transcriptions(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	r.Body = http.MaxBytesReader(w, r.Body, h.maxUpload)
	f, err := readForm(r)
	defer func() {
		if f.file != "" {
			os.Remove(f.file)
		}
	}()
	attrs := []any{"model", f.model, "bytes", f.bytes}

	fail := func(e apiError, cause error) {
		writeError(w, e)
		level := slog.LevelInfo
		if e.status >= 500 {
			level = slog.LevelError
		}
		h.log.Log(r.Context(), level, "transcription", append(attrs,
			"status", e.status, "code", e.code, "err", errString(cause), "ms", time.Since(start).Milliseconds())...)
	}

	if err != nil {
		if r.Context().Err() != nil {
			h.clientGone(attrs, start)
			return
		}
		if errors.Is(err, errTempFile) {
			fail(apiError{http.StatusInternalServerError, "internal_error", "the server cannot store the upload"}, err)
			return
		}
		var tooBig *http.MaxBytesError
		if errors.As(err, &tooBig) {
			fail(apiError{http.StatusRequestEntityTooLarge, "too_large", fmt.Sprintf("the file is larger than %d MB", h.maxUpload>>20)}, err)
			return
		}
		fail(apiError{http.StatusBadRequest, "invalid_request", err.Error()}, err)
		return
	}
	switch {
	case f.file == "":
		fail(apiError{http.StatusBadRequest, "invalid_request", "file is required"}, nil)
		return
	case f.model == "":
		fail(apiError{http.StatusBadRequest, "invalid_request", "model is required"}, nil)
		return
	case f.language != "" && f.language != "vi":
		fail(apiError{http.StatusBadRequest, "invalid_request", "language must be empty or vi"}, nil)
		return
	case f.responseFormat != "" && f.responseFormat != "json" && f.responseFormat != "text" && f.responseFormat != "verbose_json":
		fail(apiError{http.StatusBadRequest, "invalid_request", "response_format must be json, verbose_json or text"}, nil)
		return
	case !h.t.HasModel(f.model):
		fail(apiError{http.StatusNotFound, "model_not_found", fmt.Sprintf("the model %q does not exist", f.model)}, nil)
		return
	}

	res, err := h.t.Transcribe(r.Context(), f.model, f.file)
	if err != nil {
		switch {
		case r.Context().Err() != nil:
			h.clientGone(attrs, start)
		case errors.Is(err, speech.ErrUnsupportedMedia):
			fail(apiError{http.StatusBadRequest, "unsupported_media", "the file has no audio track ffmpeg can read"}, err)
		case errors.Is(err, speech.ErrTooLong):
			fail(apiError{http.StatusRequestEntityTooLarge, "too_long", "the audio is longer than the limit"}, err)
		default:
			fail(apiError{http.StatusInternalServerError, "internal_error", "transcription failed"}, err)
		}
		return
	}

	usage := map[string]any{"type": "duration", "seconds": int(math.Ceil(res.AudioSeconds))}
	switch f.responseFormat {
	case "text":
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		io.WriteString(w, res.Text)
	case "verbose_json":
		// LiteLLM asks for this format when its caller names none, and prices
		// the call from duration.
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"task": "transcribe", "language": "vietnamese", "duration": res.AudioSeconds,
			"text": res.Text, "usage": usage,
		})
	default:
		// duration lets LiteLLM price a call that asked for json.
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"text": res.Text, "duration": res.AudioSeconds, "usage": usage})
	}
	// The text stays out of the log: it is what a person said.
	h.log.Info("transcription", append(attrs,
		"status", http.StatusOK, "audio_s", res.AudioSeconds, "speech_s", res.SpeechSeconds,
		"pieces", res.Pieces, "ms", time.Since(start).Milliseconds())...)
}

func (h *handler) clientGone(attrs []any, start time.Time) {
	h.log.Info("transcription", append(attrs, "status", 499, "code", "client_gone", "ms", time.Since(start).Milliseconds())...)
}

// readForm streams the multipart body. The file part goes to a temp file, the
// known text fields are read, and other parts are skipped. The form names the
// temp file even on error, for the caller to remove.
func readForm(r *http.Request) (form, error) {
	var f form
	mr, err := r.MultipartReader()
	if err != nil {
		return f, errors.New("the body must be multipart/form-data")
	}
	for {
		p, err := mr.NextPart()
		if errors.Is(err, io.EOF) {
			return f, nil
		}
		if err != nil {
			return f, err
		}
		switch p.FormName() {
		case "file":
			if f.file != "" {
				return f, errors.New("send one file")
			}
			f.file, f.bytes, err = spool(p)
		case "model":
			f.model, err = field(p)
		case "language":
			f.language, err = field(p)
		case "response_format":
			f.responseFormat, err = field(p)
		default:
			_, err = io.Copy(io.Discard, p)
		}
		if err != nil {
			return f, err
		}
	}
}

// errTempFile marks a failure to create or write the temp file, such as a
// full disk: the server's fault, unlike an error reading the request.
var errTempFile = errors.New("temp file")

// spool copies the part to a temp file. On a copy error it still returns the
// file's name.
func spool(p *multipart.Part) (string, int64, error) {
	tmp, err := os.CreateTemp("", "ai-upload-*")
	if err != nil {
		return "", 0, fmt.Errorf("%w: %w", errTempFile, err)
	}
	n, err := io.Copy(tempWriter{tmp}, p)
	if cerr := tmp.Close(); err == nil && cerr != nil {
		err = fmt.Errorf("%w: %w", errTempFile, cerr)
	}
	return tmp.Name(), n, err
}

// tempWriter marks its write errors with errTempFile, so a full disk is not
// taken for a bad request.
type tempWriter struct{ f *os.File }

func (w tempWriter) Write(b []byte) (int, error) {
	n, err := w.f.Write(b)
	if err != nil {
		err = fmt.Errorf("%w: %w", errTempFile, err)
	}
	return n, err
}

func field(p *multipart.Part) (string, error) {
	b, err := io.ReadAll(io.LimitReader(p, 256))
	return string(b), err
}

func writeError(w http.ResponseWriter, e apiError) {
	typ := "invalid_request_error"
	if e.status >= 500 {
		typ = "server_error"
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(e.status)
	json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]string{"message": e.message, "type": typ, "code": e.code},
	})
}

func errString(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}
