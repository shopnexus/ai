package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"slices"
	"time"

	"github.com/shopnexus/ai/frames"
)

// framer cuts a video file into the frames worth keeping.
type framer interface {
	Extract(ctx context.Context, path string) (frames.Result, error)
}

// frameExtractor is package frames with the service's limits.
type frameExtractor struct{ c frames.Config }

func (f frameExtractor) Extract(ctx context.Context, path string) (frames.Result, error) {
	return frames.Extract(ctx, path, f.c)
}

type frameBody struct {
	AtSeconds float64 `json:"at_seconds"`
	Sharpness float64 `json:"sharpness"`
	JPEG      []byte  `json:"jpeg"` // encoding/json writes []byte as base64
}

// videoFrames serves POST /v1/video/frames. The server calls it directly:
// it is not an OpenAI route, and LiteLLM does not route to it.
func (h *handler) videoFrames(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	r.Body = http.MaxBytesReader(w, r.Body, h.lim.maxUpload)
	f, err := readForm(r)
	defer func() {
		if f.file != "" {
			os.Remove(f.file)
		}
	}()
	attrs := []any{"bytes", f.bytes}
	logLine := func(level slog.Level, extra ...any) {
		h.log.Log(r.Context(), level, "video_frames",
			slices.Concat(attrs, extra, []any{"ms", time.Since(start).Milliseconds()})...)
	}
	fail := func(e apiError, cause error) {
		writeError(w, e)
		level := slog.LevelInfo
		if e.status >= 500 {
			level = slog.LevelError
		}
		logLine(level, "status", e.status, "code", e.code, "err", errString(cause))
	}
	gone := func() { logLine(slog.LevelInfo, "status", 499, "code", "client_gone") }

	if err != nil {
		if r.Context().Err() != nil {
			gone()
			return
		}
		if errors.Is(err, errTempFile) {
			fail(apiError{http.StatusInternalServerError, "internal_error", "the server cannot store the upload"}, err)
			return
		}
		var tooBig *http.MaxBytesError
		if errors.As(err, &tooBig) {
			fail(apiError{http.StatusRequestEntityTooLarge, "too_large", fmt.Sprintf("the file is larger than %d MB", h.lim.maxUpload>>20)}, err)
			return
		}
		fail(apiError{http.StatusBadRequest, "invalid_request", err.Error()}, err)
		return
	}
	if f.file == "" {
		fail(apiError{http.StatusBadRequest, "invalid_request", "file is required"}, nil)
		return
	}

	select {
	case h.frameSlots <- struct{}{}:
	case <-r.Context().Done():
		gone()
		return
	}
	res, err := h.f.Extract(r.Context(), f.file)
	<-h.frameSlots
	attrs = append(attrs, "duration_s", res.Duration)
	if err != nil {
		switch {
		case r.Context().Err() != nil:
			gone()
		case errors.Is(err, frames.ErrNoVideo):
			fail(apiError{http.StatusBadRequest, "unsupported_media", "the file has no video stream ffmpeg can read"}, err)
		case errors.Is(err, frames.ErrTooShort):
			fail(apiError{http.StatusBadRequest, "too_short", "the video is shorter than the limit"}, err)
		case errors.Is(err, frames.ErrTooLong):
			fail(apiError{http.StatusRequestEntityTooLarge, "too_long", "the video is longer than the limit"}, err)
		default:
			fail(apiError{http.StatusInternalServerError, "internal_error", "frame extraction failed"}, err)
		}
		return
	}

	out := make([]frameBody, len(res.Frames))
	for i, fr := range res.Frames {
		out[i] = frameBody{AtSeconds: fr.AtSeconds, Sharpness: fr.Sharpness, JPEG: fr.JPEG}
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{"duration": res.Duration, "frames": out})
	logLine(slog.LevelInfo, "status", http.StatusOK, "decoded", res.Decoded, "blurred", res.Blurred,
		"duplicates", res.Duplicates, "kept", len(res.Frames))
}
