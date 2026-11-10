package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"image/jpeg"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/shopnexus/ai/frames"
)

// stubFramer answers res or err, records what it was given, and blocks on
// block while it is set.
type stubFramer struct {
	res   frames.Result
	err   error
	got   *[]byte // the upload, as the framer read it
	path  *string
	block chan struct{}
	calls *atomic.Int32
}

func (s stubFramer) Extract(ctx context.Context, path string) (frames.Result, error) {
	if s.calls != nil {
		s.calls.Add(1)
	}
	if s.path != nil {
		*s.path = path
	}
	if s.got != nil {
		*s.got, _ = os.ReadFile(path)
	}
	if s.block != nil {
		select {
		case <-s.block:
		case <-ctx.Done():
			return frames.Result{}, ctx.Err()
		}
	}
	return s.res, s.err
}

var frameLimits = limits{maxUpload: 1 << 20, maxInputs: 16, maxImageBytes: 10 << 20, frameJobs: 2}

func postFrames(t *testing.T, f framer, l limits, parts ...part) *httptest.ResponseRecorder {
	t.Helper()
	b, ct := body(t, parts...)
	req := httptest.NewRequest(http.MethodPost, "/v1/video/frames", b)
	req.Header.Set("Content-Type", ct)
	rec := httptest.NewRecorder()
	newHandler(&fake{}, stubEmbedder{}, f, l, slog.New(slog.DiscardHandler)).ServeHTTP(rec, req)
	return rec
}

type framesBody struct {
	Duration float64 `json:"duration"`
	Frames   []struct {
		AtSeconds float64 `json:"at_seconds"`
		Sharpness float64 `json:"sharpness"`
		JPEG      []byte  `json:"jpeg"`
	} `json:"frames"`
}

func TestVideoFramesAnswersTheFrames(t *testing.T) {
	var got []byte
	s := stubFramer{got: &got, res: frames.Result{Duration: 12.5, Frames: []frames.Frame{
		{AtSeconds: 0.5, Sharpness: 400, JPEG: []byte("first")},
		{AtSeconds: 6, Sharpness: 350.5, JPEG: []byte("second")},
	}}}
	rec := postFrames(t, s, frameLimits, file)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
		t.Errorf("Content-Type = %q", ct)
	}
	if string(got) != "fake mp4 bytes" {
		t.Errorf("the framer read %q", got)
	}
	var b framesBody
	if err := json.Unmarshal(rec.Body.Bytes(), &b); err != nil {
		t.Fatal(err)
	}
	if b.Duration != 12.5 || len(b.Frames) != 2 {
		t.Fatalf("body = %+v", b)
	}
	if string(b.Frames[0].JPEG) != "first" || b.Frames[1].AtSeconds != 6 || b.Frames[1].Sharpness != 350.5 {
		t.Errorf("frames = %+v", b.Frames)
	}
	if !strings.Contains(rec.Body.String(), `"jpeg":"Zmlyc3Q="`) {
		t.Errorf("jpeg is not base64: %s", rec.Body)
	}
}

func TestVideoFramesAnswersAnEmptyListAsAList(t *testing.T) {
	rec := postFrames(t, stubFramer{res: frames.Result{Duration: 4}}, frameLimits, file)
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"frames":[]`) {
		t.Errorf("status %d: %s", rec.Code, rec.Body)
	}
}

func TestVideoFramesErrors(t *testing.T) {
	for _, c := range []struct {
		name   string
		err    error
		parts  []part
		status int
		code   string
	}{
		{"no file", nil, []part{model}, http.StatusBadRequest, "invalid_request"},
		{"no video stream", frames.ErrNoVideo, []part{file}, http.StatusBadRequest, "unsupported_media"},
		{"too short", frames.ErrTooShort, []part{file}, http.StatusBadRequest, "too_short"},
		{"too long", frames.ErrTooLong, []part{file}, http.StatusRequestEntityTooLarge, "too_long"},
		{"ffmpeg crashed", errors.New("boom"), []part{file}, http.StatusInternalServerError, "internal_error"},
	} {
		t.Run(c.name, func(t *testing.T) {
			rec := postFrames(t, stubFramer{err: c.err}, frameLimits, c.parts...)
			var b apiErrorBody
			if err := json.Unmarshal(rec.Body.Bytes(), &b); err != nil {
				t.Fatal(err)
			}
			if rec.Code != c.status || b.Error.Code != c.code {
				t.Errorf("status %d code %q, want %d %q", rec.Code, b.Error.Code, c.status, c.code)
			}
			wantType := "invalid_request_error"
			if c.status >= 500 {
				wantType = "server_error"
			}
			if b.Error.Type != wantType {
				t.Errorf("type %q, want %q", b.Error.Type, wantType)
			}
		})
	}
}

func TestVideoFramesRefusesABodyThatIsNotMultipart(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1/video/frames", strings.NewReader("{}"))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	newHandler(&fake{}, stubEmbedder{}, stubFramer{}, frameLimits, slog.New(slog.DiscardHandler)).ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("status %d", rec.Code)
	}
}

func TestVideoFramesRefusesTooLargeAnUpload(t *testing.T) {
	var calls atomic.Int32
	l := frameLimits
	l.maxUpload = 4
	rec := postFrames(t, stubFramer{calls: &calls}, l, file)
	var b apiErrorBody
	_ = json.Unmarshal(rec.Body.Bytes(), &b)
	if rec.Code != http.StatusRequestEntityTooLarge || b.Error.Code != "too_large" {
		t.Errorf("status %d code %q", rec.Code, b.Error.Code)
	}
	if calls.Load() != 0 {
		t.Error("the framer ran")
	}
}

func TestVideoFramesRemovesTheUpload(t *testing.T) {
	var path string
	rec := postFrames(t, stubFramer{path: &path}, frameLimits, file)
	if rec.Code != http.StatusOK || path == "" {
		t.Fatalf("status %d, path %q", rec.Code, path)
	}
	if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("the temp file is still there: %v", err)
	}
}

func TestVideoFramesWaitsForASlot(t *testing.T) {
	var calls atomic.Int32
	block := make(chan struct{})
	var logs bytes.Buffer
	l := frameLimits
	l.frameJobs = 1
	h := newHandler(&fake{}, stubEmbedder{}, stubFramer{calls: &calls, block: block}, l,
		slog.New(slog.NewJSONHandler(&logs, nil)))
	request := func(ctx context.Context) *httptest.ResponseRecorder {
		b, ct := body(t, file)
		req := httptest.NewRequestWithContext(ctx, http.MethodPost, "/v1/video/frames", b)
		req.Header.Set("Content-Type", ct)
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		return rec
	}
	first := make(chan *httptest.ResponseRecorder)
	go func() { first <- request(context.Background()) }()
	for deadline := time.Now().Add(5 * time.Second); calls.Load() == 0; {
		if time.Now().After(deadline) {
			t.Fatal("the first request never reached the framer")
		}
		time.Sleep(time.Millisecond)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	request(ctx)
	if calls.Load() != 1 {
		t.Errorf("the framer ran %d times while one slot was taken", calls.Load())
	}
	close(block)
	if rec := <-first; rec.Code != http.StatusOK {
		t.Errorf("first request: status %d", rec.Code)
	}
	if !strings.Contains(logs.String(), `"status":499`) {
		t.Errorf("no client_gone line: %s", logs.String())
	}
}

// The real extractor behind the handler, on a video ffmpeg makes.
func TestVideoFramesWithFFmpeg(t *testing.T) {
	video := filepath.Join(t.TempDir(), "a.mp4")
	cmd := exec.Command("ffmpeg", "-nostdin", "-v", "error", "-y",
		"-f", "lavfi", "-i", "testsrc2=s=640x360:r=10:d=4", "-c:v", "mpeg4", video)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("ffmpeg: %v\n%s", err, out)
	}
	data, err := os.ReadFile(video)
	if err != nil {
		t.Fatal(err)
	}
	f := frameExtractor{frames.Config{MinSeconds: frames.MinSeconds, MaxSeconds: 35, MinSharpness: 30}}
	rec := postFrames(t, f, limits{maxUpload: 10 << 20, maxInputs: 16, maxImageBytes: 10 << 20, frameJobs: 1},
		part{name: "file", value: string(data), file: true})
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	var b framesBody
	if err := json.Unmarshal(rec.Body.Bytes(), &b); err != nil {
		t.Fatal(err)
	}
	if b.Duration < 3.9 || b.Duration > 4.1 || len(b.Frames) == 0 {
		t.Fatalf("duration %v, %d frames", b.Duration, len(b.Frames))
	}
	if _, err := jpeg.Decode(bytes.NewReader(b.Frames[0].JPEG)); err != nil {
		t.Errorf("the first frame is not a JPEG: %v", err)
	}
}
