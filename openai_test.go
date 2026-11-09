package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/shopnexus/ai/speech"
)

type fake struct {
	result speech.Result
	err    error
	called bool
	model  string
	upload []byte // what the file at path held during the call
	path   string
	onCall func()
}

func (f *fake) HasModel(name string) bool { return name == "gipformer-1.5" }

func (f *fake) Transcribe(ctx context.Context, model, path string) (speech.Result, error) {
	f.called, f.model, f.path = true, model, path
	f.upload, _ = os.ReadFile(path)
	if f.onCall != nil {
		f.onCall()
	}
	if err := ctx.Err(); err != nil {
		return speech.Result{}, err
	}
	return f.result, f.err
}

type part struct {
	name, value string
	file        bool
}

func body(t *testing.T, parts ...part) (*bytes.Buffer, string) {
	t.Helper()
	var b bytes.Buffer
	w := multipart.NewWriter(&b)
	for _, p := range parts {
		var err error
		if p.file {
			var fw io.Writer
			fw, err = w.CreateFormFile(p.name, "clip.mp4")
			if err == nil {
				_, err = fw.Write([]byte(p.value))
			}
		} else {
			err = w.WriteField(p.name, p.value)
		}
		if err != nil {
			t.Fatal(err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	return &b, w.FormDataContentType()
}

func call(t *testing.T, f *fake, limit int64, parts ...part) *httptest.ResponseRecorder {
	t.Helper()
	b, ct := body(t, parts...)
	req := httptest.NewRequest(http.MethodPost, "/v1/audio/transcriptions", b)
	req.Header.Set("Content-Type", ct)
	rec := httptest.NewRecorder()
	newHandler(f, limit, slog.New(slog.DiscardHandler)).ServeHTTP(rec, req)
	return rec
}

var (
	file  = part{name: "file", value: "fake mp4 bytes", file: true}
	model = part{name: "model", value: "gipformer-1.5"}
)

type apiErrorBody struct {
	Error struct{ Message, Type, Code string } `json:"error"`
}

func wantError(t *testing.T, rec *httptest.ResponseRecorder, status int, code string) {
	t.Helper()
	if rec.Code != status {
		t.Errorf("status = %d, want %d; body %s", rec.Code, status, rec.Body)
	}
	var e apiErrorBody
	if err := json.Unmarshal(rec.Body.Bytes(), &e); err != nil {
		t.Fatalf("body %q is not an OpenAI error: %v", rec.Body, err)
	}
	if e.Error.Code != code || e.Error.Message == "" {
		t.Errorf("error = %+v, want code %s", e.Error, code)
	}
	wantType := "invalid_request_error"
	if status >= 500 {
		wantType = "server_error"
	}
	if e.Error.Type != wantType {
		t.Errorf("type = %q, want %q", e.Error.Type, wantType)
	}
}

func TestJSONResponse(t *testing.T) {
	f := &fake{result: speech.Result{Text: "âm lượng tivi giảm", AudioSeconds: 41.2}}
	rec := call(t, f, 1<<20, model, file)
	if rec.Code != 200 {
		t.Fatalf("status = %d; body %s", rec.Code, rec.Body)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
		t.Errorf("Content-Type = %q", ct)
	}
	var got struct {
		Text  string
		Usage struct {
			Type    string
			Seconds int
		}
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Text != "âm lượng tivi giảm" || got.Usage.Type != "duration" || got.Usage.Seconds != 42 {
		t.Errorf("body = %+v", got)
	}
	if f.model != "gipformer-1.5" || string(f.upload) != "fake mp4 bytes" {
		t.Errorf("transcriber got model %q and file %q", f.model, f.upload)
	}
	if _, err := os.Stat(f.path); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("temp file %s is still there", f.path)
	}
}

func TestTextResponse(t *testing.T) {
	f := &fake{result: speech.Result{Text: "âm lượng tivi giảm", AudioSeconds: 2}}
	rec := call(t, f, 1<<20, file, model, part{name: "response_format", value: "text"})
	if rec.Code != 200 || rec.Body.String() != "âm lượng tivi giảm" {
		t.Errorf("status %d, body %q", rec.Code, rec.Body)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "text/plain; charset=utf-8" {
		t.Errorf("Content-Type = %q", ct)
	}
}

func TestNoSpeechIsAnEmptyText(t *testing.T) {
	rec := call(t, &fake{result: speech.Result{AudioSeconds: 3}}, 1<<20, model, file)
	if rec.Code != 200 || !strings.Contains(rec.Body.String(), `"text":""`) {
		t.Errorf("status %d, body %s", rec.Code, rec.Body)
	}
}

// LiteLLM does not document its field order; the file may come first.
func TestFileBeforeModel(t *testing.T) {
	f := &fake{result: speech.Result{Text: "x", AudioSeconds: 1}}
	if rec := call(t, f, 1<<20, file, model); rec.Code != 200 {
		t.Errorf("status = %d; body %s", rec.Code, rec.Body)
	}
}

func TestOtherOpenAIFieldsAreIgnored(t *testing.T) {
	f := &fake{result: speech.Result{Text: "x", AudioSeconds: 1}}
	rec := call(t, f, 1<<20, model, file,
		part{name: "prompt", value: "giá"}, part{name: "temperature", value: "0"}, part{name: "language", value: "vi"})
	if rec.Code != 200 {
		t.Errorf("status = %d; body %s", rec.Code, rec.Body)
	}
}

func TestRequestErrors(t *testing.T) {
	for name, tc := range map[string]struct {
		parts  []part
		status int
		code   string
	}{
		"no file":       {[]part{model}, 400, "invalid_request"},
		"no model":      {[]part{file}, 400, "invalid_request"},
		"english":       {[]part{model, file, {name: "language", value: "en"}}, 400, "invalid_request"},
		"srt":           {[]part{model, file, {name: "response_format", value: "srt"}}, 400, "invalid_request"},
		"two files":     {[]part{model, file, file}, 400, "invalid_request"},
		"unknown model": {[]part{{name: "model", value: "whisper-1"}, file}, 404, "model_not_found"},
	} {
		t.Run(name, func(t *testing.T) {
			f := &fake{}
			wantError(t, call(t, f, 1<<20, tc.parts...), tc.status, tc.code)
			if f.called {
				t.Error("the transcriber was called")
			}
		})
	}
}

func TestTranscriberErrors(t *testing.T) {
	for name, tc := range map[string]struct {
		err    error
		status int
		code   string
	}{
		"unreadable": {speech.ErrUnsupportedMedia, 400, "unsupported_media"},
		"too long":   {speech.ErrTooLong, 413, "too_long"},
		"broken":     {errors.New("onnxruntime exploded"), 500, "internal_error"},
	} {
		t.Run(name, func(t *testing.T) {
			rec := call(t, &fake{err: tc.err}, 1<<20, model, file)
			wantError(t, rec, tc.status, tc.code)
			if strings.Contains(rec.Body.String(), "onnxruntime") {
				t.Errorf("internal detail leaked: %s", rec.Body)
			}
		})
	}
}

func TestBodyOverTheLimit(t *testing.T) {
	big := part{name: "file", value: strings.Repeat("x", 4096), file: true}
	f := &fake{}
	wantError(t, call(t, f, 1024, model, big), 413, "too_large")
	if f.called {
		t.Error("the transcriber was called")
	}
}

func TestNotMultipart(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1/audio/transcriptions", strings.NewReader(`{"model":"gipformer-1.5"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	newHandler(&fake{}, 1<<20, slog.New(slog.DiscardHandler)).ServeHTTP(rec, req)
	wantError(t, rec, 400, "invalid_request")
}

func TestClientGoneLeavesNoTempFile(t *testing.T) {
	b, ct := body(t, model, file)
	ctx, cancel := context.WithCancel(context.Background())
	req := httptest.NewRequestWithContext(ctx, http.MethodPost, "/v1/audio/transcriptions", b)
	req.Header.Set("Content-Type", ct)
	f := &fake{onCall: cancel}
	var logs bytes.Buffer
	rec := httptest.NewRecorder()
	newHandler(f, 1<<20, slog.New(slog.NewJSONHandler(&logs, nil))).ServeHTTP(rec, req)
	if !f.called {
		t.Fatal("the transcriber was not called")
	}
	if _, err := os.Stat(f.path); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("temp file %s is still there", f.path)
	}
	if strings.Contains(logs.String(), `"level":"ERROR"`) {
		t.Errorf("a client that left was logged as an error: %s", logs.String())
	}
}

func TestProbes(t *testing.T) {
	h := newHandler(&fake{}, 1<<20, slog.New(slog.DiscardHandler))
	for _, p := range []string{"/healthz", "/readyz"} {
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, p, nil))
		if rec.Code != 200 {
			t.Errorf("%s = %d", p, rec.Code)
		}
	}
}

func TestTranscriptIsNeverLogged(t *testing.T) {
	var logs bytes.Buffer
	b, ct := body(t, model, file)
	req := httptest.NewRequest(http.MethodPost, "/v1/audio/transcriptions", b)
	req.Header.Set("Content-Type", ct)
	f := &fake{result: speech.Result{Text: "số điện thoại của tôi", AudioSeconds: 2, SpeechSeconds: 1.5, Pieces: 1}}
	newHandler(f, 1<<20, slog.New(slog.NewJSONHandler(&logs, nil))).ServeHTTP(httptest.NewRecorder(), req)
	if strings.Contains(logs.String(), "điện thoại") {
		t.Errorf("the transcript is in the log: %s", logs.String())
	}
	for _, k := range []string{`"model":"gipformer-1.5"`, `"audio_s":2`, `"pieces":1`, `"status":200`} {
		if !strings.Contains(logs.String(), k) {
			t.Errorf("log lacks %s: %s", k, logs.String())
		}
	}
}
