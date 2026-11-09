//go:build model

// Speech and embeddings in one process: sherpa-onnx and onnxruntime_go use
// the same libonnxruntime.so. Run from the repo root with
//
//	AI_TEST_MODELS=$PWD/tmp/models AI_TEST_WAVS=$PWD/tmp/testwavs go test -race -tags model .
package main

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/shopnexus/ai/speech"
)

func TestSpeechAndEmbeddingsShareOneProcess(t *testing.T) {
	models, wavs := os.Getenv("AI_TEST_MODELS"), os.Getenv("AI_TEST_WAVS")
	if !filepath.IsAbs(models) || !filepath.IsAbs(wavs) {
		t.Fatal("set AI_TEST_MODELS and AI_TEST_WAVS to absolute paths")
	}
	tr, err := speech.New(speech.Config{
		Models:   map[string]string{"gipformer-1.5": filepath.Join(models, "gipformer-1.5")},
		VADModel: filepath.Join(models, "silero_vad.onnx"), NumThreads: 2, MaxConcurrent: 1, MaxAudioSeconds: 300,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer tr.Close()
	if _, err := tr.Transcribe(context.Background(), "gipformer-1.5", filepath.Join(wavs, "0.wav")); err != nil {
		t.Fatal(err)
	}
	// Task 5 adds the embedder here, after the transcriber has loaded.
}
