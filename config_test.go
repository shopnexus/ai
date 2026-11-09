package main

import (
	"maps"
	"strings"
	"testing"
)

func env(over map[string]string) func(string) string {
	base := map[string]string{
		"AI_ADDR":              ":8000",
		"AI_STT_MODELS":        "gipformer-1.5=/models/gipformer-1.5",
		"AI_VAD_MODEL":         "/models/silero_vad.onnx",
		"AI_NUM_THREADS":       "4",
		"AI_MAX_CONCURRENT":    "2",
		"AI_MAX_UPLOAD_MB":     "100",
		"AI_MAX_AUDIO_SECONDS": "300",
	}
	maps.Copy(base, over)
	return func(k string) string { return base[k] }
}

func TestLoadConfig(t *testing.T) {
	c, err := loadConfig(env(nil))
	if err != nil {
		t.Fatal(err)
	}
	if c.Addr != ":8000" || c.VADModel != "/models/silero_vad.onnx" {
		t.Errorf("Addr, VADModel = %q, %q", c.Addr, c.VADModel)
	}
	if c.NumThreads != 4 || c.MaxConcurrent != 2 || c.MaxUploadMB != 100 || c.MaxAudioSeconds != 300 {
		t.Errorf("numbers = %d %d %d %d", c.NumThreads, c.MaxConcurrent, c.MaxUploadMB, c.MaxAudioSeconds)
	}
	if want := map[string]string{"gipformer-1.5": "/models/gipformer-1.5"}; !maps.Equal(c.STTModels, want) {
		t.Errorf("STTModels = %v, want %v", c.STTModels, want)
	}
}

func TestLoadConfigNamesEveryMissingVariable(t *testing.T) {
	_, err := loadConfig(env(map[string]string{"AI_ADDR": "", "AI_MAX_UPLOAD_MB": ""}))
	if err == nil {
		t.Fatal("no error")
	}
	for _, name := range []string{"AI_ADDR", "AI_MAX_UPLOAD_MB"} {
		if !strings.Contains(err.Error(), name) {
			t.Errorf("error %q does not name %s", err, name)
		}
	}
}

func TestLoadConfigRefusesBadValues(t *testing.T) {
	for name, over := range map[string]map[string]string{
		"zero threads":      {"AI_NUM_THREADS": "0"},
		"word for a number": {"AI_MAX_CONCURRENT": "two"},
		"model without dir": {"AI_STT_MODELS": "gipformer-1.5"},
		"empty model name":  {"AI_STT_MODELS": "=/models/x"},
		"duplicate model":   {"AI_STT_MODELS": "a=/x,a=/y"},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := loadConfig(env(over)); err == nil {
				t.Error("no error")
			}
		})
	}
}

func TestLoadConfigReadsSeveralModels(t *testing.T) {
	c, err := loadConfig(env(map[string]string{"AI_STT_MODELS": " a=/x , b=/y "}))
	if err != nil {
		t.Fatal(err)
	}
	if want := map[string]string{"a": "/x", "b": "/y"}; !maps.Equal(c.STTModels, want) {
		t.Errorf("STTModels = %v, want %v", c.STTModels, want)
	}
}
