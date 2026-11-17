package main

import (
	"maps"
	"strings"
	"testing"
)

// env is a lookup over the full configuration with over applied; a value of
// unset in over removes the variable.
func env(over map[string]string) func(string) (string, bool) {
	base := map[string]string{
		"AI_ADDR":                  ":8000",
		"AI_STT_MODELS":            "gipformer-1.5=/models/gipformer-1.5",
		"AI_VAD_MODEL":             "/models/silero_vad.onnx",
		"AI_NUM_THREADS":           "4",
		"AI_MAX_CONCURRENT":        "2",
		"AI_MAX_UPLOAD_MB":         "100",
		"AI_MAX_AUDIO_SECONDS":     "300",
		"AI_EMBED_MODELS":          "siglip2-so400m-patch16-384=/models/siglip2-so400m-patch16-384",
		"AI_EMBED_MAX_INPUTS":      "16",
		"AI_EMBED_MAX_CONCURRENT":  "2",
		"AI_EMBED_MAX_IMAGE_MB":    "10",
		"AI_FRAMES_MAX_CONCURRENT": "2",
		"AI_FRAMES_MAX_SECONDS":    "35",
		"AI_FRAMES_MIN_SHARPNESS":  "30",
	}
	maps.Copy(base, over)
	maps.DeleteFunc(base, func(_, v string) bool { return v == unset })
	return func(k string) (string, bool) { v, ok := base[k]; return v, ok }
}

const unset = "\x00unset"

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

func TestLoadConfigEmbed(t *testing.T) {
	c, err := loadConfig(env(nil))
	if err != nil {
		t.Fatal(err)
	}
	if want := map[string]string{"siglip2-so400m-patch16-384": "/models/siglip2-so400m-patch16-384"}; !maps.Equal(c.EmbedModels, want) {
		t.Errorf("EmbedModels = %v", c.EmbedModels)
	}
	if c.EmbedMaxInputs != 16 || c.EmbedMaxConcurrent != 2 || c.EmbedMaxImageMB != 10 {
		t.Errorf("embed numbers = %d %d %d", c.EmbedMaxInputs, c.EmbedMaxConcurrent, c.EmbedMaxImageMB)
	}
}

func TestLoadConfigNamesMissingEmbedVariables(t *testing.T) {
	_, err := loadConfig(env(map[string]string{"AI_EMBED_MODELS": unset, "AI_EMBED_MAX_IMAGE_MB": ""}))
	if err == nil || !strings.Contains(err.Error(), "AI_EMBED_MODELS") || !strings.Contains(err.Error(), "AI_EMBED_MAX_IMAGE_MB") {
		t.Errorf("err = %v", err)
	}
}

func TestLoadConfigTakesNoEmbedModels(t *testing.T) {
	c, err := loadConfig(env(map[string]string{"AI_EMBED_MODELS": ""}))
	if err != nil {
		t.Fatal(err)
	}
	if len(c.EmbedModels) != 0 {
		t.Errorf("EmbedModels = %v, want none", c.EmbedModels)
	}
}

func TestLoadConfigRefusesBadEmbedModels(t *testing.T) {
	_, err := loadConfig(env(map[string]string{"AI_EMBED_MODELS": "siglip"}))
	if err == nil || !strings.Contains(err.Error(), "AI_EMBED_MODELS") {
		t.Errorf("err = %v", err)
	}
}

func TestLoadConfigFrames(t *testing.T) {
	c, err := loadConfig(env(nil))
	if err != nil {
		t.Fatal(err)
	}
	if c.FramesMaxConcurrent != 2 || c.FramesMaxSeconds != 35 || c.FramesMinSharpness != 30 {
		t.Errorf("frames = %d %d %v", c.FramesMaxConcurrent, c.FramesMaxSeconds, c.FramesMinSharpness)
	}
	c, err = loadConfig(env(map[string]string{"AI_FRAMES_MIN_SHARPNESS": "0"}))
	if err != nil || c.FramesMinSharpness != 0 {
		t.Errorf("a floor of 0: %v, %v", c.FramesMinSharpness, err)
	}
	c, err = loadConfig(env(map[string]string{"AI_FRAMES_MIN_SHARPNESS": "12.5"}))
	if err != nil || c.FramesMinSharpness != 12.5 {
		t.Errorf("a floor of 12.5: %v, %v", c.FramesMinSharpness, err)
	}
}

func TestLoadConfigNamesMissingFramesVariables(t *testing.T) {
	_, err := loadConfig(env(map[string]string{
		"AI_FRAMES_MAX_CONCURRENT": "", "AI_FRAMES_MAX_SECONDS": "", "AI_FRAMES_MIN_SHARPNESS": "",
	}))
	for _, name := range []string{"AI_FRAMES_MAX_CONCURRENT", "AI_FRAMES_MAX_SECONDS", "AI_FRAMES_MIN_SHARPNESS"} {
		if err == nil || !strings.Contains(err.Error(), name) {
			t.Errorf("err = %v, want it to name %s", err, name)
		}
	}
}

func TestLoadConfigRefusesBadFramesValues(t *testing.T) {
	for name, over := range map[string]map[string]string{
		"no slots":           {"AI_FRAMES_MAX_CONCURRENT": "0"},
		"word for seconds":   {"AI_FRAMES_MAX_SECONDS": "many"},
		"negative floor":     {"AI_FRAMES_MIN_SHARPNESS": "-1"},
		"floor not a number": {"AI_FRAMES_MIN_SHARPNESS": "NaN"},
		"infinite floor":     {"AI_FRAMES_MIN_SHARPNESS": "Inf"},
		"word for the floor": {"AI_FRAMES_MIN_SHARPNESS": "sharp"},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := loadConfig(env(over)); err == nil {
				t.Error("no error")
			}
		})
	}
}
