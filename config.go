package main

import (
	"errors"
	"fmt"
	"strconv"
	"strings"
)

// Config holds the service's settings. Every variable is required. The chart
// and compose set them all; the image sets none.
type Config struct {
	Addr               string            // AI_ADDR
	STTModels          map[string]string // AI_STT_MODELS: name=dir pairs, comma-separated
	VADModel           string            // AI_VAD_MODEL: path of silero_vad.onnx
	NumThreads         int               // AI_NUM_THREADS: onnxruntime threads per decode
	MaxConcurrent      int               // AI_MAX_CONCURRENT: requests decoding at once
	MaxUploadMB        int               // AI_MAX_UPLOAD_MB
	MaxAudioSeconds    int               // AI_MAX_AUDIO_SECONDS
	EmbedModels        map[string]string // AI_EMBED_MODELS: name=dir pairs, comma-separated
	EmbedMaxInputs     int               // AI_EMBED_MAX_INPUTS: inputs per request
	EmbedMaxConcurrent int               // AI_EMBED_MAX_CONCURRENT: requests embedding at once
	EmbedMaxImageMB    int               // AI_EMBED_MAX_IMAGE_MB: one image after base64 decoding
}

func loadConfig(getenv func(string) string) (Config, error) {
	var missing, bad []string
	str := func(name string) string {
		v := strings.TrimSpace(getenv(name))
		if v == "" {
			missing = append(missing, name)
		}
		return v
	}
	positive := func(name string) int {
		v := str(name)
		if v == "" {
			return 0
		}
		n, err := strconv.Atoi(v)
		if err != nil || n < 1 {
			bad = append(bad, fmt.Sprintf("%s=%q is not a positive integer", name, v))
		}
		return n
	}

	var c Config
	c.Addr = str("AI_ADDR")
	if v := str("AI_STT_MODELS"); v != "" {
		m, err := parseModels(v)
		if err != nil {
			bad = append(bad, "AI_STT_MODELS: "+err.Error())
		}
		c.STTModels = m
	}
	c.VADModel = str("AI_VAD_MODEL")
	c.NumThreads = positive("AI_NUM_THREADS")
	c.MaxConcurrent = positive("AI_MAX_CONCURRENT")
	c.MaxUploadMB = positive("AI_MAX_UPLOAD_MB")
	c.MaxAudioSeconds = positive("AI_MAX_AUDIO_SECONDS")
	if v := str("AI_EMBED_MODELS"); v != "" {
		m, err := parseModels(v)
		if err != nil {
			bad = append(bad, "AI_EMBED_MODELS: "+err.Error())
		}
		c.EmbedModels = m
	}
	c.EmbedMaxInputs = positive("AI_EMBED_MAX_INPUTS")
	c.EmbedMaxConcurrent = positive("AI_EMBED_MAX_CONCURRENT")
	c.EmbedMaxImageMB = positive("AI_EMBED_MAX_IMAGE_MB")

	var errs []error
	if len(missing) > 0 {
		errs = append(errs, fmt.Errorf("missing %s", strings.Join(missing, ", ")))
	}
	for _, b := range bad {
		errs = append(errs, errors.New(b))
	}
	return c, errors.Join(errs...)
}

// parseModels reads "name=dir,name=dir".
func parseModels(v string) (map[string]string, error) {
	m := map[string]string{}
	for pair := range strings.SplitSeq(v, ",") {
		name, dir, ok := strings.Cut(strings.TrimSpace(pair), "=")
		name, dir = strings.TrimSpace(name), strings.TrimSpace(dir)
		if !ok || name == "" || dir == "" {
			return nil, fmt.Errorf("%q is not name=dir", pair)
		}
		if _, dup := m[name]; dup {
			return nil, fmt.Errorf("model %q is listed twice", name)
		}
		m[name] = dir
	}
	return m, nil
}
