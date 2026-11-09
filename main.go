// Command ai serves self-hosted models behind OpenAI's API for LiteLLM to
// route to. Today it does speech to text with Gipformer.
package main

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/shopnexus/ai/embed"
	"github.com/shopnexus/ai/speech"
)

func main() {
	log := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	if err := run(log); err != nil {
		log.Error("ai stopped", "err", err)
		os.Exit(1)
	}
}

func run(log *slog.Logger) error {
	cfg, err := loadConfig(os.Getenv)
	if err != nil {
		return err
	}
	t, err := speech.New(speech.Config{
		Models:          cfg.STTModels,
		VADModel:        cfg.VADModel,
		NumThreads:      cfg.NumThreads,
		MaxConcurrent:   cfg.MaxConcurrent,
		MaxAudioSeconds: cfg.MaxAudioSeconds,
	})
	if err != nil {
		return err
	}
	defer t.Close()

	e, err := embed.New(embed.Config{
		Models:        cfg.EmbedModels,
		NumThreads:    cfg.NumThreads,
		MaxConcurrent: cfg.EmbedMaxConcurrent,
	})
	if err != nil {
		return err
	}
	defer e.Close()

	srv := &http.Server{
		Addr: cfg.Addr,
		Handler: newHandler(t, e, limits{
			maxUpload:     int64(cfg.MaxUploadMB) << 20,
			maxInputs:     cfg.EmbedMaxInputs,
			maxImageBytes: cfg.EmbedMaxImageMB << 20,
		}, log),
		ReadHeaderTimeout: 10 * time.Second,
	}
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	done := make(chan error, 1)
	go func() {
		<-ctx.Done()
		// Requests in flight get 30 s, the grace period Kubernetes gives by default.
		sctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		done <- srv.Shutdown(sctx)
	}()

	log.Info("listening", "addr", cfg.Addr, "stt_models", t.Models(), "embed_models", e.Models())
	if err := srv.ListenAndServe(); !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return <-done
}
