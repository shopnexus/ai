# ai

ShopNexus's provider of self-hosted models. It serves OpenAI's API, and the
stack's LiteLLM routes an alias to it as it would to any provider. In the
cluster the server calls LiteLLM and LiteLLM calls `ai`.

It does speech to text with Gipformer 1.5
(`g-group-ai-lab/gipformer1.5-68M-rnnt`, MIT), a Vietnamese Zipformer
transducer with 68.6M parameters, on CPU through sherpa-onnx.

## API

`POST /v1/audio/transcriptions`, multipart:

| Field | Value |
|---|---|
| `file` | required; any audio or video file ffmpeg reads |
| `model` | required; a name from `AI_STT_MODELS` (`gipformer-1.5`) |
| `language` | empty or `vi` |
| `response_format` | `json` (default), `verbose_json` or `text` |

Other OpenAI fields are ignored. `json` answers
`{"text": "...", "duration": 41.2, "usage": {"type": "duration", "seconds": 42}}`,
where `duration` is the audio length in seconds and `seconds` that length
rounded up. `verbose_json` adds `task` and `language`. LiteLLM asks for
`verbose_json` when its caller names no format, and prices every call from
`duration`. A `text` answer carries no duration, so LiteLLM prices it at 0. The text
is lower case without punctuation. Audio with no speech gives `{"text": ""}`.

Errors use OpenAI's shape, `{"error": {"message", "type", "code"}}`:

| HTTP | `code` | Cause |
|---|---|---|
| 400 | `invalid_request` | a field is missing or has a value the service does not take |
| 404 | `model_not_found` | `model` is not in `AI_STT_MODELS` |
| 400 | `unsupported_media` | ffmpeg finds no audio track it can read |
| 413 | `too_large` | the body is over `AI_MAX_UPLOAD_MB` |
| 413 | `too_long` | the audio is over `AI_MAX_AUDIO_SECONDS` |
| 500 | `internal_error` | ffmpeg or the model failed |

`GET /healthz` and `GET /readyz` answer 200. The server starts listening after
every model has loaded.

The service has no authentication. In the cluster only pods of the same stack
reach it, and it has no Ingress.

## Request handling

The upload goes to a temporary file: an mp4 from a phone often keeps its index
at the end, and ffmpeg has to seek to read it. ffmpeg decodes the first audio
track to 16 kHz mono. Silero VAD cuts the speech at silences into pieces of at
most 20 s, and the recognizer decodes the pieces one at a time. Decoding them
in batches held more memory and was slower: two concurrent 288 s requests held
2.3 GiB and more in batches, and about 1 GiB one piece at a time. One
recognizer per model serves every request. `AI_MAX_CONCURRENT` requests
decode at once and the others wait.

Each request writes one log line with the model, the upload size, the audio
and speech seconds, the piece count, the time taken and the status. The
transcript is never logged.

## Configuration

Every variable is required, and the image sets none of them; the chart and compose set all of them.

| Variable | Meaning | Chart and compose |
|---|---|---|
| `AI_ADDR` | listen address | `:8000` |
| `AI_STT_MODELS` | `name=dir` pairs, comma-separated | `gipformer-1.5=/models/gipformer-1.5` |
| `AI_VAD_MODEL` | path of `silero_vad.onnx` | `/models/silero_vad.onnx` |
| `AI_NUM_THREADS` | onnxruntime threads per decode | `4` |
| `AI_MAX_CONCURRENT` | requests decoding at once | `2` |
| `AI_MAX_UPLOAD_MB` | largest file | `100` |
| `AI_MAX_AUDIO_SECONDS` | longest audio | `300` |

## Image

`ghcr.io/shopnexus/ai` is about 850 MB unpacked: Debian's ffmpeg takes 456 MB
and the model 271 MB. `scripts/fetch-assets.sh` downloads the model at a
pinned revision and checks every file against `scripts/assets.sha256`. An idle
container uses about 620 MiB of memory.

A push to `main` builds `:main` and `:sha-<commit>`. A `vX.Y.Z` tag adds
`X.Y.Z` to the image already built for that commit. The build runs every test,
and a failing test fails it.

## Development

The build needs Go with cgo (gcc) and ffmpeg.

```bash
scripts/fetch-assets.sh models tmp/models
scripts/fetch-assets.sh testwavs tmp/testwavs
go test ./...
AI_TEST_MODELS=$PWD/tmp/models AI_TEST_WAVS=$PWD/tmp/testwavs go test -race -tags model ./...
docker build -t shopnexus-ai:local .
```

The second `go test` loads the model. The test clips come from sherpa-onnx's
Vietnamese model repo, which took them from
`nguyenvulebinh/wav2vec2-base-vietnamese-250h` (CC BY-NC 4.0). The tests
download them; this repo and the runtime image do not contain them.
