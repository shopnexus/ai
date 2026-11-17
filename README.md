# ai

ShopNexus's AI hub. Clients (the server, the console agent) call one gateway,
`ai-gateway`, with a key per purpose and an alias such as `search` or
`embed-text`; this repo decides which model answers each alias and what each
key may spend. The gateway is LiteLLM, configured by the chart in `chart/`.
Remote models go through OpenRouter. The models ShopNexus hosts itself run in
`ai-models`, the Go service in this repo, described below.

## Chart

`chart/` deploys both into one stack's namespace: `ai-gateway` (LiteLLM, its
config rendered from `gateway.aliases`, its database in the stack's Postgres),
the Job `ai-gateway-keys` that makes every key of `gateway.keys` exist with
its daily budget, and `ai-models`. The repo's `values.yaml` holds the hub's
choices: the aliases and their models, the images, the settings of
`ai-models`. A stack's file in `infra/ai/values-<stack>.yaml` sets the stack's
name, every key's `dailyUnits` and limit, `models.embedModels` and the memory.
A budget counts tokens for aliases priced by token, seconds of audio for
`stt` (1,000,000 units is 1 of LiteLLM budget), and dollars for
`console-agent`, priced at its list price; the agent's key has 20 USD a day.

| Alias | Upstream | Model |
|---|---|---|
| `search`, `listing-read`, `scam-check` | OpenRouter | `anthropic/claude-haiku-5.5` |
| `console-agent` | OpenRouter | `anthropic/claude-sonnet-5.5` |
| `embed-text` | OpenRouter, DeepInfra only | `baai/bge-m3` |
| `embed-image` | `ai-models` | `siglip2-so400m-patch16-384` |
| `stt` | `ai-models` | `gipformer-1.5` |

`embed-text` is pinned to DeepInfra: on 2026-11-17 its bge-m3 gave the
vectors of the TEI the stacks ran before at a cosine of 0.999999, and
Parasail's gave 0.996. A key's secret is the client's: the Job reads each
key's value from the Secret its `secret` names and never prints it. After a
reseal of `litellm-secrets`, restart `ai-gateway`.

`helm unittest chart` runs the chart's tests; CI runs them on every push.

## ai-models

The rest of this README is about `ai-models`. It serves OpenAI's API, and the
stack's LiteLLM routes an alias to it as it would to any provider. In the
cluster the server calls LiteLLM and LiteLLM calls `ai`.

It does speech to text with Gipformer 1.5
(`g-group-ai-lab/gipformer1.5-68M-rnnt`, MIT), a Vietnamese Zipformer
transducer with 68.6M parameters, on CPU through sherpa-onnx. It embeds photos
and short texts with SigLIP 2 (`google/siglip2-so400m-patch16-384`, Apache
2.0), the fp32 ONNX files of `onnx-community/siglip2-so400m-patch16-384-ONNX`
at revision `29f74e673c298fd07482d8e20138179282001eb7`, on CPU through
onnxruntime.

It cuts a seller's video into its sharpest distinct frames with ffmpeg, for
the server's video read. The server calls that route directly, since it is
not an OpenAI route.

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

`POST /v1/embeddings`, JSON:

| Field | Value |
|---|---|
| `model` | required; a name from `AI_EMBED_MODELS` (`siglip2-so400m-patch16-384`) |
| `input` | required; one string, or a list of at most `AI_EMBED_MAX_INPUTS` strings |
| `encoding_format` | empty or `float` for arrays of numbers, `base64` for little-endian float32 bytes |

Other OpenAI fields are ignored. An input that starts with `data:image/jpeg;base64,`,
`data:image/png;base64,` or `data:image/webp;base64,` is a photo; any other
input is text for SigLIP's text tower, which is lowercased and cut at 64
tokens. One request may mix photos and texts. The answer is OpenAI's:
`{"object": "list", "model": "...", "data": [{"object": "embedding", "index": 0,
"embedding": [...]}], "usage": {"prompt_tokens": 580, "total_tokens": 580}}`.
Each vector has 1152 numbers and length 1. A photo counts 576 tokens, its
24 × 24 patches, and a text its token count, so LiteLLM prices a call from
`usage`. A photo is embedded as stored: EXIF orientation is not applied, and
the server sends upright photos.

| HTTP | `code` | Cause |
|---|---|---|
| 400 | `invalid_request` | a field is missing or wrong, more than `AI_EMBED_MAX_INPUTS` inputs, a data URL of another type or with bad base64, or a photo that does not decode, is under 32 px a side or over 40 megapixels; the message names the input |
| 404 | `model_not_found` | `model` is not in `AI_EMBED_MODELS` |
| 413 | `too_large` | a photo is over `AI_EMBED_MAX_IMAGE_MB` after base64 decoding |
| 500 | `internal_error` | the model failed |

`POST /v1/video/frames`, multipart, with one field `file`: a video ffmpeg
reads. The answer is the video's length in seconds and at most 8 JPEG frames
in time order:

```json
{"duration": 27.4, "frames": [{"at_seconds": 1.5, "sharpness": 412.3, "jpeg": "<base64>"}]}
```

| HTTP | `code` | Cause |
|---|---|---|
| 400 | `invalid_request` | the body is not multipart or has no `file` |
| 400 | `unsupported_media` | ffprobe finds no video stream or no duration |
| 400 | `too_short` | the video is under 3 seconds |
| 413 | `too_large` | the body is over `AI_MAX_UPLOAD_MB` |
| 413 | `too_long` | the video is over `AI_FRAMES_MAX_SECONDS` |
| 500 | `internal_error` | ffmpeg or the JPEG encoder failed |

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

An embeddings request runs its photos through the vision tower in one batch
and its texts through the text tower in another. A photo is turned to RGB,
resized to 384 × 384 and normalized the way transformers' SigLIP processor
does it. `AI_EMBED_MAX_CONCURRENT` requests embed at once and the others wait;
a caller that leaves while it waits frees its place. Each request writes one
log line with the model, the photo and text counts, the tokens, the time taken
and the status, and never an input.

A frames request spools the upload to a temporary file, as a transcription
does. ffprobe reads the duration, then ffmpeg decodes 2 frames a second of
the first video stream, turned upright by its display matrix and scaled down
to at most 1600 px on the long side; a smaller video keeps its size. Each frame is scored on a grey
copy at most 512 px on its long side: the variance of its Laplacian for
sharpness, and a 64-bit difference hash. A frame under
`AI_FRAMES_MIN_SHARPNESS` counts as blurred. The video is cut into 8 equal
spans, and each span keeps its sharpest frame that is not blurred, so at
most 8 full frames stay in memory. A kept frame whose hash is within 6 bits
of a sharper kept frame's is a duplicate and is dropped. When every frame is
blurred, the sharpest one is kept. The frames are encoded as JPEG at quality
90. `AI_FRAMES_MAX_CONCURRENT` videos are cut at once and the others wait.
Each request writes one log line with the upload size, the duration, the
frames decoded, dropped as blurred and dropped as duplicates, the frames
kept, the time taken and the status.

## Configuration

Every variable is required, and the image sets none of them; the chart and compose set all of them.

| Variable | Meaning | Chart and compose |
|---|---|---|
| `AI_ADDR` | listen address | `:8000` |
| `AI_STT_MODELS` | `name=dir` pairs, comma-separated | `gipformer-1.5=/models/gipformer-1.5` |
| `AI_VAD_MODEL` | path of `silero_vad.onnx` | `/models/silero_vad.onnx` |
| `AI_NUM_THREADS` | onnxruntime threads per decode, and per embed session | `4` |
| `AI_MAX_CONCURRENT` | requests decoding at once | `2` |
| `AI_MAX_UPLOAD_MB` | largest file | `100` |
| `AI_MAX_AUDIO_SECONDS` | longest audio | `300` |
| `AI_EMBED_MODELS` | `name=dir` pairs, comma-separated; the empty string loads no embedding model, and `/v1/embeddings` then answers 404 | `siglip2-so400m-patch16-384=/models/siglip2-so400m-patch16-384`, or empty where memory is short |
| `AI_EMBED_MAX_INPUTS` | inputs per embeddings request | `16` |
| `AI_EMBED_MAX_CONCURRENT` | embeddings requests running at once | `2` |
| `AI_EMBED_MAX_IMAGE_MB` | largest photo after base64 decoding | `10` |
| `AI_FRAMES_MAX_CONCURRENT` | videos cut into frames at once | `2` |
| `AI_FRAMES_MAX_SECONDS` | longest video | `35` |
| `AI_FRAMES_MIN_SHARPNESS` | Laplacian variance under which a frame counts as blurred | `30` |

## Image

`ghcr.io/shopnexus/ai` is about 5.4 GB unpacked: the models take 4.8 GB, 4.5 GB
of it SigLIP 2, and Debian's ffmpeg 456 MB. `scripts/fetch-assets.sh` downloads
the models at pinned revisions and checks every file against
`scripts/assets.sha256`. On 2026-11-10 the container was ready 6 s after start
and held 4.0 GiB idle; one request of 16 photos took 34 s on 4 threads and
raised it to 5.3 GiB.

The image carries SigLIP 2 in fp32. Against transformers on the six images
and eight texts of `embed/testdata`, the fp32 files gave a lowest cosine of
0.99995 and kept every ranking, at 1.33 s an input on 4 threads; the int8
files gave 0.555 and changed rankings, at 0.56 s. The texts' token ids were
identical in both, and the preprocessed pixels differ from transformers' by at
most 1/255, so the loss is in the int8 vision tower.

onnxruntime comes from sherpa-onnx's Go package, which links
`libonnxruntime.so` 1.28.2, and `embed` opens the same library.
`github.com/yalue/onnxruntime_go` stays at v1.32.1, the release whose C API
version (28) matches it; a newer release asks for an API the library does not
have. A sherpa-onnx release that ships another onnxruntime needs the
onnxruntime_go release whose `ORT_API_VERSION` equals that onnxruntime's minor
version.

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

The frames tests make their videos with ffmpeg and need `ffprobe` on the
PATH too. To see how real videos score, put them in a directory and run:

```bash
AI_TEST_VIDEOS=$PWD/tmp/videos AI_TEST_FRAMES_OUT=$PWD/tmp/frames go test -tags measure -run TestMeasureSharpness -v ./frames/
```

It logs every decoded frame's time and sharpness and writes the frame to
`tmp/frames/<video>/`, named by both.

`uv run scripts/siglip-reference.py embed/testdata` draws the embed test
images and writes `embed/testdata/reference.json` with transformers; run it
when the model or its revision changes. To compare ONNX variants with it:

```bash
scripts/fetch-assets.sh siglip-fp32 tmp/measure/fp32
scripts/fetch-assets.sh siglip-int8 tmp/measure/int8
AI_TEST_SIGLIP_DIRS=$PWD/tmp/measure/fp32,$PWD/tmp/measure/int8 go test -tags measure -run TestMeasure -v ./embed/
```

The second `go test` loads the models. The test clips come from sherpa-onnx's
Vietnamese model repo, which took them from
`nguyenvulebinh/wav2vec2-base-vietnamese-250h` (CC BY-NC 4.0). The tests
download them; this repo and the runtime image do not contain them.
