# ai

ShopNexus's provider of self-hosted models. It serves OpenAI's API, and the
stack's LiteLLM routes an alias to it as it would to any provider. In the
cluster the server calls LiteLLM and LiteLLM calls `ai`.

It does speech to text with Gipformer 1.5
(`g-group-ai-lab/gipformer1.5-68M-rnnt`, MIT), a Vietnamese Zipformer
transducer with 68.6M parameters, on CPU through sherpa-onnx. It embeds photos
and short texts with SigLIP 2 (`google/siglip2-so400m-patch16-384`, Apache
2.0), the fp32 ONNX files of `onnx-community/siglip2-so400m-patch16-384-ONNX`
at revision `29f74e673c298fd07482d8e20138179282001eb7`, on CPU through
onnxruntime.

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
| `AI_EMBED_MODELS` | `name=dir` pairs, comma-separated | `siglip2-so400m-patch16-384=/models/siglip2-so400m-patch16-384` |
| `AI_EMBED_MAX_INPUTS` | inputs per embeddings request | `16` |
| `AI_EMBED_MAX_CONCURRENT` | embeddings requests running at once | `2` |
| `AI_EMBED_MAX_IMAGE_MB` | largest photo after base64 decoding | `10` |

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
