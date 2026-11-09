#!/bin/sh
# Downloads the model files the image ships, or the clips the integration
# tests read, and checks each against assets.sha256.
#   fetch-assets.sh models <dir>
#   fetch-assets.sh testwavs <dir>
# The clips come from sherpa-onnx's Vietnamese model repo, which took them from
# nguyenvulebinh/wav2vec2-base-vietnamese-250h (CC BY-NC 4.0). They are
# downloaded for tests only. The repo and the runtime image do not hold them.
set -eu
kind=${1:?models or testwavs}
dir=${2:?target directory}
here=$(cd "$(dirname "$0")" && pwd)

GIPFORMER=https://huggingface.co/g-group-ai-lab/gipformer1.5-68M-rnnt/resolve/dd9227dcd8705c13f33bdbe59728d546ab94480f
SILERO=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
WAVS=https://huggingface.co/csukuangfj/sherpa-onnx-zipformer-vi-2025-04-20/resolve/0fc3fea3ccd9c50b439755fa8a6aba546cb3a7d4/test_wavs

mkdir -p "$dir"
cd "$dir"
case $kind in
models)
	mkdir -p gipformer-1.5
	for f in encoder.onnx decoder.onnx joiner.onnx tokens.txt; do
		curl -fsSL -o "gipformer-1.5/$f" "$GIPFORMER/$f"
	done
	curl -fsSL -o silero_vad.onnx "$SILERO"
	;;
testwavs)
	for f in 0.wav 1.wav 2.wav; do
		curl -fsSL -o "$f" "$WAVS/$f"
	done
	;;
*)
	echo "usage: $0 models|testwavs <dir>" >&2
	exit 2
	;;
esac
grep " $kind/" "$here/assets.sha256" | sed "s| $kind/| |" | sha256sum -c --quiet -
