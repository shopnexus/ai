# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers", "pillow", "sentencepiece", "protobuf"]
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Draws the embed test images and writes the reference ids and vectors.

Run once from the repo root, when the model or its revision changes:

    uv run scripts/siglip-reference.py embed/testdata

The images are drawn here, so the repo holds no third-party picture.
"""
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoModel, AutoProcessor

MODEL = "google/siglip2-so400m-patch16-384"
REVISION = "dd658faac399427308559e2c3ac1e99cbe43845d"


def draw(out: Path) -> list[str]:
    files = []

    def save(name, img):
        img.save(out / name)
        files.append(name)

    img = Image.new("RGB", (640, 480), "white")
    ImageDraw.Draw(img).ellipse((170, 90, 470, 390), fill="red")
    save("red-circle.png", img)

    img = Image.new("RGB", (500, 500), "white")
    ImageDraw.Draw(img).rectangle((100, 100, 400, 400), fill="blue")
    save("blue-square.png", img)

    img = Image.new("RGB", (800, 600), "white")
    ImageDraw.Draw(img).polygon([(400, 80), (700, 520), (100, 520)], fill="green")
    save("green-triangle.png", img)

    img = Image.new("RGB", (512, 512), "white")
    d = ImageDraw.Draw(img)
    for y in range(8):
        for x in range(8):
            if (x + y) % 2:
                d.rectangle((x * 64, y * 64, x * 64 + 63, y * 64 + 63), fill="black")
    save("checkerboard.png", img)

    img = Image.new("RGB", (600, 400), "white")
    d = ImageDraw.Draw(img)
    for y in range(0, 400, 80):
        d.rectangle((0, y, 600, y + 39), fill="yellow")
    save("yellow-stripes.png", img)

    # RGBA with a transparent corner: PIL's convert("RGB") keeps the stored RGB.
    img = Image.new("RGBA", (300, 300), (128, 0, 128, 255))
    d = ImageDraw.Draw(img)
    d.ellipse((110, 110, 190, 190), fill=(255, 140, 0, 255))
    d.rectangle((0, 0, 60, 60), fill=(0, 255, 0, 0))
    save("orange-dot.png", img)
    return files


TEXTS = [
    "a red circle",
    "a blue square",
    "a green triangle",
    "a black and white checkerboard",
    "yellow horizontal stripes",
    "an orange dot on a purple background",
    "iPhone 13 Pro Max smartphone, blue",
    # Over 64 tokens: the reference shows how truncation ends.
    "a very long description " * 20,
]


def unit(v: torch.Tensor) -> list[float]:
    v = v / v.norm(dim=-1, keepdim=True)
    return [float(x) for x in v]


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    files = draw(out)
    model = AutoModel.from_pretrained(MODEL, revision=REVISION).eval()
    proc = AutoProcessor.from_pretrained(MODEL, revision=REVISION)

    ref = {"model": MODEL, "revision": REVISION, "images": [], "texts": []}
    with torch.no_grad():
        for name in files:
            pixels = proc(images=Image.open(out / name).convert("RGB"), return_tensors="pt")
            feats = model.get_image_features(**pixels)
            feats = feats if torch.is_tensor(feats) else feats.pooler_output
            ref["images"].append({"file": name, "embedding": unit(feats[0])})
        for text in TEXTS:
            lowered = text.lower()
            enc = proc(text=[lowered], padding="max_length", max_length=64, truncation=True, return_tensors="pt")
            feats = model.get_text_features(input_ids=enc["input_ids"])
            feats = feats if torch.is_tensor(feats) else feats.pooler_output
            ids = [int(i) for i in enc["input_ids"][0]]
            # Gemma's pad id appears only as padding, so the ids before it count.
            tokens = sum(1 for i in ids if i != proc.tokenizer.pad_token_id)
            ref["texts"].append({"text": text, "input_ids": ids, "tokens": tokens, "embedding": unit(feats[0])})
    (out / "reference.json").write_text(json.dumps(ref))


if __name__ == "__main__":
    main()
