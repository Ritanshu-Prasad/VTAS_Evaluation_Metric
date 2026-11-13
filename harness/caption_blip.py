"""Generate BLIP-base captions for a COCO split (pinned model revision).

    python -m harness.caption_blip --split dev

Writes ``$VTAS_DATA/captions/blip_base_<split>.json`` = {"config": ..., "captions":
{image_id: caption}}. Captions are data and are not committed.

Decoding follows the BLIP paper's COCO captioning setup (Li et al., 2022, configs/
caption_coco.yaml): prompt "a picture of ", beam search with 3 beams, max length 20,
min length 5; the prompt is stripped from the output.
"""

from __future__ import annotations

import argparse
import json

from harness import data_dir
from harness.coco import image_path, load_split

MODEL_ID = "Salesforce/blip-image-captioning-base"
MODEL_REVISION = "82a37760796d32b1411fe092ab5d4e227313294b"
PROMPT = "a picture of "
GEN_KWARGS = {"num_beams": 3, "max_length": 20, "min_length": 5}


def generate(image_ids, batch_size: int = 16, device: str | None = None) -> dict[int, str]:
    import torch
    from PIL import Image
    from transformers import BlipForConditionalGeneration, BlipProcessor

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = BlipProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = BlipForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, torch_dtype=dtype).to(device).eval()

    out: dict[int, str] = {}
    ids = list(image_ids)
    for start in range(0, len(ids), batch_size):
        batch = ids[start:start + batch_size]
        images = [Image.open(image_path(i)).convert("RGB") for i in batch]
        inputs = processor(images=images, text=[PROMPT] * len(batch), return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
        with torch.inference_mode():
            tokens = model.generate(**inputs, **GEN_KWARGS)
        for i, text in zip(batch, processor.batch_decode(tokens, skip_special_tokens=True)):
            text = text.strip()
            if text.startswith(PROMPT.strip()):
                text = text[len(PROMPT.strip()):].strip()
            out[i] = text
        print(f"\r{min(start + batch_size, len(ids))}/{len(ids)}", end="", flush=True)
    print()
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m harness.caption_blip")
    p.add_argument("--split", default="dev")
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args(argv)

    import torch
    import transformers

    caps = generate(load_split(args.split), args.batch_size)
    config = {"model": MODEL_ID, "revision": MODEL_REVISION, "prompt": PROMPT,
              "generate": GEN_KWARGS, "split": args.split,
              "torch": torch.__version__, "transformers": transformers.__version__,
              "device": "cuda" if torch.cuda.is_available() else "cpu",
              "dtype": "float16" if torch.cuda.is_available() else "float32"}
    dest = data_dir() / "captions" / f"blip_base_{args.split}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"config": config, "captions": caps}, indent=1), encoding="utf-8")
    print(f"wrote {len(caps)} captions to {dest}")


if __name__ == "__main__":
    main()
