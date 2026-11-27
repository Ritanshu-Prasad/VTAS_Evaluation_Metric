"""Generate BLIP-base captions for a COCO split (pinned model revision).

DEFECTIVE - DO NOT USE FOR RESULTS. Kept only to document the decoding problem below.
The Hugging Face checkpoint never emits WordPiece continuation tokens ("a gi in a zoo"
for giraffe, "frc", "fire hydra", "pew pew pew ..."), so CHAIR misses those objects.
Use ``harness.caption_blip_orig`` (original Salesforce code and checkpoint) instead.

    python -m harness.caption_blip --split dev

Writes ``$VTAS_DATA/captions/blip_base_<variant>_<split>.json`` = {"config": ...,
"captions": {image_id: caption}}. Captions are data and are not committed.

Variants (declared before any CHAIR number was looked at for the second one):
  paper      - the BLIP paper's COCO captioning setup (Li et al., 2022, configs/
               caption_coco.yaml): prompt "a picture of ", 3 beams, max_length 20,
               min_length 5 (HF counts the prompt tokens in both); prompt stripped.
  unprompted - no prompt, 3 beams, max_length 30, min_length 5 (sensitivity check:
               the paper setup yields ~6.5-word captions).
Runs in float32: on a GTX 1650 float16 gave identical captions on 48 images and was
5x slower.
"""

from __future__ import annotations

import argparse
import json

from harness import data_dir
from harness.coco import image_path, load_split

MODEL_ID = "Salesforce/blip-image-captioning-base"
MODEL_REVISION = "82a37760796d32b1411fe092ab5d4e227313294b"
VARIANTS = {
    "paper": {"prompt": "a picture of ",
              "generate": {"num_beams": 3, "max_length": 20, "min_length": 5}},
    "unprompted": {"prompt": None,
                   "generate": {"num_beams": 3, "max_length": 30, "min_length": 5}},
}


def generate(image_ids, variant: str = "paper", batch_size: int = 16,
             device: str | None = None) -> dict[int, str]:
    import torch
    from PIL import Image
    from transformers import BlipForConditionalGeneration, BlipProcessor

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    prompt, gen_kwargs = VARIANTS[variant]["prompt"], VARIANTS[variant]["generate"]
    processor = BlipProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = BlipForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, torch_dtype=dtype).to(device).eval()

    out: dict[int, str] = {}
    ids = list(image_ids)
    for start in range(0, len(ids), batch_size):
        batch = ids[start:start + batch_size]
        images = [Image.open(image_path(i)).convert("RGB") for i in batch]
        text_in = [prompt] * len(batch) if prompt else None
        inputs = processor(images=images, text=text_in, return_tensors="pt").to(device)
        with torch.inference_mode():
            tokens = model.generate(**inputs, **gen_kwargs)
        for i, text in zip(batch, processor.batch_decode(tokens, skip_special_tokens=True)):
            text = text.strip()
            if prompt and text.startswith(prompt.strip()):
                text = text[len(prompt.strip()):].strip()
            out[i] = text
        print(f"\r{min(start + batch_size, len(ids))}/{len(ids)}", end="", flush=True)
    print()
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m harness.caption_blip")
    p.add_argument("--split", default="dev")
    p.add_argument("--variant", choices=list(VARIANTS), default="paper")
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args(argv)

    import torch
    import transformers

    caps = generate(load_split(args.split), args.variant, args.batch_size)
    config = {"model": MODEL_ID, "revision": MODEL_REVISION, "variant": args.variant,
              **VARIANTS[args.variant], "split": args.split, "dtype": "float32",
              "torch": torch.__version__, "transformers": transformers.__version__,
              "device": "cuda" if torch.cuda.is_available() else "cpu"}
    dest = data_dir() / "captions" / f"blip_base_{args.variant}_{args.split}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"config": config, "captions": caps}, indent=1), encoding="utf-8")
    print(f"wrote {len(caps)} captions to {dest}")


if __name__ == "__main__":
    main()
