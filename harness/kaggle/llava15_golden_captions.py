"""LLaVA-1.5-7B captions for the VTAS golden set. Runs on Kaggle (GPU T4 x2), not locally.

Self-contained: no repo imports, so it can be pasted into a single Kaggle notebook cell.

Kaggle setup
  1. New notebook -> Settings: Accelerator "GPU T4 x2", Internet ON.
  2. Cell 1 (install pinned versions, then restart the session if Kaggle asks):
         !pip install -q "transformers==4.46.3" "accelerate==1.1.1"
  3. Cell 2: paste this whole file and run it. About 30-40 min.
  4. Download /kaggle/working/llava15_7b_golden.json and put it at
         <repo>/data/captions/llava15_7b_golden.json
     then run locally:  python -m harness.golden.llava_check

What it does (fixed before any caption was seen)
  * model llava-hf/llava-1.5-7b-hf at a pinned revision, float16, device_map="auto",
    batch size 1, greedy decoding (do_sample=False, num_beams=1).
  * short: "Provide a one-sentence caption for the provided image." (the LLaVA-1.5
    COCO-caption evaluation prompt), max_new_tokens 64.
  * long: "Please describe this image in detail." (the usual CHAIR prompt for LVLMs),
    max_new_tokens 512.
  * images: the 100 golden-set LLaVA images (harness/golden/golden_ids.json,
    llava_image_ids), downloaded from the official COCO server; their SHA-256 is saved
    so the local check can confirm the same files were captioned.
Both prompts run on all 100 images; the golden set uses short on 100 and long on the
50 paired images.
"""

import hashlib
import io
import json
import platform
import time
import urllib.request

import torch
import transformers
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
MODEL_REVISION = "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
IMAGE_URL = "http://images.cocodataset.org/val2014/COCO_val2014_{:012d}.jpg"
OUT = "/kaggle/working/llava15_7b_golden.json"
PROMPTS = {
    "short": {"text": "Provide a one-sentence caption for the provided image.",
              "max_new_tokens": 64},
    "long": {"text": "Please describe this image in detail.", "max_new_tokens": 512},
}
TEMPLATE = "USER: <image>\n{text} ASSISTANT:"   # LLaVA-1.5 conversation format

# Must equal llava_image_ids in harness/golden/golden_ids.json (checked by a unit test).
IMAGE_IDS = [
    2453, 10400, 11181, 20598, 30925, 31965, 39081, 40446, 49014, 49688,
    74209, 75173, 75990, 84735, 87399, 87617, 106048, 108338, 108853, 113223,
    114941, 125071, 151790, 166124, 168093, 182683, 183014, 185639, 188077, 195648,
    198176, 201207, 206851, 208516, 211120, 212285, 229014, 231685, 249623, 252220,
    260966, 264347, 266977, 267349, 270505, 274398, 280909, 288202, 291664, 292170,
    295728, 301817, 305000, 313182, 325153, 325991, 326344, 326968, 360610, 361860,
    364166, 375285, 375324, 391199, 393777, 397042, 403421, 407072, 409217, 420546,
    424880, 443887, 444302, 451972, 453297, 458755, 468935, 468993, 471641, 475403,
    491497, 495081, 499198, 506201, 506458, 520108, 525568, 529592, 533013, 539784,
    539904, 540280, 541478, 552564, 554749, 562592, 564091, 571990, 577866, 580591,
]


def fetch(image_id, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(IMAGE_URL.format(image_id), timeout=60) as r:
                return r.read()
        except Exception:  # noqa: BLE001 - network retry
            if k == tries - 1:
                raise
            time.sleep(2 ** k)


def main():
    assert transformers.__version__ == "4.46.3", transformers.__version__
    assert torch.cuda.is_available(), "enable a GPU accelerator"
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.float16, device_map="auto").eval()

    captions = {name: {} for name in PROMPTS}
    image_sha256 = {}
    t0 = time.time()
    for n, image_id in enumerate(IMAGE_IDS, 1):
        raw = fetch(image_id)
        image_sha256[str(image_id)] = hashlib.sha256(raw).hexdigest()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        for name, cfg in PROMPTS.items():
            inputs = processor(images=image, text=TEMPLATE.format(text=cfg["text"]),
                               return_tensors="pt").to(model.device, torch.float16)
            with torch.inference_mode():
                out = model.generate(**inputs, do_sample=False, num_beams=1,
                                     max_new_tokens=cfg["max_new_tokens"])
            new_tokens = out[0, inputs["input_ids"].shape[1]:]
            captions[name][str(image_id)] = processor.decode(
                new_tokens, skip_special_tokens=True).strip()
        print(f"{n}/{len(IMAGE_IDS)}  {time.time() - t0:.0f}s  "
              f"{captions['short'][str(image_id)][:80]}", flush=True)

    config = {
        "model": MODEL_ID, "revision": MODEL_REVISION, "template": TEMPLATE,
        "prompts": PROMPTS,
        "generate": {"do_sample": False, "num_beams": 1, "batch_size": 1},
        "dtype": "float16", "device_map": "auto",
        "torch": torch.__version__, "transformers": transformers.__version__,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "python": platform.python_version(),
        "image_source": IMAGE_URL,
        "n_images": len(IMAGE_IDS),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"config": config, "captions": captions, "image_sha256": image_sha256},
                  f, indent=1)
    print(f"wrote {OUT}")


main()
