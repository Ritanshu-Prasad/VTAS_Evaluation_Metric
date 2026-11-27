"""Which captions go into the golden set. Deterministic from the dev split.

    python -m harness.golden.design      # writes harness/golden/golden_ids.json

Design: 200 dev images, seed 0.
  * paired block P: 50 images, each captioned by BLIP, LLaVA-1.5 (short) and a human
    reference -> 150 captions. Lets sources be compared on the same images.
  * unpaired block U: 150 further images, 50 per source -> 150 captions. More images
    and scenes, e.g. for fitting the several / group / crowd / many count intervals.
  * core set = P + U = 100 BLIP + 100 LLaVA-short + 100 human = 300 captions.
  * extension block L: LLaVA-1.5 long ("describe in detail") on the 50 P images. Kept
    separate so short- and long-caption behaviour of the metric can be compared; it
    is not part of the 300-caption core and not part of the kappa gate (Gate F).
The human caption is one of the image's official references (captions_val2014.json),
drawn with the same seed; its annotation id is stored so the text can be rebuilt.

Blinding: each caption also gets an opaque id ``gid`` (g001..g350, seeded shuffle), and
annotator files show only gids and images, in gid order, so the source (BLIP, LLaVA,
human) is not shown. This file is the key; annotators should not open it.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from harness import HARNESS_DIR
from harness.coco import annotation_path, load_split

SEED = 0
N_PAIRED = 50
N_UNPAIRED_PER_SOURCE = 50
CORE_SOURCES = ("blip", "llava_short", "human")
EXTENSION_SOURCES = ("llava_long",)
IDS_PATH = HARNESS_DIR / "golden" / "golden_ids.json"


def caption_id(source: str, image_id: int) -> str:
    return f"{source}-{int(image_id)}"


def select_items(dev_ids: list[int], seed: int = SEED) -> list[dict]:
    """Golden-set items (without the human reference choice) in a fixed order."""
    need = N_PAIRED + N_UNPAIRED_PER_SOURCE * len(CORE_SOURCES)
    if len(set(dev_ids)) < need:
        raise ValueError(f"need {need} distinct dev ids, got {len(set(dev_ids))}")
    order = np.random.default_rng(seed).permutation(sorted(set(int(i) for i in dev_ids)))
    paired = sorted(int(i) for i in order[:N_PAIRED])
    items = []
    for src in CORE_SOURCES:
        items += [{"source": src, "block": "P", "image_id": i} for i in paired]
    rest = order[N_PAIRED:need]
    for k, src in enumerate(CORE_SOURCES):
        chunk = sorted(int(i) for i in rest[k * N_UNPAIRED_PER_SOURCE:(k + 1) * N_UNPAIRED_PER_SOURCE])
        items += [{"source": src, "block": "U", "image_id": i} for i in chunk]
    items += [{"source": "llava_long", "block": "L", "image_id": i} for i in paired]
    for it in items:
        it["caption_id"] = caption_id(it["source"], it["image_id"])
    blind = np.random.default_rng(seed + 2).permutation(len(items))
    for it, k in zip(items, blind):
        it["gid"] = f"g{int(k) + 1:03d}"
    return items


def choose_references(items: list[dict], captions_json: dict, seed: int = SEED) -> None:
    """Pick one official reference per human item (in place); stores ann id and text."""
    by_image: dict[int, list[dict]] = {}
    for a in captions_json["annotations"]:
        by_image.setdefault(a["image_id"], []).append(a)
    rng = np.random.default_rng(seed + 1)
    for it in items:
        if it["source"] != "human":
            continue
        anns = sorted(by_image[it["image_id"]], key=lambda a: a["id"])
        a = anns[int(rng.integers(len(anns)))]
        it["ref_annotation_id"] = a["id"]


def build(seed: int = SEED) -> dict:
    items = select_items(load_split("dev"), seed)
    with open(annotation_path("captions_val2014.json"), encoding="utf-8") as f:
        choose_references(items, json.load(f), seed)
    images = sorted({it["image_id"] for it in items})
    return {
        "description": "Golden set captions. See harness/golden/design.py.",
        "split": "dev",
        "seed": seed,
        "n_images": len(images),
        "core_sources": list(CORE_SOURCES),
        "extension_sources": list(EXTENSION_SOURCES),
        "counts": {b: sum(it["block"] == b for it in items) for b in ("P", "U", "L")},
        "llava_image_ids": sorted({it["image_id"] for it in items if it["source"].startswith("llava")}),
        "items": items,
    }


def load_design(path=IDS_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m harness.golden.design")
    p.add_argument("--out", default=str(IDS_PATH))
    args = p.parse_args(argv)
    d = build()
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(d, f, indent=1)
        f.write("\n")
    print(f"wrote {len(d['items'])} items on {d['n_images']} images to {args.out}")


if __name__ == "__main__":
    main()
