"""Official COCO 2014 files (real image ids) and the fixed calib/dev/test splits.

Usage (from the repo root):
    python -m harness.coco annotations          # official annotation zip, checksummed
    python -m harness.coco images --split dev   # official val2014 images for one split
    python -m harness.coco make-splits          # regenerate split id lists (deterministic)

Split design. The 5,000 split images are the Karpathy *test* split (all val2014). The
Karpathy train split (used to fine-tune BLIP and most COCO captioners) contains 30,504
val2014 images ("restval"), so a random val2014 sample would contain training images.
The 5,000 ids are shuffled with a fixed seed and cut into calib 1,000 / dev 1,000 /
test 3,000. Only the id lists are committed (harness/splits/*.txt).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from harness import SPLITS_DIR, data_dir
from harness.download import extract_zip, fetch

ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
ANNOTATIONS_SHA256 = "031296bbc80c45a1d1f76bf9a90ead27e94e99ec629208449507a4917a3bf009"
FILE_SHA256 = {
    "instances_val2014.json": "e1b4d1bc54906827b1a029981b1c77f2088bd4462569b7615db7929070c3ab47",
    "captions_val2014.json": "7da4cc45cb76ee1642d163e846e1e0a33feffac2e841c9c426d585ea499165a4",
}
IMAGE_URL = "http://images.cocodataset.org/val2014/COCO_val2014_{:012d}.jpg"

KARPATHY_URL = "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"
KARPATHY_SHA256 = "4cfd70132527b80933105e5829dc9034eaab9573482e2e680abbab6130244817"

SPLIT_SEED = 0
SPLIT_SIZES = {"calib": 1000, "dev": 1000, "test": 3000}


def coco_dir() -> Path:
    return data_dir() / "coco"


def annotation_path(name: str) -> Path:
    return coco_dir() / "annotations" / name


def image_path(image_id: int) -> Path:
    return coco_dir() / "val2014" / f"COCO_val2014_{image_id:012d}.jpg"


def download_annotations() -> dict[str, Path]:
    z = fetch(ANNOTATIONS_URL, coco_dir() / "annotations_trainval2014.zip", ANNOTATIONS_SHA256)
    names = [f"annotations/{n}" for n in FILE_SHA256]
    extract_zip(z, names, coco_dir())
    from harness.download import verify
    return {n: verify(annotation_path(n), s) for n, s in FILE_SHA256.items()}


def download_images(image_ids, workers: int = 16) -> list[Path]:
    """Fetch official val2014 images one by one (no 6 GB zip needed)."""
    ids = [int(i) for i in image_ids]

    def one(i):
        return fetch(IMAGE_URL.format(i), image_path(i))

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, ids))


# ---- splits ---------------------------------------------------------------------------

def karpathy_test_ids() -> list[int]:
    z = fetch(KARPATHY_URL, data_dir() / "karpathy" / "caption_datasets.zip", KARPATHY_SHA256)
    extract_zip(z, ["dataset_coco.json"], z.parent)
    with open(z.parent / "dataset_coco.json", encoding="utf-8") as f:
        images = json.load(f)["images"]
    return sorted(im["cocoid"] for im in images if im["split"] == "test")


def make_splits(ids: list[int], seed: int = SPLIT_SEED) -> dict[str, list[int]]:
    ids = sorted(set(int(i) for i in ids))
    if len(ids) != sum(SPLIT_SIZES.values()):
        raise ValueError(f"expected {sum(SPLIT_SIZES.values())} ids, got {len(ids)}")
    order = np.random.default_rng(seed).permutation(len(ids))
    out, start = {}, 0
    for name, size in SPLIT_SIZES.items():
        out[name] = sorted(ids[k] for k in order[start:start + size])
        start += size
    return out


def _ids_sha256(ids: list[int]) -> str:
    return hashlib.sha256("".join(f"{i}\n" for i in ids).encode()).hexdigest()


def write_splits(splits: dict[str, list[int]], seed: int, out_dir: Path = SPLITS_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in splits.items():
        (out_dir / f"coco_{name}.txt").write_text("".join(f"{i}\n" for i in ids),
                                                  encoding="ascii", newline="\n")
    manifest = {
        "source": "COCO val2014 image ids from the Karpathy test split (dataset_coco.json)",
        "karpathy_zip": KARPATHY_URL,
        "karpathy_zip_sha256": KARPATHY_SHA256,
        "method": "sorted ids -> numpy.random.default_rng(seed).permutation -> "
                  "calib, dev, test in that order; each list written sorted",
        "seed": seed,
        "splits": {n: {"n": len(ids), "sha256": _ids_sha256(ids)} for n, ids in splits.items()},
    }
    (out_dir / "coco_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                encoding="utf-8", newline="\n")


def load_split(name: str, splits_dir: Path = SPLITS_DIR) -> list[int]:
    """Image ids of a committed split ("calib", "dev" or "test")."""
    if name not in SPLIT_SIZES:
        raise ValueError(f"unknown split {name!r}; choose from {list(SPLIT_SIZES)}")
    text = (splits_dir / f"coco_{name}.txt").read_text(encoding="ascii")
    return [int(line) for line in text.split()]


def load_references(image_ids) -> dict[int, list[str]]:
    """Human reference captions (official captions_val2014.json) for the given ids."""
    wanted = set(int(i) for i in image_ids)
    with open(annotation_path("captions_val2014.json"), encoding="utf-8") as f:
        caps = json.load(f)
    refs: dict[int, list[str]] = {i: [] for i in sorted(wanted)}
    for a in sorted(caps["annotations"], key=lambda a: a["id"]):
        if a["image_id"] in wanted:
            refs[a["image_id"]].append(a["caption"])
    return refs


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m harness.coco")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("annotations")
    im = sub.add_parser("images")
    im.add_argument("--split", choices=list(SPLIT_SIZES) + ["all"], default="dev")
    ms = sub.add_parser("make-splits")
    ms.add_argument("--seed", type=int, default=SPLIT_SEED)
    args = p.parse_args(argv)

    if args.cmd == "annotations":
        for n, path in download_annotations().items():
            print(f"ok {n} -> {path}")
    elif args.cmd == "images":
        names = list(SPLIT_SIZES) if args.split == "all" else [args.split]
        ids = [i for n in names for i in load_split(n)]
        paths = download_images(ids)
        print(f"ok {len(paths)} images in {coco_dir() / 'val2014'}")
    elif args.cmd == "make-splits":
        ids = karpathy_test_ids()
        with open(download_annotations()["captions_val2014.json"], encoding="utf-8") as f:
            val_ids = {im["id"] for im in json.load(f)["images"]}
        missing = set(ids) - val_ids
        if missing:
            raise SystemExit(f"{len(missing)} Karpathy test ids are not in val2014")
        write_splits(make_splits(ids, args.seed), args.seed)
        print(f"wrote splits to {SPLITS_DIR}")


if __name__ == "__main__":
    main()
