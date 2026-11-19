"""Check the LLaVA-1.5 caption file brought back from Kaggle before it is used.

    python -m harness.golden.llava_check [path]   # default $VTAS_DATA/captions/llava15_7b_golden.json

Checks: pinned model and revision, prompts and decoding as declared in
harness/kaggle/llava15_golden_captions.py, exactly the 100 golden-set LLaVA images,
non-empty captions for both prompts, and (when the local COCO image exists) that the
image Kaggle captioned has the same SHA-256 as the local file.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

from harness import REPO_ROOT, data_dir
from harness.coco import image_path
from harness.golden.design import load_design

KAGGLE_SCRIPT = REPO_ROOT / "harness" / "kaggle" / "llava15_golden_captions.py"


def script_constants(path: Path = KAGGLE_SCRIPT) -> dict:
    """Module-level constants of the Kaggle script, read without executing it."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


def check(doc: dict) -> list[str]:
    """Problems found (empty list = file is usable)."""
    want = script_constants()
    cfg, caps = doc.get("config", {}), doc.get("captions", {})
    problems = []
    for key, const in (("model", "MODEL_ID"), ("revision", "MODEL_REVISION"),
                       ("template", "TEMPLATE"), ("prompts", "PROMPTS")):
        if cfg.get(key) != want[const]:
            problems.append(f"config.{key} = {cfg.get(key)!r}, expected {want[const]!r}")
    if cfg.get("generate", {}).get("do_sample") is not False:
        problems.append("decoding is not greedy")
    ids = {str(i) for i in load_design()["llava_image_ids"]}
    for name in ("short", "long"):
        got = caps.get(name, {})
        if set(got) != ids:
            problems.append(f"{name}: {len(set(got) & ids)}/100 golden images, "
                            f"{len(set(got) - ids)} extra")
        empty = [k for k, v in got.items() if not str(v).strip()]
        if empty:
            problems.append(f"{name}: {len(empty)} empty captions, e.g. {empty[:3]}")
    checked = 0
    for k, sha in doc.get("image_sha256", {}).items():
        p = image_path(int(k))
        if p.exists():
            checked += 1
            if hashlib.sha256(p.read_bytes()).hexdigest() != sha:
                problems.append(f"image {k}: SHA-256 differs from local file")
    if not checked:
        problems.append("no local images to compare SHA-256 with (download dev images)")
    return problems


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m harness.golden.llava_check")
    p.add_argument("path", nargs="?",
                   default=str(data_dir() / "captions" / "llava15_7b_golden.json"))
    args = p.parse_args(argv)
    doc = json.loads(Path(args.path).read_text(encoding="utf-8"))
    problems = check(doc)
    for name in ("short", "long"):
        vals = list(doc.get("captions", {}).get(name, {}).values())
        if vals:
            words = sum(len(v.split()) for v in vals) / len(vals)
            print(f"{name}: {len(vals)} captions, {words:.1f} words on average")
    print("OK" if not problems else "PROBLEMS:\n  " + "\n  ".join(problems))
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
