"""CHAIR-GT for a captions file on a committed COCO split.

    python -m harness.eval_chair CAPTIONS.json --split dev --out harness/results/x.json

CAPTIONS.json is {image_id: caption} or {"config": {...}, "captions": {image_id: caption}}.
The summary (CHAIRs, CHAIRi with bootstrap 95% CIs over images) is small and may be
committed; per-caption details are only written with --details (to data/, not git).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness.chair_gt import ChairGT, word_tokenize
from harness.coco import annotation_path, load_split


def tokenizer_check(texts) -> dict:
    """Compare harness word_tokenize with nltk.word_tokenize (needs Punkt data)."""
    import nltk

    try:
        nltk.word_tokenize("a test.")
    except LookupError:
        return {"checked": False, "reason": "nltk punkt_tab data not installed"}
    diff = [t for t in texts if word_tokenize(t.lower()) != nltk.word_tokenize(t.lower())]
    return {"checked": True, "n_texts": len(texts), "n_token_differences": len(diff),
            "examples": diff[:5]}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m harness.eval_chair")
    p.add_argument("captions")
    p.add_argument("--split", required=True, choices=["calib", "dev", "test"])
    p.add_argument("--out", required=True)
    p.add_argument("--details", help="optional per-caption JSON (keep out of git)")
    p.add_argument("--check-tokenizer", action="store_true")
    p.add_argument("--resamples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    raw = json.loads(Path(args.captions).read_text(encoding="utf-8"))
    config = raw.get("config", {}) if "captions" in raw else {}
    captions = {int(k): v for k, v in (raw["captions"] if "captions" in raw else raw).items()}
    ids = load_split(args.split)
    if set(captions) != set(ids):
        raise SystemExit(f"captions cover {len(captions)} images; split {args.split} has "
                         f"{len(ids)} (missing {len(set(ids) - set(captions))})")

    chair = ChairGT.from_coco(annotation_path("instances_val2014.json"),
                              annotation_path("captions_val2014.json"), ids)
    res = chair.evaluate({i: captions[i] for i in ids})
    summary = {"metric": "CHAIR-GT", "split": args.split, "captioner": config,
               **res.summary(args.resamples, seed=args.seed)}
    if args.check_tokenizer:
        refs = json.loads(annotation_path("captions_val2014.json").read_text(encoding="utf-8"))
        ref_texts = [a["caption"] for a in refs["annotations"] if a["image_id"] in set(ids)]
        summary["tokenizer_check"] = tokenizer_check(list(captions.values()) + ref_texts)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: summary[k] for k in ("CHAIRs", "CHAIRi", "n_captions")}, indent=2))

    if args.details:
        rows = [{"image_id": c.image_id, "caption": c.caption, "objects": c.objects,
                 "hallucinated": c.hallucinated, "gt_objects": c.gt_objects}
                for c in res.captions]
        Path(args.details).write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
