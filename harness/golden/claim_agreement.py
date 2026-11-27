"""Claim-extraction agreement between a parser and the golden-set claim lists.

Definition (fixed before any parser was run):

* A claim is the pair (type, normalised text). Types: object, count, attribute, relation, scene.
* Normalisation of claim text: lower-case; every article (a, an, the) removed; punctuation other
  than the " / " and ": " separators of relation and attribute claims removed; whitespace
  collapsed. Nothing else is folded: plural forms, word order and verb forms must match.
* Per caption, parser claims and reference claims are matched as multisets (a claim that
  appears twice in the reference must be produced twice).
* Counts are summed over captions (micro): TP = matched claims, P = parser claims, R = reference
  claims. precision = TP / P, recall = TP / R, agreement = F1 = 2 TP / (P + R).
* Reported per source (blip, llava_short, human, llava_long) and per claim type. The gate
  is pooled agreement over the 300-caption core (blip + llava_short + human) >= 0.95; the
  llava_long block is an extension and is reported but not part of the gate.

Diagnostic categories for unmatched claims (never used to score): ``variant`` (same type, token
Jaccard >= 0.5 with a leftover claim of the other list), ``type`` (same normalised text,
different type), ``missing`` (reference claim with no counterpart), ``spurious`` (parser claim
with no counterpart).

Held-out rule: ``claim_parser_split.json`` fixes a dev part (used for rule development) and a
held-out part (reported, never inspected claim by claim). ``failures`` refuses the held-out part.

    python -m harness.golden.claim_agreement --make-split          # once; writes the split file
    python -m harness.golden.claim_agreement --model sm --part dev --failures
    python -m harness.golden.claim_agreement --model sm --part heldout
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict

import numpy as np

from harness import HARNESS_DIR
from harness.golden.annotate import load_caption_texts, load_claims
from harness.golden.design import CORE_SOURCES, load_design

SPLIT_PATH = HARNESS_DIR / "golden" / "claim_parser_split.json"
GATE = 0.95
SEED = 0
SOURCES = (*CORE_SOURCES, "llava_long")
TYPES = ("object", "count", "attribute", "relation", "scene")
_ARTICLES = re.compile(r"\b(?:a|an|the)\b")
_KEEP = re.compile(r"[^a-z0-9 /:'-]")


def normalise(text: str) -> str:
    t = _ARTICLES.sub(" ", text.lower())
    t = _KEEP.sub(" ", t)
    t = re.sub(r"\s*/\s*", " / ", t)
    t = re.sub(r"\s*:\s*", ": ", t)
    return " ".join(t.split())


def key(claim) -> tuple[str, str]:
    return (claim[0], normalise(claim[1]))


def source_of(caption_id: str) -> str:
    return caption_id.rsplit("-", 1)[0]


# ---- scoring ----------------------------------------------------------------------------------

def match_caption(pred: list, ref: list) -> dict:
    """Multiset match for one caption. Returns tp list and leftover lists (keys)."""
    p, r = Counter(key(c) for c in pred), Counter(key(c) for c in ref)
    tp = p & r
    return {"tp": tp, "fp": p - tp, "fn": r - tp}


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.replace("/", " ").replace(":", " ").split()), set(b.replace("/", " ").replace(":", " ").split())
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def categorise(fp: Counter, fn: Counter) -> list[tuple[str, tuple, tuple | None]]:
    """(category, parser claim or None, reference claim or None) for leftovers of one caption."""
    fp, fn = Counter(fp), Counter(fn)
    out = []
    for t in TYPES:  # variants: greedy best-Jaccard pairing within a type
        while True:
            best = (0.0, None, None)
            for a in (k for k in fp.elements() if k[0] == t):
                for b in (k for k in fn.elements() if k[0] == t):
                    j = _jaccard(a[1], b[1])
                    if j > best[0]:
                        best = (j, a, b)
            if best[0] < 0.5:
                break
            fp[best[1]] -= 1
            fn[best[2]] -= 1
            out.append(("variant", best[1], best[2]))
    fp, fn = +fp, +fn
    for a in list(fp.elements()):  # same text, other type
        hit = next((b for b in fn.elements() if b[1] == a[1]), None)
        if hit:
            fp[a] -= 1
            fn[hit] -= 1
            out.append(("type", a, hit))
    out += [("spurious", a, None) for a in (+fp).elements()]
    out += [("missing", None, b) for b in (+fn).elements()]
    return out


def evaluate(pred: dict, ref: dict, ids: list[str]) -> dict:
    """Micro counts per (source, type) plus the leftovers per caption."""
    cnt = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    left = {}
    for cid in ids:
        m = match_caption(pred.get(cid, []), ref[cid])
        s = source_of(cid)
        for kind in ("tp", "fp", "fn"):
            for (t, _), n in m[kind].items():
                cnt[(s, t)][kind] += n
        left[cid] = categorise(m["fp"], m["fn"])
    return {"counts": dict(cnt), "left": left}


def _prf(c: dict) -> dict:
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    p_n, r_n = tp + fp, tp + fn
    return {"tp": tp, "n_parser": p_n, "n_ref": r_n,
            "precision": tp / p_n if p_n else None, "recall": tp / r_n if r_n else None,
            "agreement": 2 * tp / (p_n + r_n) if p_n + r_n else None}


def summarise(res: dict) -> dict:
    cnt = res["counts"]

    def pool(sel):
        agg = {"tp": 0, "fp": 0, "fn": 0}
        for (s, t), c in cnt.items():
            if sel(s, t):
                for k in agg:
                    agg[k] += c[k]
        return _prf(agg)

    out = {"core": pool(lambda s, t: s in CORE_SOURCES), "all": pool(lambda s, t: True),
           "by_source": {s: pool(lambda x, t, s=s: x == s) for s in SOURCES},
           "by_type_core": {t: pool(lambda s, x, t=t: s in CORE_SOURCES and x == t) for t in TYPES},
           "by_source_type": {f"{s}/{t}": pool(lambda x, y, s=s, t=t: (x, y) == (s, t))
                              for s in SOURCES for t in TYPES}}
    out["gate"] = {"threshold": GATE, "metric": "pooled core agreement (F1)",
                   "pass": (out["core"]["agreement"] or 0) >= GATE}
    return out


def bootstrap_ci(pred: dict, ref: dict, ids: list[str], n: int = 1000, seed: int = SEED) -> list[float]:
    """95% CI of pooled core agreement, resampling images (all captions of an image together)."""
    by_img = defaultdict(list)
    for cid in ids:
        if source_of(cid) in CORE_SOURCES:
            by_img[cid.rsplit("-", 1)[1]].append(cid)
    per = {}
    for img, cids in by_img.items():
        tot = np.zeros(3)
        for cid in cids:
            m = match_caption(pred.get(cid, []), ref[cid])
            tp = sum(m["tp"].values())
            tot += (tp, tp + sum(m["fp"].values()), tp + sum(m["fn"].values()))
        per[img] = tot
    imgs, arr = list(per), np.array(list(per.values()))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        s = arr[rng.integers(0, len(imgs), len(imgs))].sum(axis=0)
        vals.append(2 * s[0] / (s[1] + s[2]) if s[1] + s[2] else float("nan"))
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


# ---- split ------------------------------------------------------------------------------------

def make_split(seed: int = SEED) -> dict:
    """50/50 dev / held-out by IMAGE (all captions of an image in the same part), stratified by
    block (P images; U images per source) so both parts see every source."""
    design = load_design()
    strata = defaultdict(set)
    for it in design["items"]:
        if it["block"] == "P" or it["block"] == "L":
            strata["P"].add(it["image_id"])
        else:
            strata[f"U-{it['source']}"].add(it["image_id"])
    rng = np.random.default_rng(seed)
    part_of = {}
    for name in sorted(strata):
        ids = sorted(strata[name])
        order = rng.permutation(len(ids))
        for rank, k in enumerate(order):
            part_of[ids[k]] = "dev" if rank < len(ids) // 2 else "heldout"
    parts = {"dev": [], "heldout": []}
    for it in sorted(design["items"], key=lambda i: i["caption_id"]):
        parts[part_of[it["image_id"]]].append(it["caption_id"])
    return {"description": "Dev / held-out split of the golden-set captions for the claim parser. "
                           "Split by image id (all captions of one image in the same "
                           "part), stratified by block and source, seed 0. Rules are developed on "
                           "dev only; held-out is reported in aggregate and never read claim by claim.",
            "seed": seed, "unit": "image_id",
            "counts": {k: len(v) for k, v in parts.items()}, **parts}


def load_split() -> dict:
    return json.loads(SPLIT_PATH.read_text(encoding="utf-8"))


# ---- CLI --------------------------------------------------------------------------------------

def _fmt(x):
    return "  n/a" if x is None else f"{100 * x:5.1f}"


def print_report(summ: dict, ci=None) -> None:
    print(f"{'':22s} {'agree':>6s} {'prec':>6s} {'recall':>6s} {'n_ref':>6s} {'n_par':>6s}")
    rows = [("CORE (gate)", summ["core"])] + [(s, summ["by_source"][s]) for s in SOURCES]
    rows += [(f"core/{t}", summ["by_type_core"][t]) for t in TYPES]
    for name, r in rows:
        print(f"{name:22s} {_fmt(r['agreement']):>6s} {_fmt(r['precision']):>6s} {_fmt(r['recall']):>6s} "
              f"{r['n_ref']:6d} {r['n_parser']:6d}")
    print(f"gate pooled core >= {100 * GATE:.0f}%: {'PASS' if summ['gate']['pass'] else 'FAIL'}"
          + (f"   95% CI [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]" if ci else ""))


def print_failures(res: dict, texts: dict, limit_per_cat: int = 400) -> None:
    cats = Counter()
    for cid, items in res["left"].items():
        for cat, a, b in items:
            cats[(source_of(cid), (a or b)[0], cat)] += 1
    print("\nfailure counts (source, type, category):")
    for k, n in sorted(cats.items(), key=lambda kv: -kv[1]):
        print(f"  {n:4d}  {k[0]:12s} {k[1]:10s} {k[2]}")
    shown = Counter()
    print("\nexamples:")
    for cid in sorted(res["left"]):
        for cat, a, b in res["left"][cid]:
            if shown[cat] >= limit_per_cat:
                continue
            shown[cat] += 1
            print(f"[{cat}] {cid}: {texts.get(cid, '')[:110]!r}\n      parser: {a}\n      ref   : {b}")


def run_parser(model: str, ids: list[str], texts: dict) -> dict:
    from vtas.claims import ClaimParser
    parser = ClaimParser(model=model)
    return {cid: [(c.type, c.text) for c in parser.parse(texts[cid]).claims] for cid in ids}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--make-split", action="store_true")
    ap.add_argument("--model", default="sm", choices=["sm", "trf"])
    ap.add_argument("--part", default="dev", choices=["dev", "heldout", "all"])
    ap.add_argument("--failures", action="store_true", help="list unmatched claims (dev only)")
    ap.add_argument("--json", help="write the summary JSON here")
    a = ap.parse_args(argv)
    if a.make_split:
        if SPLIT_PATH.exists():
            raise SystemExit(f"{SPLIT_PATH} exists; the split is fixed and is not overwritten")
        SPLIT_PATH.write_text(json.dumps(make_split(), indent=1) + "\n", encoding="utf-8")
        print("wrote", SPLIT_PATH)
        return
    if a.failures and a.part != "dev":
        raise SystemExit("--failures is only allowed for --part dev (held-out is never inspected)")
    design, ref = load_design(), load_claims()
    texts = load_caption_texts(design)
    split = load_split()
    ids = split["dev"] + split["heldout"] if a.part == "all" else split[a.part]
    pred = run_parser(a.model, ids, texts)
    res = evaluate(pred, ref, ids)
    summ = summarise(res)
    print(f"model={a.model} part={a.part} captions={len(ids)}")
    print_report(summ, bootstrap_ci(pred, ref, ids))
    if a.failures:
        print_failures(res, texts)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"model": a.model, "part": a.part, **summ}, f, indent=1)


if __name__ == "__main__":
    main()
