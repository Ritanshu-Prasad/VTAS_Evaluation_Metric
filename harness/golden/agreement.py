"""Inter-annotator agreement on the golden set: Cohen's kappa and Krippendorff's alpha.

    python -m harness.golden.agreement annotator_A.xlsx annotator_B.csv \
        [--field verdict] [--out summary.json]

Inputs are two annotators' files (.csv or .xlsx, sheet "claims") with at least the
columns ``claim_id`` and the labelled field (default ``verdict``). Annotator files are
blinded (claim ids start with an opaque caption id such as ``g042``); source and block
are looked up in harness/golden/golden_ids.json for the per-source breakdown and
``--blocks``. Rows are matched on
``claim_id``. A claim counts only when both annotators labelled it; claims missing from
one file or left blank are reported, not imputed. Claude pre-label columns are ignored:
Claude is never an annotator.

Statistics (nominal data):
  * Cohen's kappa (two raters):  (p_o - p_e) / (1 - p_e).
  * Krippendorff's alpha, nominal, from the coincidence matrix (Krippendorff 2011,
    "Computing Krippendorff's alpha-reliability"); handles any number of coders and
    missing values, so units labelled by only one coder are dropped automatically.
  * 95% percentile bootstrap CI for kappa, resampling images (claims of one caption,
    and the three captions of a paired image, are not independent; the unit is the
    ``image_id`` column, else the caption), 1,000 resamples, seed 0.
The gate F threshold for ``annotator_cohen_kappa_verdicts`` is read from
harness/gates.yaml, never hard-coded here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Hashable, Iterable, Sequence

import numpy as np

from harness import HARNESS_DIR

VERDICTS = ("supported", "contradicted", "unverifiable")
CLAIM_TYPES = ("object", "count", "attribute", "relation", "scene")
ALLOWED = {"verdict": VERDICTS, "type": CLAIM_TYPES}
SHEET = "claims"
GATES_PATH = HARNESS_DIR / "gates.yaml"


# ---- statistics ------------------------------------------------------------------------

def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    """Cohen's kappa for two raters on the same items. NaN when chance agreement is 1."""
    if len(a) != len(b):
        raise ValueError("raters must label the same number of items")
    n = len(a)
    if n == 0:
        return math.nan
    p_o = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    p_e = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    if p_e == 1.0:
        return math.nan
    return (p_o - p_e) / (1.0 - p_e)


def krippendorff_alpha_nominal(units: Iterable[Sequence[Hashable | None]]) -> float:
    """Krippendorff's alpha for nominal data.

    ``units`` is an iterable of per-unit value lists, one entry per coder, with ``None``
    for a missing value. Units with fewer than two values are not pairable and are
    dropped. NaN when fewer than two distinct values occur (alpha is undefined).
    """
    o: dict[tuple, float] = defaultdict(float)   # coincidence matrix
    for unit in units:
        vals = [v for v in unit if v is not None]
        m = len(vals)
        if m < 2:
            continue
        for i, c in enumerate(vals):
            for j, k in enumerate(vals):
                if i != j:
                    o[(c, k)] += 1.0 / (m - 1)
    n_c: dict[Hashable, float] = defaultdict(float)
    for (c, _k), v in o.items():
        n_c[c] += v
    n = sum(n_c.values())
    if n <= 1 or len(n_c) < 2:
        return math.nan
    d_o = sum(v for (c, k), v in o.items() if c != k)
    d_e = sum(n_c[c] * n_c[k] for c in n_c for k in n_c if c != k)
    return 1.0 - (n - 1) * d_o / d_e


def confusion(a: Sequence[str], b: Sequence[str], labels: Sequence[str]) -> list[list[int]]:
    idx = {l: i for i, l in enumerate(labels)}
    m = [[0] * len(labels) for _ in labels]
    for x, y in zip(a, b):
        m[idx[x]][idx[y]] += 1
    return m


def cluster_bootstrap_kappa(a: Sequence[str], b: Sequence[str], clusters: Sequence[Hashable],
                            n_resamples: int = 1000, ci: float = 0.95, seed: int = 0) -> dict:
    """Percentile CI for kappa, resampling clusters (images) with replacement."""
    groups: dict[Hashable, list[int]] = defaultdict(list)
    for i, c in enumerate(clusters):
        groups[c].append(i)
    keys = sorted(groups, key=str)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_resamples):
        pick = rng.integers(0, len(keys), size=len(keys))
        idx = [i for p in pick for i in groups[keys[p]]]
        k = cohen_kappa([a[i] for i in idx], [b[i] for i in idx])
        if not math.isnan(k):
            samples.append(k)
    point = cohen_kappa(a, b)
    if not samples:
        return {"point": point, "low": math.nan, "high": math.nan, "level": ci}
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(samples, [alpha, 1.0 - alpha])
    return {"point": point, "low": float(lo), "high": float(hi), "level": ci,
            "resamples": n_resamples, "seed": seed, "unit": "image"}


# ---- file handling ---------------------------------------------------------------------

def read_annotations(path: str | Path) -> list[dict[str, str]]:
    """Rows of an annotation file (.csv, or .xlsx sheet "claims") as string dicts."""
    path = Path(path)
    if path.suffix.lower() == ".xlsx":
        from openpyxl import load_workbook

        wb = load_workbook(path, read_only=True, data_only=True)
        ws = wb[SHEET] if SHEET in wb.sheetnames else wb.worksheets[0]
        rows = ws.iter_rows(values_only=True)
        header = [str(h).strip() if h is not None else "" for h in next(rows)]
        out = [{h: ("" if v is None else str(v)) for h, v in zip(header, r)} for r in rows]
        wb.close()
        return [r for r in out if r.get("claim_id", "").strip()]
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [r for r in csv.DictReader(f) if (r.get("claim_id") or "").strip()]


def _norm(value: str | None) -> str:
    return (value or "").strip().lower()


def pair_labels(rows_a: list[dict], rows_b: list[dict], field: str = "verdict") -> dict:
    """Match two annotators' rows on claim_id and validate the labels."""
    allowed = ALLOWED.get(field)

    def index(rows, who):
        d = {}
        for r in rows:
            cid = r["claim_id"].strip()
            if cid in d:
                raise ValueError(f"annotator {who}: duplicate claim_id {cid!r}")
            v = _norm(r.get(field))
            if v and allowed and v not in allowed:
                raise ValueError(f"annotator {who}: claim {cid!r} has {field}={v!r}; "
                                 f"allowed: {', '.join(allowed)}")
            d[cid] = (v, r)
        return d

    ia, ib = index(rows_a, "A"), index(rows_b, "B")
    common = sorted(set(ia) & set(ib))
    both = [c for c in common if ia[c][0] and ib[c][0]]
    return {
        "claim_ids": both,
        "a": [ia[c][0] for c in both],
        "b": [ib[c][0] for c in both],
        "rows": [ia[c][1] for c in both],
        "only_in_a": sorted(set(ia) - set(ib)),
        "only_in_b": sorted(set(ib) - set(ia)),
        "unlabelled_a": sum(1 for c in common if not ia[c][0]),
        "unlabelled_b": sum(1 for c in common if not ib[c][0]),
    }


def _cluster_of(claim_id: str, row: dict) -> str:
    """Bootstrap unit: the image, else the caption (claim ids are '<caption_id>#NN')."""
    return ((row.get("image_id") or "").strip() or (row.get("caption_id") or "").strip()
            or claim_id.split("#")[0])


def attach_key(rows: list[dict], design: dict | None) -> list[dict]:
    """Fill source / block / image_id for blinded rows from the design key (gid prefix of
    claim_id). Rows that already carry a source are left as they are."""
    if not design:
        return rows
    by_gid = {it["gid"]: it for it in design.get("items", []) if "gid" in it}
    out = []
    for r in rows:
        it = by_gid.get(r["claim_id"].strip().split("#")[0])
        if it and not (r.get("source") or "").strip():
            r = {**r, "source": it["source"], "block": it["block"],
                 "image_id": r.get("image_id") or str(it["image_id"])}
        out.append(r)
    return out


def _default_design() -> dict | None:
    path = HARNESS_DIR / "golden" / "golden_ids.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def gate_threshold(path: Path = GATES_PATH) -> float | None:
    """Gate F threshold for annotator kappa on verdicts, from gates.yaml."""
    try:
        import yaml
    except ImportError:
        return None
    with open(path, encoding="utf-8") as f:
        gates = yaml.safe_load(f)
    for crit in gates.get("gates", {}).get("F", {}).get("criteria", []):
        if crit.get("metric") == "annotator_cohen_kappa_verdicts":
            return float(crit["threshold"])
    return None


def agreement(rows_a: list[dict], rows_b: list[dict], field: str = "verdict",
              n_resamples: int = 1000, seed: int = 0) -> dict:
    p = pair_labels(rows_a, rows_b, field)
    a, b, rows = p["a"], p["b"], p["rows"]
    labels = list(ALLOWED.get(field) or sorted(set(a) | set(b)))

    def block(idx: list[int]) -> dict:
        xa, xb = [a[i] for i in idx], [b[i] for i in idx]
        return {
            "n": len(idx),
            "percent_agreement": (sum(x == y for x, y in zip(xa, xb)) / len(idx)) if idx else math.nan,
            "cohen_kappa": cohen_kappa(xa, xb),
            "krippendorff_alpha": krippendorff_alpha_nominal(zip(xa, xb)),
        }

    everything = list(range(len(a)))
    out = {
        "field": field,
        **block(everything),
        "kappa_ci": cluster_bootstrap_kappa(
            a, b, [_cluster_of(c, r) for c, r in zip(p["claim_ids"], rows)], n_resamples, seed=seed),
        "labels": labels,
        "confusion_rows_A_cols_B": confusion(a, b, labels),
        "by_source": {},
        "by_type": {},
        "excluded": {
            "only_in_a": len(p["only_in_a"]), "only_in_b": len(p["only_in_b"]),
            "unlabelled_a": p["unlabelled_a"], "unlabelled_b": p["unlabelled_b"],
        },
    }
    for key, col in (("by_source", "source"), ("by_type", "type")):
        groups: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(rows):
            g = _norm(r.get(col))
            if g:
                groups[g].append(i)
        out[key] = {g: block(idx) for g, idx in sorted(groups.items())}
    if field == "verdict":
        thr = gate_threshold()
        k = out["cohen_kappa"]
        out["gate_F_kappa"] = {
            "metric": "annotator_cohen_kappa_verdicts", "threshold": thr,
            "pass": (None if thr is None or math.isnan(k) else bool(k >= thr)),
            "note": "Valid for the gate only on the full core set (300 captions, "
                    "blocks P and U), labelled independently by two humans.",
        }
    return out


def _clean(obj):
    """NaN -> None so the JSON is valid."""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(prog="python -m harness.golden.agreement")
    p.add_argument("annotator_a")
    p.add_argument("annotator_b")
    p.add_argument("--field", default="verdict", help="column to compare (verdict, type, ...)")
    p.add_argument("--blocks", default=None,
                   help="comma list of blocks to keep, e.g. P,U (default: all rows)")
    p.add_argument("--resamples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="write the summary JSON here")
    args = p.parse_args(argv)

    design = _default_design()
    rows_a = attach_key(read_annotations(args.annotator_a), design)
    rows_b = attach_key(read_annotations(args.annotator_b), design)
    if args.blocks:
        keep = {b.strip().upper() for b in args.blocks.split(",")}
        rows_a = [r for r in rows_a if (r.get("block") or "").strip().upper() in keep]
        rows_b = [r for r in rows_b if (r.get("block") or "").strip().upper() in keep]
    res = _clean(agreement(rows_a, rows_b, args.field, args.resamples, args.seed))
    res["inputs"] = {"annotator_a": Path(args.annotator_a).name,
                     "annotator_b": Path(args.annotator_b).name, "blocks": args.blocks}
    text = json.dumps(res, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    ci = res["kappa_ci"]
    print(f"{args.field}: n={res['n']}  kappa={res['cohen_kappa']}  "
          f"[{ci['low']}, {ci['high']}]  alpha={res['krippendorff_alpha']}  "
          f"agreement={res['percent_agreement']}")
    if "gate_F_kappa" in res:
        print(f"gate F (kappa >= {res['gate_F_kappa']['threshold']}): {res['gate_F_kappa']['pass']}")
    return res


if __name__ == "__main__":
    main()
