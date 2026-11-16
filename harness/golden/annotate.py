"""Golden-set annotation files (plan S0.6).

    python -m harness.golden.annotate template          # committed blank template
    python -m harness.golden.annotate build [--force] [--partial]  # annotator + adjudication files
    python -m harness.golden.annotate merge A.xlsx B.xlsx  # copy A/B labels into adjudication

Inputs
  * harness/golden/golden_ids.json            which captions (see design.py)
  * harness/golden/claims_src/<source>.txt    frozen claim segmentation (units, no verdicts)
  * caption text: $VTAS_DATA/captions/blip_base_paper_dev.json (BLIP),
    $VTAS_DATA/captions/llava15_7b_golden.json (LLaVA-1.5, Kaggle run) and the official
    captions_val2014.json (human references).
Outputs (under $VTAS_DATA/golden/, never committed: they hold caption text and labels)
  * annotator_A.xlsx, annotator_B.xlsx - identical, blinded (opaque caption ids, no
    source, no Claude column). Sheets: README, claims, missing_claims.
  * adjudication.xlsx - real caption ids and sources, both annotators' labels side by
    side, Claude pre-labels in their OWN columns, and the final (adjudicated) columns.
Claude pre-labels are never an annotator: they are not in the annotator files and the
agreement script never reads them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness import HARNESS_DIR, data_dir
from harness.golden.agreement import CLAIM_TYPES, VERDICTS, read_annotations
from harness.golden.design import load_design

CLAIMS_SRC = HARNESS_DIR / "golden" / "claims_src"
TEMPLATE_DIR = HARNESS_DIR / "golden" / "templates"
TYPE_CODES = {"o": "object", "c": "count", "a": "attribute", "r": "relation", "s": "scene"}
FLAGS = ("not_a_claim", "wrong_type", "should_split", "should_merge", "unclear_text")
IMAGE_URL = "http://images.cocodataset.org/val2014/COCO_val2014_{:012d}.jpg"
RUBRIC = "docs/GOLDEN_SET_RUBRIC.md"

ANNOTATOR_COLUMNS = ["claim_id", "caption_gid", "image_id", "image_url", "caption", "claim_no",
                     "type", "claim_text", "verdict", "visible_count", "flag", "note"]
MISSING_COLUMNS = ["caption_gid", "type", "claim_text", "verdict", "visible_count", "note"]
ADJ_COLUMNS = ["claim_id", "caption_gid", "caption_id", "source", "block", "image_id",
               "image_url", "caption", "claim_no", "type", "claim_text",
               "verdict_A", "verdict_B", "visible_count_A", "visible_count_B",
               "flag_A", "flag_B", "note_A", "note_B",
               "claude_prelabel_verdict", "claude_prelabel_visible_count", "claude_prelabel_note",
               "final_verdict", "final_visible_count", "adjudication_note"]
WIDTHS = {"caption": 48, "claim_text": 36, "note": 30, "image_url": 22, "adjudication_note": 30}


# ---- claims ------------------------------------------------------------------------------

def parse_claims(text: str) -> dict[str, list[tuple[str, str]]]:
    """caption_id -> [(type, claim_text), ...] from the claims_src line format."""
    out: dict[str, list[tuple[str, str]]] = {}
    for n, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cid, *parts = [p.strip() for p in line.split("|")]
        if cid in out:
            raise ValueError(f"line {n}: duplicate caption {cid!r}")
        claims = []
        for p in parts:
            code, _, body = p.partition(" ")
            if code not in TYPE_CODES or not body.strip():
                raise ValueError(f"line {n}: bad claim {p!r} (want '<o|c|a|r|s> text')")
            claims.append((TYPE_CODES[code], body.strip()))
        if not claims:
            raise ValueError(f"line {n}: caption {cid!r} has no claims")
        out[cid] = claims
    return out


def load_claims(src_dir: Path = CLAIMS_SRC) -> dict[str, list[tuple[str, str]]]:
    claims: dict[str, list[tuple[str, str]]] = {}
    for f in sorted(src_dir.glob("*.txt")):
        for cid, c in parse_claims(f.read_text(encoding="utf-8")).items():
            if cid in claims:
                raise ValueError(f"{f.name}: caption {cid!r} already segmented in another file")
            claims[cid] = c
    return claims


# ---- caption text ------------------------------------------------------------------------

def load_caption_texts(design: dict, root: Path | None = None) -> dict[str, str]:
    """caption_id -> text for every caption whose source file is present."""
    root = root or data_dir()
    texts: dict[str, str] = {}
    blip_p = root / "captions" / "blip_base_paper_dev.json"
    llava_p = root / "captions" / "llava15_7b_golden.json"
    refs_p = root / "coco" / "annotations" / "captions_val2014.json"
    blip = json.loads(blip_p.read_text(encoding="utf-8"))["captions"] if blip_p.exists() else {}
    llava = json.loads(llava_p.read_text(encoding="utf-8"))["captions"] if llava_p.exists() else {}
    refs = {}
    if refs_p.exists():
        with open(refs_p, encoding="utf-8") as f:
            refs = {a["id"]: a["caption"] for a in json.load(f)["annotations"]}
    for it in design["items"]:
        img, src = str(it["image_id"]), it["source"]
        if src == "blip" and img in blip:
            texts[it["caption_id"]] = blip[img]
        elif src.startswith("llava_") and img in llava.get(src.split("_", 1)[1], {}):
            texts[it["caption_id"]] = llava[src.split("_", 1)[1]][img]
        elif src == "human" and it.get("ref_annotation_id") in refs:
            texts[it["caption_id"]] = refs[it["ref_annotation_id"]]
    return {k: " ".join(v.split()) for k, v in texts.items()}


def build_rows(design: dict, texts: dict[str, str], claims: dict) -> tuple[list[dict], dict]:
    """Rows in gid order, plus a report of captions that could not be included."""
    rows, missing_text, missing_claims = [], [], []
    for it in sorted(design["items"], key=lambda i: i["gid"]):
        cid = it["caption_id"]
        if cid not in texts:
            missing_text.append(cid)
            continue
        if cid not in claims:
            missing_claims.append(cid)
            continue
        for k, (ctype, ctext) in enumerate(claims[cid], 1):
            rows.append({
                "claim_id": f"{it['gid']}#{k:02d}", "caption_gid": it["gid"], "caption_id": cid,
                "source": it["source"], "block": it["block"], "image_id": it["image_id"],
                "image_url": IMAGE_URL.format(it["image_id"]), "caption": texts[cid],
                "claim_no": k, "type": ctype, "claim_text": ctext,
            })
    unused = sorted(set(claims) - {i["caption_id"] for i in design["items"]})
    return rows, {"no_caption_text": missing_text, "not_segmented": missing_claims,
                  "claims_for_unknown_captions": unused}


# ---- workbook writing --------------------------------------------------------------------

README_LINES = [
    "VTAS golden set - annotator workbook (plan S0.6)",
    "",
    f"Read the rubric first: {RUBRIC} in the repository.",
    "Label every row of the 'claims' sheet on your own. Do not discuss labels with the other",
    "annotator until both files are finished.",
    "",
    "verdict: supported | contradicted | unverifiable (dropdown)",
    "visible_count: only for type = count. Integer number of visible instances; if more than",
    "  20, your best estimate; leave blank and explain in note if impossible to count.",
    "flag: only if the claim unit itself is wrong (not_a_claim, wrong_type, should_split,",
    "  should_merge, unclear_text). Still give a verdict for the claim as written.",
    "note: free text. Required for every contradicted or unverifiable verdict.",
    "missing_claims sheet: claims the caption makes that have no row. They are adjudicated",
    "  but do not enter the agreement statistics.",
    "Do not edit, sort-and-save, insert or delete rows or the claim_id column.",
]


def _write_sheet(ws, columns: list[str], rows: list[dict]) -> None:
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    ws.append(columns)
    for c in ws[1]:
        c.font = Font(bold=True)
        c.fill = PatternFill("solid", fgColor="DDDDDD")
    for r in rows:
        ws.append([r.get(c, "") for c in columns])
    ws.freeze_panes = "A2"
    for i, col in enumerate(columns, 1):
        ws.column_dimensions[get_column_letter(i)].width = WIDTHS.get(col, max(10, len(col) + 2))
        if col in ("caption", "claim_text", "note", "adjudication_note"):
            for cell in ws[get_column_letter(i)][1:]:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
    if "image_url" in columns:
        for cell in ws[get_column_letter(columns.index("image_url") + 1)][1:]:
            if cell.value:
                cell.hyperlink = cell.value


def _add_list_validation(ws, columns: list[str], col: str, values, n_rows: int) -> None:
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation

    if col not in columns:
        return
    letter = get_column_letter(columns.index(col) + 1)
    dv = DataValidation(type="list", formula1='"' + ",".join(values) + '"', allow_blank=True,
                        showErrorMessage=True, errorTitle="Invalid value",
                        error="Choose a value from the list.")
    ws.add_data_validation(dv)
    dv.add(f"{letter}2:{letter}{max(n_rows + 1, 2) + 200}")


def write_annotator_workbook(path: Path, rows: list[dict]) -> None:
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "README"
    for line in README_LINES:
        ws.append([line])
    ws.column_dimensions["A"].width = 100
    claims = wb.create_sheet("claims")
    _write_sheet(claims, ANNOTATOR_COLUMNS, rows)
    _add_list_validation(claims, ANNOTATOR_COLUMNS, "verdict", VERDICTS, len(rows))
    _add_list_validation(claims, ANNOTATOR_COLUMNS, "flag", FLAGS, len(rows))
    miss = wb.create_sheet("missing_claims")
    _write_sheet(miss, MISSING_COLUMNS, [])
    _add_list_validation(miss, MISSING_COLUMNS, "type", CLAIM_TYPES, 0)
    _add_list_validation(miss, MISSING_COLUMNS, "verdict", VERDICTS, 0)
    wb.active = 1
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def write_adjudication_workbook(path: Path, rows: list[dict]) -> None:
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "claims"
    _write_sheet(ws, ADJ_COLUMNS, rows)
    _add_list_validation(ws, ADJ_COLUMNS, "final_verdict", VERDICTS, len(rows))
    _add_list_validation(ws, ADJ_COLUMNS, "claude_prelabel_verdict", VERDICTS, len(rows))
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


# ---- commands ----------------------------------------------------------------------------

TEMPLATE_ROWS = [  # illustrative only, not from the golden set
    {"claim_id": "g000#01", "caption_gid": "g000", "image_id": 0, "image_url": "",
     "caption": "two dogs playing on a red couch", "claim_no": 1, "type": "object",
     "claim_text": "dogs", "verdict": "supported", "visible_count": "", "flag": "", "note": ""},
    {"claim_id": "g000#02", "caption_gid": "g000", "image_id": 0, "image_url": "",
     "caption": "two dogs playing on a red couch", "claim_no": 2, "type": "count",
     "claim_text": "two dogs", "verdict": "contradicted", "visible_count": 3, "flag": "",
     "note": "three dogs visible; third partly behind cushion"},
    {"claim_id": "g000#03", "caption_gid": "g000", "image_id": 0, "image_url": "",
     "caption": "two dogs playing on a red couch", "claim_no": 3, "type": "attribute",
     "claim_text": "dogs: playing", "verdict": "unverifiable", "visible_count": "", "flag": "",
     "note": "dogs lying still; playing cannot be confirmed or ruled out from one frame"},
    {"claim_id": "g000#04", "caption_gid": "g000", "image_id": 0, "image_url": "",
     "caption": "two dogs playing on a red couch", "claim_no": 4, "type": "relation",
     "claim_text": "dogs / on / couch", "verdict": "supported", "visible_count": "",
     "flag": "", "note": ""},
    {"claim_id": "g000#05", "caption_gid": "g000", "image_id": 0, "image_url": "",
     "caption": "two dogs playing on a red couch", "claim_no": 5, "type": "attribute",
     "claim_text": "couch: red", "verdict": "contradicted", "visible_count": "", "flag": "",
     "note": "couch is clearly brown"},
]


def cmd_template(out_dir: Path = TEMPLATE_DIR) -> list[Path]:
    import csv

    out_dir.mkdir(parents=True, exist_ok=True)
    xlsx = out_dir / "golden_annotation_template.xlsx"
    write_annotator_workbook(xlsx, TEMPLATE_ROWS)
    csv_p = out_dir / "golden_annotation_template.csv"
    with open(csv_p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ANNOTATOR_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(TEMPLATE_ROWS)
    return [xlsx, csv_p]


def cmd_build(force: bool = False, partial: bool = False, out_dir: Path | None = None,
              design: dict | None = None, texts: dict | None = None,
              claims: dict | None = None) -> dict:
    """Write the annotator and adjudication workbooks.

    Refuses when any golden-set caption lacks text or segmentation, unless ``partial``;
    a partial build goes to ``<out_dir>/preview/`` and must not be labelled.
    """
    design = design if design is not None else load_design()
    texts = texts if texts is not None else load_caption_texts(design)
    claims = claims if claims is not None else load_claims()
    rows, report = build_rows(design, texts, claims)
    incomplete = bool(report["no_caption_text"] or report["not_segmented"])
    if report["claims_for_unknown_captions"]:
        raise SystemExit(f"claims for captions not in the design: "
                         f"{report['claims_for_unknown_captions'][:5]}")
    if incomplete and not partial:
        raise SystemExit(f"golden set incomplete: {len(report['no_caption_text'])} captions "
                         f"without text, {len(report['not_segmented'])} not segmented. "
                         "Use --partial for a preview (not for labelling).")
    out_dir = out_dir or data_dir() / "golden"
    if partial:
        out_dir = out_dir / "preview"
    targets = [out_dir / "annotator_A.xlsx", out_dir / "annotator_B.xlsx",
               out_dir / "adjudication.xlsx"]
    existing = [t for t in targets if t.exists()]
    if existing and not (force or partial):
        raise SystemExit(f"refusing to overwrite {', '.join(t.name for t in existing)} "
                         "(labels may be in progress); pass --force")
    blinded = [{k: r[k] for k in ANNOTATOR_COLUMNS if k in r} for r in rows]
    write_annotator_workbook(targets[0], blinded)
    write_annotator_workbook(targets[1], blinded)
    write_adjudication_workbook(targets[2], rows)
    return {
        "complete": not incomplete,
        "n_claims": len(rows),
        "n_captions": len({r["caption_id"] for r in rows}),
        "by_source": {s: sum(r["source"] == s for r in rows)
                      for s in ("blip", "llava_short", "human", "llava_long")},
        "by_type": {t: sum(r["type"] == t for r in rows) for t in CLAIM_TYPES},
        "captions_without_text": _summarise(report["no_caption_text"]),
        "captions_not_segmented": _summarise(report["not_segmented"]),
        "out_dir": str(out_dir),
    }


def _summarise(caption_ids: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in caption_ids:
        src = c.rsplit("-", 1)[0]
        out[src] = out.get(src, 0) + 1
    return out


def cmd_merge(file_a: str, file_b: str, adjudication: Path | None = None) -> int:
    """Copy both annotators' labels into adjudication.xlsx (A/B columns only)."""
    from openpyxl import load_workbook

    adjudication = adjudication or data_dir() / "golden" / "adjudication.xlsx"
    labels = {}
    for tag, f in (("A", file_a), ("B", file_b)):
        labels[tag] = {r["claim_id"]: r for r in read_annotations(f)}
    wb = load_workbook(adjudication)
    ws = wb["claims"]
    header = [c.value for c in ws[1]]
    col = {h: i for i, h in enumerate(header)}
    n = 0
    for row in ws.iter_rows(min_row=2):
        cid = row[col["claim_id"]].value
        for tag in ("A", "B"):
            r = labels[tag].get(cid)
            if r is None:
                continue
            for field in ("verdict", "visible_count", "flag", "note"):
                row[col[f"{field}_{tag}"]].value = (r.get(field) or "").strip() or None
        n += 1
    wb.save(adjudication)
    return n


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m harness.golden.annotate")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("template")
    b = sub.add_parser("build")
    b.add_argument("--force", action="store_true", help="overwrite existing annotator files")
    b.add_argument("--partial", action="store_true",
                   help="preview build when captions are missing (written to preview/)")
    m = sub.add_parser("merge")
    m.add_argument("annotator_a")
    m.add_argument("annotator_b")
    args = p.parse_args(argv)
    if args.cmd == "template":
        for path in cmd_template():
            print(f"wrote {path}")
    elif args.cmd == "build":
        print(json.dumps(cmd_build(args.force, args.partial), indent=1))
    else:
        print(f"merged labels into {cmd_merge(args.annotator_a, args.annotator_b)} rows")


if __name__ == "__main__":
    main()
