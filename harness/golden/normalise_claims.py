"""Apply the agreed annotation conventions (data/golden/conventions_decision.xlsx, 14 rows) to the
claims_src lists. Deterministic; back up claims_src first. Rewrites the files in place and prints
a change report.

    python -m harness.golden.normalise_claims [--dry-run]

Choices applied (A = recommended unless noted): posture verbs -> attribute (animate subjects only)
+ bare preposition; parked/posted -> attribute + bare preposition; "walks with" -> "with" +
attribute; group words keep one count claim; "piece of X" -> X; "in <clothing>" -> wearing;
body-part rows dropped; coordinated subjects merged; attribute verbs in -ing form; row 12 = B:
a room-type place gets one combined relation "bathroom / with / toilet and sink" (3+: "toilet, sink and mirror").
Not automated (left as drafted): row 11 "next to" attachment, row 8 compound names.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from harness.golden.annotate import CLAIMS_SRC, TYPE_CODES, load_caption_texts, parse_claims
from harness.golden.design import load_design

LEX = json.loads((Path(__file__).resolve().parents[2] / "src/vtas/claims/lexicon.json").read_text(encoding="utf-8"))
ANIMATE = set(LEX["animate_nouns"]) | {"men", "women", "children", "kids", "people", "persons", "ladies", "guys"}
ROOMS = {"kitchen", "bathroom", "bedroom", "living room", "room", "church", "garden", "station",
         "train station", "market", "street market", "museum"}
BODY = set(LEX["body_parts"]) - {"back"}
CLOTHING = set(LEX["clothing"])
POSTURE = {"sit": "sitting", "sits": "sitting", "sitting": "sitting", "sat": "sitting",
           "stand": "standing", "stands": "standing", "standing": "standing",
           "lay": "laying", "lays": "laying", "laying": "laying", "lie": "lying", "lies": "lying",
           "lying": "lying"}
STATE = {"parked": "parked", "posted": "posted"}
GERUND = {"walks": "walking", "walk": "walking", "sits": "sitting", "sit": "sitting", "stands": "standing",
          "stand": "standing", "lays": "laying", "lies": "lying", "rides": "riding", "runs": "running",
          "flies": "flying", "jumps": "jumping", "swims": "swimming", "sleeps": "sleeping"}
SCENE_ALIAS = {"bath room": "bathroom", "bed room": "bedroom"}
CODE = {v: k for k, v in TYPE_CODES.items()}
STOP_AFTER = r"(?:on|in|at|next|near|by|under|over|behind|beside|with|sitting|standing|while|that|which)\b"
PIECE = re.compile(r"\bpieces? of ")


def animate(entity: str) -> bool:
    w = entity.lower().replace(" and ", " ").split()
    tail = " ".join(w[-2:])
    return any(x in ANIMATE or x.rstrip("s") in ANIMATE for x in w[-2:]) or tail in ANIMATE


def normalise_caption(claims: list, caption: str, log: Counter) -> list:
    out, seen = [], set()

    def add(t, x):
        if (t, x) not in seen:
            seen.add((t, x))
            out.append((t, x))

    for t, x in claims:
        if t == "attribute":
            ent, _, val = x.partition(": ")
            w = val.split()
            if w and w[0] in POSTURE and not animate(ent):
                log["inanimate posture attribute dropped"] += 1
                continue
            if len(w) == 1 and w[0] in GERUND:
                log["attribute verb -> -ing"] += 1
                add(t, f"{ent}: {GERUND[w[0]]}")
                continue
            if ent.split()[-1] in BODY:
                log["body-part row dropped"] += 1
                continue
            add(t, PIECE.sub("", x))
        elif t == "relation":
            s, p, o = [q.strip() for q in x.split(" / ")]
            if re.search(r"\b(its|his|her|their) (mouth|hand|hands|head|paws?)\b", p) or o.split()[-1] in BODY \
                    or s.split()[-1] in BODY:
                log["body-part row dropped"] += 1
                continue
            w = p.split()
            if w[0] in POSTURE and len(w) > 1:
                add("relation", f"{s} / {' '.join(w[1:])} / {o}")
                if animate(s):
                    add("attribute", f"{s}: {POSTURE[w[0]]}")
                log["posture verb -> attribute + bare preposition"] += 1
                continue
            if w[0] in STATE and len(w) > 1:
                add("relation", f"{s} / {' '.join(w[1:])} / {o}")
                add("attribute", f"{s}: {STATE[w[0]]}")
                log["parked/posted split"] += 1
                continue
            if p in ("walks with", "walking with", "walk with"):
                add("relation", f"{s} / with / {o}")
                add("attribute", f"{s}: walking")
                log["walks with -> with + walking"] += 1
                continue
            if p == "in" and o.split()[-1].rstrip("s") in CLOTHING:
                add("relation", f"{s} / wearing / {o}")
                log["in clothing -> wearing"] += 1
                continue
            add(t, PIECE.sub("", x))
        elif t == "object":
            if x.split()[-1] in BODY:
                log["body-part row dropped"] += 1
                continue
            add(t, PIECE.sub("", x))
        else:
            add(t, x)
    # group words: keep the one count claim ("a group of children"), drop the bare plural twin
    for t, x in [r for r in out if r[0] == "count"]:
        m = re.search(r"\b(?:group|bunch|herd|crowd|couple|pair|flock) of (.+)$", x)
        if m and ("count", m.group(1)) in seen and m.group(1) != x:
            out.remove(("count", m.group(1)))
            seen.discard(("count", m.group(1)))
            log["duplicate plural count removed"] += 1
    # coordinated subjects: "a man and a woman standing"
    by_val: dict = {}
    for t, x in out:
        if t == "attribute":
            ent, _, val = x.partition(": ")
            by_val.setdefault(val, []).append(ent)
    cap = caption.lower()
    for val, ents in by_val.items():
        if len(ents) == 2 and re.search(rf"\b{re.escape(ents[0])}s? and (a |an |the )?{re.escape(ents[1])}s?\b", cap):
            a, b = ents
            out = [(t, x) for t, x in out if not (t == "attribute" and x in (f"{a}: {val}", f"{b}: {val}"))]
            out.append(("attribute", f"{a} and {b}: {val}"))
            log["coordinated subjects merged"] += 1
    # place with ...: "bathroom with a toilet and sink" (row 12 = B)
    cap2 = cap
    for k, v in SCENE_ALIAS.items():
        cap2 = cap2.replace(k, v)
    objs = [x for t, x in out if t == "object"]
    for _, sc in [r for r in out if r[0] == "scene" and r[1] in ROOMS]:
        m = re.search(rf"\b{re.escape(sc)} with (.+?)(?:\s{STOP_AFTER}|[.]|$)", cap2)
        if not m:
            continue
        seg = m.group(1)
        found = sorted((seg.find(o), o) for o in objs if re.search(rf"\b{re.escape(o)}\b", seg))
        found = [o for i, o in found if i >= 0]
        if found:
            joined = found[0] if len(found) == 1 else ", ".join(found[:-1]) + " and " + found[-1]
            add("relation", f"{sc} / with / {joined}")
            log["place with ... relation added"] += 1
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    texts = load_caption_texts(load_design())
    log, n_before, n_after = Counter(), 0, 0
    for f in sorted(CLAIMS_SRC.glob("*.txt")):
        raw = f.read_text(encoding="utf-8")
        parsed = parse_claims(raw)
        lines = []
        for line in raw.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                lines.append(line)
                continue
            cid = s.split("|")[0].strip()
            new = normalise_caption(parsed[cid], texts[cid], log)
            n_before += len(parsed[cid])
            n_after += len(new)
            lines.append("|".join([cid] + [f"{CODE[t]} {x}" for t, x in new]))
        if not a.dry_run:
            f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"claims {n_before} -> {n_after}")
    for k, v in log.most_common():
        print(f"  {v:4d}  {k}")


if __name__ == "__main__":
    main()
