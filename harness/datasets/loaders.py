"""Loaders for the external evaluation sets. Each returns plain dicts.

Every ``load_*`` verifies the SHA-256 of its files (see ``harness.datasets.REGISTRY``)
before parsing. ``parse_*`` functions take file paths and do no checking, so they can be
unit-tested on small fixtures in the native format.
"""

from __future__ import annotations

import io
import json
import re
import zipfile

from harness.datasets import get

_COCO_ID = re.compile(r"(\d{12})\.jpg$")


def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---- FOIL -------------------------------------------------------------------------------

def parse_foil(path) -> list[dict]:
    """One row per caption: original captions have foil=False (target/foil word 'ORIG')."""
    return [{"image_id": a["image_id"], "caption": a["caption"], "foil": bool(a["foil"]),
             "target_word": a["target_word"], "foil_word": a["foil_word"],
             "id": a["id"], "foil_id": a["foil_id"]}
            for a in _read_json(path)["annotations"]]


def load_foil(verify: bool = True) -> list[dict]:
    return parse_foil(get("foil").local("foilv1.0_test_2017.json", verify))


# ---- THumB 1.0 ----------------------------------------------------------------------------

def parse_thumb(ratings_path, references_path) -> list[dict]:
    """One row per (system, image): P, R, penalties and human_score, with the references."""
    refs = {r["seg_id"]: r["refs"] for r in _read_jsonl(references_path)}
    rows = []
    for r in _read_jsonl(ratings_path):
        rows.append({"system": r["SYS"], "seg_id": r["seg_id"], "image": r["image"],
                     "image_id": int(_COCO_ID.search(r["image"]).group(1)),
                     "caption": r["hyp"], "precision": r["P"], "recall": r["R"],
                     "fluency": r["Fl"], "inclusive": r["Inc"], "conciseness": r["Con"],
                     "human_score": r["human_score"], "references": refs[r["seg_id"]]})
    return rows


def load_thumb(verify: bool = True) -> list[dict]:
    d = get("thumb")
    return parse_thumb(d.local("mscoco_THumB-1.0.jsonl", verify),
                       d.local("mscoco_references.json", verify))


# ---- Flickr8k-Expert and Flickr8k-CF ------------------------------------------------------

def _zip_text(z: zipfile.ZipFile, name: str) -> list[str]:
    with z.open(name) as f:
        return io.TextIOWrapper(f, encoding="utf-8").read().splitlines()


def parse_flickr8k(zip_path) -> dict[str, list[dict]]:
    """{"expert": [...], "crowdflower": [...]}.

    Expert rows: three expert scores (1-4) for an (image, candidate caption) pair; the
    candidate caption may belong to another image. CF rows: share of crowd "yes" votes.
    """
    with zipfile.ZipFile(zip_path) as z:
        captions = {}
        for line in _zip_text(z, "Flickr8k.token.txt"):
            if line.strip():
                cid, text = line.split("\t", 1)
                captions[cid] = text.strip()
        expert, cf = [], []
        for line in _zip_text(z, "ExpertAnnotations.txt"):
            if line.strip():
                img, cid, *scores = line.split("\t")
                expert.append({"image": img, "caption_id": cid, "caption": captions[cid],
                               "scores": [int(s) for s in scores]})
        for line in _zip_text(z, "CrowdFlowerAnnotations.txt"):
            if line.strip():
                img, cid, p_yes, n_yes, n_no = line.split("\t")
                cf.append({"image": img, "caption_id": cid, "caption": captions[cid],
                           "p_yes": float(p_yes), "n_yes": int(n_yes), "n_no": int(n_no)})
    return {"expert": expert, "crowdflower": cf}


def load_flickr8k(verify: bool = True) -> dict[str, list[dict]]:
    return parse_flickr8k(get("flickr8k").local("Flickr8k_text.zip", verify))


# ---- Pascal-50S ---------------------------------------------------------------------------

PASCAL_CATEGORIES = {1: "HC", 2: "HI", 3: "HM", 4: "MM"}


def _unwrap(x):
    """Strip MATLAB cell/array nesting (size-1 arrays) down to a Python str or number."""
    import numpy as np

    while isinstance(x, np.ndarray):
        if x.size == 0:
            return ""
        if x.size != 1:
            raise ValueError(f"expected a scalar cell, got shape {x.shape}")
        x = x.reshape(-1)[0]
    return x.item() if isinstance(x, np.generic) else x


def parse_pascal50s(pair_mat, consensus_mat, triplets_per_pair: int = 48) -> list[dict]:
    """One row per candidate pair with the human votes for A and B (ties kept as is).

    Follows the layout used by CLIPScore/Polos (jmhessel/clipscore issue 4):
    pair_pascal.mat: new_input[i] = (image, caption A, caption B), category[i] in 1..4;
    consensus_pascal.mat: triplets[j] = (reference, cand 1, cand 2, choice), 48 per pair.
    """
    from scipy.io import loadmat

    pairs = loadmat(pair_mat)
    data, cats = pairs["new_input"].reshape(-1), pairs["category"].reshape(-1)
    trip = loadmat(consensus_mat)["triplets"].reshape(-1)
    rows = []
    for i, item in enumerate(data):
        item = item.reshape(-1)
        image, cap_a, cap_b = (_unwrap(item[k]).strip() for k in range(3))
        votes, refs = {}, []
        for j in range(i * triplets_per_pair, (i + 1) * triplets_per_pair):
            ref, c1, c2, choice = (_unwrap(v) for v in trip[j].reshape(-1)[:4])
            refs.append(str(ref).strip())
            chosen = (c1 if int(choice) == 1 else c2).strip()
            votes[chosen] = votes.get(chosen, 0) + 1
        rows.append({"image": image, "caption_a": cap_a, "caption_b": cap_b,
                     "category": PASCAL_CATEGORIES[int(cats[i])],
                     "votes_a": votes.get(cap_a, 0), "votes_b": votes.get(cap_b, 0),
                     "references": refs})
    return rows


def load_pascal50s(verify: bool = True) -> list[dict]:
    d = get("pascal50s")
    return parse_pascal50s(d.local("pair_pascal.mat", verify),
                           d.local("consensus_pascal.mat", verify))


# ---- VSR -------------------------------------------------------------------------------

def parse_vsr(path) -> list[dict]:
    return [{"image": r["image"], "image_link": r["image_link"], "caption": r["caption"],
             "label": int(r["label"]), "relation": r["relation"]} for r in _read_jsonl(path)]


def load_vsr(split: str = "test", variant: str = "random", verify: bool = True) -> list[dict]:
    if variant not in ("random", "zeroshot") or split not in ("train", "dev", "test"):
        raise ValueError("variant in {random, zeroshot}, split in {train, dev, test}")
    return parse_vsr(get("vsr").local(f"{variant}_{split}.jsonl", verify))


# ---- What'sUp -----------------------------------------------------------------------------

WHATSUP_SUBSETS = {
    "controlled_a": "controlled_images_dataset.json",   # on / under / left / right
    "controlled_b": "controlled_clevr_dataset.json",    # left / right / front / behind
    "coco_qa_one": "coco_qa_one_obj.json",               # left / right / top / bottom
    "coco_qa_two": "coco_qa_two_obj.json",
    "vg_qa_one": "vg_qa_one_obj.json",
    "vg_qa_two": "vg_qa_two_obj.json",
}


def parse_whatsup(path) -> list[dict]:
    """Rows with the caption options; option 0 is the correct one in all six subsets
    (README of amitakamath/whatsup_vlms)."""
    rows = []
    for r in _read_json(path):
        if isinstance(r, dict):
            rows.append({"image": r["image_path"], "options": list(r["caption_options"]),
                         "correct": 0})
        else:
            rows.append({"image": str(r[0]), "options": list(r[1:]), "correct": 0})
    return rows


def load_whatsup(subset: str, verify: bool = True) -> list[dict]:
    return parse_whatsup(get("whatsup").local(WHATSUP_SUBSETS[subset], verify))


# ---- AMBER --------------------------------------------------------------------------------

def parse_amber(queries_path, annotations_path) -> list[dict]:
    """Queries joined with their annotation by id. Generative rows carry truth/hallu object
    lists; discriminative rows carry a yes/no truth."""
    ann = {a["id"]: a for a in _read_json(annotations_path)}
    rows = []
    for q in _read_json(queries_path):
        a = ann[q["id"]]
        rows.append({"id": q["id"], "image": q["image"], "query": q["query"],
                     "type": a["type"], "truth": a["truth"], "hallu": a.get("hallu")})
    return rows


def load_amber(task: str | None = "generative", verify: bool = True) -> list[dict]:
    d = get("amber")
    rows = parse_amber(d.local("query_all.json", verify), d.local("annotations.json", verify))
    return rows if task is None else [r for r in rows if r["type"].startswith(task)]


# ---- nocaps -------------------------------------------------------------------------------

def parse_nocaps(path) -> list[dict]:
    data = _read_json(path)
    refs: dict[int, list[str]] = {}
    for a in sorted(data["annotations"], key=lambda a: a["id"]):
        refs.setdefault(a["image_id"], []).append(a["caption"])
    return [{"image_id": im["id"], "open_images_id": im["open_images_id"],
             "file_name": im["file_name"], "domain": im["domain"],
             "references": refs.get(im["id"], [])} for im in data["images"]]


def load_nocaps(verify: bool = True) -> list[dict]:
    return parse_nocaps(get("nocaps").local("nocaps_val_4500_captions.json", verify))


LOADERS = {
    "foil": load_foil, "thumb": load_thumb, "flickr8k": load_flickr8k,
    "pascal50s": load_pascal50s, "vsr": load_vsr, "whatsup": load_whatsup,
    "amber": load_amber, "nocaps": load_nocaps,
}

