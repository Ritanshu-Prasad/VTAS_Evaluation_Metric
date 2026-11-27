"""Loader tests.

Two layers per dataset:
  * fixture tests: a tiny file in the dataset's native format is parsed (always run, CI);
  * real-file tests: the downloaded file must match its pinned SHA-256 and parse to the
    published size. These run wherever the data is present and skip otherwise
    (``python -m harness.datasets fetch`` downloads it).
"""

import json
import re
import zipfile

import numpy as np
import pytest

from harness.datasets import REGISTRY, get
from harness.datasets import loaders as L
from harness.download import ChecksumError, sha256_file

HEX64 = re.compile(r"^[0-9a-f]{64}$")


# ---- registry and checksum mechanism --------------------------------------------------

def test_registry_covers_the_plan_datasets():
    assert set(REGISTRY) == {"foil", "thumb", "flickr8k", "pascal50s", "vsr", "whatsup",
                             "amber", "nocaps"}
    assert set(L.LOADERS) == set(REGISTRY)


@pytest.mark.parametrize("key", sorted(REGISTRY))
def test_every_fetchable_file_is_pinned(key):
    d = REGISTRY[key]
    for f in d.files:
        if d.fetchable:
            assert HEX64.match(f.sha256) and f.size > 0, f.name
            assert f.url.startswith("https://")
            assert "/master/" not in f.url and "/main/" not in f.url  # pinned revisions


def test_non_public_sets_are_flagged():
    flagged = {k for k, d in REGISTRY.items() if not d.public}
    assert flagged == {"pascal50s", "flickr8k", "nocaps"}
    assert not REGISTRY["pascal50s"].fetchable
    for k in flagged:
        assert REGISTRY[k].access


def test_checksum_mismatch_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("VTAS_DATA", str(tmp_path))
    d = get("thumb")
    d.root().mkdir(parents=True)
    d.path("mscoco_THumB-1.0.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ChecksumError):
        d.local("mscoco_THumB-1.0.jsonl")
    assert d.local("mscoco_THumB-1.0.jsonl", verify_checksum=False).exists()


def test_missing_file_names_the_fix(tmp_path, monkeypatch):
    monkeypatch.setenv("VTAS_DATA", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="harness.datasets fetch foil"):
        get("foil").local("foilv1.0_test_2017.json")
    with pytest.raises(FileNotFoundError, match="no download available"):
        get("pascal50s").local("pair_pascal.mat")


def test_unpinned_file_cannot_pass_verification(tmp_path, monkeypatch):
    monkeypatch.setenv("VTAS_DATA", str(tmp_path))
    d = get("pascal50s")
    d.root().mkdir(parents=True)
    d.path("pair_pascal.mat").write_bytes(b"x")
    with pytest.raises(ChecksumError, match="no pinned sha256"):
        d.local("pair_pascal.mat")


# ---- fixture parsing (native formats) --------------------------------------------------

def _write_json(path, obj):
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_parse_foil(tmp_path):
    p = _write_json(tmp_path / "foil.json", {"images": [], "annotations": [
        {"id": 1, "foil_id": 10, "image_id": 391895, "caption": "a man on a motorcycle",
         "target_word": "ORIG", "foil_word": "ORIG", "foil": False},
        {"id": 1, "foil_id": 11, "image_id": 391895, "caption": "a man on a bicycle",
         "target_word": "motorcycle", "foil_word": "bicycle", "foil": True}]})
    rows = L.parse_foil(p)
    assert [r["foil"] for r in rows] == [False, True]
    assert rows[1]["foil_word"] == "bicycle" and rows[1]["image_id"] == 391895


def test_parse_thumb(tmp_path):
    ratings = _write_jsonl(tmp_path / "r.jsonl", [
        {"SYS": "Up-Down", "seg_id": "974", "set_id": 5, "hyp": "people on an elephant.",
         "image": "COCO_val2014_000000000974.jpg", "P": 4.0, "R": 3.0, "Fl": -0.0,
         "Inc": -0.0, "Con": -0.5, "human_score": 3.0}])
    refs = _write_jsonl(tmp_path / "refs.json", [{"seg_id": "974", "set_id": 5,
                                                  "refs": ["a", "b"]}])
    (row,) = L.parse_thumb(ratings, refs)
    assert row["image_id"] == 974 and row["system"] == "Up-Down"
    assert (row["precision"], row["recall"], row["human_score"]) == (4.0, 3.0, 3.0)
    assert row["references"] == ["a", "b"]


def test_parse_flickr8k(tmp_path):
    p = tmp_path / "Flickr8k_text.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("Flickr8k.token.txt", "a.jpg#0\tA dog runs .\nb.jpg#2\tA cat sits .\n")
        z.writestr("ExpertAnnotations.txt", "a.jpg\tb.jpg#2\t1\t2\t1\n")
        z.writestr("CrowdFlowerAnnotations.txt", "a.jpg\ta.jpg#0\t1\t3\t0\n")
    out = L.parse_flickr8k(p)
    assert out["expert"] == [{"image": "a.jpg", "caption_id": "b.jpg#2",
                              "caption": "A cat sits .", "scores": [1, 2, 1]}]
    assert out["crowdflower"][0]["p_yes"] == 1.0 and out["crowdflower"][0]["n_yes"] == 3


def test_parse_pascal50s(tmp_path):
    scipy_io = pytest.importorskip("scipy.io")

    def cell(*items):
        arr = np.empty((1, len(items)), dtype=object)
        for k, v in enumerate(items):
            arr[0, k] = v
        return arr

    pairs = cell(cell("2008_000032.jpg", "cap a", "cap b"))
    triplets = cell(*([cell("ref one", "cap a", "cap b", 1.0)] * 3
                      + [cell("ref two", "cap a", "cap b", 2.0)]))
    scipy_io.savemat(tmp_path / "pair.mat", {"new_input": pairs, "category": np.array([[3]])})
    scipy_io.savemat(tmp_path / "cons.mat", {"triplets": triplets})
    (row,) = L.parse_pascal50s(tmp_path / "pair.mat", tmp_path / "cons.mat",
                               triplets_per_pair=4)
    assert row["image"] == "2008_000032.jpg" and row["category"] == "HM"
    assert (row["votes_a"], row["votes_b"]) == (3, 1)
    assert row["references"] == ["ref one"] * 3 + ["ref two"]


def test_parse_vsr(tmp_path):
    p = _write_jsonl(tmp_path / "test.jsonl", [
        {"image": "000000451431.jpg", "image_link": "http://images.cocodataset.org/x.jpg",
         "caption": "The person is inside the refrigerator.", "label": 1,
         "relation": "inside", "annotator_id": 29}])
    (row,) = L.parse_vsr(p)
    assert row["label"] == 1 and row["relation"] == "inside"


def test_parse_whatsup_both_layouts(tmp_path):
    a = _write_json(tmp_path / "a.json", [{"image_path": "data/x/mug_on_table.jpeg",
                                           "caption_options": ["on", "under", "left", "right"]}])
    b = _write_json(tmp_path / "b.json", [[397133, "on the bottom", "on the top"]])
    assert L.parse_whatsup(a)[0]["options"][0] == "on"
    assert L.parse_whatsup(b) == [{"image": "397133", "options": ["on the bottom", "on the top"],
                                   "correct": 0}]


def test_parse_amber(tmp_path):
    q = _write_json(tmp_path / "q.json", [
        {"id": 1, "image": "AMBER_1.jpg", "query": "Describe this image."},
        {"id": 1005, "image": "AMBER_1.jpg", "query": "Is the sky sunny in this image?"}])
    a = _write_json(tmp_path / "a.json", [
        {"id": 1, "type": "generative", "truth": ["sky"], "hallu": ["dog"]},
        {"id": 1005, "type": "discriminative-attribute-state", "truth": "yes"}])
    rows = L.parse_amber(q, a)
    assert rows[0]["hallu"] == ["dog"] and rows[1]["truth"] == "yes" and rows[1]["hallu"] is None


def test_parse_nocaps(tmp_path):
    p = _write_json(tmp_path / "n.json", {
        "images": [{"id": 0, "open_images_id": "0013ea2087020901", "domain": "in-domain",
                    "file_name": "0013ea2087020901.jpg"}],
        "annotations": [{"image_id": 0, "id": 1, "caption": "second"},
                        {"image_id": 0, "id": 0, "caption": "first"}]})
    (row,) = L.parse_nocaps(p)
    assert row["references"] == ["first", "second"] and row["domain"] == "in-domain"


# ---- real files: pinned checksum + published size ---------------------------------------

def _present(key):
    d = REGISTRY[key]
    return all(d.path(f.name).exists() for f in d.files)


REAL = {
    "foil": (lambda: L.load_foil(), lambda r: (len(r), sum(x["foil"] for x in r)), (198960, 99480)),
    "thumb": (lambda: L.load_thumb(), lambda r: (len(r), len({x["image_id"] for x in r})), (2500, 500)),
    "flickr8k": (lambda: L.load_flickr8k(),
                 lambda r: (len(r["expert"]), len(r["crowdflower"])), (5822, 47830)),
    "pascal50s": (lambda: L.load_pascal50s(), len, 4000),
    "vsr": (lambda: [len(L.load_vsr(s)) for s in ("train", "dev", "test")], list, [7680, 1097, 2195]),
    "whatsup": (lambda: [len(L.load_whatsup(s)) for s in L.WHATSUP_SUBSETS], list,
                [412, 408, 2247, 440, 1160, 291]),
    "amber": (lambda: L.load_amber(None), lambda r: (len(r), sum(x["type"] == "generative" for x in r)),
              (15220, 1004)),
    "nocaps": (lambda: L.load_nocaps(),
               lambda r: (len(r), sum(len(x["references"]) for x in r)), (4500, 45000)),
}


@pytest.mark.parametrize("key", sorted(REAL))
def test_real_file_checksum_and_size(key):
    if not _present(key):
        pytest.skip(f"{key} not downloaded ({REGISTRY[key].access})")
    for f in REGISTRY[key].files:
        assert sha256_file(REGISTRY[key].path(f.name)) == f.sha256
    load, measure, expected = REAL[key]
    assert measure(load()) == expected
