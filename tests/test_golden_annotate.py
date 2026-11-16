"""Golden-set design, claim files and annotation workbooks (harness/golden)."""

from collections import Counter

import pytest

from harness.coco import load_split
from harness.golden.agreement import attach_key, main as agreement_main, read_annotations
from harness.golden.annotate import (
    ANNOTATOR_COLUMNS,
    build_rows,
    cmd_build,
    cmd_merge,
    load_claims,
    parse_claims,
)
from harness.golden.design import load_design, select_items

# ---- design ------------------------------------------------------------------------------


def test_design_is_deterministic_and_balanced():
    dev = load_split("dev")
    items = select_items(dev)
    assert items == select_items(dev)
    assert Counter(i["source"] for i in items) == {
        "blip": 100, "llava_short": 100, "human": 100, "llava_long": 50}
    assert Counter(i["block"] for i in items) == {"P": 150, "U": 150, "L": 50}
    assert len({i["image_id"] for i in items}) == 200
    assert set(i["image_id"] for i in items) <= set(dev)
    paired = {i["image_id"] for i in items if i["block"] == "P"}
    for src in ("blip", "llava_short", "human"):
        assert {i["image_id"] for i in items if i["block"] == "P" and i["source"] == src} == paired
    assert {i["image_id"] for i in items if i["block"] == "L"} == paired
    unpaired = [i["image_id"] for i in items if i["block"] == "U"]
    assert len(unpaired) == len(set(unpaired)) == 150 and not set(unpaired) & paired
    gids = [i["gid"] for i in items]
    assert sorted(gids) == [f"g{k:03d}" for k in range(1, 351)]


def test_committed_design_matches_code():
    d = load_design()
    fresh = select_items(load_split("dev"), d["seed"])
    strip = [{k: v for k, v in i.items() if k != "ref_annotation_id"} for i in d["items"]]
    assert strip == fresh
    assert all("ref_annotation_id" in i for i in d["items"] if i["source"] == "human")


# ---- claim files -------------------------------------------------------------------------


def test_parse_claims_format_and_errors():
    text = "# comment\nhuman-1|o dog|c two dogs|a dog: brown|r dog / on / couch|s living room\n"
    assert parse_claims(text) == {"human-1": [
        ("object", "dog"), ("count", "two dogs"), ("attribute", "dog: brown"),
        ("relation", "dog / on / couch"), ("scene", "living room")]}
    with pytest.raises(ValueError, match="bad claim"):
        parse_claims("x-1|q nonsense")
    with pytest.raises(ValueError, match="duplicate"):
        parse_claims("x-1|o a\nx-1|o b")
    with pytest.raises(ValueError, match="no claims"):
        parse_claims("x-1")


def test_committed_claims_cover_known_captions_only():
    claims = load_claims()
    design_ids = {i["caption_id"] for i in load_design()["items"]}
    assert set(claims) <= design_ids
    human = {i["caption_id"] for i in load_design()["items"] if i["source"] == "human"}
    assert human <= set(claims)          # all 100 human captions are segmented


# ---- workbooks end to end ----------------------------------------------------------------


@pytest.fixture
def tiny():
    pytest.importorskip("openpyxl")
    design = {"items": [
        {"source": "blip", "block": "P", "image_id": 1, "caption_id": "blip-1", "gid": "g002"},
        {"source": "human", "block": "P", "image_id": 1, "caption_id": "human-1", "gid": "g001"},
        {"source": "llava_short", "block": "U", "image_id": 2, "caption_id": "llava_short-2",
         "gid": "g003"},
    ]}
    texts = {"blip-1": "a dog on a couch", "human-1": "two brown dogs", "llava_short-2": "a cat"}
    claims = {"blip-1": [("object", "dog"), ("relation", "dog / on / couch")],
              "human-1": [("object", "dogs"), ("count", "two dogs")],
              "llava_short-2": [("object", "cat")]}
    return design, texts, claims


def test_build_rows_order_and_report(tiny):
    design, texts, claims = tiny
    rows, report = build_rows(design, {k: v for k, v in texts.items() if k != "blip-1"}, claims)
    assert [r["claim_id"] for r in rows] == ["g001#01", "g001#02", "g003#01"]
    assert report["no_caption_text"] == ["blip-1"]


def test_build_refuses_incomplete_and_blinds_annotators(tiny, tmp_path):
    design, texts, claims = tiny
    with pytest.raises(SystemExit, match="incomplete"):
        cmd_build(out_dir=tmp_path, design=design, texts={}, claims=claims)
    rep = cmd_build(out_dir=tmp_path, design=design, texts=texts, claims=claims)
    assert rep["complete"] and rep["n_claims"] == 5
    a = read_annotations(tmp_path / "annotator_A.xlsx")
    assert [r["claim_id"] for r in a] == ["g001#01", "g001#02", "g002#01", "g002#02", "g003#01"]
    assert set(a[0]) == set(ANNOTATOR_COLUMNS)              # no source, no Claude column
    assert not any("blip" in v or "human" in v for r in a for v in r.values())
    adj = read_annotations(tmp_path / "adjudication.xlsx")
    assert adj[0]["source"] == "human" and "claude_prelabel_verdict" in adj[0]
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        cmd_build(out_dir=tmp_path, design=design, texts=texts, claims=claims)


def test_label_merge_and_agreement_roundtrip(tiny, tmp_path):
    from openpyxl import load_workbook

    design, texts, claims = tiny
    cmd_build(out_dir=tmp_path, design=design, texts=texts, claims=claims)
    verdicts = {"A": ["supported", "contradicted", "supported", "unverifiable", "supported"],
                "B": ["supported", "contradicted", "supported", "supported", "supported"]}
    for tag, vs in verdicts.items():
        p = tmp_path / f"annotator_{tag}.xlsx"
        wb = load_workbook(p)
        ws = wb["claims"]
        col = [c.value for c in ws[1]].index("verdict") + 1
        for i, v in enumerate(vs, start=2):
            ws.cell(row=i, column=col, value=v)
        wb.save(p)
    n = cmd_merge(str(tmp_path / "annotator_A.xlsx"), str(tmp_path / "annotator_B.xlsx"),
                  tmp_path / "adjudication.xlsx")
    assert n == 5
    adj = read_annotations(tmp_path / "adjudication.xlsx")
    assert [r["verdict_B"] for r in adj] == verdicts["B"]

    keyed = attach_key(read_annotations(tmp_path / "annotator_A.xlsx"), design)
    assert [r["source"] for r in keyed] == ["human", "human", "blip", "blip", "llava_short"]
    res = agreement_main([str(tmp_path / "annotator_A.xlsx"), str(tmp_path / "annotator_B.xlsx"),
                          "--resamples", "50"])
    assert res["n"] == 5 and res["percent_agreement"] == pytest.approx(0.8)


# ---- Kaggle LLaVA script -----------------------------------------------------------------


def test_kaggle_script_ids_and_pins_match_design():
    from harness.golden.llava_check import check, script_constants

    c = script_constants()
    assert c["IMAGE_IDS"] == load_design()["llava_image_ids"]
    assert c["MODEL_REVISION"] == "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
    assert set(c["PROMPTS"]) == {"short", "long"}

    fake = {"config": {"model": c["MODEL_ID"], "revision": c["MODEL_REVISION"],
                       "template": c["TEMPLATE"], "prompts": c["PROMPTS"],
                       "generate": {"do_sample": False}},
            "captions": {n: {str(i): "a cat" for i in c["IMAGE_IDS"]} for n in ("short", "long")},
            "image_sha256": {}}
    problems = check(fake)
    assert problems == ["no local images to compare SHA-256 with (download dev images)"]
    fake["config"]["revision"] = "main"
    assert any("revision" in p for p in check(fake))
