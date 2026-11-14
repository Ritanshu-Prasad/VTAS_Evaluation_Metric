"""The committed COCO split id lists are fixed, disjoint and match their manifest."""

import json

import pytest

from harness import SPLITS_DIR
from harness.coco import SPLIT_SIZES, _ids_sha256, load_split, make_splits

MANIFEST = json.loads((SPLITS_DIR / "coco_manifest.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", list(SPLIT_SIZES))
def test_split_matches_manifest(name):
    ids = load_split(name)
    assert len(ids) == SPLIT_SIZES[name] == MANIFEST["splits"][name]["n"]
    assert ids == sorted(set(ids))
    assert _ids_sha256(ids) == MANIFEST["splits"][name]["sha256"]


def test_splits_are_disjoint():
    sets = [set(load_split(n)) for n in SPLIT_SIZES]
    assert len(set.union(*sets)) == sum(SPLIT_SIZES.values())


def test_splits_regenerate_from_their_union():
    ids = [i for n in SPLIT_SIZES for i in load_split(n)]
    regenerated = make_splits(ids, MANIFEST["seed"])
    assert regenerated == {n: load_split(n) for n in SPLIT_SIZES}


def test_make_splits_rejects_wrong_size():
    with pytest.raises(ValueError):
        make_splits(list(range(10)))


def test_unknown_split_name():
    with pytest.raises(ValueError):
        load_split("train")
