"""Original-BLIP captioner (harness/caption_blip_orig.py): pins, and real files when present."""

import re

import pytest

from harness import caption_blip_orig as cbo
from harness.download import sha256_file


def test_pins_are_full_hashes():
    assert re.fullmatch(r"[0-9a-f]{40}", cbo.BLIP_COMMIT)
    assert re.fullmatch(r"[0-9a-f]{40}", cbo.TOKENIZER_REVISION)
    assert re.fullmatch(r"[0-9a-f]{64}", cbo.CHECKPOINT_SHA256)
    assert all(re.fullmatch(r"[0-9a-f]{64}", h) for h in cbo.BLIP_FILES.values())
    assert {"models/blip.py", "models/med.py", "models/vit.py",
            "configs/med_config.json"} <= set(cbo.BLIP_FILES)
    assert cbo.BLIP_COMMIT in cbo.BLIP_RAW


def test_paper_decoding_matches_blip_caption_coco_yaml():
    v = cbo.VARIANTS["paper"]
    assert v["prompt"] == "a picture of "
    assert v["generate"] == {"num_beams": 3, "max_length": 20, "min_length": 5,
                             "repetition_penalty": 1.0}
    assert cbo.IMAGE_SIZE == 384


def test_downloaded_files_match_pins():
    root = cbo.blip_dir()
    present = [rel for rel in cbo.BLIP_FILES if (root / rel).exists()]
    if not present:
        pytest.skip("BLIP source files not downloaded")
    for rel in present:
        assert sha256_file(root / rel) == cbo.BLIP_FILES[rel], rel
