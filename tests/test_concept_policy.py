"""Structure checks for the Concept Policy v1 draft (policy/concept_policy_v1.yaml)."""

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

POLICY = Path(__file__).resolve().parent.parent / "policy" / "concept_policy_v1.yaml"


@pytest.fixture(scope="module")
def policy():
    return yaml.safe_load(POLICY.read_text(encoding="utf-8"))


def test_profiles_and_calibration_fpr(policy):
    p = policy["profiles"]
    assert p["default"] == "standard"
    assert {k: p[k]["calibration_fpr"] for k in ("strict", "standard", "lenient")} == {
        "strict": 0.02, "standard": 0.05, "lenient": 0.10}


def test_vague_quantifiers_are_left_to_data(policy):
    ex = policy["counts"]["expressions"]
    for name in ("several", "group", "crowd", "many"):
        assert ex[name]["interval"] == "fit_from_data", name


def test_fixed_count_intervals_match_plan_table(policy):
    ex = policy["counts"]["expressions"]
    assert ex["singular"]["interval"] == {"strict": [1, 1], "standard": [1, 2], "lenient": [1, None]}
    assert ex["numeral"]["tolerance"]["standard"] == 0 and ex["numeral"]["tolerance"]["lenient"] == 1
    assert ex["plural"]["interval"]["standard"] == [2, None]
    assert ex["pair"]["interval"]["standard"] == [2, 2] and ex["pair"]["interval"]["lenient"] == [2, 3]


def test_verdicts_and_types(policy):
    assert policy["verdicts"] == ["SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"]
    assert policy["claim_types"] == ["object", "count", "attribute", "relation", "scene"]
    assert policy["out_of_vocabulary"]["trusted_vocabulary"]["per_class_recall_min"] == 0.6
