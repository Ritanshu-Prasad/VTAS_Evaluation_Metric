"""Claim parser tests. Uses en_core_web_sm unless VTAS_TEST_MODEL=trf.
Skipped when spaCy or the model is missing."""

import os
import shutil
from pathlib import Path

import pytest

pytest.importorskip("spacy")
pytest.importorskip("yaml")

from vtas.claims import ClaimParser, ConceptPolicy  # noqa: E402

MODEL = os.environ.get("VTAS_TEST_MODEL", "sm")
POLICY = Path(__file__).resolve().parent.parent / "policy" / "concept_policy_v1.yaml"


@pytest.fixture(scope="module")
def parser():
    try:
        return ClaimParser(model=MODEL, policy=ConceptPolicy.load(POLICY))
    except OSError as e:  # model not installed
        pytest.skip(str(e))


def objects(r):
    return r.texts("object")


# ---- U2: compounds ----------------------------------------------------------------------------

COMPOUNDS = ["fire hydrant", "cell phone", "hot dog", "stop sign", "teddy bear", "wine glass",
             "traffic light"]
CONTEXTS = ["a {} on the table", "a {}", "two people next to a {}", "A {} is next to a tree."]


@pytest.mark.parametrize("compound", COMPOUNDS)
@pytest.mark.parametrize("ctx", CONTEXTS)
def test_u2_compound_is_one_object_claim(parser, compound, ctx):
    objs = objects(parser.parse(ctx.format(compound)))
    assert objs.count(compound) == 1, objs
    assert not any(w in objs for w in compound.split()), objs  # no fragment claims


def test_u2_plural_compound(parser):
    assert "teddy bears" in objects(parser.parse("three teddy bears on a bed"))


# ---- U3: counts -------------------------------------------------------------------------------

_POLICY = ConceptPolicy.load(POLICY) if POLICY.exists() else None


@pytest.mark.parametrize("caption, text, kind, lo, hi", [
    ("two dogs on a couch", "two dogs", "numeral", 2, 2),
    ("three cats", "three cats", "numeral", 3, 3),
    ("a couple of cats on a bed", "a couple of cats", "pair", 2, 2),
    ("several people on a street", "several people", "several", 3, None),
    ("a crowd of people", "a crowd of people", "crowd", 6, None),
    ("a group of children at a table", "a group of children", "group", 3, None),
    ("many birds in the sky", "many birds", "many", 6, None),
    ("some trees behind a house", "some trees", "plural", 2, None),
    ("horses in a field", "horses", "plural", 2, None),
])
def test_u3_count_intervals_follow_policy(parser, caption, text, kind, lo, hi):
    r = parser.parse(caption)
    got = {c.text: c for c in r.of_type("count")}
    assert text in got, (list(got), caption)
    c = got[text]
    iv = c.interval(_POLICY)
    assert (c.quantity.kind, iv.min, iv.max) == (kind, lo, hi)


def test_u3_numeral_tolerance_by_profile(parser):
    c = [c for c in parser.parse("two dogs").of_type("count")][0]
    assert (c.interval(_POLICY, "standard").min, c.interval(_POLICY, "standard").max) == (2, 2)
    assert (c.interval(_POLICY, "lenient").min, c.interval(_POLICY, "lenient").max) == (1, 3)


def test_u3_vague_quantifier_is_flagged_provisional(parser):
    c = parser.parse("several people").of_type("count")[0]
    assert c.interval(_POLICY).provisional is True
    assert parser.parse("two people").of_type("count")[0].interval(_POLICY).provisional is False


def test_u3_singular_has_no_count_claim_but_a_quantity(parser):
    r = parser.parse("a dog on a couch")
    assert r.of_type("count") == []
    dog = [c for c in r.of_type("object") if c.text == "dog"][0]
    iv = dog.interval(_POLICY)
    assert (dog.quantity.kind, iv.min, iv.max) == ("singular", 1, 2)
    assert (dog.interval(_POLICY, "strict").min, dog.interval(_POLICY, "strict").max) == (1, 1)


@pytest.mark.parametrize("caption", ["a pair of scissors on a table", "water in a glass",
                                     "a man eating fries"])
def test_u3_no_count_for_one_item_and_mass_nouns(parser, caption):
    assert parser.parse(caption).of_type("count") == []


# ---- U4: attributes and relations -------------------------------------------------------------

def relations(r):
    return {(c.subject, c.predicate, c.object) for c in r.of_type("relation")}


def attributes(r):
    return {(c.entity, c.attribute) for c in r.of_type("attribute")}


def test_u4_red_car_under_a_tree(parser):
    r = parser.parse("a red car under a tree")
    assert ("car", "red") in attributes(r)
    assert ("car", "under", "tree") in relations(r)
    assert "car: red" in r.texts("attribute") and "car / under / tree" in r.texts("relation")
    assert {"car", "tree"} <= set(objects(r))


def test_u4_each_adjective_is_one_attribute(parser):
    assert attributes(parser.parse("a small open window")) >= {("window", "small"), ("window", "open")}


def test_u4_action_with_object_is_relation_without_is_attribute(parser):
    r = parser.parse("a man riding a horse")
    assert ("man", "riding", "horse") in relations(r)
    assert attributes(parser.parse("a man walking")) >= {("man", "walking")}


def test_u4_place_phrase_is_scene_not_relation(parser):
    r = parser.parse("a dog on the beach")
    assert "beach" in r.texts("scene")
    assert relations(r) == set()


def test_u4_multiword_preposition_and_clean_object(parser):
    r = parser.parse("a vase in the middle of a table")
    assert ("vase", "in the middle of", "table") in relations(r)


# ---- U8: policy file is the source of truth ---------------------------------------------------

def test_u8_changing_yaml_changes_intervals_without_code_change(tmp_path):
    text = POLICY.read_text(encoding="utf-8")
    p1 = tmp_path / "p1.yaml"
    shutil.copy(POLICY, p1)
    assert ConceptPolicy.load(p1).interval("several").min == 3
    # freeze a fitted interval for 'several' and loosen the numeral tolerance: only the YAML changes
    new = text.replace("      interval: fit_from_data\n      plan_prior: [3, null]         # Section 4.2 table; replaced by the fitted interval",
                       "      interval: [4, 9]\n      plan_prior: [3, null]", 1)
    new = new.replace("        standard: 0\n        lenient: 1", "        standard: 1\n        lenient: 2", 1)
    assert new != text
    p2 = tmp_path / "p2.yaml"
    p2.write_text(new, encoding="utf-8")
    pol = ConceptPolicy.load(p2)
    iv = pol.interval("several")
    assert (iv.min, iv.max, iv.provisional) == (4, 9, False)
    assert (pol.interval("numeral", 3).min, pol.interval("numeral", 3).max) == (2, 4)


def test_u8_parser_uses_the_policy_it_was_given(parser, tmp_path):
    text = POLICY.read_text(encoding="utf-8")
    p = tmp_path / "p.yaml"
    p.write_text(text.replace("        standard: 0\n        lenient: 1", "        standard: 1\n        lenient: 2", 1),
                 encoding="utf-8")
    c = parser.parse("two dogs").of_type("count")[0]
    assert (c.interval(ConceptPolicy.load(p)).min, c.interval(ConceptPolicy.load(p)).max) == (1, 3)
    assert (c.interval(_POLICY).min, c.interval(_POLICY).max) == (2, 2)


def test_parse_terminates_when_root_is_an_adjective_tagged_noun(parser):
    # sm tags "locomotive" as a root ADJ; the preposition climb once looped forever on it
    r = parser.parse("A locomotive and a passenger car next to each other at the station")
    assert "passenger car" in objects(r)
