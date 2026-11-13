"""Pure-math tests for the scoring step of VTASEvaluator.score().

No model is loaded: detection, parsing, matching and CLIP are stubbed, and
every expected value below is worked out by hand in the comments.
"""

import pytest

from conftest import det, noun

TOL = 1e-4


def test_empty_caption_scores_zero_with_precision_one(make_evaluator, image_path):
    # No claims => P = 1.0 (no false claims). Nothing mentioned => both objects
    # missed, R = (0.5 - 0.5) / 0.5 = 0. F1 = 2*1*0 / (1+0) = 0.
    dets = [det("person", 0.3), det("dog", 0.2)]
    alignment = {"matched": [], "hallucinated": [], "missed": dets, "similarity_matrix": {}}
    r = make_evaluator(dets, [], alignment).score(image_path, "")
    assert r["precision"] == pytest.approx(1.0)
    assert r["recall"] == pytest.approx(0.0)
    assert r["vtas_score"] == pytest.approx(0.0)


def test_empty_detections_and_empty_caption_is_perfect(make_evaluator, image_path):
    # P = 1.0 (no claims), R = 1.0 (nothing to miss) => F1 = 1.0.
    alignment = {"matched": [], "hallucinated": [], "missed": [], "similarity_matrix": {}}
    r = make_evaluator([], [], alignment).score(image_path, "")
    assert r["vtas_score"] == pytest.approx(1.0)


def test_empty_detections_unrescued_noun_scores_zero(make_evaluator, image_path):
    # One claim, no detections, CLIP rejects it: P = 0, R = 1 (no saliency) => F1 = 0.
    nouns = [noun("dog")]
    alignment = {"matched": [], "hallucinated": nouns, "missed": [], "similarity_matrix": {}}
    ev = make_evaluator([], nouns, alignment, clip_hallucinated=[("dog", 0.1)])
    r = ev.score(image_path, "a dog")
    assert r["precision"] == pytest.approx(0.0)
    assert r["recall"] == pytest.approx(1.0)
    assert r["vtas_score"] == pytest.approx(0.0)
    assert r["hallucinated"] == ["dog"]


def test_empty_detections_clip_rescue_gives_full_score(make_evaluator, image_path):
    # CLIP rescues the only claim: P = 1. Saliency = 0 + 0.10 bonus, nothing
    # missed => R = 0.10 / 0.10 = 1. F1 = 1.
    nouns = [noun("dog")]
    alignment = {"matched": [], "hallucinated": nouns, "missed": [], "similarity_matrix": {}}
    ev = make_evaluator([], nouns, alignment, clip_grounded=[("dog", 0.3)])
    assert ev.score(image_path, "a dog")["vtas_score"] == pytest.approx(1.0)


def test_all_hallucinated_scores_zero(make_evaluator, image_path):
    # Two claims both fail both tiers: P = 1 - 2/2 = 0. Both objects missed:
    # R = 0. Denominator is 0 => score is defined as 0.0 (no ZeroDivisionError).
    dets = [det("person", 0.6), det("car", 0.4)]
    nouns = [noun("unicorn"), noun("dragon")]
    alignment = {"matched": [], "hallucinated": nouns, "missed": dets, "similarity_matrix": {}}
    ev = make_evaluator(dets, nouns, alignment,
                        clip_hallucinated=[("unicorn", 0.0), ("dragon", 0.0)])
    r = ev.score(image_path, "a unicorn and a dragon")
    assert r["precision"] == 0.0
    assert r["recall"] == 0.0
    assert r["vtas_score"] == 0.0
    assert r["hallucinated"] == ["unicorn", "dragon"]


@pytest.mark.parametrize(
    "beta, expected",
    [
        (1.0, 8 / 11),    # 2PR/(P+R)          = 16/22
        (2.0, 10 / 13),   # 5PR/(4P+R)         = (8/3)/(52/15)
        (0.5, 20 / 29),   # 1.25PR/(0.25P+R)   = (2/3)/(29/30)
    ],
)
def test_f_beta_values_by_hand(make_evaluator, image_path, beta, expected):
    # Detected: person .5, dog .3, cup .2 (total saliency 1.0).
    # Caption: man, dog, unicorn. man~person and dog~dog match; unicorn is
    # hallucinated (CLIP rejects it); cup is missed (saliency 0.2).
    # P = 1 - 1/3 = 2/3; R = (1.0 - 0.2) / 1.0 = 4/5.
    dets = [det("person", 0.5), det("dog", 0.3), det("cup", 0.2)]
    nouns = [noun("man"), noun("dog"), noun("unicorn")]
    alignment = {
        "matched": [(noun("man"), dets[0], 0.8), (noun("dog"), dets[1], 0.99)],
        "hallucinated": [noun("unicorn")],
        "missed": [dets[2]],
        "similarity_matrix": {},
    }
    ev = make_evaluator(dets, nouns, alignment,
                        clip_hallucinated=[("unicorn", 0.05)], beta=beta)
    r = ev.score(image_path, "a man, a dog and a unicorn")
    assert r["precision"] == pytest.approx(2 / 3, abs=TOL)
    assert r["recall"] == pytest.approx(4 / 5, abs=TOL)
    assert r["vtas_score"] == pytest.approx(expected, abs=TOL)
    assert r["missed"] == [dets[2]]


def test_plural_exemption_is_reflected_in_recall(make_evaluator, image_path):
    # Three persons (.2 each). "men" (plural) matches one; the semantic bridge
    # exempts the other two, so `missed` is empty => R = 1, P = 1, F1 = 1.
    # With the singular "man" the bridge reports the other two as missed:
    # R = (0.6 - 0.4) / 0.6 = 1/3, F1 = 2*(1/3)/(4/3) = 0.5.
    dets = [det("person", 0.2) for _ in range(3)]
    plural = {"matched": [(noun("man", True), dets[0], 0.9)], "hallucinated": [],
              "missed": [], "similarity_matrix": {}}
    singular = {"matched": [(noun("man", False), dets[0], 0.9)], "hallucinated": [],
                "missed": dets[1:], "similarity_matrix": {}}
    assert make_evaluator(dets, [noun("man", True)], plural).score(
        image_path, "men")["vtas_score"] == pytest.approx(1.0)
    r = make_evaluator(dets, [noun("man")], singular).score(image_path, "a man")
    assert r["recall"] == pytest.approx(1 / 3, abs=TOL)
    assert r["vtas_score"] == pytest.approx(0.5, abs=TOL)


def test_clip_rescue_removes_weak_synonym_from_missed_and_adds_bonus(
        make_evaluator, image_path):
    # Detected: person .6, bowl .4. "woman" fails Tier 1 (sim .62) and is
    # rescued by CLIP. Sim to "person" (.62) > 0.30 so "person" leaves `missed`.
    # Missed saliency = bowl .4; total = 1.0 + 0.10 bonus = 1.1.
    # R = (1.1 - 0.4) / 1.1 = 7/11; P = 1. F1 = 2*(7/11)/(1+7/11) = 7/9.
    dets = [det("person", 0.6), det("bowl", 0.4)]
    nouns = [noun("woman")]
    alignment = {"matched": [], "hallucinated": nouns, "missed": dets,
                 "similarity_matrix": {"woman": {"person": 0.62, "bowl": 0.1}}}
    ev = make_evaluator(dets, nouns, alignment, clip_grounded=[("woman", 0.3)])
    r = ev.score(image_path, "a woman")
    assert r["missed"] == [dets[1]]
    assert r["recall"] == pytest.approx(7 / 11, abs=TOL)
    assert r["vtas_score"] == pytest.approx(7 / 9, abs=TOL)


def test_overcount_duplicate_noun_is_not_sent_to_clip(make_evaluator, image_path):
    # "banana" twice, DETR found one. The second copy is a strict
    # hallucination (never rescued by CLIP): P = 1 - 1/2 = 0.5, R = 1.
    # F1 = 2*0.5*1/1.5 = 2/3.
    dets = [det("banana", 0.5)]
    nouns = [noun("banana"), noun("banana")]
    alignment = {"matched": [(noun("banana"), dets[0], 0.99)],
                 "hallucinated": [noun("banana")], "missed": [],
                 "similarity_matrix": {}}
    ev = make_evaluator(dets, nouns, alignment)
    r = ev.score(image_path, "a banana and a banana")
    assert ev.clip_calls == [[]]
    assert r["hallucinated"] == ["banana"]
    assert r["vtas_score"] == pytest.approx(2 / 3, abs=TOL)


def test_missing_area_defaults_to_unit_saliency(make_evaluator, image_path):
    # Detections without 'area' count as 1.0 each. Two objects, one missed:
    # R = 1/2, P = 1 (no claims hallucinated) => F1 = 2*0.5/1.5 = 2/3.
    dets = [{"label": "cat"}, {"label": "sofa"}]
    nouns = [noun("cat")]
    alignment = {"matched": [(noun("cat"), dets[0], 0.99)], "hallucinated": [],
                 "missed": [dets[1]], "similarity_matrix": {}}
    r = make_evaluator(dets, nouns, alignment).score(image_path, "a cat")
    assert r["recall"] == pytest.approx(0.5)
    assert r["vtas_score"] == pytest.approx(2 / 3, abs=TOL)


def test_result_has_documented_keys(make_evaluator, image_path):
    alignment = {"matched": [], "hallucinated": [], "missed": [], "similarity_matrix": {}}
    r = make_evaluator([], [], alignment).score(image_path, "")
    assert {"vtas_score", "precision", "recall", "detected_objects", "text_nouns",
            "matched", "clip_grounded", "hallucinated", "missed",
            "similarity_matrix", "clip_scores"} <= set(r)
    assert "visual_recall" not in r and "hallucination_rate" not in r
