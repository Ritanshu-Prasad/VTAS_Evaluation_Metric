"""Tests for the linguistic and semantic-bridge modules.

* ``test_bridge_*`` use a deterministic stub encoder (one-hot vectors, with
  synonyms sharing a vector) so the Hungarian matching and plural-exemption
  logic run fast with no model download.
* Tests marked ``integration`` load the real spaCy / MiniLM models and run
  only with ``pytest -m integration``.
"""

from types import SimpleNamespace

import pytest

from conftest import det, noun

torch = pytest.importorskip("torch")
pytest.importorskip("sentence_transformers")
from vtas.semantic_bridge import SemanticBridgeModule  # noqa: E402


def _labels(items):
    """Labels/texts of a list of claim or detection dicts."""
    return [i["text"] if "text" in i else i["label"] for i in items]


def _stub_bridge(groups, threshold=0.65):
    """A SemanticBridgeModule whose encoder maps each phrase to a one-hot
    vector; phrases in the same synonym group share a vector (cosine 1.0),
    all other pairs are orthogonal (cosine 0.0)."""
    index = {}
    for gi, group in enumerate(groups):
        for word in group:
            index[f"a {word}"] = gi
    dim = len(groups) + 8  # spare dims for words outside every group

    def encode(phrases, convert_to_tensor=True):
        out = torch.zeros(len(phrases), dim)
        for i, p in enumerate(phrases):
            out[i, index.get(p, len(groups) + (hash(p) % 8))] = 1.0
        return out

    bridge = SemanticBridgeModule.__new__(SemanticBridgeModule)
    bridge.threshold = threshold
    bridge.model = SimpleNamespace(encode=encode)
    return bridge


GROUPS = [("person",), ("dog",), ("car",), ("couch", "sofa"), ("tv", "television")]


def test_bridge_perfect_match():
    bridge = _stub_bridge(GROUPS)
    dets = [det("person", 0.3), det("dog", 0.2)]
    result = bridge.compute_alignment(dets, [noun("person"), noun("dog")])
    assert result["hallucinated"] == []
    assert result["missed"] == []
    assert sorted(_labels([m[0] for m in result["matched"]])) == ["dog", "person"]


def test_bridge_synonyms_match():
    bridge = _stub_bridge(GROUPS)
    dets = [det("couch", 0.4), det("tv", 0.2), det("person", 0.3)]
    nouns = [noun("sofa"), noun("television"), noun("man")]  # man -> person (ontology)
    result = bridge.compute_alignment(dets, nouns)
    assert result["hallucinated"] == []
    assert result["missed"] == []
    assert len(result["matched"]) == 3


def test_bridge_hallucination_and_miss():
    bridge = _stub_bridge(GROUPS)
    dets = [det("person", 0.5), det("car", 0.3)]
    result = bridge.compute_alignment(dets, [noun("person"), noun("dragon")])
    assert _labels(result["hallucinated"]) == ["dragon"]
    assert _labels(result["missed"]) == ["car"]


def test_bridge_empty_inputs():
    bridge = _stub_bridge(GROUPS)
    empty_caption = bridge.compute_alignment([det("person", 0.5)], [])
    assert empty_caption["matched"] == []
    assert empty_caption["hallucinated"] == []
    assert _labels(empty_caption["missed"]) == ["person"]

    empty_dets = bridge.compute_alignment([], [noun("person")])
    assert empty_dets["matched"] == []
    assert _labels(empty_dets["hallucinated"]) == ["person"]
    assert empty_dets["missed"] == []


def test_bridge_plural_noun_exempts_remaining_instances():
    bridge = _stub_bridge(GROUPS)
    dets = [det("person", 0.2) for _ in range(3)]
    plural = bridge.compute_alignment(dets, [noun("man", is_plural=True)])
    assert len(plural["matched"]) == 1
    assert plural["missed"] == []

    singular = bridge.compute_alignment(dets, [noun("man", is_plural=False)])
    assert len(singular["matched"]) == 1
    assert len(singular["missed"]) == 2


# --------------------------------------------------------------------------
# Integration tests (real models; `pytest -m integration`)
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def nlp_module():
    from vtas.linguistic_extraction import LinguisticExtractionModule
    return LinguisticExtractionModule()


@pytest.fixture(scope="module")
def real_bridge():
    return SemanticBridgeModule(threshold=0.65)


@pytest.mark.integration
def test_extraction_filters_generic_nouns(nlp_module):
    texts = _labels(nlp_module.extract("A group of people taking a photo of a view."))
    assert "group" not in texts
    assert "photo" not in texts
    assert "view" not in texts
    assert "people" in texts or "person" in texts


@pytest.mark.integration
def test_extraction_lemmatizes_and_flags_plurals(nlp_module):
    claims = nlp_module.extract("Two dogs chase three cats near the cars.")
    by_text = {c["text"]: c for c in claims}
    assert {"dog", "cat", "car"} <= set(by_text)
    assert "dogs" not in by_text
    assert by_text["dog"]["is_plural"] and by_text["cat"]["is_plural"]


@pytest.mark.integration
def test_extraction_strips_caption_prefix(nlp_module):
    texts = _labels(nlp_module.extract("detailed caption: a man throws a frisbee"))
    assert "caption" not in texts
    assert "man" in texts and "frisbee" in texts


@pytest.mark.integration
def test_real_bridge_synonyms(real_bridge):
    dets = [det("couch", 0.4), det("tv", 0.2), det("person", 0.3)]
    nouns = [noun("sofa"), noun("television"), noun("man")]
    result = real_bridge.compute_alignment(dets, nouns)
    assert result["hallucinated"] == []
    assert len(result["matched"]) == 3


@pytest.mark.integration
def test_real_bridge_hallucination(real_bridge):
    dets = [det("person", 0.5), det("car", 0.3)]
    result = real_bridge.compute_alignment(dets, [noun("person"), noun("dragon")])
    assert _labels(result["hallucinated"]) == ["dragon"]
    assert _labels(result["missed"]) == ["car"]
