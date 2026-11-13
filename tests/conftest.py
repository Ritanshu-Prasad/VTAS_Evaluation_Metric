"""Shared fixtures. Fast tests need no model downloads; integration tests do."""

from types import SimpleNamespace

import pytest
from PIL import Image

from vtas.vtas import VTASEvaluator


def det(label, area):
    """A DETR-style detection dict."""
    return {"label": label, "area": area}


def noun(text, is_plural=False):
    """A linguistic-extraction-style claim dict."""
    return {"text": text, "is_plural": is_plural}


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "blank.png"
    Image.new("RGB", (8, 8)).save(path)
    return str(path)


@pytest.fixture
def make_evaluator():
    """Build a VTASEvaluator whose four modules are stubs (no models loaded).

    The stubs return exactly what the real modules would, so only the
    scoring arithmetic inside ``score()`` is exercised.
    """

    def _make(detections, nouns, alignment, clip_grounded=(), clip_hallucinated=(),
              beta=1.0):
        ev = VTASEvaluator.__new__(VTASEvaluator)  # skip __init__ (model loading)
        ev.beta = beta
        ev.visual_module = SimpleNamespace(detect=lambda image: detections)
        ev.linguistic_module = SimpleNamespace(extract=lambda caption: nouns)
        ev.semantic_module = SimpleNamespace(
            compute_alignment=lambda detected_objects, text_nouns: alignment
        )
        calls = []

        def verify(image, to_check):
            calls.append(list(to_check))
            return {
                "grounded": list(clip_grounded),
                "hallucinated": list(clip_hallucinated),
                "scores": {},
            }

        ev.clip_module = SimpleNamespace(verify=verify)
        ev.clip_calls = calls
        return ev

    return _make
