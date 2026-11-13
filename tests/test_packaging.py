"""Guards against the version/metadata drift found in 1.3.0."""

import re
from pathlib import Path

import vtas
from vtas import _spacy_model

ROOT = Path(__file__).resolve().parent.parent


def test_citation_version_matches_package():
    text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert re.search(rf'^version:\s*"{re.escape(vtas.__version__)}"', text, re.M)


def test_no_hardcoded_version_in_pyproject():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'dynamic = ["version"]' in text
    assert not re.search(r'^version\s*=\s*"\d', text, re.M)


def test_metadata_has_no_placeholder_email_or_wrong_summary():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "example.com" not in text
    assert "Visual-Truth Alignment Score" in text
    assert "Visual-Text" not in text


def test_missing_spacy_model_message_names_the_fix():
    msg = _spacy_model.missing_message("en_core_web_sm")
    assert "vtas download" in msg and "en_core_web_sm" in msg
