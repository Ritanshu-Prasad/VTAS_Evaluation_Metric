"""Helpers for locating and downloading the spaCy model VTAS needs."""

import subprocess
import sys

DEFAULT_MODEL = "en_core_web_sm"


class SpacyModelMissingError(OSError):
    """Raised when the required spaCy model is not installed."""


def is_installed(model_name: str = DEFAULT_MODEL) -> bool:
    import importlib.util

    return importlib.util.find_spec(model_name) is not None


def missing_message(model_name: str = DEFAULT_MODEL) -> str:
    return (
        f"The spaCy model '{model_name}' is not installed. "
        f"Install it with:  vtas download   "
        f"(or: python -m spacy download {model_name})"
    )


def download(model_name: str = DEFAULT_MODEL) -> None:
    """Download the spaCy model into the current environment."""
    subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])


def ensure(model_name: str = DEFAULT_MODEL) -> None:
    """Download the model if it is not already installed."""
    if not is_installed(model_name):
        print(f"[VTAS] spaCy model '{model_name}' not found; downloading...")
        download(model_name)
