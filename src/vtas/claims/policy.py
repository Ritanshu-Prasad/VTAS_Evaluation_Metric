"""Concept policy loader: count intervals come from the YAML file, never from code (test U8).

Expression kinds (policy ``counts.expressions``): singular, numeral (needs n), plural, pair,
several, group, crowd, many. A value of ``fit_from_data`` falls back to ``plan_prior`` and the
returned interval is flagged ``provisional``; once the golden-set fit is frozen into the YAML
as a list or a per-profile mapping, it is used with no code change.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel

from .schema import CountInterval

DEFAULT_POLICY = "concept_policy_v1.yaml"


def default_policy_path() -> Path:
    """VTAS_POLICY, else policy/ in the repository checkout, else policy/ shipped in the package."""
    if os.environ.get("VTAS_POLICY"):
        return Path(os.environ["VTAS_POLICY"])
    here = Path(__file__).resolve()
    for root in (here.parents[3], here.parents[1]):
        p = root / "policy" / DEFAULT_POLICY
        if p.exists():
            return p
    raise FileNotFoundError(f"{DEFAULT_POLICY} not found; set VTAS_POLICY")


def _pair(v) -> tuple[int, int | None]:
    lo, hi = v
    return int(lo), (None if hi is None else int(hi))


class ConceptPolicy(BaseModel):
    version: str
    default_profile: str
    profiles: list[str]
    expressions: dict
    path: str = ""

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ConceptPolicy":
        p = Path(path) if path else default_policy_path()
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        prof = raw["profiles"]
        return cls(version=str(raw["version"]), default_profile=prof["default"],
                   profiles=[k for k in prof if k != "default"],
                   expressions=raw["counts"]["expressions"], path=str(p))

    def interval(self, kind: str, n: int | None = None, profile: str | None = None) -> CountInterval:
        profile = profile or self.default_profile
        if profile not in self.profiles:
            raise ValueError(f"unknown profile {profile!r}; have {self.profiles}")
        if kind not in self.expressions:
            raise KeyError(f"policy has no count expression {kind!r}")
        e = self.expressions[kind]
        if kind == "numeral":
            if n is None:
                raise ValueError("numeral needs n")
            tol = int(e["tolerance"][profile])
            return CountInterval(min=max(0, n - tol), max=n + tol, provisional=False)
        iv, provisional = e["interval"], False
        if iv == "fit_from_data":
            iv, provisional = e["plan_prior"], True
        if isinstance(iv, dict):
            iv = iv[profile]
        lo, hi = _pair(iv)
        return CountInterval(min=lo, max=hi, provisional=provisional)
