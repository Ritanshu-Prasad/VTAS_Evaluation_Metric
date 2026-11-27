"""Typed claim objects produced by the claim parser."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

ClaimType = Literal["object", "count", "attribute", "relation", "scene"]
QuantityKind = Literal["singular", "numeral", "plural", "pair", "several", "group", "crowd", "many"]


class CountInterval(BaseModel):
    """Inclusive interval of visible instances; max None = no upper bound."""
    min: int
    max: Optional[int]
    provisional: bool = False  # True while the policy value is still the plan prior

    def contains(self, n: int) -> bool:
        return n >= self.min and (self.max is None or n <= self.max)


class Quantity(BaseModel):
    kind: QuantityKind
    n: Optional[int] = None      # numeral value
    expression: str = ""         # the caption's own quantity words


class Claim(BaseModel):
    type: ClaimType
    text: str                    # claim_text in the golden-set rubric shape (docs/GOLDEN_SET_RUBRIC.md 2)
    entity: Optional[str] = None      # object, count, attribute, scene: the entity name
    attribute: Optional[str] = None   # attribute claims
    subject: Optional[str] = None     # relation claims
    predicate: Optional[str] = None
    object: Optional[str] = None
    quantity: Optional[Quantity] = None  # count claims; singular object claims
    start: int = -1              # character span in the caption
    end: int = -1

    def interval(self, policy, profile: str | None = None) -> Optional[CountInterval]:
        """Count interval under a policy and strictness profile; None if the claim has no quantity."""
        if self.quantity is None:
            return None
        return policy.interval(self.quantity.kind, self.quantity.n, profile)


class ParseResult(BaseModel):
    text: str
    model: str
    policy_version: str
    claims: list[Claim] = Field(default_factory=list)

    def of_type(self, t: ClaimType) -> list[Claim]:
        return [c for c in self.claims if c.type == t]

    def texts(self, t: ClaimType | None = None) -> list[str]:
        return [c.text for c in self.claims if t is None or c.type == t]
