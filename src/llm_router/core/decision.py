"""Core data types flowing through the routing pipeline.

These are *value* types: immutable, serializable, and free of business
logic. Anything that does work belongs in `router.py` or a layer module.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Tier(str, Enum):
    """Logical capability tier. Concrete model resolution is downstream."""

    WEAK = "weak"
    MID = "mid"
    STRONG = "strong"


class DecisionLayer(str, Enum):
    """Which layer in the pipeline produced the final decision."""

    TENANT_OVERRIDE = "tenant_override"
    SESSION_STICKINESS = "session_stickiness"
    RULE = "rule"
    CLASSIFIER = "classifier"
    DEFAULT = "default"


class RoutingRequest(BaseModel):
    """Input to the router. All fields except `prompt` are optional."""

    model_config = ConfigDict(extra="forbid")

    prompt: str
    session_id: str | None = None
    tenant_id: str | None = None
    # Latest-turn-only is fine for v0; full conversation can be added later.
    history_turns: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RoutingDecision(BaseModel):
    """Output of the router. Always emitted; never None."""

    model_config = ConfigDict(extra="forbid")

    tier: Tier
    layer: DecisionLayer
    reason: str
    """Human-readable, machine-stable. e.g. `rule:short_query`,
    `classifier:p_strong=0.23`, `tenant:forced_strong`."""

    confidence: float = Field(ge=0.0, le=1.0)
    """Routing confidence. 1.0 for deterministic layers (rule, override).
    For the classifier this is `max(p, 1-p)` in binary form."""

    classifier_score: float | None = None
    """Calibrated `p(needs_strong)` when the classifier ran; else None."""

    rules_evaluated: list[str] = Field(default_factory=list)
    """Names of rules that were evaluated in order. Useful for debugging
    why a decision was NOT made by an earlier rule."""

    metadata: dict[str, Any] = Field(default_factory=dict)


class RuleResult(BaseModel):
    """What a single rule emits. `tier=None` means 'pass through'."""

    model_config = ConfigDict(extra="forbid")

    tier: Tier | None = None
    reason: str = ""
    confidence: float = 1.0
