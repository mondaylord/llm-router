"""HTTP request/response schemas. Decoupled from internal types so the
wire contract can evolve without touching the core library."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from llm_router.core.decision import DecisionLayer, RoutingDecision, Tier


class RouteRequestBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str
    session_id: str | None = None
    tenant_id: str | None = None
    history_turns: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RouteResponseBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tier: Tier
    layer: DecisionLayer
    reason: str
    confidence: float
    classifier_score: float | None = None
    rules_evaluated: list[str]
    elapsed_ms: float

    @classmethod
    def from_decision(cls, d: RoutingDecision) -> RouteResponseBody:
        return cls(
            tier=d.tier,
            layer=d.layer,
            reason=d.reason,
            confidence=d.confidence,
            classifier_score=d.classifier_score,
            rules_evaluated=d.rules_evaluated,
            elapsed_ms=float(d.metadata.get("elapsed_ms", 0.0)),
        )


class HealthResponse(BaseModel):
    status: str
    classifier_loaded: bool
    rules_count: int
