"""HTTP request/response schemas. Decoupled from internal types so the
wire contract can evolve without touching the core library."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from llm_router.core.decision import (
    AgentStepType,
    DecisionLayer,
    Message,
    Outcome,
    RoutingDecision,
    Tier,
)


class RouteRequestBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Either prompt or messages.
    prompt: str | None = None
    messages: list[Message] | None = None

    session_id: str | None = None
    tenant_id: str | None = None

    # Agent fields. All optional — chat-only clients can ignore.
    agent_step_type: AgentStepType | None = None
    available_tools: list[str] = Field(default_factory=list)
    planned_tool: str | None = None
    last_tool_called: str | None = None
    recent_outcomes: list[Outcome] = Field(default_factory=list)
    total_context_tokens: int = 0

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
    inferred_step_type: AgentStepType | None = None

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
            inferred_step_type=d.inferred_step_type,
        )


class HealthResponse(BaseModel):
    status: str
    classifier_loaded: bool
    rules_count: int
