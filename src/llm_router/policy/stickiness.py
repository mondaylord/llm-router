"""Session stickiness.

Default policy: once a session has been served by a higher tier on a
given step type, do not downgrade in subsequent turns of the same
step type. This avoids the "model got dumber mid-conversation" UX bug.

Crucially, stickiness is keyed by `(session_id, step_type)`. So:

  - planning step that escalated to STRONG -> next planning step STRONG
  - planning step STRONG -> next tool_call step CAN go WEAK
                            (different step type, no carryover)

This is the behavior agent-style workflows want. Chat-only callers
(no step_type) all share a single bucket and behave like classical
"global stickiness".

Setting `upgrade_only=False` in config disables stickiness entirely.
We never recommend a "downgrade allowed" mode for production.
"""

from __future__ import annotations

from llm_router.core.config import StickinessConfig
from llm_router.core.decision import (
    AgentStepType,
    DecisionLayer,
    RoutingDecision,
    Tier,
)
from llm_router.session.store import _GLOBAL_STEP, SessionState, SessionStore

_TIER_RANK = {Tier.WEAK: 0, Tier.MID: 1, Tier.STRONG: 2}


class StickinessPolicy:
    def __init__(self, config: StickinessConfig) -> None:
        self.config = config

    def apply(
        self,
        session_store: SessionStore,
        session_id: str,
        proposed: RoutingDecision,
        step_type: AgentStepType | None = None,
    ) -> RoutingDecision:
        step_key = step_type.value if step_type is not None else _GLOBAL_STEP
        existing = session_store.get(session_id)

        # Non-sticky step types: don't track or pin. Each turn stands alone.
        if step_key in set(self.config.non_sticky_step_types):
            return proposed

        if existing is None:
            session_store.upsert(
                session_id,
                SessionState(tier_by_step={step_key: proposed.tier}),
            )
            return proposed

        if not self.config.upgrade_only:
            # No stickiness; just record current tier for this step.
            state = SessionState(tier_by_step=dict(existing.tier_by_step))
            state.tier_by_step[step_key] = proposed.tier
            session_store.upsert(session_id, state)
            return proposed

        prev_for_step = existing.tier_by_step.get(step_key)
        if prev_for_step is None or _TIER_RANK[proposed.tier] >= _TIER_RANK[prev_for_step]:
            # Equal or upgrade — accept and remember.
            new_step_map = dict(existing.tier_by_step)
            new_step_map[step_key] = proposed.tier
            session_store.upsert(
                session_id, SessionState(tier_by_step=new_step_map)
            )
            return proposed

        # Sticky upgrade: keep the higher tier for this step.
        adjusted = RoutingDecision(
            tier=prev_for_step,
            layer=DecisionLayer.SESSION_STICKINESS,
            reason=(
                f"sticky:kept_{prev_for_step.value}_was_{proposed.tier.value}"
                f"[step={step_key}]"
            ),
            confidence=1.0,
            classifier_score=proposed.classifier_score,
            rules_evaluated=proposed.rules_evaluated,
            inferred_step_type=proposed.inferred_step_type,
            metadata={"original_tier": proposed.tier.value, **proposed.metadata},
        )
        new_step_map = dict(existing.tier_by_step)
        new_step_map[step_key] = prev_for_step
        session_store.upsert(
            session_id, SessionState(tier_by_step=new_step_map)
        )
        return adjusted
