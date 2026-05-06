"""Session stickiness.

Default policy: once a session has been served by a higher tier, do not
downgrade in subsequent turns. This avoids the "model got dumber
mid-conversation" UX bug.

Setting `upgrade_only=False` in config disables stickiness entirely.
We never recommend a "downgrade allowed" mode.
"""

from __future__ import annotations

from llm_router.core.config import StickinessConfig
from llm_router.core.decision import (
    DecisionLayer,
    RoutingDecision,
    Tier,
)
from llm_router.session.store import SessionState, SessionStore

_TIER_RANK = {Tier.WEAK: 0, Tier.MID: 1, Tier.STRONG: 2}


class StickinessPolicy:
    def __init__(self, config: StickinessConfig) -> None:
        self.config = config

    def apply(
        self,
        session_store: SessionStore,
        session_id: str,
        proposed: RoutingDecision,
    ) -> RoutingDecision:
        existing = session_store.get(session_id)

        # If no prior turn, just record and return as-is.
        if existing is None:
            session_store.upsert(
                session_id, SessionState(highest_tier=proposed.tier, turn_count=1)
            )
            return proposed

        if not self.config.upgrade_only:
            # No stickiness; just update state.
            session_store.upsert(
                session_id, SessionState(highest_tier=proposed.tier)
            )
            return proposed

        proposed_rank = _TIER_RANK[proposed.tier]
        existing_rank = _TIER_RANK[existing.highest_tier]

        if proposed_rank >= existing_rank:
            session_store.upsert(
                session_id, SessionState(highest_tier=proposed.tier)
            )
            return proposed

        # Stick to the higher tier.
        adjusted = RoutingDecision(
            tier=existing.highest_tier,
            layer=DecisionLayer.SESSION_STICKINESS,
            reason=f"sticky:kept_{existing.highest_tier.value}_was_{proposed.tier.value}",
            confidence=1.0,
            classifier_score=proposed.classifier_score,
            rules_evaluated=proposed.rules_evaluated,
            metadata={"original_tier": proposed.tier.value, **proposed.metadata},
        )
        session_store.upsert(
            session_id, SessionState(highest_tier=existing.highest_tier)
        )
        return adjusted
