"""Session state store.

Holds per-session routing state used by the stickiness policy. The
in-memory implementation is fine for single-process deployments and
tests; production should swap in a Redis-backed implementation behind
the same `SessionStore` interface.

State is keyed by `(session_id, step_type)`. This matters because in
agent flows, the right tier for a given session depends on what kind of
step is currently running — a planning step that escalated to STRONG
should not pin the next safe tool_call step to STRONG. Chat-style
callers that don't pass a step type all use a single bucket
(`_GLOBAL_STEP`) and so behave exactly as before.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from llm_router.core.decision import Tier

_GLOBAL_STEP = "_global"


@dataclass
class SessionState:
    """Per-session minimum state.

    `tier_by_step` maps step-type-string to the highest tier seen for
    that step in this session. `_global` is the bucket used for chat
    requests where no step type is meaningful."""

    tier_by_step: dict[str, Tier] = field(default_factory=dict)
    turn_count: int = 0
    last_seen_unix: float = field(default_factory=time.time)

    # Convenience for callers that only care about the overall high-water
    # mark (e.g. metrics, dashboards).
    @property
    def highest_tier_overall(self) -> Tier | None:
        if not self.tier_by_step:
            return None
        rank = {Tier.WEAK: 0, Tier.MID: 1, Tier.STRONG: 2}
        return max(self.tier_by_step.values(), key=lambda t: rank[t])


class SessionStore(ABC):
    @abstractmethod
    def get(self, session_id: str) -> SessionState | None: ...

    @abstractmethod
    def upsert(self, session_id: str, state: SessionState) -> None: ...


class InMemorySessionStore(SessionStore):
    """Trivial dict store with TTL-based eviction.

    Eviction runs lazily on read/write; OK for moderate cardinalities.
    Replace with Redis at production scale (or partition by tenant)."""

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self.ttl_seconds = ttl_seconds
        self._data: dict[str, SessionState] = {}

    def _evict_if_stale(self, session_id: str) -> None:
        st = self._data.get(session_id)
        if st is None:
            return
        if time.time() - st.last_seen_unix > self.ttl_seconds:
            self._data.pop(session_id, None)

    def get(self, session_id: str) -> SessionState | None:
        self._evict_if_stale(session_id)
        return self._data.get(session_id)

    def upsert(self, session_id: str, state: SessionState) -> None:
        state.last_seen_unix = time.time()
        existing = self._data.get(session_id)
        if existing is not None:
            state.turn_count = existing.turn_count + 1
        else:
            state.turn_count = max(state.turn_count, 1)
        self._data[session_id] = state

    # convenience for tests
    def __len__(self) -> int:
        return len(self._data)
