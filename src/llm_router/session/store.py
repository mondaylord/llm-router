"""Session state store.

Holds per-session routing state used by the stickiness policy. The
in-memory implementation is fine for single-process deployments and
tests; production should swap in a Redis-backed implementation
behind the same `SessionStore` interface.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from llm_router.core.decision import Tier


@dataclass
class SessionState:
    """Per-session minimum state.

    Tracks the highest tier reached so we can implement upgrade-only
    stickiness without rewriting older history.
    """

    highest_tier: Tier
    turn_count: int = 0
    last_seen_unix: float = field(default_factory=time.time)


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
        self._tier_rank = {Tier.WEAK: 0, Tier.MID: 1, Tier.STRONG: 2}

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
        existing = self.get(session_id)
        if existing is not None:
            # Always keep the highest tier ever seen.
            new_rank = self._tier_rank[state.highest_tier]
            old_rank = self._tier_rank[existing.highest_tier]
            if new_rank < old_rank:
                state.highest_tier = existing.highest_tier
            state.turn_count = existing.turn_count + 1
        else:
            state.turn_count = max(state.turn_count, 1)
        state.last_seen_unix = time.time()
        self._data[session_id] = state

    # convenience for tests
    def __len__(self) -> int:
        return len(self._data)
