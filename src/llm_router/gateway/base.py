"""Downstream gateway abstraction.

The router's job is to decide which tier should serve a request. Actually
calling the model is delegated to a `Gateway`. This separation means:

- you can use the router as a decision-only library (no upstream calls)
  by leaving the gateway as `NoopGateway`;
- the LiteLLM adapter is the recommended production path; you wire it
  up by mapping tier → concrete model id in config.gateway.tier_to_model.

A LiteLLM adapter sketch is in `litellm_adapter.py`. It is a stub
until you decide on auth/credential handling, which is environment-
specific.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from llm_router.core.config import GatewayConfig
from llm_router.core.decision import RoutingDecision


class Gateway(ABC):
    """Translates a (decision, request_payload) pair into a model call."""

    @abstractmethod
    def invoke(
        self,
        decision: RoutingDecision,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call the model. Returns provider-shaped response dict."""


class NoopGateway(Gateway):
    """Default: do nothing. Returns a synthetic envelope so the
    framework can be exercised without a live provider."""

    def __init__(self, config: GatewayConfig | None = None) -> None:
        self.config = config

    def invoke(
        self,
        decision: RoutingDecision,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "noop": True,
            "tier": decision.tier.value,
            "reason": decision.reason,
            "messages_received": len(messages),
            "would_call_model": (
                self.config.tier_to_model.get(decision.tier)
                if self.config
                else None
            ),
        }
