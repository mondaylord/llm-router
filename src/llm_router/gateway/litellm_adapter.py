"""LiteLLM gateway adapter.

LiteLLM (https://github.com/BerriAI/litellm) is a mature multi-provider
gateway with retry, fallback, observability, and budget tracking. We
delegate model invocation to it and keep our own surface limited to
"which tier should this go to".

This file is a working stub: it imports lazily so the package can be
installed without LiteLLM, and exposes the right call shape. Wire up
api keys, base urls, and provider-specific kwargs in your deployment.
"""

from __future__ import annotations

from typing import Any

from llm_router.core.config import GatewayConfig
from llm_router.core.decision import RoutingDecision
from llm_router.gateway.base import Gateway


class LiteLLMGateway(Gateway):
    def __init__(self, config: GatewayConfig) -> None:
        try:
            import litellm  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "litellm not installed. Install with `pip install 'llm-router[gateway]'`."
            ) from e
        self.config = config

    def invoke(
        self,
        decision: RoutingDecision,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        import litellm

        model = self.config.tier_to_model.get(decision.tier)
        if not model:
            raise ValueError(
                f"No concrete model configured for tier {decision.tier.value}. "
                f"Set gateway.tier_to_model in your RouterConfig."
            )
        response = litellm.completion(
            model=model,
            messages=messages,
            **kwargs,
        )
        return response  # type: ignore[return-value]
