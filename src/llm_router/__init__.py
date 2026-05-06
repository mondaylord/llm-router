"""llm-router: a production-oriented model routing layer.

Public API surface kept intentionally small. Importing the package does
not import heavy ML dependencies; those are loaded lazily by the
classifier module on first use.
"""

from llm_router.core.config import RouterConfig
from llm_router.core.decision import (
    DecisionLayer,
    RoutingDecision,
    RoutingRequest,
    Tier,
)
from llm_router.core.router import Router

__all__ = [
    "DecisionLayer",
    "Router",
    "RouterConfig",
    "RoutingDecision",
    "RoutingRequest",
    "Tier",
]

__version__ = "0.1.0"
