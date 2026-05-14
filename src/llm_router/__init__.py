"""llm-router: a production-oriented model routing layer.

Public API surface kept intentionally small. Importing the package does
not import heavy ML dependencies; those are loaded lazily by the
classifier module on first use.
"""

from llm_router.core.config import RouterConfig
from llm_router.core.decision import (
    AgentStepType,
    DecisionLayer,
    Message,
    Outcome,
    OutcomeKind,
    RoutingDecision,
    RoutingRequest,
    Tier,
    ToolCall,
)
from llm_router.core.router import Router

__all__ = [
    "AgentStepType",
    "DecisionLayer",
    "Message",
    "Outcome",
    "OutcomeKind",
    "Router",
    "RouterConfig",
    "RoutingDecision",
    "RoutingRequest",
    "Tier",
    "ToolCall",
]

__version__ = "0.1.0"
