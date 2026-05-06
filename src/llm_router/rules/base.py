"""Rule interface.

A rule inspects a `RoutingRequest` and either emits a `Tier` (early-exit
signal) or passes through. Rules should be:
  - cheap (target < 0.5ms each, total <5ms across the ruleset);
  - high-precision (false positives degrade quality silently);
  - explainable (reason string must indicate WHICH heuristic fired).

Authors of new rules: prefer to err on the side of NOT firing — let
the classifier handle ambiguity. Rules are for the obvious cases.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from llm_router.core.decision import RoutingRequest, RuleResult


class Rule(ABC):
    """Abstract base class for a routing rule."""

    name: str = "unnamed_rule"

    @abstractmethod
    def evaluate(self, request: RoutingRequest) -> RuleResult: ...


class FunctionRule(Rule):
    """Wrap a plain function as a Rule. Convenient for simple heuristics."""

    def __init__(self, name: str, fn) -> None:
        self.name = name
        self._fn = fn

    def evaluate(self, request: RoutingRequest) -> RuleResult:
        return self._fn(request)
