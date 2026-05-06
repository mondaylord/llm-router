"""The router pipeline.

Layers run in this fixed order:

  1. Pre-router: tenant policy, session state lookup
  2. Tenant override (forced tier short-circuit)
  3. Rule engine (Layer 1)
  4. Classifier (Layer 2)
  5. Default (configured `default_tier`, typically STRONG)
  6. Stickiness adjustment (post-process; never downgrades by default)
  7. Tenant blocked-tier guard (post-process)

Each layer either emits a decision (early-exit) or passes through.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from llm_router.core.config import RouterConfig
from llm_router.core.decision import (
    DecisionLayer,
    RoutingDecision,
    RoutingRequest,
    Tier,
)
from llm_router.observability.logger import get_logger
from llm_router.policy.stickiness import StickinessPolicy
from llm_router.policy.tenant import TenantPolicyResolver
from llm_router.rules.base import Rule
from llm_router.rules.builtin import default_ruleset
from llm_router.session.store import InMemorySessionStore, SessionStore

if TYPE_CHECKING:
    from llm_router.classifier.predictor import ClassifierPredictor

log = get_logger(__name__)


class Router:
    """Main entry point. Construct via `Router.from_config(...)` typically."""

    def __init__(
        self,
        config: RouterConfig,
        rules: list[Rule] | None = None,
        classifier: "ClassifierPredictor | None" = None,
        session_store: SessionStore | None = None,
    ) -> None:
        self.config = config
        self.rules = rules if rules is not None else default_ruleset()
        # Filter rules by config if requested.
        if config.rules.enabled_rule_names is not None:
            wanted = set(config.rules.enabled_rule_names)
            self.rules = [r for r in self.rules if r.name in wanted]

        self.classifier = classifier
        self.session_store = session_store or InMemorySessionStore(
            ttl_seconds=config.stickiness.ttl_seconds
        )
        self.tenant = TenantPolicyResolver(config.tenants)
        self.stickiness = StickinessPolicy(config.stickiness)

    @classmethod
    def from_config(cls, config: RouterConfig) -> Router:
        """Build with default rules and lazy-loaded classifier."""
        classifier = None
        if config.classifier.enabled and config.classifier.artifact_path:
            from llm_router.classifier.predictor import ClassifierPredictor

            classifier = ClassifierPredictor.load(
                artifact_path=config.classifier.artifact_path,
                embedding_model=config.classifier.embedding_model,
            )
        return cls(config=config, classifier=classifier)

    # ------------------------------------------------------------------
    # main entry point
    # ------------------------------------------------------------------
    def route(self, request: RoutingRequest) -> RoutingDecision:
        t0 = time.perf_counter()
        decision = self._route_inner(request)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        decision.metadata["elapsed_ms"] = round(elapsed_ms, 2)

        if self._should_log(request):
            log.info(
                "routing_decision",
                tier=decision.tier.value,
                layer=decision.layer.value,
                reason=decision.reason,
                confidence=decision.confidence,
                p_strong=decision.classifier_score,
                elapsed_ms=decision.metadata["elapsed_ms"],
                tenant_id=request.tenant_id,
                session_id=request.session_id,
                prompt_len=len(request.prompt),
            )
        return decision

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _route_inner(self, request: RoutingRequest) -> RoutingDecision:
        tenant_policy = self.tenant.resolve(request.tenant_id)

        # Layer 0: tenant override (short-circuits everything)
        if tenant_policy.forced_tier is not None:
            return self._maybe_apply_stickiness(
                request,
                RoutingDecision(
                    tier=tenant_policy.forced_tier,
                    layer=DecisionLayer.TENANT_OVERRIDE,
                    reason=f"tenant:forced_{tenant_policy.forced_tier.value}",
                    confidence=1.0,
                ),
            )

        # Layer 1: rules
        rules_evaluated: list[str] = []
        if self.config.rules.enabled:
            for rule in self.rules:
                rules_evaluated.append(rule.name)
                result = rule.evaluate(request)
                if result.tier is not None:
                    decision = RoutingDecision(
                        tier=result.tier,
                        layer=DecisionLayer.RULE,
                        reason=f"rule:{rule.name}" if not result.reason else result.reason,
                        confidence=result.confidence,
                        rules_evaluated=rules_evaluated,
                    )
                    return self._guard_and_stick(request, decision, tenant_policy)

        # Layer 2: classifier (skip if tenant requires strict latency)
        if (
            self.config.classifier.enabled
            and self.classifier is not None
            and not tenant_policy.latency_strict
        ):
            try:
                p_strong = self.classifier.predict_proba_strong(request.prompt)
            except Exception as exc:  # fail open to strong
                log.warning("classifier_failed", error=str(exc))
                p_strong = None

            if p_strong is not None:
                p_high = (
                    tenant_policy.classifier_p_high_override
                    if tenant_policy.classifier_p_high_override is not None
                    else self.config.classifier.p_high
                )
                p_low = self.config.classifier.p_low

                if p_strong >= p_high:
                    chosen = Tier.STRONG
                    reason = f"classifier:p_strong={p_strong:.3f}>=high"
                elif p_strong <= p_low:
                    chosen = Tier.WEAK
                    reason = f"classifier:p_strong={p_strong:.3f}<=low"
                else:
                    # Uncertainty band → safe default (strong)
                    chosen = Tier.STRONG
                    reason = f"classifier:p_strong={p_strong:.3f}_uncertain"

                decision = RoutingDecision(
                    tier=chosen,
                    layer=DecisionLayer.CLASSIFIER,
                    reason=reason,
                    confidence=max(p_strong, 1.0 - p_strong),
                    classifier_score=p_strong,
                    rules_evaluated=rules_evaluated,
                )
                return self._guard_and_stick(request, decision, tenant_policy)

        # Layer 3: default
        decision = RoutingDecision(
            tier=self.config.default_tier,
            layer=DecisionLayer.DEFAULT,
            reason=f"default:{self.config.default_tier.value}",
            confidence=1.0,
            rules_evaluated=rules_evaluated,
        )
        return self._guard_and_stick(request, decision, tenant_policy)

    # ------------------------------------------------------------------
    # post-processing helpers
    # ------------------------------------------------------------------
    def _guard_and_stick(
        self,
        request: RoutingRequest,
        decision: RoutingDecision,
        tenant_policy,
    ) -> RoutingDecision:
        # Tenant blocked-tier guard: bump up to strong if blocked.
        if decision.tier in tenant_policy.blocked_tiers:
            decision = RoutingDecision(
                tier=Tier.STRONG,
                layer=DecisionLayer.TENANT_OVERRIDE,
                reason=f"tenant:blocked_{decision.tier.value}_upgraded",
                confidence=1.0,
                classifier_score=decision.classifier_score,
                rules_evaluated=decision.rules_evaluated,
            )
        return self._maybe_apply_stickiness(request, decision)

    def _maybe_apply_stickiness(
        self, request: RoutingRequest, decision: RoutingDecision
    ) -> RoutingDecision:
        if not self.config.stickiness.enabled or request.session_id is None:
            return decision
        adjusted = self.stickiness.apply(
            session_store=self.session_store,
            session_id=request.session_id,
            proposed=decision,
        )
        return adjusted

    # ------------------------------------------------------------------
    def _should_log(self, request: RoutingRequest) -> bool:
        rate = self.config.logging.sample_rate
        if rate >= 1.0:
            return True
        if rate <= 0.0:
            return False
        # Deterministic by session/prompt to avoid log flicker per turn.
        key = request.session_id or request.prompt
        return (hash(key) % 10000) / 10000.0 < rate
