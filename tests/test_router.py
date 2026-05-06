"""End-to-end pipeline tests for the Router (without classifier)."""

from __future__ import annotations

from llm_router import Router, RouterConfig, RoutingRequest, Tier
from llm_router.core.config import (
    ClassifierConfig,
    StickinessConfig,
    TenantConfig,
    TenantPolicyEntry,
)
from llm_router.core.decision import DecisionLayer


def _build(config: RouterConfig | None = None) -> Router:
    cfg = config or RouterConfig.default()
    cfg.classifier = ClassifierConfig(enabled=False, artifact_path=None)
    return Router.from_config(cfg)


def test_short_greeting_routes_to_weak_via_rule():
    router = _build()
    d = router.route(RoutingRequest(prompt="hi"))
    assert d.tier is Tier.WEAK
    assert d.layer is DecisionLayer.RULE


def test_long_prompt_routes_to_strong_via_rule():
    router = _build()
    d = router.route(RoutingRequest(prompt="x" * 5000))
    assert d.tier is Tier.STRONG
    assert d.layer is DecisionLayer.RULE


def test_unmatched_falls_through_to_default_strong():
    router = _build()
    d = router.route(RoutingRequest(
        prompt="Tell me about the latest in quantum computing research."
    ))
    assert d.tier is Tier.STRONG
    assert d.layer is DecisionLayer.DEFAULT


def test_tenant_forced_tier_short_circuits():
    cfg = RouterConfig.default()
    cfg.classifier = ClassifierConfig(enabled=False, artifact_path=None)
    cfg.tenants = TenantConfig(
        default_policy=TenantPolicyEntry(),
        overrides={"acme": TenantPolicyEntry(forced_tier=Tier.STRONG)},
    )
    router = Router.from_config(cfg)
    d = router.route(RoutingRequest(prompt="hi", tenant_id="acme"))
    assert d.tier is Tier.STRONG
    assert d.layer is DecisionLayer.TENANT_OVERRIDE


def test_tenant_blocked_tier_upgrades_decision():
    cfg = RouterConfig.default()
    cfg.classifier = ClassifierConfig(enabled=False, artifact_path=None)
    cfg.tenants = TenantConfig(
        default_policy=TenantPolicyEntry(),
        overrides={"a": TenantPolicyEntry(blocked_tiers=[Tier.WEAK])},
    )
    router = Router.from_config(cfg)
    d = router.route(RoutingRequest(prompt="hi", tenant_id="a"))
    assert d.tier is Tier.STRONG
    assert d.layer is DecisionLayer.TENANT_OVERRIDE
    assert "blocked" in d.reason


def test_session_stickiness_prevents_downgrade():
    cfg = RouterConfig.default()
    cfg.classifier = ClassifierConfig(enabled=False, artifact_path=None)
    cfg.stickiness = StickinessConfig(enabled=True, upgrade_only=True)
    router = Router.from_config(cfg)

    # Turn 1: long prompt -> STRONG
    d1 = router.route(RoutingRequest(
        prompt="x" * 5000,
        session_id="sess-1",
    ))
    assert d1.tier is Tier.STRONG

    # Turn 2: would be WEAK on its own
    d2 = router.route(RoutingRequest(
        prompt="hi",
        session_id="sess-1",
    ))
    assert d2.tier is Tier.STRONG  # stickiness kept us on strong
    assert d2.layer is DecisionLayer.SESSION_STICKINESS


def test_session_stickiness_allows_upgrade():
    cfg = RouterConfig.default()
    cfg.classifier = ClassifierConfig(enabled=False, artifact_path=None)
    cfg.stickiness = StickinessConfig(enabled=True, upgrade_only=True)
    router = Router.from_config(cfg)

    d1 = router.route(RoutingRequest(prompt="hi", session_id="sess-2"))
    assert d1.tier is Tier.WEAK

    d2 = router.route(RoutingRequest(prompt="x" * 5000, session_id="sess-2"))
    assert d2.tier is Tier.STRONG  # upgrade allowed


def test_session_stickiness_off_allows_downgrade():
    cfg = RouterConfig.default()
    cfg.classifier = ClassifierConfig(enabled=False, artifact_path=None)
    cfg.stickiness = StickinessConfig(enabled=False)
    router = Router.from_config(cfg)

    router.route(RoutingRequest(prompt="x" * 5000, session_id="s"))
    d2 = router.route(RoutingRequest(prompt="hi", session_id="s"))
    assert d2.tier is Tier.WEAK


def test_decision_includes_elapsed_ms_and_rules_evaluated():
    router = _build()
    d = router.route(RoutingRequest(prompt="hello"))
    assert "elapsed_ms" in d.metadata
    assert d.metadata["elapsed_ms"] >= 0
    assert isinstance(d.rules_evaluated, list)
