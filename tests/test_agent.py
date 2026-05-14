"""Tests for agent-mode routing: step-type detection, agent rules,
per-step-type stickiness, failure escalation."""

from __future__ import annotations

import pytest

from llm_router import (
    AgentStepType,
    DecisionLayer,
    Message,
    Outcome,
    OutcomeKind,
    Router,
    RouterConfig,
    RoutingRequest,
    Tier,
    ToolCall,
)
from llm_router.core.config import AgentConfig, ClassifierConfig, StickinessConfig
from llm_router.core.messages import detect_step_type


def _make_router(
    weak_safe: list[str] | None = None,
    requires_strong: list[str] | None = None,
    long_threshold: int = 8000,
    non_sticky: list[str] | None = None,
) -> Router:
    cfg = RouterConfig.default()
    cfg.classifier = ClassifierConfig(enabled=False)
    cfg.agent = AgentConfig(
        enabled=True,
        weak_safe_tools=weak_safe or ["read_file", "grep", "ls"],
        requires_strong_tools=requires_strong or ["edit_file", "run_shell"],
        long_context_threshold_tokens=long_threshold,
        failure_escalation_enabled=True,
    )
    cfg.stickiness = StickinessConfig(
        enabled=True,
        upgrade_only=True,
        non_sticky_step_types=non_sticky or [],
    )
    return Router.from_config(cfg)


# ----------------------------------------------------------------------
# step-type detection
# ----------------------------------------------------------------------


def test_detect_step_type_tool_result():
    req = RoutingRequest(
        messages=[
            Message(role="user", content="run grep"),
            Message(role="assistant", tool_calls=[ToolCall(name="grep", arguments={})]),
            Message(role="tool", name="grep", content="hits: 1"),
        ]
    )
    assert detect_step_type(req) is AgentStepType.TOOL_RESULT


def test_detect_step_type_tool_call_from_assistant_msg():
    req = RoutingRequest(
        messages=[
            Message(role="user", content="please grep"),
            Message(role="assistant", tool_calls=[ToolCall(name="grep", arguments={})]),
        ]
    )
    assert detect_step_type(req) is AgentStepType.TOOL_CALL


def test_detect_step_type_planning_from_system_or_user():
    req = RoutingRequest(prompt="Plan how to refactor the auth module step by step.")
    assert detect_step_type(req) is AgentStepType.PLANNING


def test_detect_step_type_edit_from_user_cue():
    req = RoutingRequest(prompt="Please edit the file auth.py and remove dead code.")
    assert detect_step_type(req) is AgentStepType.EDIT


def test_detect_step_type_summarize():
    req = RoutingRequest(prompt="TL;DR this thread please.")
    assert detect_step_type(req) is AgentStepType.SUMMARIZE


def test_detect_step_type_defaults_to_chat():
    req = RoutingRequest(prompt="hello")
    assert detect_step_type(req) is AgentStepType.CHAT


def test_detect_step_type_tool_call_default_when_tools_available():
    req = RoutingRequest(prompt="anything", available_tools=["grep", "read_file"])
    assert detect_step_type(req) is AgentStepType.TOOL_CALL


# ----------------------------------------------------------------------
# agent rules
# ----------------------------------------------------------------------


def test_planning_routes_strong():
    router = _make_router()
    d = router.route(
        RoutingRequest(
            prompt="Plan how to refactor the auth middleware step by step.",
        )
    )
    assert d.tier is Tier.STRONG
    assert "planning" in d.reason


def test_safe_tool_call_routes_weak():
    router = _make_router(weak_safe=["read_file"])
    d = router.route(
        RoutingRequest(
            prompt="read auth.py",
            agent_step_type=AgentStepType.TOOL_CALL,
            planned_tool="read_file",
        )
    )
    assert d.tier is Tier.WEAK
    assert "safe_tool" in d.reason


def test_requires_strong_tool_forces_strong():
    router = _make_router(requires_strong=["edit_file"])
    d = router.route(
        RoutingRequest(
            prompt="edit auth.py to add validation",
            agent_step_type=AgentStepType.TOOL_CALL,
            planned_tool="edit_file",
        )
    )
    assert d.tier is Tier.STRONG
    assert "requires_strong_tool" in d.reason


def test_tool_result_interpretation_routes_weak_when_small():
    router = _make_router()
    d = router.route(
        RoutingRequest(
            messages=[
                Message(role="user", content="?"),
                Message(role="tool", name="grep", content="hit: line 12"),
            ]
        )
    )
    assert d.tier is Tier.WEAK
    assert "tool_result_small" in d.reason


def test_long_context_routes_strong():
    router = _make_router(long_threshold=1000)
    d = router.route(
        RoutingRequest(
            prompt="anything",
            total_context_tokens=5000,
        )
    )
    assert d.tier is Tier.STRONG
    assert "long_context" in d.reason


def test_recent_failure_escalates_to_strong():
    router = _make_router(weak_safe=["read_file"])
    # Without failure, this would be WEAK via safe-tool rule.
    d = router.route(
        RoutingRequest(
            prompt="retry",
            agent_step_type=AgentStepType.TOOL_CALL,
            planned_tool="read_file",
            recent_outcomes=[
                Outcome(kind=OutcomeKind.TOOL_SCHEMA_ERROR, tool_name="read_file"),
            ],
        )
    )
    assert d.tier is Tier.STRONG
    assert "recent_failure" in d.reason


def test_summarize_routes_weak():
    router = _make_router()
    d = router.route(RoutingRequest(prompt="Please summarize this thread."))
    assert d.tier is Tier.WEAK
    assert "summarize" in d.reason


def test_edit_step_small_routes_weak():
    router = _make_router()
    d = router.route(
        RoutingRequest(
            prompt="edit the file: rename x to y",  # short
            agent_step_type=AgentStepType.EDIT,
        )
    )
    assert d.tier is Tier.WEAK


def test_edit_step_large_routes_strong():
    router = _make_router()
    d = router.route(
        RoutingRequest(
            prompt="edit the file " + ("blah " * 200),  # large
            agent_step_type=AgentStepType.EDIT,
        )
    )
    assert d.tier is Tier.STRONG


# ----------------------------------------------------------------------
# per-step-type stickiness
# ----------------------------------------------------------------------


def test_stickiness_isolates_step_types():
    router = _make_router(weak_safe=["read_file"], requires_strong=["edit_file"])
    sid = "agent-sess"

    d1 = router.route(
        RoutingRequest(
            prompt="plan a refactor step by step",
            session_id=sid,
        )
    )
    assert d1.tier is Tier.STRONG  # planning -> strong

    # Subsequent tool_call step in same session — should NOT inherit strong.
    d2 = router.route(
        RoutingRequest(
            prompt="read it",
            session_id=sid,
            agent_step_type=AgentStepType.TOOL_CALL,
            planned_tool="read_file",
        )
    )
    assert d2.tier is Tier.WEAK
    assert d2.layer is not DecisionLayer.SESSION_STICKINESS


def test_stickiness_persists_within_step_type():
    # Without non_sticky_step_types, tool_call sticks across turns.
    router = _make_router(
        weak_safe=["read_file"],
        requires_strong=["edit_file"],
        non_sticky=[],
    )
    sid = "agent-sess-2"

    d1 = router.route(
        RoutingRequest(
            prompt="edit it",
            session_id=sid,
            agent_step_type=AgentStepType.TOOL_CALL,
            planned_tool="edit_file",
        )
    )
    assert d1.tier is Tier.STRONG

    d2 = router.route(
        RoutingRequest(
            prompt="read it",
            session_id=sid,
            agent_step_type=AgentStepType.TOOL_CALL,
            planned_tool="read_file",
        )
    )
    # tool_call step already pinned to strong -> stickiness keeps it strong.
    assert d2.tier is Tier.STRONG
    assert d2.layer is DecisionLayer.SESSION_STICKINESS


def test_non_sticky_step_types_allow_re_decision():
    router = _make_router(
        weak_safe=["read_file"],
        requires_strong=["edit_file"],
        non_sticky=["tool_call"],
    )
    sid = "agent-sess-3"

    d1 = router.route(
        RoutingRequest(
            prompt="edit",
            session_id=sid,
            agent_step_type=AgentStepType.TOOL_CALL,
            planned_tool="edit_file",
        )
    )
    assert d1.tier is Tier.STRONG

    # tool_call is non-sticky → next safe tool call gets re-decided.
    d2 = router.route(
        RoutingRequest(
            prompt="read",
            session_id=sid,
            agent_step_type=AgentStepType.TOOL_CALL,
            planned_tool="read_file",
        )
    )
    assert d2.tier is Tier.WEAK


# ----------------------------------------------------------------------
# backward compat: chat-only requests still work
# ----------------------------------------------------------------------


def test_chat_only_request_still_works_with_agent_enabled():
    router = _make_router()
    d = router.route(RoutingRequest(prompt="hi"))
    assert d.tier is Tier.WEAK  # short_greeting builtin rule
    assert d.layer is DecisionLayer.RULE


def test_request_validation_requires_text():
    with pytest.raises(ValueError):
        RoutingRequest()  # neither prompt nor messages


def test_messages_only_request_uses_effective_text():
    router = _make_router()
    d = router.route(
        RoutingRequest(
            messages=[
                Message(role="user", content="hi"),
            ]
        )
    )
    assert d.tier is Tier.WEAK
