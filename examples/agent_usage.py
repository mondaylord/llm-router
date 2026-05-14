"""Agent-mode end-to-end demo.

Walks through a synthetic Cursor-style auto-mode session:

  1. user asks for a refactor              -> PLANNING -> strong
  2. agent calls `read_file`               -> safe tool -> weak
  3. agent interprets the read result      -> small tool result -> weak
  4. agent calls `grep`                    -> safe tool -> weak
  5. agent calls `edit_file`               -> requires strong -> strong
  6. agent reports a tool schema error
     on the next turn                      -> failure escalation -> strong
  7. follow-up plain chat in same session  -> stickiness keeps strong
     ONLY for the chat step bucket once
     escalated; tool_call bucket can still
     go weak                                  (see step 8)
  8. agent calls `read_file` again         -> still weak (per-step-type stickiness)
"""

from __future__ import annotations

from llm_router import (
    AgentStepType,
    Message,
    Outcome,
    OutcomeKind,
    Router,
    RouterConfig,
    RoutingRequest,
    ToolCall,
)
from llm_router.core.config import AgentConfig, ClassifierConfig, StickinessConfig
from llm_router.observability.logger import configure_logging


def main() -> None:
    configure_logging(level="WARNING", json=False)

    cfg = RouterConfig.default()
    cfg.classifier = ClassifierConfig(enabled=False)
    cfg.agent = AgentConfig(
        enabled=True,
        weak_safe_tools=["read_file", "list_dir", "grep", "ls"],
        requires_strong_tools=["edit_file", "write_file", "run_shell", "run_tests"],
        long_context_threshold_tokens=8000,
        failure_escalation_enabled=True,
    )
    # Tool calls are atomic; don't pin a session's tool_call tier across turns.
    cfg.stickiness = StickinessConfig(
        enabled=True,
        upgrade_only=True,
        non_sticky_step_types=["tool_call", "tool_result", "summarize"],
    )
    router = Router.from_config(cfg)

    sid = "agent-session-1"

    def show(label: str, req: RoutingRequest):
        d = router.route(req)
        step = d.inferred_step_type.value if d.inferred_step_type else "-"
        print(f"[{d.tier.value:>6}] [{d.layer.value:>18}] step={step:<12} "
              f"reason={d.reason}")
        print(f"            -> {label}")
        print()

    # 1. Planning
    show(
        "user asks for a refactor (planning)",
        RoutingRequest(
            prompt="Plan how to refactor the auth middleware into smaller modules.",
            session_id=sid,
        ),
    )

    # 2. Safe tool call: read_file
    show(
        "agent decides to read_file",
        RoutingRequest(
            messages=[
                Message(role="system", content="You are a coding agent."),
                Message(role="user", content="Refactor auth middleware."),
                Message(
                    role="assistant",
                    content=None,
                    tool_calls=[ToolCall(name="read_file", arguments={"path": "auth.py"})],
                ),
            ],
            session_id=sid,
            agent_step_type=AgentStepType.TOOL_CALL,
            planned_tool="read_file",
        ),
    )

    # 3. Tool result interpretation
    show(
        "agent interprets a small read_file result",
        RoutingRequest(
            messages=[
                Message(role="user", content="Refactor auth middleware."),
                Message(role="assistant", tool_calls=[ToolCall(name="read_file", arguments={})]),
                Message(role="tool", name="read_file", content="def authenticate(req): ..."),
            ],
            session_id=sid,
            last_tool_called="read_file",
        ),
    )

    # 4. Another safe tool call: grep
    show(
        "agent calls grep",
        RoutingRequest(
            prompt="Find usages of authenticate.",
            session_id=sid,
            agent_step_type=AgentStepType.TOOL_CALL,
            planned_tool="grep",
        ),
    )

    # 5. High-stakes tool call: edit_file
    show(
        "agent calls edit_file (requires strong)",
        RoutingRequest(
            prompt="Now edit auth.py to extract the validation logic.",
            session_id=sid,
            agent_step_type=AgentStepType.TOOL_CALL,
            planned_tool="edit_file",
        ),
    )

    # 6. Failure escalation: previous tool call had a schema error
    show(
        "next turn reports a tool schema error -> escalate to strong",
        RoutingRequest(
            prompt="Try again.",
            session_id=sid,
            agent_step_type=AgentStepType.TOOL_CALL,
            planned_tool="edit_file",
            recent_outcomes=[
                Outcome(kind=OutcomeKind.TOOL_SCHEMA_ERROR, tool_name="edit_file"),
            ],
        ),
    )

    # 7. Follow-up safe tool — still weak because stickiness is per step type
    show(
        "later safe tool call: still weak (per-step-type stickiness)",
        RoutingRequest(
            prompt="Read the new file once more.",
            session_id=sid,
            agent_step_type=AgentStepType.TOOL_CALL,
            planned_tool="read_file",
        ),
    )


if __name__ == "__main__":
    main()
