"""Agent-mode rules.

These run *before* the chat-mode rules in `builtin.py` (set up by the
router). They consult the structured agent context — `agent_step_type`,
`available_tools`, `planned_tool`, `recent_outcomes`,
`total_context_tokens` — which the text-based builtin rules don't see.

Design rules:

- Failure escalation runs FIRST. Once we have evidence the last step
  failed, no other heuristic should be allowed to downgrade.
- Tool whitelists/blacklists are configurable (see `AgentConfig`).
  We do not hardcode which tools are "safe for weak" because that's
  product-specific.
- Auto-detected step types are inputs to these rules; the caller's
  explicit `agent_step_type` always wins (handled in router.py).
"""

from __future__ import annotations

from llm_router.core.decision import AgentStepType, RoutingRequest, RuleResult, Tier
from llm_router.rules.base import Rule


# ----------------------------------------------------------------------
# Failure-driven escalation (the heart of agent cascade)
# ----------------------------------------------------------------------


class RecentFailureRule(Rule):
    """If a prior turn flagged a failure, route this turn to STRONG.

    This is how cascade is realized: the caller observes a failure
    (tool schema error, parse error, validation error, etc.) and passes
    it back in `recent_outcomes`. The router then escalates without
    needing any inline scoring of model output."""

    name = "agent_recent_failure"

    def evaluate(self, request: RoutingRequest) -> RuleResult:
        if request.has_recent_failure:
            kinds = ",".join({o.kind.value for o in request.recent_outcomes})
            return RuleResult(
                tier=Tier.STRONG,
                reason=f"agent:recent_failure[{kinds}]",
                confidence=1.0,
            )
        return RuleResult()


# ----------------------------------------------------------------------
# Step-type-aware rules
# ----------------------------------------------------------------------


class PlanningStepRule(Rule):
    """Planning steps need the strong model. Cheap models skip planning
    steps or produce shallow ones, which then cascades into bad tool
    calls and bad edits."""

    name = "agent_planning_step"

    def evaluate(self, request: RoutingRequest) -> RuleResult:
        # Read the step type from metadata stamped by the router
        # (router sets `request.metadata['detected_step']` before
        # evaluating rules so we don't recompute it per rule).
        step = request.metadata.get("detected_step")
        if step == AgentStepType.PLANNING.value:
            return RuleResult(
                tier=Tier.STRONG, reason="agent:planning_step", confidence=1.0
            )
        return RuleResult()


class SafeToolCallRule(Rule):
    """For TOOL_CALL steps invoking a known-safe tool, route to WEAK.

    The list of safe tools comes from `AgentConfig.weak_safe_tools`.
    Common entries: read_file, list_dir, grep, ls, get_url, fetch_metadata.
    """

    name = "agent_safe_tool_call"

    def __init__(self, weak_safe_tools: list[str]) -> None:
        self._safe = set(weak_safe_tools)

    def evaluate(self, request: RoutingRequest) -> RuleResult:
        step = request.metadata.get("detected_step")
        if step != AgentStepType.TOOL_CALL.value:
            return RuleResult()
        if not self._safe:
            return RuleResult()
        tool = request.planned_tool
        if tool is None:
            # If every available tool is in the safe set, we can still
            # treat the upcoming call as safe.
            if request.available_tools and all(t in self._safe for t in request.available_tools):
                return RuleResult(
                    tier=Tier.WEAK,
                    reason="agent:safe_tools_only_in_scope",
                    confidence=1.0,
                )
            return RuleResult()
        if tool in self._safe:
            return RuleResult(
                tier=Tier.WEAK,
                reason=f"agent:safe_tool[{tool}]",
                confidence=1.0,
            )
        return RuleResult()


class RequiresStrongToolRule(Rule):
    """For TOOL_CALL steps invoking a high-stakes tool, force STRONG.
    Tool list comes from `AgentConfig.requires_strong_tools`."""

    name = "agent_requires_strong_tool"

    def __init__(self, requires_strong_tools: list[str]) -> None:
        self._strong = set(requires_strong_tools)

    def evaluate(self, request: RoutingRequest) -> RuleResult:
        step = request.metadata.get("detected_step")
        if step != AgentStepType.TOOL_CALL.value:
            return RuleResult()
        if not self._strong:
            return RuleResult()
        tool = request.planned_tool
        if tool and tool in self._strong:
            return RuleResult(
                tier=Tier.STRONG,
                reason=f"agent:requires_strong_tool[{tool}]",
                confidence=1.0,
            )
        return RuleResult()


class ToolResultInterpretationRule(Rule):
    """Interpreting a small tool result is a cheap-tier task. If the
    result is huge (token-heavy file dump, big JSON), defer to other
    rules / classifier — interpretation of a 100k-char dump is not
    a weak-tier job."""

    name = "agent_tool_result_interpretation"

    SMALL_RESULT_CHARS = 4000

    def evaluate(self, request: RoutingRequest) -> RuleResult:
        step = request.metadata.get("detected_step")
        if step != AgentStepType.TOOL_RESULT.value:
            return RuleResult()
        last = request.last_message
        if last is None or last.role != "tool":
            return RuleResult()
        size = len(last.content or "")
        if size <= self.SMALL_RESULT_CHARS:
            return RuleResult(
                tier=Tier.WEAK,
                reason="agent:tool_result_small",
                confidence=1.0,
            )
        return RuleResult()


class LongContextRule(Rule):
    """Long assembled contexts route to STRONG. Catches agent runs that
    have accumulated lots of history regardless of the current message."""

    name = "agent_long_context"

    def __init__(self, threshold_tokens: int) -> None:
        self.threshold_tokens = threshold_tokens

    def evaluate(self, request: RoutingRequest) -> RuleResult:
        if request.total_context_tokens >= self.threshold_tokens:
            return RuleResult(
                tier=Tier.STRONG,
                reason=f"agent:long_context_{request.total_context_tokens}t",
                confidence=1.0,
            )
        return RuleResult()


class EditStepRule(Rule):
    """Code-edit steps default to STRONG unless they are very small.
    A small edit (< 500 chars of context, no multi-file directive) can
    go to weak; everything else should not risk a botched patch."""

    name = "agent_edit_step"

    SMALL_EDIT_CHARS = 500

    def evaluate(self, request: RoutingRequest) -> RuleResult:
        step = request.metadata.get("detected_step")
        if step != AgentStepType.EDIT.value:
            return RuleResult()
        if len(request.effective_text) <= self.SMALL_EDIT_CHARS:
            return RuleResult(
                tier=Tier.WEAK,
                reason="agent:edit_small",
                confidence=1.0,
            )
        return RuleResult(
            tier=Tier.STRONG, reason="agent:edit_default_strong", confidence=1.0
        )


class SummarizeStepRule(Rule):
    """Summarization is a weak-tier task by default. Long summarizations
    of very long context are still handled by LongContextRule above."""

    name = "agent_summarize_step"

    def evaluate(self, request: RoutingRequest) -> RuleResult:
        step = request.metadata.get("detected_step")
        if step == AgentStepType.SUMMARIZE.value:
            return RuleResult(
                tier=Tier.WEAK, reason="agent:summarize_step", confidence=1.0
            )
        return RuleResult()


def default_agent_ruleset(
    weak_safe_tools: list[str],
    requires_strong_tools: list[str],
    long_context_threshold: int,
    failure_escalation_enabled: bool,
) -> list[Rule]:
    """Default agent ruleset in priority order.

    Order rationale:
      1. recent-failure escalation — strongest signal, must win
      2. high-stakes tool requirement — cannot be downgraded
      3. long context — preempts step-type heuristics
      4. planning — early-binds plans to strong
      5. safe-tool call — downgrades obvious cheap tool calls
      6. tool-result interpretation — downgrades simple result reads
      7. edit step — defaults edits, sized
      8. summarize step — downgrades summaries
    """
    rules: list[Rule] = []
    if failure_escalation_enabled:
        rules.append(RecentFailureRule())
    rules.extend(
        [
            RequiresStrongToolRule(requires_strong_tools),
            LongContextRule(long_context_threshold),
            PlanningStepRule(),
            SafeToolCallRule(weak_safe_tools),
            ToolResultInterpretationRule(),
            EditStepRule(),
            SummarizeStepRule(),
        ]
    )
    return rules
