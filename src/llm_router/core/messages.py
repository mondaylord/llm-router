"""Auto-detect `AgentStepType` from a request.

This is a best-effort heuristic. Callers should pass `agent_step_type`
explicitly when they know — the agent framework usually does. Auto-
detection exists for the case where the router is dropped in front of
an existing endpoint that emits OpenAI-style messages and we don't have
clean step labels yet.

Detection priority (first match wins):

  1. last message role is `tool`               -> TOOL_RESULT
  2. recent_outcomes carry RETRY_ATTEMPT       -> (caller's prior step type
                                                  preferred; fallback to
                                                  TOOL_CALL)
  3. last assistant message has tool_calls     -> TOOL_CALL
  4. system prompt contains strong planning
     cues                                      -> PLANNING
  5. user text contains explicit edit cues
     ("apply this diff", "edit the file", ...) -> EDIT
  6. user text contains summarize cues          -> SUMMARIZE
  7. available_tools is non-empty               -> TOOL_CALL  (default
                                                  for agent contexts
                                                  with tools)
  8. otherwise                                  -> CHAT
"""

from __future__ import annotations

import re

from llm_router.core.decision import AgentStepType, RoutingRequest

_PLAN_RE = re.compile(
    r"\b(plan|break\s+down|strategy|decompose|outline|"
    r"think\s+step\s+by\s+step|first\s+make\s+a\s+plan)\b|"
    r"(规划|拆解|分解|策略|先制定计划)",
    re.IGNORECASE,
)

_EDIT_RE = re.compile(
    r"\b(apply\s+this\s+diff|edit\s+the\s+file|patch\s+this|"
    r"modify\s+the\s+file|refactor\s+this|rewrite\s+the\s+function)\b|"
    r"(修改文件|改写函数|重构这段|应用补丁)",
    re.IGNORECASE,
)

_SUMMARIZE_RE = re.compile(
    r"\b(summarize|summarise|tldr|tl;dr|condense|compress\s+the)\b|"
    r"(总结|摘要|概括)",
    re.IGNORECASE,
)


def detect_step_type(request: RoutingRequest) -> AgentStepType:
    """Best-effort inference. Explicit `request.agent_step_type` should
    be preferred by the caller; this function does NOT consult it."""
    msgs = request.messages or []

    # 1. Tool result interpretation
    if msgs and msgs[-1].role == "tool":
        return AgentStepType.TOOL_RESULT

    # 3. Tool call about to be issued (or just issued)
    for m in reversed(msgs):
        if m.role == "assistant" and m.tool_calls:
            return AgentStepType.TOOL_CALL
        if m.role == "user":
            break  # don't look past the most recent user turn

    # System prompt scan for planning cue
    sys_text = " ".join(m.content or "" for m in msgs if m.role == "system")
    text = request.effective_text or ""
    combined = f"{sys_text}\n{text}"

    # 4. Planning
    if _PLAN_RE.search(combined):
        return AgentStepType.PLANNING

    # 5. Edit
    if _EDIT_RE.search(text):
        return AgentStepType.EDIT

    # 6. Summarize
    if _SUMMARIZE_RE.search(text):
        return AgentStepType.SUMMARIZE

    # 7. Tool-calling context but no specific cue
    if request.available_tools:
        return AgentStepType.TOOL_CALL

    # 8. Default
    return AgentStepType.CHAT
