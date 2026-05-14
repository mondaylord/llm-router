"""Core data types flowing through the routing pipeline.

These are *value* types: immutable, serializable, and free of business
logic. Anything that does work belongs in `router.py` or a layer module.

This module is the public schema seen by every caller and every layer,
so it must support BOTH chat-style requests (`prompt: str`) and
agent-style requests (a `messages` array + tool context + outcome
signals from prior turns).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Tier(str, Enum):
    """Logical capability tier. Concrete model resolution is downstream."""

    WEAK = "weak"
    MID = "mid"
    STRONG = "strong"


class DecisionLayer(str, Enum):
    """Which layer in the pipeline produced the final decision."""

    TENANT_OVERRIDE = "tenant_override"
    SESSION_STICKINESS = "session_stickiness"
    RULE = "rule"
    CLASSIFIER = "classifier"
    DEFAULT = "default"


class AgentStepType(str, Enum):
    """Coarse classification of an agent-loop step.

    Set explicitly by the caller when known; otherwise the router will
    try to infer from messages + tools (see `messages.detect_step_type`).
    """

    CHAT = "chat"
    """Normal Q&A or single-turn answer — non-agent."""

    PLANNING = "planning"
    """Decomposing a task, choosing strategy, multi-step reasoning."""

    TOOL_CALL = "tool_call"
    """About to emit a tool call. Routing depends on which tool."""

    TOOL_RESULT = "tool_result"
    """Interpreting / summarizing a tool's output."""

    EDIT = "edit"
    """Producing a code edit / file patch."""

    SUMMARIZE = "summarize"
    """Compressing long context into a summary."""


class OutcomeKind(str, Enum):
    """Failure / quality signals from a prior turn, passed back by the
    caller so the router can escalate. These power agent cascade."""

    TOOL_SCHEMA_ERROR = "tool_schema_error"
    """Generated tool call's arguments did not validate."""

    TOOL_EXECUTION_ERROR = "tool_execution_error"
    """Tool ran but returned an error / non-zero exit."""

    PARSE_ERROR = "parse_error"
    """Output was supposed to be structured (JSON/diff) but failed to parse."""

    VALIDATION_ERROR = "validation_error"
    """Output parsed but failed semantic validation (tests, lint, types)."""

    USER_NEGATIVE_FEEDBACK = "user_negative_feedback"
    """User signaled dissatisfaction (thumbs down, retry, copy-then-edit)."""

    RETRY_ATTEMPT = "retry_attempt"
    """Caller is explicitly retrying the previous step."""

    GENERIC_FAILURE = "generic_failure"
    """Caller knows something went wrong but doesn't have a specific kind."""


class ToolCall(BaseModel):
    """Single tool invocation. `arguments` is left as raw dict/str so we
    don't tie ourselves to a specific provider's schema."""

    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    name: str
    arguments: dict[str, Any] | str = Field(default_factory=dict)


class Message(BaseModel):
    """OpenAI/Anthropic-compatible chat message shape."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class Outcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: OutcomeKind
    detail: str | None = None
    # Helpful for replay debugging; not required.
    tool_name: str | None = None


class RoutingRequest(BaseModel):
    """Input to the router.

    Backward-compatible: `RoutingRequest(prompt="...")` still works.
    For agent flows pass `messages`, `agent_step_type`, `available_tools`,
    `planned_tool`, and (importantly) `recent_outcomes` from the prior
    turn so failure-driven escalation can fire.
    """

    model_config = ConfigDict(extra="forbid")

    # --- text input (one of these is required) ---
    prompt: str | None = None
    messages: list[Message] | None = None

    # --- identity / multi-tenancy ---
    session_id: str | None = None
    tenant_id: str | None = None

    # --- agent context ---
    agent_step_type: AgentStepType | None = None
    """Explicit step-type from the caller. Wins over auto-detection."""

    available_tools: list[str] = Field(default_factory=list)
    """Tool names the agent is allowed to call this turn."""

    planned_tool: str | None = None
    """If the caller already knows which tool will be invoked (e.g. the
    agent framework decided based on policy), pass it here for clean
    routing. Otherwise the router infers from messages and tool list."""

    last_tool_called: str | None = None
    """The tool just executed; populates tool-result interpretation routing."""

    recent_outcomes: list[Outcome] = Field(default_factory=list)
    """Outcomes from previous turns in this session. The most recent few
    are what drive escalation; old outcomes are stale and should be
    pruned by the caller (router does NOT remember outcomes across
    requests)."""

    total_context_tokens: int = 0
    """Approximate token count of the full assembled context, if known.
    Used by the LongContextRule. 0 means 'unknown' (rule falls back to
    len(prompt))."""

    history_turns: int = 0
    """How many turns have happened in this session. 0 = first turn."""

    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _require_text_input(self) -> RoutingRequest:
        if not self.prompt and not self.messages:
            raise ValueError("RoutingRequest needs either `prompt` or `messages`")
        return self

    # ------------------------------------------------------------------
    # Derived helpers used by rules and the classifier.
    # ------------------------------------------------------------------
    @property
    def effective_text(self) -> str:
        """Text used for the classifier and text-based rules.

        Preference order:
          1. `prompt` if set
          2. last user message content
          3. last message with content
          4. empty string
        """
        if self.prompt:
            return self.prompt
        if not self.messages:
            return ""
        for m in reversed(self.messages):
            if m.role == "user" and m.content:
                return m.content
        for m in reversed(self.messages):
            if m.content:
                return m.content
        return ""

    @property
    def last_message(self) -> Message | None:
        if not self.messages:
            return None
        return self.messages[-1]

    @property
    def has_recent_failure(self) -> bool:
        """True if any of the (caller-curated) recent outcomes is a failure."""
        failure_kinds = {
            OutcomeKind.TOOL_SCHEMA_ERROR,
            OutcomeKind.TOOL_EXECUTION_ERROR,
            OutcomeKind.PARSE_ERROR,
            OutcomeKind.VALIDATION_ERROR,
            OutcomeKind.RETRY_ATTEMPT,
            OutcomeKind.GENERIC_FAILURE,
            OutcomeKind.USER_NEGATIVE_FEEDBACK,
        }
        return any(o.kind in failure_kinds for o in self.recent_outcomes)


class RoutingDecision(BaseModel):
    """Output of the router. Always emitted; never None."""

    model_config = ConfigDict(extra="forbid")

    tier: Tier
    layer: DecisionLayer
    reason: str
    """Human-readable, machine-stable. e.g. `rule:short_query`,
    `classifier:p_strong=0.23`, `tenant:forced_strong`."""

    confidence: float = Field(ge=0.0, le=1.0)
    """Routing confidence. 1.0 for deterministic layers (rule, override).
    For the classifier this is `max(p, 1-p)` in binary form."""

    classifier_score: float | None = None
    """Calibrated `p(needs_strong)` when the classifier ran; else None."""

    rules_evaluated: list[str] = Field(default_factory=list)
    """Names of rules that were evaluated in order. Useful for debugging
    why a decision was NOT made by an earlier rule."""

    inferred_step_type: AgentStepType | None = None
    """The step type used by the router (after auto-detect / explicit)."""

    metadata: dict[str, Any] = Field(default_factory=dict)


class RuleResult(BaseModel):
    """What a single rule emits. `tier=None` means 'pass through'."""

    model_config = ConfigDict(extra="forbid")

    tier: Tier | None = None
    reason: str = ""
    confidence: float = 1.0
