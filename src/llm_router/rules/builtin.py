"""Builtin rules.

Each rule is intentionally narrow. The full set should NOT classify
most traffic — by design, the rule layer aims to catch ~30-50% of
obviously-cheap or obviously-expensive prompts and let the rest fall
through to Layer 2.

Add new rules conservatively. Test on real traffic before enabling.
"""

from __future__ import annotations

import re

from llm_router.core.decision import RoutingRequest, RuleResult, Tier
from llm_router.rules.base import Rule

# ----------------------------------------------------------------------
# Cheap-tier rules (route to WEAK)
# ----------------------------------------------------------------------


class ShortGreetingRule(Rule):
    """Single-token greetings, acknowledgements, etc."""

    name = "short_greeting"

    _GREETING_PATTERNS = re.compile(
        r"^\s*(hi|hello|hey|thanks|thank you|ok|okay|yes|no|sure|"
        r"你好|您好|嗨|谢谢|好的|是|不|不是|对|嗯)\s*[!.?。！？]?\s*$",
        re.IGNORECASE,
    )

    def evaluate(self, request: RoutingRequest) -> RuleResult:
        if self._GREETING_PATTERNS.match(request.prompt):
            return RuleResult(tier=Tier.WEAK, reason="rule:short_greeting")
        return RuleResult()


class VeryShortQueryRule(Rule):
    """Very short prompts with no special markers — likely casual."""

    name = "very_short_query"

    MAX_CHARS = 30

    def evaluate(self, request: RoutingRequest) -> RuleResult:
        text = request.prompt.strip()
        if len(text) > self.MAX_CHARS:
            return RuleResult()
        # Don't fire on short code/math snippets.
        if re.search(r"[`$=<>{}]|```", text):
            return RuleResult()
        return RuleResult(tier=Tier.WEAK, reason="rule:very_short_query")


class SimpleYesNoRule(Rule):
    """Yes/no questions ending with '?' under a length cap."""

    name = "simple_yes_no"

    _PREFIXES = (
        "is ", "are ", "do ", "does ", "did ", "can ", "could ",
        "will ", "would ", "should ", "has ", "have ",
    )
    MAX_CHARS = 80

    def evaluate(self, request: RoutingRequest) -> RuleResult:
        text = request.prompt.strip().lower()
        if not text.endswith("?"):
            return RuleResult()
        if len(text) > self.MAX_CHARS:
            return RuleResult()
        if not text.startswith(self._PREFIXES):
            return RuleResult()
        return RuleResult(tier=Tier.WEAK, reason="rule:simple_yes_no")


# ----------------------------------------------------------------------
# Strong-tier rules (route to STRONG)
# ----------------------------------------------------------------------


class LongPromptRule(Rule):
    """Very long prompts — probably document analysis or rich context."""

    name = "long_prompt"

    MIN_CHARS = 4000

    def evaluate(self, request: RoutingRequest) -> RuleResult:
        if len(request.prompt) >= self.MIN_CHARS:
            return RuleResult(tier=Tier.STRONG, reason="rule:long_prompt")
        return RuleResult()


class CodeBlockRule(Rule):
    """Prompts with sizable code blocks. Cheap models are unreliable on
    real-world coding tasks past trivial scope."""

    name = "code_block"

    MIN_CODE_CHARS = 200
    _FENCE = re.compile(r"```([\s\S]*?)```")

    def evaluate(self, request: RoutingRequest) -> RuleResult:
        matches = self._FENCE.findall(request.prompt)
        if not matches:
            return RuleResult()
        total = sum(len(m) for m in matches)
        if total >= self.MIN_CODE_CHARS:
            return RuleResult(tier=Tier.STRONG, reason="rule:code_block")
        return RuleResult()


class HighStakesKeywordRule(Rule):
    """Domain keywords where wrong answers carry real cost (legal /
    medical / financial advice). Default to strong; tenants can disable."""

    name = "high_stakes_keywords"

    _KEYWORDS = re.compile(
        r"\b(diagnose|prognosis|prescription|lawsuit|litigation|"
        r"contract\s+terms|securities|tax\s+(advice|implications)|"
        r"clinical|medical\s+history)\b|"
        r"(诊断|处方|起诉|合同条款|证券|税务建议|临床)",
        re.IGNORECASE,
    )

    def evaluate(self, request: RoutingRequest) -> RuleResult:
        if self._KEYWORDS.search(request.prompt):
            return RuleResult(tier=Tier.STRONG, reason="rule:high_stakes_keywords")
        return RuleResult()


class MultiStepReasoningRule(Rule):
    """Prompts that ask for explicit step-by-step reasoning, proofs, or
    multi-part analysis. Cheap models tend to skip steps."""

    name = "multi_step_reasoning"

    _CUES = re.compile(
        r"\b(step[- ]by[- ]step|prove that|derive|chain of thought|"
        r"reason carefully|explain why|justify|trade[- ]off)\b|"
        r"(逐步|证明|推导|权衡|为什么)",
        re.IGNORECASE,
    )

    def evaluate(self, request: RoutingRequest) -> RuleResult:
        if self._CUES.search(request.prompt):
            return RuleResult(tier=Tier.STRONG, reason="rule:multi_step_reasoning")
        return RuleResult()


class StructuredOutputRule(Rule):
    """Tight JSON-schema or function-call-style requests are usually
    safe on cheap models if they're short. Acts as a `WEAK` shortcut
    for short structured-output prompts."""

    name = "structured_output_short"

    MAX_CHARS = 400
    _CUES = re.compile(
        r"(return\s+(only\s+)?(valid\s+)?json|"
        r"output\s+as\s+json|"
        r"respond\s+with\s+a\s+single\s+(word|number)|"
        r"true\s+or\s+false)",
        re.IGNORECASE,
    )

    def evaluate(self, request: RoutingRequest) -> RuleResult:
        if len(request.prompt) > self.MAX_CHARS:
            return RuleResult()
        if self._CUES.search(request.prompt):
            return RuleResult(tier=Tier.WEAK, reason="rule:structured_output_short")
        return RuleResult()


def default_ruleset() -> list[Rule]:
    """Order matters: cheaper-to-evaluate and higher-confidence rules first.
    Strong-tier rules run before weak-tier rules so we don't accidentally
    downgrade a high-stakes short prompt."""
    return [
        # Strong-tier first (safety bias)
        HighStakesKeywordRule(),
        LongPromptRule(),
        CodeBlockRule(),
        MultiStepReasoningRule(),
        # Weak-tier
        ShortGreetingRule(),
        SimpleYesNoRule(),
        StructuredOutputRule(),
        VeryShortQueryRule(),
    ]
