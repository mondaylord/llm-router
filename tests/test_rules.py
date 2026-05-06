"""Test the rule layer in isolation."""

from __future__ import annotations

from llm_router.core.decision import RoutingRequest, Tier
from llm_router.rules.builtin import (
    CodeBlockRule,
    HighStakesKeywordRule,
    LongPromptRule,
    MultiStepReasoningRule,
    ShortGreetingRule,
    SimpleYesNoRule,
    StructuredOutputRule,
    VeryShortQueryRule,
    default_ruleset,
)


def _r(p: str) -> RoutingRequest:
    return RoutingRequest(prompt=p)


def test_short_greeting_fires_on_common_greetings():
    rule = ShortGreetingRule()
    for g in ["hi", "hello", "Thanks", "你好", "谢谢", "OK"]:
        assert rule.evaluate(_r(g)).tier is Tier.WEAK


def test_short_greeting_does_not_fire_on_question():
    rule = ShortGreetingRule()
    assert rule.evaluate(_r("hello, what is the capital of France?")).tier is None


def test_very_short_query_fires():
    rule = VeryShortQueryRule()
    assert rule.evaluate(_r("what day is it")).tier is Tier.WEAK


def test_very_short_query_skips_code_markers():
    rule = VeryShortQueryRule()
    assert rule.evaluate(_r("`x = 1`")).tier is None


def test_simple_yes_no_fires():
    rule = SimpleYesNoRule()
    assert rule.evaluate(_r("Is the sky blue?")).tier is Tier.WEAK
    assert rule.evaluate(_r("Are dolphins mammals?")).tier is Tier.WEAK


def test_simple_yes_no_skips_long_followups():
    rule = SimpleYesNoRule()
    long_q = "Is this contract clause enforceable in California, and if so, " * 3 + "?"
    assert rule.evaluate(_r(long_q)).tier is None


def test_long_prompt_fires():
    rule = LongPromptRule()
    text = "x" * 5000
    assert rule.evaluate(_r(text)).tier is Tier.STRONG


def test_long_prompt_does_not_fire_on_short():
    rule = LongPromptRule()
    assert rule.evaluate(_r("hi")).tier is None


def test_code_block_fires_on_real_code():
    rule = CodeBlockRule()
    text = "```python\n" + ("def f():\n    return 1\n" * 30) + "```"
    assert rule.evaluate(_r(text)).tier is Tier.STRONG


def test_code_block_does_not_fire_on_inline():
    rule = CodeBlockRule()
    assert rule.evaluate(_r("just `x = 1` inline")).tier is None


def test_high_stakes_fires_on_keywords():
    rule = HighStakesKeywordRule()
    assert rule.evaluate(_r("Can you diagnose this rash?")).tier is Tier.STRONG
    assert rule.evaluate(_r("帮我看下这个诊断")).tier is Tier.STRONG


def test_multi_step_reasoning_fires():
    rule = MultiStepReasoningRule()
    assert rule.evaluate(_r("Prove that x+y = y+x step by step.")).tier is Tier.STRONG
    assert rule.evaluate(_r("逐步推导一下")).tier is Tier.STRONG


def test_structured_output_fires_on_short_json_request():
    rule = StructuredOutputRule()
    p = "Return only JSON: {\"answer\": yes or no}. Are cats mammals?"
    assert rule.evaluate(_r(p)).tier is Tier.WEAK


def test_default_ruleset_returns_strong_rules_first():
    rules = default_ruleset()
    names = [r.name for r in rules]
    assert names.index("high_stakes_keywords") < names.index("very_short_query")
    assert names.index("code_block") < names.index("short_greeting")
