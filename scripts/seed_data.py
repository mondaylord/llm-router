"""Generate a small synthetic dataset so the rest of the pipeline can be
exercised end-to-end without real labeled traffic.

Two kinds of prompts are produced:

- "easy" prompts (label=0, should route to weak): short greetings,
  yes/no questions, simple lookup, basic translation.
- "hard" prompts (label=1, should route to strong): code reviews,
  multi-step reasoning, long context, high-stakes domains.

Both training labels (easy=0/hard=1) and eval triples
(prompt × {weak, strong} × quality, cost) are written.

This is a TOY dataset. Real production should replace this with the
Phase-0 labeled set from EVAL.md. Do NOT tune thresholds against this
dataset and ship.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from llm_router.eval.dataset import EvalRecord, write_jsonl

EASY_TEMPLATES = [
    "hello",
    "thanks",
    "ok",
    "你好",
    "谢谢",
    "Is the sky blue?",
    "Are dolphins mammals?",
    "What is 2 + 2?",
    "Translate 'good morning' to Spanish.",
    "Translate 'thank you' to Japanese.",
    "Return only JSON: {{\"answer\": yes or no}}. Are cats mammals?",
    "Respond with a single word: capital of France?",
    "True or false: water boils at 100C at sea level?",
    "What day comes after Monday?",
    "Convert 5 km to miles.",
    "Spell 'accommodate'.",
    "How many days in February in a leap year?",
    "What is the past tense of 'go'?",
]

HARD_TEMPLATES = [
    "Walk through, step by step, why this code has a race condition:\n```python\n"
    + "import threading\nx = 0\ndef inc():\n    global x\n    x = x + 1\nts = [threading.Thread(target=inc) for _ in range(1000)]\nfor t in ts: t.start()\nfor t in ts: t.join()\nprint(x)\n```\n"
    + "Then propose two fixes and analyze their performance trade-offs.",
    "Prove that the sum of the first n odd numbers equals n^2 by induction. Show every step.",
    "Derive the closed-form solution for ridge regression starting from the loss function.",
    "Review this contract clause for ambiguity and potential dispute risk in the context of "
    "an enterprise SaaS agreement: 'Either party may terminate for convenience with thirty "
    "days notice provided no material breach has occurred within the prior period.'",
    "Given the following clinical history, list possible differential diagnoses and explain "
    "your reasoning for each: 45yo male, 3 weeks of intermittent epigastric pain, weight loss "
    "8 lbs, family history of pancreatic cancer, mildly elevated lipase.",
    "Refactor the following Python class to follow SOLID principles. Walk me through each "
    "principle and the trade-offs of your refactor.\n```python\n"
    + ("class UserManager:\n    def create(self, data):\n        if not data.get('email'): raise ValueError\n"
       "        # send email, write db, log audit, generate report\n        pass\n" * 5)
    + "```",
    "Explain the trade-off between Lagrangian and Eulerian formulations for fluid simulation, "
    "and recommend which is better for an interactive game engine. Justify with at least "
    "three considerations: memory locality, numerical stability, and parallelizability.",
    "Given the following 2000-token product spec... " + ("lorem ipsum dolor sit amet " * 250)
    + " ... summarize the top 5 risks and assign owners.",
    "请证明对于任意正整数 n，1+2+...+n = n(n+1)/2，并说明数学归纳法每一步的逻辑。",
    "请逐步分析这段交易策略代码的潜在问题，并给出至少两种改进方案，权衡它们各自的延迟和滑点风险。",
    "这位 56 岁女性病人有以下症状：胸痛三周，活动后加重，伴有冷汗和左臂麻木。请列出鉴别诊断并说明推理。",
]


def make_dataset(n_easy: int, n_hard: int, seed: int):
    rng = random.Random(seed)
    prompts: list[tuple[str, int]] = []
    for _ in range(n_easy):
        prompts.append((rng.choice(EASY_TEMPLATES), 0))
    for _ in range(n_hard):
        prompts.append((rng.choice(HARD_TEMPLATES), 1))
    rng.shuffle(prompts)
    return prompts


def make_eval_records(prompts: list[tuple[str, int]]):
    """Synthesize plausible quality/cost triples per (prompt, model)."""
    records: list[EvalRecord] = []
    rng = random.Random(0)
    # Cost-per-token-ish constants (relative).
    cost_weak = 0.0001
    cost_strong = 0.003
    for i, (text, label) in enumerate(prompts):
        prompt_id = f"p_{i:05d}"
        token_count = max(1, len(text) // 4)
        if label == 0:
            q_weak = max(0.0, min(1.0, rng.gauss(0.85, 0.05)))
            q_strong = max(0.0, min(1.0, rng.gauss(0.93, 0.03)))
        else:
            q_weak = max(0.0, min(1.0, rng.gauss(0.55, 0.10)))
            q_strong = max(0.0, min(1.0, rng.gauss(0.91, 0.05)))
        records.append(
            EvalRecord(
                prompt_id=prompt_id,
                prompt=text,
                model="weak",
                quality=q_weak,
                cost=cost_weak * token_count,
            )
        )
        records.append(
            EvalRecord(
                prompt_id=prompt_id,
                prompt=text,
                model="strong",
                quality=q_strong,
                cost=cost_strong * token_count,
            )
        )
    return records


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="data")
    p.add_argument("--n-easy", type=int, default=400)
    p.add_argument("--n-hard", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    prompts = make_dataset(args.n_easy, args.n_hard, args.seed)

    # Training labels file (jsonl)
    train_path = out / "train.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for text, label in prompts:
            import json

            f.write(json.dumps({"prompt": text, "label": label}, ensure_ascii=False) + "\n")

    # Eval records (jsonl)
    records = make_eval_records(prompts)
    write_jsonl(records, out / "eval.jsonl")

    print(f"wrote {len(prompts)} training prompts -> {train_path}")
    print(f"wrote {len(records)} eval records  -> {out / 'eval.jsonl'}")


if __name__ == "__main__":
    main()
