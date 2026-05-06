"""Eval dataset I/O.

JSONL schema (one record per line):

    {"prompt_id": "p_001", "prompt": "...", "model": "weak",
     "quality": 0.71, "cost": 0.0003}

A complete dataset has multiple records per `prompt_id`, one per
candidate model, so the harness can compute the optimal-router
Pareto curve.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvalRecord:
    prompt_id: str
    prompt: str
    model: str
    quality: float
    cost: float

    def to_dict(self) -> dict:
        return {
            "prompt_id": self.prompt_id,
            "prompt": self.prompt,
            "model": self.model,
            "quality": self.quality,
            "cost": self.cost,
        }

    @classmethod
    def from_dict(cls, d: dict) -> EvalRecord:
        return cls(
            prompt_id=d["prompt_id"],
            prompt=d["prompt"],
            model=d["model"],
            quality=float(d["quality"]),
            cost=float(d["cost"]),
        )


def load_jsonl(path: str | Path) -> list[EvalRecord]:
    out: list[EvalRecord] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(EvalRecord.from_dict(json.loads(line)))
    return out


def write_jsonl(records: Iterable[EvalRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")


def group_by_prompt(
    records: list[EvalRecord],
) -> dict[str, dict[str, EvalRecord]]:
    """Returns prompt_id -> { model -> record }."""
    out: dict[str, dict[str, EvalRecord]] = defaultdict(dict)
    for r in records:
        out[r.prompt_id][r.model] = r
    return dict(out)
