"""Breakpoints and their conditions.

Conditions are a closed, declarative union rather than an expression string, so
this is dict dispatch with no parser and no `eval` -- and no sandbox to escape.
The client builds a real form from the same union, and
`client/src/app/core/api/mock/mock-engine.ts` implements identical semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from .pipeline import StageKind, parse_stage_id
from .runner import StageOutcome

#: Conditions every stage can evaluate.
ALWAYS_VALID = ("token_index", "sequence_length", "hit_count")

#: Extra conditions by stage kind. See docs/api-contract.md section 5.
CONDITIONS_BY_STAGE: dict[StageKind, tuple[str, ...]] = {
    "tokenize": (),
    "embed": ("residual_norm",),
    "attention": ("residual_norm", "attention_max"),
    "ffn": ("residual_norm",),
    "final_norm": ("residual_norm",),
    "lm_head": ("top1_prob", "logit_entropy"),
    "sample": ("top1_prob", "logit_entropy"),
    "emit": ("top1_prob", "logit_entropy", "emitted_token_text", "emitted_token_id"),
}


def conditions_for_stage(kind: StageKind) -> tuple[str, ...]:
    return ALWAYS_VALID + CONDITIONS_BY_STAGE.get(kind, ())


def is_condition_valid_at(kind: StageKind, condition_kind: str) -> bool:
    return condition_kind in conditions_for_stage(kind)


_COMPARATORS: dict[str, Callable[[Any, Any], bool]] = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
}


@dataclass
class Breakpoint:
    id: str
    stage_id: str
    enabled: bool = True
    condition: dict[str, Any] | None = None
    one_shot: bool = False
    hit_count: int = 0

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "Breakpoint":
        return cls(
            id=str(raw.get("id") or ""),
            stage_id=str(raw["stageId"]),
            enabled=bool(raw.get("enabled", True)),
            condition=raw.get("condition"),
            one_shot=bool(raw.get("oneShot", False)),
            hit_count=int(raw.get("hitCount", 0)),
        )

    def as_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "stageId": self.stage_id,
            "enabled": self.enabled,
            "oneShot": self.one_shot,
            "hitCount": self.hit_count,
        }
        if self.condition is not None:
            out["condition"] = self.condition
        return out


class ConditionError(ValueError):
    """A condition that references a value its stage does not have.

    Rejected when the breakpoint is set rather than silently never firing,
    which would look like a broken debugger.
    """


def validate(breakpoint: Breakpoint) -> None:
    condition = breakpoint.condition
    if condition is None:
        return
    kind = parse_stage_id(breakpoint.stage_id).kind
    condition_kind = condition.get("kind")
    if not is_condition_valid_at(kind, str(condition_kind)):
        allowed = ", ".join(conditions_for_stage(kind)) or "none"
        raise ConditionError(
            f"condition '{condition_kind}' is not available at stage "
            f"'{breakpoint.stage_id}'; valid here: {allowed}"
        )
    if condition_kind in ("emitted_token_text",):
        if condition.get("op") not in ("equals", "contains"):
            raise ConditionError("emitted_token_text supports only 'equals' and 'contains'")
    elif condition.get("op") not in _COMPARATORS:
        raise ConditionError(f"unknown comparison operator {condition.get('op')!r}")


@dataclass
class StageValues:
    """What a condition can be evaluated against at one halt point.

    Values are pulled from the outcome lazily-ish: everything here is already
    computed by the stage, so this is just naming rather than work.
    """

    outcome: StageOutcome
    display_token: Callable[[int], str]
    hit_count: int = 0
    _cache: dict[str, Any] = field(default_factory=dict)

    @property
    def token_index(self) -> int:
        return self.outcome.step

    @property
    def sequence_length(self) -> int:
        return self.outcome.sequence_length

    @property
    def top1_prob(self) -> float:
        sample = self.outcome.sample
        return float(sample.top_probs[0]) if sample and sample.top_probs else 0.0

    @property
    def logit_entropy(self) -> float:
        sample = self.outcome.sample
        return float(sample.entropy) if sample else 0.0

    @property
    def residual_norm(self) -> float:
        residual = self.outcome.residual
        if residual is None:
            return 0.0
        return float(torch.linalg.vector_norm(residual.to(torch.float32)))

    def attention_max(self, head: int | None) -> float:
        values = self.outcome.head_max_attention
        if values is None or values.numel() == 0:
            return 0.0
        if head is None:
            return float(values.max())
        if head < 0 or head >= values.numel():
            return 0.0
        return float(values[head])

    @property
    def emitted_token_id(self) -> int | None:
        return self.outcome.emitted_token_id

    @property
    def emitted_token_text(self) -> str:
        token_id = self.outcome.emitted_token_id
        return "" if token_id is None else self.display_token(token_id)


def evaluate(condition: dict[str, Any], values: StageValues) -> bool:
    """Dict dispatch over the condition union. Unknown kinds never fire."""
    kind = condition.get("kind")
    op = condition.get("op")
    expected = condition.get("value")

    if kind == "emitted_token_text":
        actual = values.emitted_token_text
        text = str(expected)
        return actual == text if op == "equals" else text in actual

    if kind == "emitted_token_id":
        return values.emitted_token_id == expected

    compare = _COMPARATORS.get(str(op))
    if compare is None:
        return False

    if kind == "token_index":
        return compare(values.token_index, expected)
    if kind == "sequence_length":
        return compare(values.sequence_length, expected)
    if kind == "hit_count":
        return compare(values.hit_count, expected)
    if kind == "top1_prob":
        return compare(values.top1_prob, expected)
    if kind == "logit_entropy":
        return compare(values.logit_entropy, expected)
    if kind == "residual_norm":
        return compare(values.residual_norm, expected)
    if kind == "attention_max":
        head = condition.get("head")
        return compare(values.attention_max(None if head is None else int(head)), expected)

    return False
