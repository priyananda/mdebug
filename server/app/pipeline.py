"""The pipeline: the stages a decode step walks through, and how they are named.

This mirrors `client/src/app/core/models/pipeline.model.ts` exactly. The two
files are the same contract expressed twice; changing one means changing the
other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

StageKind = Literal[
    "tokenize",
    "embed",
    "attention",
    "ffn",
    "final_norm",
    "lm_head",
    "sample",
    "emit",
]

PER_LAYER_STAGES: tuple[StageKind, ...] = ("attention", "ffn")
PRE_LAYER_STAGES: tuple[StageKind, ...] = ("embed",)
POST_LAYER_STAGES: tuple[StageKind, ...] = ("final_norm", "lm_head", "sample", "emit")


@dataclass(frozen=True)
class StageRef:
    kind: StageKind
    #: Present iff the stage is per-layer (attention, ffn).
    layer: int | None = None

    @property
    def id(self) -> str:
        return stage_id(self.kind, self.layer)

    def as_json(self) -> dict[str, object]:
        out: dict[str, object] = {"kind": self.kind}
        if self.layer is not None:
            out["layer"] = self.layer
        return out


def stage_id(kind: StageKind, layer: int | None = None) -> str:
    """Canonical string form: 'embed', 'L7.attention', 'sample'."""
    return kind if layer is None else f"L{layer}.{kind}"


def parse_stage_id(value: str) -> StageRef:
    dot = value.find(".")
    if dot < 0:
        return StageRef(value)  # type: ignore[arg-type]
    return StageRef(value[dot + 1 :], int(value[1:dot]))  # type: ignore[arg-type]


def is_per_layer(kind: StageKind) -> bool:
    return kind in PER_LAYER_STAGES


def stage_sequence(num_layers: int, step: int) -> list[StageRef]:
    """Halt points for one decode step, in execution order.

    `tokenize` exists only on step 0 -- the prompt is tokenized once -- so step 0
    has 2*num_layers + 6 stages and every later step has 2*num_layers + 5. At
    num_layers=20 that is 46 and 45.
    """
    stages: list[StageRef] = []
    if step == 0:
        stages.append(StageRef("tokenize"))
    stages.extend(StageRef(kind) for kind in PRE_LAYER_STAGES)
    for layer in range(num_layers):
        stages.extend(StageRef(kind, layer) for kind in PER_LAYER_STAGES)
    stages.extend(StageRef(kind) for kind in POST_LAYER_STAGES)
    return stages


def all_stage_ids(num_layers: int) -> list[str]:
    return [s.id for s in stage_sequence(num_layers, 0)]


#: The stage catalog reported in ModelInfo. The client renders the graph from
#: this rather than from constants of its own.
STAGE_DESCRIPTORS: list[dict[str, object]] = [
    {
        "kind": "tokenize",
        "label": "Tokenize",
        "perLayer": False,
        "breakpointable": True,
        "description": "Byte-level BPE splits the prompt into token ids. Runs once.",
    },
    {
        "kind": "embed",
        "label": "Embed",
        "perLayer": False,
        "breakpointable": True,
        "description": (
            "Token embedding plus learned absolute position embedding. This model adds "
            "both, then applies RoPE inside attention as well."
        ),
    },
    {
        "kind": "attention",
        "label": "Attention",
        "perLayer": True,
        "breakpointable": True,
        "description": (
            "RMSNorm, then multi-head self-attention with RoPE on the first 16 dims of "
            "each head, then a residual add."
        ),
    },
    {
        "kind": "ffn",
        "label": "FFN",
        "perLayer": True,
        "breakpointable": True,
        "description": "RMSNorm, then a SiLU feed-forward (512 -> 1536 -> 512), then a residual add.",
    },
    {
        "kind": "final_norm",
        "label": "Final norm",
        "perLayer": False,
        "breakpointable": True,
        "description": "The last RMSNorm before the output projection.",
    },
    {
        "kind": "lm_head",
        "label": "LM head",
        "perLayer": False,
        "breakpointable": True,
        "description": "Linear projection from the hidden size to one logit per vocabulary entry.",
    },
    {
        "kind": "sample",
        "label": "Sample",
        "perLayer": False,
        "breakpointable": True,
        "description": "Apply temperature and top-k, then draw the next token from the distribution.",
    },
    {
        "kind": "emit",
        "label": "Emit",
        "perLayer": False,
        "breakpointable": True,
        "description": "Append the chosen token to the sequence and begin the next decode step.",
    },
]
