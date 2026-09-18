"""A steppable, instrumented forward pass with a real KV cache.

`QwenModel.forward` runs start to finish and returns only logits: it cannot be
halted between layers, and it throws away the attention weights. This module
re-expresses the same computation as a generator that yields once per pipeline
stage, using the model's own modules and weights.

`src/model.py` is left untouched on purpose -- it is vendored code with its own
attribution, and the debugger should observe the model rather than rewrite it.
The cost of that choice is that the arithmetic here has to match `model.py`
exactly; `tests/test_runner_parity.py` asserts that it does, against the real
checkpoint.

Two things this adds that the original does not have:

* **A KV cache.** `server/infer.py` re-runs the whole prefix for every token,
  which is O(T) slower and leaves the KV panel with nothing to show. Keys and
  values are cached here after rotation, so each step computes one new position.
* **Position-aware RoPE.** `RotaryEmbedding.forward` derives its positions from
  the input's own length, so a cached query of length 1 would be rotated as
  though it sat at position 0. Rotation here takes explicit absolute positions,
  which is what makes the cache correct rather than merely fast.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterator

import torch
import torch.nn.functional as F

from .pipeline import StageRef, stage_sequence


@dataclass
class SamplingState:
    """Mutable so that `patch_config` can change it between stages."""

    max_new_tokens: int = 32
    mode: str = "temperature"  # greedy | temperature | top_k
    temperature: float = 0.8
    top_k: int = 40
    seed: int | None = None


@dataclass
class LayerCache:
    """Rotated keys and values for one layer, shaped [B, H, P, Hd]."""

    keys: torch.Tensor | None = None
    values: torch.Tensor | None = None

    def append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        self.keys = k if self.keys is None else torch.cat([self.keys, k], dim=2)
        self.values = v if self.values is None else torch.cat([self.values, v], dim=2)

    @property
    def length(self) -> int:
        return 0 if self.keys is None else int(self.keys.shape[2])

    @property
    def nbytes(self) -> int:
        total = 0
        for t in (self.keys, self.values):
            if t is not None:
                total += t.numel() * t.element_size()
        return total


#: How many candidates the run *keeps* per step. Server-side memory only -- a
#: few hundred bytes a step -- and it is the ceiling for `GET .../logits?k=`,
#: which can only narrow this list. Distinct from `session.TOP_K`, which is how
#: many are pushed to the client unasked.
TOP_K_STORED = 50


@dataclass
class SampleResult:
    token_id: int
    chosen_rank: int
    entropy: float
    temperature: float
    top_ids: list[int]
    top_logits: list[float]
    top_probs: list[float]


@dataclass
class StageOutcome:
    """Everything the session needs to decide whether to halt, and what to send."""

    step: int
    stage: StageRef
    sequence_length: int
    #: Hidden state at the position just computed, after this stage. [hidden]
    residual: torch.Tensor | None = None
    #: Set at `sample` and `emit`.
    sample: SampleResult | None = None
    #: Set at `emit`: the token id appended to the sequence.
    emitted_token_id: int | None = None
    #: Max attention weight on the current query row, per head. [num_heads]
    head_max_attention: torch.Tensor | None = None
    #: Wall-clock cost of this stage.
    elapsed_ms: float = 0.0


class AttentionStore:
    """Accumulated attention rows, already in the wire format.

    A causal matrix packed lower-triangular *is* the concatenation of its rows,
    so appending each query row's quantized bytes builds exactly the payload the
    client decodes -- no duplication between steps, and a tile for sequence
    length T is simply the first T*(T+1)/2 bytes.

    At the default cap of 192 positions that is 18.5 kB per (layer, head), or
    about 3 MB for a 20x8 model across an entire run.
    """

    def __init__(self, num_layers: int, num_heads: int) -> None:
        self._rows: list[list[bytearray]] = [
            [bytearray() for _ in range(num_heads)] for _ in range(num_layers)
        ]

    def append_rows(self, layer: int, rows: torch.Tensor) -> None:
        """Append query rows for one layer. `rows` is [heads, queries, keys].

        Rows arrive in absolute position order, so the query that is `n` rows
        into the buffer attends to keys 0..n and contributes n+1 bytes. That
        invariant is what keeps the buffer a valid `causal_lower` payload.
        """
        heads, queries, _ = rows.shape
        for head in range(heads):
            buffer = self._rows[layer][head]
            for q in range(queries):
                valid = rows[head, q][: _row_count(len(buffer)) + 1]
                quantized = (
                    (valid.clamp_min(0.0).sqrt() * 255.0).round().clamp(0, 255).to(torch.uint8)
                )
                buffer.extend(quantized.numpy().tobytes())

    def tile(self, layer: int, head: int, sequence_length: int) -> bytes | None:
        """Packed lower-triangular bytes for a [T, T] matrix, or None if unavailable."""
        needed = sequence_length * (sequence_length + 1) // 2
        buffer = self._rows[layer][head]
        if len(buffer) < needed:
            return None
        return bytes(buffer[:needed])

    def rows_stored(self, layer: int, head: int) -> int:
        return _row_count(len(self._rows[layer][head]))

    @property
    def nbytes(self) -> int:
        return sum(len(b) for layer in self._rows for b in layer)


def _row_count(packed_length: int) -> int:
    """Inverse of n(n+1)/2."""
    n = int((math.isqrt(8 * packed_length + 1) - 1) // 2)
    return n


def rope_rotate(x: torch.Tensor, positions: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """RoPE at explicit absolute positions.

    Identical to `RotaryEmbedding.forward` when `positions` is `arange(T)`; the
    difference is that a cached decode can pass the one position it is actually
    computing.
    """
    freqs = positions.to(inv_freq.dtype)[:, None] * inv_freq[None, :]
    cos = freqs.cos()[None, :, :]
    sin = freqs.sin()[None, :, :]

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    out = torch.zeros_like(x)
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out


@dataclass
class DecodeRun:
    """One generation run, exposed as a generator over pipeline stages.

    Advancing the generator by one is exactly one stage of execution, which is
    what lets the session halt between any two of them without threads or locks
    in the model code.
    """

    model: torch.nn.Module
    num_layers: int
    num_heads: int
    head_dim: int
    hidden_size: int
    prompt_ids: list[int]
    sampling: SamplingState
    eos_token_id: int | None = None
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def __post_init__(self) -> None:
        self.caches: list[LayerCache] = [LayerCache() for _ in range(self.num_layers)]
        self.attention = AttentionStore(self.num_layers, self.num_heads)
        self.generated_ids: list[int] = []
        #: Entropy of the current query row per (layer, head), carried between
        #: steps so layers below the program counter show their last known value
        #: rather than a fabricated one.
        self.head_entropy = torch.full(
            (self.num_layers, self.num_heads), math.log(max(2, len(self.prompt_ids)))
        )
        #: Residual stream by layer and stage, indexed by absolute position.
        self.residuals: dict[tuple[int, str], list[torch.Tensor]] = {}
        self.key_norms: list[list[float]] = [[] for _ in range(self.num_layers)]
        #: Full logits per step, for `GET /logits?k=0`. 120 kB each.
        self.logits_by_step: dict[int, torch.Tensor] = {}
        #: The sampling decision taken at each step, for replaying past steps.
        self.samples_by_step: dict[int, SampleResult] = {}
        #: Positions whose K/V are resident, per layer. Drives the KV grid.
        self.cache_fill: list[int] = [0] * self.num_layers
        self._generator: Iterator[StageOutcome] | None = None
        self._rng = torch.Generator(device="cpu")
        if self.sampling.seed is not None:
            self._rng.manual_seed(int(self.sampling.seed))
        else:
            self._rng.seed()

    # -- public -----------------------------------------------------------

    @property
    def sequence_length(self) -> int:
        return len(self.prompt_ids) + len(self.generated_ids)

    def stages(self) -> Iterator[StageOutcome]:
        if self._generator is None:
            self._generator = self._run()
        return self._generator

    def close(self) -> None:
        if self._generator is not None:
            self._generator.close()
            self._generator = None

    def kv_bytes(self) -> int:
        return sum(c.nbytes for c in self.caches)

    # -- the run ----------------------------------------------------------

    def _run(self) -> Iterator[StageOutcome]:
        import time

        step = 0
        while step < self.sampling.max_new_tokens:
            # Positions computed on this step: the whole prompt on step 0, then
            # one new token per step thereafter.
            if step == 0:
                positions = list(range(len(self.prompt_ids)))
                input_ids = list(self.prompt_ids)
            else:
                positions = [len(self.prompt_ids) + step - 1]
                input_ids = [self.generated_ids[-1]]

            seq_len = len(self.prompt_ids) + step
            hidden: torch.Tensor | None = None
            logits: torch.Tensor | None = None
            sample: SampleResult | None = None

            for stage in stage_sequence(self.num_layers, step):
                started = time.perf_counter()

                if stage.kind == "tokenize":
                    outcome = StageOutcome(step, stage, seq_len)

                elif stage.kind == "embed":
                    hidden = self._embed(input_ids, positions)
                    outcome = StageOutcome(step, stage, seq_len, residual=hidden[0, -1].clone())

                elif stage.kind == "attention":
                    assert hidden is not None
                    hidden, head_max = self._attention(stage.layer or 0, hidden, positions)
                    self._store_residual(stage.layer or 0, "post_attention", positions, hidden)
                    outcome = StageOutcome(
                        step,
                        stage,
                        seq_len,
                        residual=hidden[0, -1].clone(),
                        head_max_attention=head_max,
                    )

                elif stage.kind == "ffn":
                    assert hidden is not None
                    hidden = self._ffn(stage.layer or 0, hidden)
                    self._store_residual(stage.layer or 0, "post_ffn", positions, hidden)
                    outcome = StageOutcome(step, stage, seq_len, residual=hidden[0, -1].clone())

                elif stage.kind == "final_norm":
                    assert hidden is not None
                    with torch.no_grad():
                        hidden = self.model.norm(hidden)
                    outcome = StageOutcome(step, stage, seq_len, residual=hidden[0, -1].clone())

                elif stage.kind == "lm_head":
                    assert hidden is not None
                    with torch.no_grad():
                        logits = self.model.head(hidden)
                    self.logits_by_step[step] = logits[0, -1].to(torch.float32).cpu().clone()
                    sample = self._sample(logits[0, -1])
                    outcome = StageOutcome(step, stage, seq_len, sample=sample)

                elif stage.kind == "sample":
                    assert logits is not None
                    # Re-draw so that a temperature changed while halted at
                    # lm_head takes effect on this very step.
                    sample = self._sample(logits[0, -1])
                    self.samples_by_step[step] = sample
                    outcome = StageOutcome(step, stage, seq_len, sample=sample)

                else:  # emit
                    assert sample is not None
                    self.generated_ids.append(sample.token_id)
                    outcome = StageOutcome(
                        step,
                        stage,
                        seq_len,
                        sample=sample,
                        emitted_token_id=sample.token_id,
                    )

                outcome.elapsed_ms = (time.perf_counter() - started) * 1000.0
                yield outcome

            if (
                self.eos_token_id is not None
                and self.generated_ids
                and self.generated_ids[-1] == self.eos_token_id
            ):
                return

            step += 1

    # -- stages -----------------------------------------------------------

    def _embed(self, input_ids: list[int], positions: list[int]) -> torch.Tensor:
        ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        pos = torch.tensor([positions], dtype=torch.long, device=self.device)
        with torch.no_grad():
            return self.model.token_emb(ids) + self.model.pos_emb(pos)

    def _attention(
        self, layer: int, x: torch.Tensor, positions: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        block = self.model.layers[layer]
        attn = block.attn
        cache = self.caches[layer]
        pos = torch.tensor(positions, dtype=torch.long, device=self.device)

        with torch.no_grad():
            normed = block.norm1(x)
            b, t, _ = normed.shape

            q = attn.to_q(normed).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
            k = attn.to_k(normed).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
            v = attn.to_v(normed).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

            rope_dim = attn.rope_dim
            inv_freq = attn.rotary.inv_freq
            q = torch.cat([rope_rotate(q[..., :rope_dim], pos, inv_freq), q[..., rope_dim:]], -1)
            k = torch.cat([rope_rotate(k[..., :rope_dim], pos, inv_freq), k[..., rope_dim:]], -1)

            # Cache the rotated keys, which is what makes a cached decode
            # equivalent to recomputing the prefix.
            cache.append(k, v)
            keys, values = cache.keys, cache.values
            assert keys is not None and values is not None

            scores = (q @ keys.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # Queries occupy the last `t` cached positions, so query i may see
            # keys up to offset + i.
            total = keys.shape[2]
            offset = total - t
            causal = torch.arange(total, device=self.device)[None, :] <= (
                torch.arange(t, device=self.device)[:, None] + offset
            )
            scores = scores.masked_fill(~causal[None, None, :, :], float("-inf"))

            weights = F.softmax(scores, dim=-1)
            out = weights @ values
            out = out.transpose(1, 2).contiguous().view(b, t, self.num_heads * self.head_dim)
            result = x + attn.proj(out)

            rows = weights[0].to(torch.float32).cpu()
            self.attention.append_rows(layer, rows)

            probs = rows[:, -1, :]
            safe = torch.where(probs > 0, probs, torch.ones_like(probs))
            self.head_entropy[layer] = -(probs * safe.log()).sum(dim=-1)

            self.cache_fill[layer] = total
            self.key_norms[layer].extend(
                torch.linalg.vector_norm(k[0].to(torch.float32), dim=-1).mean(dim=0).cpu().tolist()
            )

            return result, probs.max(dim=-1).values

    def _ffn(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        block = self.model.layers[layer]
        with torch.no_grad():
            return x + block.ff(block.norm2(x))

    def _store_residual(
        self, layer: int, stage: str, positions: list[int], hidden: torch.Tensor
    ) -> None:
        bucket = self.residuals.setdefault((layer, stage), [])
        values = hidden[0].to(torch.float32).cpu()
        for index, _position in enumerate(positions):
            bucket.append(values[index].clone())

    def _sample(self, logits: torch.Tensor) -> SampleResult:
        s = self.sampling
        scores = logits.to(torch.float32)

        if s.mode == "greedy":
            temperature = 0.0
            probs = torch.zeros_like(scores)
            probs[int(scores.argmax())] = 1.0
        else:
            temperature = max(1e-4, float(s.temperature))
            scaled = scores / temperature
            if s.mode == "top_k" and s.top_k > 0:
                cutoff = torch.topk(scaled, min(int(s.top_k), scaled.numel())).values[-1]
                scaled = scaled.masked_fill(scaled < cutoff, float("-inf"))
            probs = F.softmax(scaled, dim=-1)

        k = min(TOP_K_STORED, probs.numel())
        top = torch.topk(probs, k)
        top_ids = [int(i) for i in top.indices]
        top_probs = [float(p) for p in top.values]

        if s.mode == "greedy":
            token_id = top_ids[0]
        else:
            token_id = int(torch.multinomial(probs, 1, generator=self._rng))

        if token_id in top_ids:
            chosen_rank = top_ids.index(token_id)
        else:
            # Sampling can land outside the stored list -- at temperature 2 and
            # above it usually does. Keep the token anyway: showing where the
            # emitted token sat in the distribution is the panel's whole job,
            # and a list that omits it is the one case that must not happen.
            # Its true rank is one comparison, so report that rather than a
            # sentinel the client would have to special-case.
            chosen_rank = int((probs > probs[token_id]).sum())
            top_ids.append(token_id)
            top_probs.append(float(probs[token_id]))

        nonzero = probs[probs > 0]
        entropy = float(-(nonzero * nonzero.log()).sum())

        return SampleResult(
            token_id=token_id,
            chosen_rank=chosen_rank,
            entropy=entropy,
            temperature=temperature,
            top_ids=top_ids,
            top_logits=[float(scores[i]) for i in top_ids],
            top_probs=top_probs,
        )
