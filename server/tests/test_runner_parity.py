"""The instrumented runner must compute what the model computes.

`app/runner.py` re-expresses `QwenModel.forward` as a steppable generator with a
KV cache, so the debugger is only trustworthy if the two agree numerically.
These tests check that against the real checkpoint: everything the UI shows
rests on them.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.runner import DecodeRun, SamplingState, _row_count, rope_rotate  # noqa: E402

CHECKPOINT = ROOT / "checkpoints" / "ckpt_5000.pt"
PROMPT = "The key to happiness is not found in wealth but in attention"


@pytest.fixture(scope="module")
def model():
    from src.config import config
    from src.model import QwenModel

    if not CHECKPOINT.exists():
        pytest.skip(f"checkpoint not present at {CHECKPOINT}")

    m = QwenModel(config)
    m.eval()
    state = torch.load(CHECKPOINT, map_location="cpu")
    m.load_state_dict(state["model_state"])
    return m


@pytest.fixture(scope="module")
def prompt_ids():
    from tokenizers import ByteLevelBPETokenizer

    tok = ByteLevelBPETokenizer(
        str(ROOT / "data" / "tokenizer" / "vocab.json"),
        str(ROOT / "data" / "tokenizer" / "merges.txt"),
    )
    return tok.encode(PROMPT).ids


def make_run(model, prompt_ids, **sampling):
    from src.config import config

    return DecodeRun(
        model=model,
        num_layers=config["num_hidden_layers"],
        num_heads=config["num_attention_heads"],
        head_dim=config["hidden_size"] // config["num_attention_heads"],
        hidden_size=config["hidden_size"],
        prompt_ids=list(prompt_ids),
        sampling=SamplingState(**sampling),
    )


def run_until(run, predicate):
    """Advance the generator until `predicate(outcome)`, returning that outcome."""
    for outcome in run.stages():
        if predicate(outcome):
            return outcome
    raise AssertionError("generator finished without matching")


class TestRopeParity:
    def test_matches_the_models_own_rotary_at_arange_positions(self, model):
        rotary = model.layers[0].attn.rotary
        x = torch.randn(1, 8, 7, 16)

        expected = rotary(x)
        actual = rope_rotate(x, torch.arange(7), rotary.inv_freq)

        assert torch.allclose(expected, actual, atol=1e-6)

    def test_rotating_one_position_matches_that_row_of_a_full_rotation(self, model):
        """The property the KV cache depends on.

        `RotaryEmbedding.forward` derives positions from the input length, so a
        single cached query would be rotated as position 0. Rotating at the
        explicit position has to reproduce the corresponding row of a full
        rotation, or cached decoding silently diverges.
        """
        rotary = model.layers[0].attn.rotary
        x = torch.randn(1, 8, 12, 16)
        full = rope_rotate(x, torch.arange(12), rotary.inv_freq)

        for position in (0, 1, 5, 11):
            single = rope_rotate(x[:, :, position : position + 1], torch.tensor([position]), rotary.inv_freq)
            assert torch.allclose(full[:, :, position : position + 1], single, atol=1e-6)


class TestPrefillParity:
    def test_logits_match_the_models_forward_pass(self, model, prompt_ids):
        """The whole debugger rests on this: stage-by-stage == all at once."""
        run = make_run(model, prompt_ids, max_new_tokens=1, mode="greedy")
        run_until(run, lambda o: o.stage.kind == "lm_head")

        with torch.no_grad():
            expected = model(torch.tensor([prompt_ids]))[0, -1]

        actual = run.logits_by_step[0]
        assert actual.shape == expected.shape
        assert torch.allclose(actual, expected, atol=2e-4), (
            f"max abs diff {float((actual - expected).abs().max())}"
        )

    def test_greedy_choice_matches_the_models_argmax(self, model, prompt_ids):
        run = make_run(model, prompt_ids, max_new_tokens=1, mode="greedy")
        outcome = run_until(run, lambda o: o.stage.kind == "sample")

        with torch.no_grad():
            expected = int(model(torch.tensor([prompt_ids]))[0, -1].argmax())

        assert outcome.sample is not None
        assert outcome.sample.token_id == expected


class TestCachedDecodeParity:
    """A KV cache is only worth having if it changes speed and nothing else."""

    def test_cached_generation_matches_full_recomputation(self, model, prompt_ids):
        steps = 6
        run = make_run(model, prompt_ids, max_new_tokens=steps, mode="greedy")
        for _ in run.stages():
            pass
        cached = run.generated_ids

        # The naive loop server/infer.py uses: re-run the whole prefix each time.
        ids = list(prompt_ids)
        recomputed = []
        for _ in range(steps):
            with torch.no_grad():
                logits = model(torch.tensor([ids]))
            nxt = int(logits[0, -1].argmax())
            recomputed.append(nxt)
            ids.append(nxt)

        assert cached == recomputed

    def test_cached_logits_match_recomputation_at_every_step(self, model, prompt_ids):
        steps = 4
        run = make_run(model, prompt_ids, max_new_tokens=steps, mode="greedy")
        for _ in run.stages():
            pass

        ids = list(prompt_ids)
        for step in range(steps):
            with torch.no_grad():
                expected = model(torch.tensor([ids]))[0, -1]
            actual = run.logits_by_step[step]
            diff = float((actual - expected).abs().max())
            assert diff < 5e-3, f"step {step}: max abs diff {diff}"
            ids.append(run.generated_ids[step])


class TestAttentionCapture:
    def test_rows_are_causal_and_sum_to_one(self, model, prompt_ids):
        run = make_run(model, prompt_ids, max_new_tokens=3, mode="greedy")
        for _ in run.stages():
            pass

        t = len(prompt_ids) + 3 - 1
        packed = run.attention.tile(layer=0, head=0, sequence_length=t)
        assert packed is not None
        assert len(packed) == t * (t + 1) // 2

        # Undo the wire encoding the way the client does, then check the rows
        # are real distributions.
        values = torch.frombuffer(bytearray(packed), dtype=torch.uint8).to(torch.float32)
        values = (values / 255.0) ** 2
        cursor = 0
        for row in range(t):
            width = row + 1
            total = float(values[cursor : cursor + width].sum())
            cursor += width
            # u8 + sqrt quantization costs a little mass; half a step per cell.
            assert abs(total - 1.0) < 0.02, f"row {row} sums to {total}"

    def test_a_tile_is_a_prefix_of_a_later_one(self, model, prompt_ids):
        """Appending rows must never rewrite history, or past steps would shift."""
        run = make_run(model, prompt_ids, max_new_tokens=3, mode="greedy")
        for _ in run.stages():
            pass

        early = run.attention.tile(0, 0, len(prompt_ids))
        later = run.attention.tile(0, 0, len(prompt_ids) + 1)
        assert early is not None and later is not None
        assert later[: len(early)] == early

    def test_head_entropy_is_within_bounds(self, model, prompt_ids):
        run = make_run(model, prompt_ids, max_new_tokens=2, mode="greedy")
        for _ in run.stages():
            pass

        ceiling = math.log(len(prompt_ids) + 1) + 1e-6
        assert torch.all(run.head_entropy >= 0)
        assert torch.all(run.head_entropy <= ceiling)


class TestRowCount:
    @pytest.mark.parametrize("n", [0, 1, 2, 5, 13, 192])
    def test_inverts_triangular_numbers(self, n):
        assert _row_count(n * (n + 1) // 2) == n
