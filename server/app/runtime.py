"""Loading the model and tokenizer once, and describing them honestly."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from tokenizers import ByteLevelBPETokenizer

from .pipeline import STAGE_DESCRIPTORS

SERVER_ROOT = Path(__file__).resolve().parents[1]

#: Reserved ids from train_tokenizer.py: <s> <pad> </s> <unk> <mask>.
SPECIAL_TOKEN_COUNT = 5


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw) if raw else default


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in ("", "0", "false", "no", "off")


CHECKPOINT_PATH = _env_path("MDEBUG_CHECKPOINT", SERVER_ROOT / "checkpoints" / "ckpt_5000.pt")
TOKENIZER_DIR = _env_path("MDEBUG_TOKENIZER_DIR", SERVER_ROOT / "data" / "tokenizer")


def _byte_decoder() -> dict[str, int]:
    """Reverse of GPT-2's bytes_to_unicode table."""
    printable = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    mapping = {b: b for b in printable}
    spare = 0
    for byte in range(256):
        if byte not in mapping:
            mapping[byte] = 256 + spare
            spare += 1
    return {chr(code): byte for byte, code in mapping.items()}


BYTE_DECODER = _byte_decoder()


def display_token(text: str) -> str:
    """The human-readable form of a token piece.

    Byte-level BPE works in an alphabet where every byte is a printable
    character, so any token holding a multi-byte character arrives looking like
    mojibake -- a curly quote is the three characters U+00E2 U+0122 U+013E.
    Mapping back through the byte table and decoding as UTF-8 restores it. A
    token that is only part of a character decodes to U+FFFD, which is the
    honest answer: it really is a fragment.

    Whitespace then becomes visible glyphs, because rendering real spaces would
    hide exactly where the tokenizer cut.
    """
    try:
        raw = bytes(BYTE_DECODER[c] for c in text)
    except KeyError:
        # Not in the byte alphabet (a special token such as <s>): show as-is.
        return text
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError:
        # A token that is only part of a character. U+FFFD would hide which
        # bytes those are; this audience reads hex, and the fragment is the
        # interesting fact.
        return "".join(f"\\x{b:02X}" for b in raw)
    return decoded.replace(" ", "\u00b7").replace("\n", "\u23ce").replace("\t", "\u21e5")


class Tokenizer:
    def __init__(self, directory: Path) -> None:
        vocab = directory / "vocab.json"
        merges = directory / "merges.txt"
        if not vocab.exists() or not merges.exists():
            raise FileNotFoundError(f"tokenizer files not found in {directory}")
        self._tok = ByteLevelBPETokenizer(str(vocab), str(merges))
        self.size = self._tok.get_vocab_size()

    def encode(self, text: str) -> list[tuple[int, str]]:
        encoded = self._tok.encode(text)
        return list(zip(encoded.ids, encoded.tokens))

    def id_to_token(self, token_id: int) -> str:
        return self._tok.id_to_token(token_id) or "<unk>"

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)

    def is_special(self, token_id: int) -> bool:
        return token_id < SPECIAL_TOKEN_COUNT

    def tokens(self, text: str, origin: str, start: int = 0) -> list[dict[str, Any]]:
        return [
            {
                "id": token_id,
                "text": piece,
                "display": display_token(piece),
                "position": start + index,
                "isSpecial": self.is_special(token_id),
                "origin": origin,
            }
            for index, (token_id, piece) in enumerate(self.encode(text))
        ]

    def token(self, token_id: int, position: int, origin: str) -> dict[str, Any]:
        piece = self.id_to_token(token_id)
        return {
            "id": token_id,
            "text": piece,
            "display": display_token(piece),
            "position": position,
            "isSpecial": self.is_special(token_id),
            "origin": origin,
        }


@dataclass
class Runtime:
    model: torch.nn.Module
    tokenizer: Tokenizer
    config: dict[str, Any]
    model_info: dict[str, Any]
    checkpoint_step: int | None


#: Product limits, not technical ones. A 1024x1024 attention matrix is 84 MB as
#: u8 for the whole tensor and nobody can read one; the tool is for
#: understanding, so the useful range is bounded well below `seq_len`.
LIMITS = {
    "maxPromptTokens": int(os.environ.get("MDEBUG_MAX_PROMPT_TOKENS", 128)),
    "maxNewTokens": int(os.environ.get("MDEBUG_MAX_NEW_TOKENS", 64)),
    "maxTotalTokens": int(os.environ.get("MDEBUG_MAX_TOTAL_TOKENS", 192)),
    "attentionWarnThreshold": int(os.environ.get("MDEBUG_ATTENTION_WARN", 512)),
}


def build_model_info(config: dict[str, Any], tokenizer: Tokenizer, parameters: int) -> dict[str, Any]:
    """What the client renders every label and the whole graph from.

    `notes` is not decoration. This model's class names do not describe what it
    does, and the audience reads labels literally, so the discrepancies are
    reported rather than smoothed over.
    """
    head_dim = config["hidden_size"] // config["num_attention_heads"]
    return {
        "name": (
            f"QwenModel (from scratch, {config['num_hidden_layers']}L/"
            f"{config['num_attention_heads']}H/{config['hidden_size']}d, "
            f"{parameters / 1e6:.0f}M params)"
        ),
        "numLayers": config["num_hidden_layers"],
        "numHeads": config["num_attention_heads"],
        "hiddenSize": config["hidden_size"],
        "headDim": head_dim,
        "intermediateSize": config["intermediate_size"],
        "vocabSize": config["vocab_size"],
        "tokenizerVocabSize": tokenizer.size,
        "maxPositionEmbeddings": config["max_position_embeddings"],
        "ropePct": config["rotary_pct"],
        "ropeDim": int(head_dim * config["rotary_pct"]),
        # Both true: this server adds a real cache and really captures attention.
        "hasKvCache": True,
        "capturesAttention": True,
        "limits": LIMITS,
        "stages": STAGE_DESCRIPTORS,
        "notes": [
            "GroupedQueryAttention is plain multi-head attention - there is no KV-head grouping.",
            (
                "RMSNorm divides by the L2 norm rather than sqrt(mean(x^2)), so it returns "
                "vectors of norm ~1 instead of ~sqrt(512)=22.6. Measured consequence: "
                "attention scores span ~0.007 instead of ~1.35, so softmax is nearly flat "
                "and every head attends almost uniformly. That is why the head marks in the "
                "layer ladder are pale -- the attention really has collapsed, and the "
                "entropy of each head's current row sits within 0.001 of log(T)."
            ),
            "The model adds learned absolute position embeddings AND applies RoPE inside attention.",
            (
                f"vocab_size is {config['vocab_size']} but the trained tokenizer has "
                f"{tokenizer.size} entries, so the remaining logits are unreachable."
            ),
            (
                "The KV cache and attention capture live in the debugger's runner, not in "
                "src/model.py, which is unmodified. Outputs are asserted equal to the "
                "model's own forward pass."
            ),
            "Trained for 5000 steps on a single book, so its continuations repeat heavily.",
        ],
    }


@lru_cache(maxsize=1)
def get_runtime() -> Runtime:
    """Loads the checkpoint once per process. Roughly a second on CPU."""
    import sys

    if str(SERVER_ROOT) not in sys.path:
        sys.path.insert(0, str(SERVER_ROOT))

    from src.config import config
    from src.model import QwenModel

    # torch sizes its pools from the host's core count, not the container's CPU
    # limit, which oversubscribes a small Cloud Run instance. OMP_NUM_THREADS in
    # the image covers the OpenMP runtime, which is initialised before we get
    # here; this covers torch's own intra/inter-op pools.
    threads = os.environ.get("MDEBUG_TORCH_THREADS")
    if threads:
        torch.set_num_threads(int(threads))
        try:
            torch.set_num_interop_threads(int(threads))
        except RuntimeError:
            pass  # already started; only settable before the first parallel op

    tokenizer = Tokenizer(TOKENIZER_DIR)

    model = QwenModel(config)
    model.eval()

    checkpoint_step: int | None = None
    if CHECKPOINT_PATH.exists():
        state = torch.load(CHECKPOINT_PATH, map_location="cpu")
        model.load_state_dict(state["model_state"])
        checkpoint_step = state.get("step")
    else:
        # Refusing to start would make the API impossible to develop against
        # without a 978 MB file, but silently serving noise would be worse.
        # A deployment sets MDEBUG_REQUIRE_CHECKPOINT so that a missing file
        # fails the revision instead of promoting a service that answers
        # 200 OK from /api/health while generating garbage.
        if _env_flag("MDEBUG_REQUIRE_CHECKPOINT"):
            raise RuntimeError(
                f"MDEBUG_REQUIRE_CHECKPOINT is set but there is no checkpoint at {CHECKPOINT_PATH}"
            )
        print(
            f"WARNING: no checkpoint at {CHECKPOINT_PATH}; serving randomly initialised weights",
            flush=True,
        )

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    parameters = sum(p.numel() for p in model.parameters())
    info = build_model_info(config, tokenizer, parameters)
    if checkpoint_step is None:
        info["notes"].insert(0, "NO CHECKPOINT LOADED - weights are random and outputs are noise.")
    else:
        info["name"] += f" @ step {checkpoint_step}"

    return Runtime(
        model=model,
        tokenizer=tokenizer,
        config=dict(config),
        model_info=info,
        checkpoint_step=checkpoint_step,
    )
