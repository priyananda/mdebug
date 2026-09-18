"""Produce a weights-only checkpoint for the deployed image.

`src/utils.py:save_checkpoint` writes the optimizer and scheduler state next to
the weights, so `ckpt_5000.pt` is 978 MB of which only 326 MB is the model.
Shipping the rest would triple the container layer and add ~680 MB to the peak
RSS of `torch.load` at startup, for state that inference never reads.

    python scripts/strip_checkpoint.py            # skips if already current
    python scripts/strip_checkpoint.py --force

The source checkpoint is not in git (see .gitignore) and there is no LFS, so it
has to come from training or an out-of-band copy.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = REPO_ROOT / "server" / "checkpoints" / "ckpt_5000.pt"
DEFAULT_DEST = REPO_ROOT / "server" / "checkpoints" / "ckpt_5000.weights.pt"

#: `app/runtime.py` reads `step` back out to label the model and to answer
#: /api/health's `checkpointStep`, so it has to survive the strip.
KEEP = ("step", "model_state")


def _load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except Exception as exc:  # optimizer/scheduler pickles are not always allowlisted
        print(f"note: strict load failed ({exc.__class__.__name__}), retrying unrestricted", flush=True)
        return torch.load(path, map_location="cpu", mmap=True, weights_only=False)


def _verify(dest: Path, original: dict) -> None:
    """Reload what we just wrote and prove it matches, before anything ships it."""
    reloaded = torch.load(dest, map_location="cpu", weights_only=True)
    if set(reloaded) != set(KEEP):
        raise SystemExit(f"verify failed: {dest} has keys {sorted(reloaded)}, expected {sorted(KEEP)}")
    if reloaded["step"] != original["step"]:
        raise SystemExit("verify failed: step did not round-trip")

    before, after = original["model_state"], reloaded["model_state"]
    if set(before) != set(after):
        raise SystemExit("verify failed: tensor names differ")
    for name, tensor in before.items():
        if tensor.shape != after[name].shape or tensor.dtype != after[name].dtype:
            raise SystemExit(f"verify failed: {name} changed shape or dtype")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--force", action="store_true", help="rewrite even if the output looks current")
    args = parser.parse_args()

    if not args.src.exists():
        print(
            f"error: no checkpoint at {args.src}\n"
            "It is gitignored and not in LFS -- copy it in or retrain with server/train.py.",
            file=sys.stderr,
        )
        return 1

    # Mtime rather than size: a retrained checkpoint of identical size must not
    # be mistaken for the one we already stripped.
    if args.dest.exists() and not args.force:
        if args.dest.stat().st_mtime >= args.src.stat().st_mtime:
            print(f"up to date: {args.dest} ({args.dest.stat().st_size / 1e6:.0f} MB)")
            return 0

    print(f"reading {args.src} ({args.src.stat().st_size / 1e6:.0f} MB)", flush=True)
    full = _load(args.src)
    missing = [key for key in KEEP if key not in full]
    if missing:
        print(f"error: {args.src} is missing {missing}", file=sys.stderr)
        return 1

    stripped = {key: full[key] for key in KEEP}
    dropped = sorted(set(full) - set(KEEP))

    # Write beside the target then rename, so an interrupt cannot leave a
    # truncated file that the Dockerfile's `test -s` would happily accept.
    tmp = args.dest.with_suffix(".pt.tmp")
    args.dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stripped, tmp)
    os.replace(tmp, args.dest)

    _verify(args.dest, stripped)

    parameters = sum(t.numel() for t in stripped["model_state"].values())
    print(
        f"wrote {args.dest} ({args.dest.stat().st_size / 1e6:.0f} MB)\n"
        f"  step={stripped['step']}  tensors={len(stripped['model_state'])}  parameters={parameters:,}\n"
        f"  dropped={dropped}  verified=ok"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
