"""A debugging session: one run, its breakpoints, and the events it emits.

The execution model is breakpoints-only. There is no step-into or step-over,
because a stage is the smallest unit of execution and there is nothing nested
to step into; `step` advances exactly one stage and `run_to_cursor` installs a
one-shot breakpoint.

Concurrency: advancing the generator by one stage is a blocking torch call, so
it runs in the default executor. Everything else -- breakpoint evaluation,
halting, event fan-out -- stays on the event loop, which means no locks and no
shared mutable state between threads beyond the generator itself.
"""

from __future__ import annotations

import asyncio
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterator

import torch

from . import encoding
from .breakpoints import Breakpoint, StageValues, evaluate, validate
from .pipeline import StageRef, parse_stage_id
from .runner import DecodeRun, SampleResult, SamplingState, StageOutcome
from .runtime import Runtime

#: The contract's guarantee: `stage_entered` is coalesced to at most this many
#: per second. It is pure animation data at ~45 events per token, and a stage
#: that causes a halt is never dropped.
STAGE_EVENT_RATE_LIMIT = 30
_MIN_STAGE_INTERVAL = 1.0 / STAGE_EVENT_RATE_LIMIT

#: How many candidates ride along in `halted` and `token_emitted`.
#:
#: Cut from 50 after measuring: this model puts ~98.3 % of the mass on the
#: top-1 token, and ranks 2-50 share the remaining 0.7 % between them, so the
#: tail was costing real bytes on every token to show nothing. Twelve still
#: covers ~98.7 % and leaves eleven visible alternatives, which is what the
#: panel is for.
#:
#: `runner.TOP_K_STORED` is deliberately larger: the full list stays on the
#: server so `GET .../logits?k=` can still widen it on demand, and `k=0` still
#: returns the whole vocabulary. This is the contract's rule 1 -- eager
#: payloads are bounded, depth is fetched.
TOP_K = 12


class SessionClosed(Exception):
    pass


@dataclass
class SessionConfig:
    prompt: str = ""
    max_new_tokens: int = 32
    sampling_mode: str = "temperature"
    temperature: float | None = 0.8
    top_k: int | None = 40
    seed: int | None = None
    capture_attention: bool = True

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "SessionConfig":
        return cls(
            prompt=str(raw.get("prompt", "")),
            max_new_tokens=int(raw.get("maxNewTokens", 32)),
            sampling_mode=str(raw.get("samplingMode", "temperature")),
            temperature=raw.get("temperature", 0.8),
            top_k=raw.get("topK", 40),
            seed=raw.get("seed"),
            capture_attention=bool(raw.get("captureAttention", True)),
        )

    def merge(self, patch: dict[str, Any]) -> None:
        if "prompt" in patch:
            self.prompt = str(patch["prompt"])
        if "maxNewTokens" in patch:
            self.max_new_tokens = int(patch["maxNewTokens"])
        if "samplingMode" in patch:
            self.sampling_mode = str(patch["samplingMode"])
        if "temperature" in patch:
            self.temperature = patch["temperature"]
        if "topK" in patch:
            self.top_k = patch["topK"]
        if "seed" in patch:
            self.seed = patch["seed"]
        if "captureAttention" in patch:
            self.capture_attention = bool(patch["captureAttention"])

    def as_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "prompt": self.prompt,
            "maxNewTokens": self.max_new_tokens,
            "samplingMode": self.sampling_mode,
            "captureAttention": self.capture_attention,
        }
        if self.temperature is not None:
            out["temperature"] = self.temperature
        if self.top_k is not None:
            out["topK"] = self.top_k
        if self.seed is not None:
            out["seed"] = self.seed
        return out


@dataclass
class HaltPosition:
    step: int
    stage: StageRef
    sequence_length: int
    reason: str
    breakpoint_id: str | None = None

    def as_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "step": self.step,
            "stage": self.stage.as_json(),
            "stageId": self.stage.id,
            "sequenceLength": self.sequence_length,
            "reason": self.reason,
        }
        if self.breakpoint_id is not None:
            out["breakpointId"] = self.breakpoint_id
        return out


class Session:
    def __init__(self, runtime: Runtime, config: SessionConfig) -> None:
        self.id = uuid.uuid4().hex[:10]
        self.runtime = runtime
        self.config = config
        self.created_at = _now_iso()
        self.updated_at = self.created_at
        self.status = "idle"
        self.breakpoints: list[Breakpoint] = []

        self.prompt_tokens: list[dict[str, Any]] = runtime.tokenizer.tokens(config.prompt, "prompt")
        self.generated_tokens: list[dict[str, Any]] = []
        self.current_step = 0
        self.halt: HaltPosition | None = None

        self._run: DecodeRun | None = None
        self._sampling = SamplingState()
        self._task: asyncio.Task[None] | None = None
        self._resume = asyncio.Event()
        self._stop_requested = False
        self._step_budget: float = math.inf
        self._next_halt_reason = "step"
        self._subscribers: list[asyncio.Queue[dict[str, Any]]] = []
        self._last_stage_event = 0.0
        self._prompt_tokens_sent = False
        self._generated_tokens_sent = 0
        self._last_hit_breakpoint: str | None = None
        self._closed = False

    # -- subscriptions ----------------------------------------------------

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=512)
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        message = {"v": 1, "id": uuid.uuid4().hex, "ts": int(time.time() * 1000),
                   "type": event_type, "payload": payload}
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                # A client that cannot keep up loses animation frames, not
                # state: it resyncs from `session_state` on reconnect.
                pass

    # -- serialisation ----------------------------------------------------

    def as_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
            "status": self.status,
            "config": self.config.as_json(),
            "model": self.runtime.model_info,
            "promptTokens": self.prompt_tokens,
            "generatedTokens": self.generated_tokens,
            "currentStep": self.current_step,
            "halt": self.halt.as_json() if self.halt else None,
            "breakpoints": [b.as_json() for b in self.breakpoints],
        }

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "createdAt": self.created_at,
            "status": self.status,
            "promptPreview": self.config.prompt[:60],
            "tokensGenerated": len(self.generated_tokens),
        }

    def emit_state(self) -> None:
        self._emit("session_state", self.as_json())

    # -- commands ---------------------------------------------------------

    def start(self, config_patch: dict[str, Any] | None = None) -> None:
        """Start always restarts, the way it does in a real debugger.

        `continue` is what resumes; there is no "run on from where I stopped".
        """
        if self._task and not self._task.done():
            return
        if config_patch:
            self.config.merge(config_patch)

        limits = self.runtime.model_info["limits"]
        self.config.max_new_tokens = max(1, min(self.config.max_new_tokens, limits["maxNewTokens"]))

        self.prompt_tokens = self.runtime.tokenizer.tokens(self.config.prompt, "prompt")
        if len(self.prompt_tokens) > limits["maxPromptTokens"]:
            self.prompt_tokens = self.prompt_tokens[: limits["maxPromptTokens"]]
        self.generated_tokens = []
        self.current_step = 0
        self.halt = None
        self._prompt_tokens_sent = False
        self._generated_tokens_sent = 0
        self._stop_requested = False
        self._step_budget = math.inf
        self._next_halt_reason = "breakpoint"

        self._sampling = SamplingState(
            max_new_tokens=self.config.max_new_tokens,
            mode=self.config.sampling_mode,
            temperature=float(self.config.temperature or 1.0),
            top_k=int(self.config.top_k or 0),
            seed=self.config.seed,
        )
        info = self.runtime.model_info
        self._run = DecodeRun(
            model=self.runtime.model,
            num_layers=info["numLayers"],
            num_heads=info["numHeads"],
            head_dim=info["headDim"],
            hidden_size=info["hiddenSize"],
            prompt_ids=[t["id"] for t in self.prompt_tokens],
            sampling=self._sampling,
        )
        self._task = asyncio.create_task(self._drive())

    def continue_run(self) -> None:
        self._step_budget = math.inf
        self._next_halt_reason = "breakpoint"
        self._resume.set()

    def step(self, count: int = 1) -> None:
        self._step_budget = max(1, int(count))
        self._next_halt_reason = "step"
        self._resume.set()

    def stop(self) -> None:
        self._stop_requested = True
        self._resume.set()

    def set_breakpoints(self, raw: list[dict[str, Any]]) -> list[Breakpoint]:
        previous = {b.id: b.hit_count for b in self.breakpoints}
        parsed: list[Breakpoint] = []
        for entry in raw:
            breakpoint = Breakpoint.from_json(entry)
            if not breakpoint.id:
                breakpoint.id = uuid.uuid4().hex[:8]
            validate(breakpoint)
            breakpoint.hit_count = previous.get(breakpoint.id, breakpoint.hit_count)
            parsed.append(breakpoint)
        self.breakpoints = parsed
        self._emit("breakpoints_changed", {"breakpoints": [b.as_json() for b in parsed]})
        return parsed

    def run_to_cursor(self, stage_id: str, step: int | None = None) -> None:
        condition = None if step is None else {"kind": "token_index", "op": "==", "value": step}
        self.breakpoints.append(
            Breakpoint(
                id=f"oneshot-{uuid.uuid4().hex[:6]}",
                stage_id=stage_id,
                enabled=True,
                condition=condition,
                one_shot=True,
            )
        )
        self._emit("breakpoints_changed", {"breakpoints": [b.as_json() for b in self.breakpoints]})
        if self._task and not self._task.done():
            self.continue_run()
        else:
            self.start()

    def patch_config(self, patch: dict[str, Any]) -> None:
        self.config.merge(patch)
        # The runner reads sampling from this object at the moment it samples,
        # so a temperature changed while halted applies to the current step.
        if "temperature" in patch and patch["temperature"] is not None:
            self._sampling.temperature = float(patch["temperature"])
        if "samplingMode" in patch:
            self._sampling.mode = str(patch["samplingMode"])
        if "topK" in patch and patch["topK"] is not None:
            self._sampling.top_k = int(patch["topK"])
        if "maxNewTokens" in patch:
            self._sampling.max_new_tokens = int(patch["maxNewTokens"])
        self.emit_state()

    async def close(self) -> None:
        self._closed = True
        self.stop()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        if self._run:
            self._run.close()
        self.status = "closed"

    # -- the run loop -----------------------------------------------------

    async def _drive(self) -> None:
        assert self._run is not None
        loop = asyncio.get_running_loop()
        stages: Iterator[StageOutcome] = self._run.stages()
        self.status = "running"
        self._emit("run_started", {"step": 0})
        step_started = time.perf_counter()
        reason = "max_tokens"

        try:
            while True:
                outcome = await loop.run_in_executor(None, _advance, stages)
                if outcome is None:
                    break
                if self._stop_requested:
                    reason = "stopped"
                    break

                self.current_step = outcome.step
                self.updated_at = _now_iso()

                if outcome.stage.kind == "emit" and outcome.emitted_token_id is not None:
                    token = self.runtime.tokenizer.token(
                        outcome.emitted_token_id,
                        len(self.prompt_tokens) + outcome.step,
                        "generated",
                    )
                    self.generated_tokens.append(token)
                    self._emit(
                        "token_emitted",
                        {
                            "step": outcome.step,
                            "token": token,
                            "topK": self._top_k(outcome.sample, outcome.step),
                        },
                    )

                halt_reason = self._halt_reason(outcome)
                self._announce_stage(outcome, forced=halt_reason is not None)

                if halt_reason is not None:
                    step_ms = (time.perf_counter() - step_started) * 1000.0
                    await self._halt(outcome, halt_reason, step_ms)
                    step_started = time.perf_counter()
                    if self._stop_requested:
                        reason = "stopped"
                        break
        except asyncio.CancelledError:
            raise
        except Exception as error:  # pragma: no cover - surfaced to the client
            self.status = "error"
            self._emit(
                "error",
                {
                    "code": "runner_failure",
                    "message": f"{type(error).__name__}: {error}",
                    "fatal": True,
                },
            )
            return

        self._finish(reason)

    def _finish(self, reason: str) -> None:
        self.status = "idle" if reason == "stopped" else "finished"
        self.halt = None
        ids = [t["id"] for t in self.prompt_tokens] + [t["id"] for t in self.generated_tokens]
        self._emit(
            "finished",
            {
                "reason": reason,
                "totalSteps": len(self.generated_tokens),
                "text": self.runtime.tokenizer.decode(ids),
            },
        )

    def _announce_stage(self, outcome: StageOutcome, forced: bool) -> None:
        now = time.monotonic()
        if not forced and (now - self._last_stage_event) < _MIN_STAGE_INTERVAL:
            return
        self._last_stage_event = now
        self._emit(
            "stage_entered",
            {
                "step": outcome.step,
                "stageId": outcome.stage.id,
                "sequenceLength": outcome.sequence_length,
            },
        )

    async def _halt(self, outcome: StageOutcome, reason: str, step_ms: float) -> None:
        position = HaltPosition(
            step=outcome.step,
            stage=outcome.stage,
            sequence_length=outcome.sequence_length,
            reason=reason,
            breakpoint_id=self._last_hit_breakpoint,
        )
        self.status = "halted"
        self.halt = position
        self._emit("halted", self._halt_payload(position, outcome, step_ms))

        self._resume.clear()
        await self._resume.wait()

        if self._stop_requested:
            return
        self.status = "running"
        self.halt = None
        self._emit("resumed", {"from": position.as_json()})

    def _halt_reason(self, outcome: StageOutcome) -> str | None:
        self._last_hit_breakpoint = None
        stage_id = outcome.stage.id

        for breakpoint in list(self.breakpoints):
            if not breakpoint.enabled or breakpoint.stage_id != stage_id:
                continue
            if breakpoint.condition is not None:
                values = StageValues(
                    outcome=outcome,
                    display_token=lambda i: self.runtime.tokenizer.token(i, 0, "generated")["display"],
                    hit_count=breakpoint.hit_count,
                )
                if not evaluate(breakpoint.condition, values):
                    continue

            breakpoint.hit_count += 1
            self._last_hit_breakpoint = breakpoint.id
            if breakpoint.one_shot:
                self.breakpoints = [b for b in self.breakpoints if b.id != breakpoint.id]
                self._emit(
                    "breakpoints_changed",
                    {"breakpoints": [b.as_json() for b in self.breakpoints]},
                )
                return "run_to_cursor"
            return "breakpoint"

        if self._step_budget != math.inf:
            self._step_budget -= 1
            if self._step_budget <= 0:
                return self._next_halt_reason
        return None

    # -- payloads ---------------------------------------------------------

    def _halt_payload(
        self, position: HaltPosition, outcome: StageOutcome, step_ms: float
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "position": position.as_json(),
            "sequence": {
                "generatedTokens": self.generated_tokens[self._generated_tokens_sent :],
            },
            "timings": {"stageMs": round(outcome.elapsed_ms, 3), "stepMs": round(step_ms, 3)},
        }
        if not self._prompt_tokens_sent:
            payload["sequence"]["promptTokens"] = self.prompt_tokens
            self._prompt_tokens_sent = True
        self._generated_tokens_sent = len(self.generated_tokens)

        assert self._run is not None
        payload["headSummary"] = encoding.encode_f32(self._run.head_entropy)

        if outcome.residual is not None:
            payload["residual"] = encoding.encode_f32(outcome.residual)
            payload["residualStats"] = encoding.vector_stats(outcome.residual)

        payload["kv"] = self._kv_snapshot(outcome.sequence_length)

        if outcome.stage.kind in ("lm_head", "sample", "emit"):
            top = self._top_k(outcome.sample, outcome.step)
            if top is not None:
                payload["topK"] = top
        return payload

    def _kv_snapshot(self, sequence_length: int) -> dict[str, Any]:
        assert self._run is not None
        run = self._run
        layers = run.num_layers
        t = max(1, sequence_length)
        newest = t - 1

        occupancy = torch.zeros((layers, t), dtype=torch.uint8)
        norms = torch.zeros((layers, t), dtype=torch.float32)
        for layer in range(layers):
            filled = min(run.cache_fill[layer], t)
            if filled:
                # 2 = resident from an earlier step, 1 = written on this one.
                occupancy[layer, :filled] = 2
                if filled > newest:
                    occupancy[layer, newest] = 1
                stored = run.key_norms[layer][:filled]
                if stored:
                    norms[layer, : len(stored)] = torch.tensor(stored, dtype=torch.float32)

        return {
            "numLayers": layers,
            "sequenceLength": t,
            "occupancy": encoding.encode_u8_exact(occupancy, [layers, t]),
            "keyNorms": encoding.encode_f32(norms, [layers, t]),
            "bytesResident": run.kv_bytes(),
            # A real cache, so this is measurement rather than reconstruction.
            "simulated": False,
        }

    def _top_k(
        self, sample: SampleResult | None, step: int, limit: int = TOP_K
    ) -> dict[str, Any] | None:
        if sample is None:
            return None
        tokenizer = self.runtime.tokenizer

        ids = sample.top_ids[:limit]
        logits = sample.top_logits[:limit]
        probs = sample.top_probs[:limit]

        # Truncating must never hide the token that was actually emitted, or
        # the panel loses the one row it exists to highlight. Under greedy
        # sampling the chosen token is rank 0 and this never fires; under
        # temperature it can fall outside a short list. Its probability is by
        # construction no greater than anything kept, so appending preserves
        # the descending order the client renders in.
        if sample.token_id not in ids and sample.token_id in sample.top_ids:
            at = sample.top_ids.index(sample.token_id)
            ids = ids + [sample.top_ids[at]]
            logits = logits + [sample.top_logits[at]]
            probs = probs + [sample.top_probs[at]]

        entries = []
        for token_id, logit, prob in zip(ids, logits, probs):
            piece = tokenizer.id_to_token(token_id)
            entries.append(
                {
                    "tokenId": token_id,
                    "text": piece,
                    "display": tokenizer.token(token_id, 0, "generated")["display"],
                    "logit": logit,
                    "prob": prob,
                }
            )
        return {
            "step": step,
            "k": len(entries),
            "entries": entries,
            "entropy": sample.entropy,
            "temperature": sample.temperature,
            "chosenTokenId": sample.token_id,
            "chosenRank": sample.chosen_rank,
        }

    # -- on-demand tensors ------------------------------------------------

    def attention_tile(self, step: int, layer: int, head: int) -> dict[str, Any] | None:
        if self._run is None:
            return None
        t = len(self.prompt_tokens) + step
        packed = self._run.attention.tile(layer, head, t)
        if packed is None:
            return None

        values = torch.frombuffer(bytearray(packed), dtype=torch.uint8).to(torch.float32)
        weights = (values / 255.0) ** 2
        max_weight = float(weights.max()) if weights.numel() else 0.0

        entropies: list[float] = []
        sink = 0.0
        cursor = 0
        for row in range(t):
            width = row + 1
            chunk = weights[cursor : cursor + width]
            cursor += width
            positive = chunk[chunk > 0]
            entropies.append(float(-(positive * positive.log()).sum()) if positive.numel() else 0.0)
            sink += float(chunk[0]) if width else 0.0

        return {
            "step": step,
            "layer": layer,
            "head": head,
            "sequenceLength": t,
            "weights": encoding.encoded_attention(packed, t),
            "stats": {
                "maxWeight": max_weight,
                "meanEntropy": sum(entropies) / len(entropies) if entropies else 0.0,
                "sinkMass": sink / t if t else 0.0,
            },
        }

    def residual(self, step: int, layer: int, stage: str, position: int | None) -> dict[str, Any] | None:
        if self._run is None:
            return None
        bucket = self._run.residuals.get((layer, stage))
        if not bucket:
            return None
        t = len(self.prompt_tokens) + step
        available = min(t, len(bucket))
        if available <= 0:
            return None

        if position is None:
            block = torch.stack(bucket[:available])
            return {
                "step": step,
                "layer": layer,
                "stage": stage,
                "values": encoding.encode_f32(block, [available, block.shape[-1]]),
            }

        if position < 0 or position >= available:
            return None
        values = bucket[position]
        return {
            "step": step,
            "layer": layer,
            "position": position,
            "stage": stage,
            "values": encoding.encode_f32(values),
            "stats": encoding.vector_stats(values),
        }

    def kv_snapshot(self, step: int) -> dict[str, Any] | None:
        if self._run is None:
            return None
        return self._kv_snapshot(len(self.prompt_tokens) + step)

    def logits(self, step: int, k: int) -> dict[str, Any] | None:
        if self._run is None:
            return None
        sample = self._run.samples_by_step.get(step)
        if k <= 0:
            full = self._run.logits_by_step.get(step)
            if full is None:
                return None
            return {"step": step, "k": 0, "values": encoding.encode_f32(full)}
        if sample is None:
            return None
        # The stored list is the ceiling; `k` beyond it simply returns it all.
        return self._top_k(sample, step, limit=k)


def _advance(stages: Iterator[StageOutcome]) -> StageOutcome | None:
    """One stage of execution. Runs in the executor; blocking torch work."""
    try:
        return next(stages)
    except StopIteration:
        return None


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
