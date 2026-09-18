"""The HTTP and WebSocket surface described in docs/api-contract.md.

Two rules from the contract shape this file:

* **Control flow is WebSocket; data retrieval is HTTP.** The socket carries
  commands and events. Tensors come over plain GETs so they are cacheable,
  independently cancellable, and unaffected by socket health.
* **Halt payloads are bounded; unbounded tensors are fetched on demand.** A
  single (layer, head) attention tile is the unit of transfer; the full tensor
  is never sent at any sequence length.
"""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Query, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field

from .breakpoints import ConditionError
from .metering import MeteringMiddleware, enabled_by_env, meter, set_enabled, wire_meter
from .registry import SessionNotFound, registry
from .runtime import get_runtime
from .session import Session, SessionConfig

#: Completed steps never change, so their tensors are safe to cache forever.
IMMUTABLE = "public, max-age=31536000, immutable"


def _origins() -> list[str]:
    configured = os.environ.get("MDEBUG_CORS_ORIGINS")
    if configured:
        return [o.strip() for o in configured.split(",") if o.strip()]
    return [
        "http://localhost:4200",
        "http://127.0.0.1:4200",
        # GitHub Pages is HTTPS-only, so a deployed client can only reach an
        # https/wss origin -- there is no "deployed client, local server".
        "https://priyananda.github.io",
    ]


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()

    # The default executor is sized min(32, cpu + 4), so on a two-core instance
    # several sessions can each run a forward pass at once and thrash. Bounding
    # it serialises inference, which is faster in aggregate than oversubscribing.
    workers = int(os.environ.get("MDEBUG_EXECUTOR_WORKERS", "0"))
    if workers:
        loop.set_default_executor(
            ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mdebug")
        )

    # Load the checkpoint at startup rather than on the first request, so the
    # first session does not pay for it.
    await loop.run_in_executor(None, get_runtime)
    yield
    await registry.shutdown()


app = FastAPI(title="mdebug inference server", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Order matters, and `add_middleware` makes the LAST call the outermost layer.
# The stack ends up: wire meter -> gzip -> payload meter -> CORS -> routes.
#
# The two meters straddle compression on purpose: the inner one reports the
# payload a change to `session.py` would shrink, the outer one reports what the
# client actually downloads. Both are pass-throughs unless enabled.
app.add_middleware(MeteringMiddleware, byte_meter=meter)

# Responses are JSON, and mostly base64 of quantised tensors. Measured on the
# at-cap benchmark profile this takes HTTP from ~178 kB to ~10 kB a session --
# the largest single reduction available, for one line.
#
# The socket is deliberately not covered: uvicorn already negotiates
# permessage-deflate with context takeover there, and GZipMiddleware ignores
# non-HTTP scopes anyway.
app.add_middleware(GZipMiddleware, minimum_size=500)

app.add_middleware(MeteringMiddleware, byte_meter=wire_meter)
set_enabled(enabled_by_env())


def _not_found(session_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={"code": "session_not_found", "message": f"Session {session_id} is not open"},
    )


def _session_or_404(session_id: str) -> Session | JSONResponse:
    try:
        return registry.get(session_id)
    except SessionNotFound:
        return _not_found(session_id)


# --- model -----------------------------------------------------------------


@app.get("/api/model")
async def get_model() -> dict[str, Any]:
    return get_runtime().model_info


@app.get("/api/health")
async def health() -> dict[str, Any]:
    runtime = get_runtime()
    return {
        "status": "ok",
        "checkpointStep": runtime.checkpoint_step,
        "sessions": len(registry.list()),
    }


# --- tokenize --------------------------------------------------------------


class TokenizeRequest(BaseModel):
    text: str = ""


@app.post("/api/tokenize")
async def tokenize(request: TokenizeRequest) -> list[dict[str, Any]]:
    """Session-free, so the prompt box can preview before a run exists."""
    return get_runtime().tokenizer.tokens(request.text, "prompt")


# --- sessions --------------------------------------------------------------


class CreateSessionRequest(BaseModel):
    config: dict[str, Any] = Field(default_factory=dict)


@app.post("/api/sessions")
async def create_session(request: CreateSessionRequest) -> dict[str, Any]:
    session = await registry.create(SessionConfig.from_json(request.config))
    return session.as_json()


@app.get("/api/sessions")
async def list_sessions() -> list[dict[str, Any]]:
    return [s.summary() for s in registry.list()]


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str) -> Any:
    session = _session_or_404(session_id)
    return session if isinstance(session, JSONResponse) else session.as_json()


# No `status_code=204` on the decorator: this route also answers 404 with a
# body, and FastAPI forbids declaring a bodyless status for that.
@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str) -> Any:
    try:
        await registry.delete(session_id)
    except SessionNotFound:
        return _not_found(session_id)
    return Response(status_code=204)


class BreakpointsRequest(BaseModel):
    breakpoints: list[dict[str, Any]] = Field(default_factory=list)


@app.put("/api/sessions/{session_id}/breakpoints")
async def put_breakpoints(session_id: str, request: BreakpointsRequest) -> Any:
    session = _session_or_404(session_id)
    if isinstance(session, JSONResponse):
        return session
    try:
        return [b.as_json() for b in session.set_breakpoints(request.breakpoints)]
    except ConditionError as error:
        # Rejected at set time rather than silently never firing.
        return JSONResponse(status_code=422, content={"code": "invalid_condition", "message": str(error)})


# --- on-demand tensors -----------------------------------------------------


@app.get("/api/sessions/{session_id}/steps/{step}/attention")
async def get_attention(session_id: str, step: int, layer: int = Query(...), head: int = Query(...)) -> Any:
    session = _session_or_404(session_id)
    if isinstance(session, JSONResponse):
        return session
    tile = session.attention_tile(step, layer, head)
    if tile is None:
        return JSONResponse(
            status_code=404,
            content={"code": "not_captured", "message": f"no attention for step {step}, L{layer} head {head}"},
        )
    return JSONResponse(content=tile, headers={"Cache-Control": IMMUTABLE})


@app.get("/api/sessions/{session_id}/steps/{step}/residual")
async def get_residual(
    session_id: str,
    step: int,
    layer: int = Query(...),
    stage: str = Query("post_ffn"),
    position: int | None = Query(None),
) -> Any:
    session = _session_or_404(session_id)
    if isinstance(session, JSONResponse):
        return session
    payload = session.residual(step, layer, stage, position)
    if payload is None:
        return JSONResponse(
            status_code=404,
            content={"code": "not_captured", "message": f"no residual for step {step}, L{layer} {stage}"},
        )
    return JSONResponse(content=payload, headers={"Cache-Control": IMMUTABLE})


@app.get("/api/sessions/{session_id}/steps/{step}/kv")
async def get_kv(session_id: str, step: int) -> Any:
    session = _session_or_404(session_id)
    if isinstance(session, JSONResponse):
        return session
    payload = session.kv_snapshot(step)
    if payload is None:
        return JSONResponse(status_code=404, content={"code": "not_captured", "message": "no run yet"})
    return payload


@app.get("/api/sessions/{session_id}/steps/{step}/logits")
async def get_logits(session_id: str, step: int, k: int = Query(50)) -> Any:
    session = _session_or_404(session_id)
    if isinstance(session, JSONResponse):
        return session
    payload = session.logits(step, k)
    if payload is None:
        return JSONResponse(
            status_code=404,
            content={"code": "not_captured", "message": f"no logits for step {step}"},
        )
    return JSONResponse(content=payload, headers={"Cache-Control": IMMUTABLE})


# --- control channel -------------------------------------------------------


@app.websocket("/api/sessions/{session_id}/ws")
async def session_socket(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    try:
        session = registry.get(session_id)
    except SessionNotFound:
        await websocket.send_json(
            {
                "v": 1,
                "id": "0",
                "ts": 0,
                "type": "error",
                "payload": {
                    "code": "session_not_found",
                    "message": f"Session {session_id} is not open",
                    "fatal": True,
                },
            }
        )
        await websocket.close()
        return

    queue = session.subscribe()
    # Full state first, so a reconnecting client resyncs before anything else.
    session.emit_state()

    async def pump() -> None:
        while True:
            message = await queue.get()
            await websocket.send_json(message)

    pump_task = asyncio.create_task(pump())
    try:
        while True:
            envelope = await websocket.receive_json()
            await _handle_command(session, websocket, envelope)
    except WebSocketDisconnect:
        pass
    except Exception as error:  # pragma: no cover - transport level
        try:
            await websocket.send_json(
                {
                    "v": 1,
                    "id": "0",
                    "ts": 0,
                    "type": "error",
                    "payload": {"code": "socket_error", "message": str(error), "fatal": False},
                }
            )
        except Exception:
            pass
    finally:
        pump_task.cancel()
        session.unsubscribe(queue)


async def _handle_command(session: Session, websocket: WebSocket, envelope: dict[str, Any]) -> None:
    command = envelope.get("type")
    payload = envelope.get("payload") or {}

    if command == "start":
        session.start(payload.get("config"))
    elif command == "continue":
        session.continue_run()
    elif command == "step":
        session.step(int(payload.get("count", 1) or 1))
    elif command == "stop":
        session.stop()
    elif command == "set_breakpoints":
        try:
            session.set_breakpoints(payload.get("breakpoints", []))
        except ConditionError as error:
            await websocket.send_json(
                {
                    "v": 1,
                    "id": "0",
                    "ts": 0,
                    "type": "error",
                    "payload": {"code": "invalid_condition", "message": str(error), "fatal": False},
                }
            )
    elif command == "run_to_cursor":
        session.run_to_cursor(str(payload.get("stageId")), payload.get("step"))
    elif command == "patch_config":
        session.patch_config(payload)
    elif command == "ping":
        await websocket.send_json(
            {"v": 1, "id": envelope.get("id", "0"), "ts": 0, "type": "pong", "payload": {},
             "replyTo": envelope.get("id")}
        )
