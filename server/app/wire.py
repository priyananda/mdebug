"""Small helpers shared by the tests and `bench/session_bytes.py`.

`drain` lives here rather than in `tests/` so the benchmark can drive a socket
the same way the contract tests do, without one importing the other.
"""

from __future__ import annotations

from typing import Any


def envelope(message_id: str, message_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """The client -> server envelope from docs/api-contract.md section 8."""
    return {"v": 1, "id": message_id, "ts": 0, "type": message_type, "payload": payload or {}}


def drain(ws: Any, wanted: str, limit: int = 4000) -> dict[str, Any]:
    """Read events until one of `wanted` arrives."""
    for _ in range(limit):
        message = ws.receive_json()
        if message["type"] == wanted:
            return message
        if message["type"] == "error" and wanted != "error":
            raise AssertionError(f"server error: {message['payload']}")
    raise AssertionError(f"never saw {wanted}")
