"""Byte accounting for the client/server wire, used by `bench/session_bytes.py`.

Off unless switched on, and a straight pass-through when off: this is a
measuring instrument, not a feature.

Both channels are compressed, but by different mechanisms with very different
behaviour, so one number would mislead:

* HTTP goes through `GZipMiddleware`. Each response is compressed alone -- there
  is no dictionary shared between requests.
* The socket goes through uvicorn's permessage-deflate, whose default
  `ws_per_message_deflate` is True and which negotiates *context takeover*, so
  one compression dictionary spans every message on the connection. A field
  that is identical at every halt therefore costs its full size once and almost
  nothing after that.

So every row records:

* ``raw``     -- what the application serialised at this point in the stack.
* ``gzip``    -- each message compressed alone, no shared dictionary.
* ``deflate`` -- one streaming deflate stream shared by every message on the
                 channel, flushed per message. RFC 7692 with context takeover,
                 i.e. what the socket really costs.

`TestClient` is in-process and does no WebSocket framing, so the deflate column
has to be synthesised here rather than observed. HTTP needs no such trick: the
outer of the two meters sits above `GZipMiddleware` and sees the real bytes.
"""

from __future__ import annotations

import gzip
import json
import os
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable

#: Channels are metered separately: they have different compressors in reality,
#: so sharing one dictionary between them would invent savings that don't exist.
Channel = str  # 'http' | 'ws'
Direction = str  # 'out' (server -> client) | 'in' (client -> server)


@dataclass
class Row:
    channel: Channel
    direction: Direction
    label: str
    count: int = 0
    raw: int = 0
    gzip: int = 0
    deflate: int = 0

    def as_json(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "direction": self.direction,
            "label": self.label,
            "count": self.count,
            "raw": self.raw,
            "gzip": self.gzip,
            "deflate": self.deflate,
        }


class ByteMeter:
    """Accumulates per-label byte totals across one scripted session.

    The deflate column needs a compressor per (channel, direction) that lives
    for the whole session, because context takeover is precisely the effect of
    *not* resetting it between messages. Messages are fed to it in arrival
    order, which is why `record` must be called from the send/receive path
    rather than reconstructed afterwards.
    """

    def __init__(self) -> None:
        self.enabled = False
        self.rows: dict[tuple[Channel, Direction, str], Row] = {}
        self._streams: dict[tuple[Channel, Direction], Any] = {}
        #: Raw bodies of `halted` frames, kept for the per-field breakdown.
        self.halted_bodies: list[bytes] = []
        #: Ordered log of every message, for debugging a surprising total.
        self.log: list[tuple[Channel, Direction, str, int]] = []

    def reset(self) -> None:
        self.rows.clear()
        self._streams.clear()
        self.halted_bodies.clear()
        self.log.clear()

    def _stream(self, channel: Channel, direction: Direction) -> Any:
        key = (channel, direction)
        if key not in self._streams:
            # wbits=-15 is a raw deflate stream with no zlib header, which is
            # what RFC 7692 specifies.
            self._streams[key] = zlib.compressobj(9, zlib.DEFLATED, -15)
        return self._streams[key]

    def record(self, channel: Channel, direction: Direction, label: str, body: bytes) -> None:
        if not body:
            return
        key = (channel, direction, label)
        row = self.rows.get(key)
        if row is None:
            row = Row(channel, direction, label)
            self.rows[key] = row

        stream = self._stream(channel, direction)
        compressed = stream.compress(body) + stream.flush(zlib.Z_SYNC_FLUSH)

        row.count += 1
        row.raw += len(body)
        row.gzip += len(gzip.compress(body, 6))
        row.deflate += len(compressed)
        self.log.append((channel, direction, label, len(body)))

        if channel == "ws" and direction == "out" and label == "halted":
            self.halted_bodies.append(body)

    # -- reporting --------------------------------------------------------

    def totals(self) -> dict[str, dict[str, int]]:
        """Per channel+direction totals, plus a grand total."""
        out: dict[str, dict[str, int]] = {}
        for row in self.rows.values():
            for key in (f"{row.channel}/{row.direction}", "total"):
                bucket = out.setdefault(key, {"count": 0, "raw": 0, "gzip": 0, "deflate": 0})
                bucket["count"] += row.count
                bucket["raw"] += row.raw
                bucket["gzip"] += row.gzip
                bucket["deflate"] += row.deflate
        return out

    def sorted_rows(self) -> list[Row]:
        return sorted(self.rows.values(), key=lambda r: -r.raw)

    def as_json(self) -> dict[str, Any]:
        return {
            "totals": self.totals(),
            "rows": [r.as_json() for r in self.sorted_rows()],
        }


#: Two meters, because there are two questions and they have different answers.
#:
#: `meter` sits *under* GZipMiddleware and records what the application
#: serialised -- the number you reduce by changing a payload. `wire_meter` sits
#: *above* it and records what actually leaves the process, which on HTTP is now
#: gzipped. GZipMiddleware ignores non-HTTP scopes, so on the socket the two are
#: identical and the real cost is the synthesised `deflate` column instead.
#:
#: The bench imports these directly rather than the server exposing an endpoint,
#: so the measured API surface stays exactly the real API surface.
meter = ByteMeter()
wire_meter = ByteMeter()

METERS = (meter, wire_meter)


def set_enabled(flag: bool) -> None:
    for m in METERS:
        m.enabled = flag


def reset_all() -> None:
    for m in METERS:
        m.reset()


def _http_label(scope: dict[str, Any]) -> str:
    """`METHOD /api/sessions/{session_id}/steps/{step}/attention`.

    The route template rather than the concrete path, so a table row is not
    fragmented by session ids and step numbers.
    """
    route = scope.get("route")
    path = getattr(route, "path", None) or scope.get("path", "?")
    return f"{scope.get('method', '?')} {path}"


def _ws_label(body: bytes) -> str:
    """The envelope's `type`, which is how the protocol names its messages."""
    try:
        parsed = json.loads(body)
    except Exception:
        return "unparsed"
    return str(parsed.get("type", "unknown")) if isinstance(parsed, dict) else "unknown"


def _body_of(message: dict[str, Any]) -> bytes:
    """ASGI carries text frames as `text` and everything else as `body`."""
    if message.get("text") is not None:
        return str(message["text"]).encode()
    return bytes(message.get("body") or b"")


class MeteringMiddleware:
    """Pure ASGI, so one instance covers both `http` and `websocket` scopes.

    `main.py` installs two of these, straddling `GZipMiddleware`, so the
    instance decides what it means: the inner one measures the payload the
    application built, the outer one the bytes that leave the process.
    """

    def __init__(self, app: Callable[..., Any], byte_meter: ByteMeter | None = None) -> None:
        self.app = app
        self.meter = byte_meter if byte_meter is not None else meter

    async def __call__(self, scope: dict[str, Any], receive: Callable, send: Callable) -> None:
        kind = scope.get("type")
        if not self.meter.enabled or kind not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        channel = "http" if kind == "http" else "ws"

        async def metered_send(message: dict[str, Any]) -> None:
            kind_ = message.get("type")
            if kind_ == "http.response.body":
                # The route is only attached to the scope once the router has
                # matched, which has happened by the time a body is sent.
                self.meter.record(channel, "out", _http_label(scope), _body_of(message))
            elif kind_ == "websocket.send":
                body = _body_of(message)
                self.meter.record(channel, "out", _ws_label(body), body)
            await send(message)

        async def metered_receive() -> dict[str, Any]:
            message = await receive()
            kind_ = message.get("type")
            if kind_ == "http.request":
                self.meter.record(channel, "in", _http_label(scope), _body_of(message))
            elif kind_ == "websocket.receive":
                body = _body_of(message)
                self.meter.record(channel, "in", _ws_label(body), body)
            return message

        await self.app(scope, metered_receive, metered_send)


def enabled_by_env() -> bool:
    return os.environ.get("MDEBUG_METER", "").strip().lower() not in ("", "0", "false", "no", "off")
