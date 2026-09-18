"""How many bytes does a debugging session cost?

Drives the real app with the real checkpoint through `TestClient`, running a
scripted scenario -- two breakpoints, some steps, some continues -- and prints
where every byte went.

    .venv/bin/python -m bench.session_bytes                  # both profiles
    .venv/bin/python -m bench.session_bytes --profile typical
    .venv/bin/python -m bench.session_bytes --write-baseline

Two profiles, because payload sizes scale with sequence length and a single
number would be misleading: `typical` is the short prompt the contract tests
use, `at-cap` runs against the documented 128-prompt / 192-total ceiling.

Both channels are compressed, differently, so the report separates the payload
the application built from the bytes that actually leave:

* **HTTP** is measured twice, above and below `GZipMiddleware`, so the "wire"
  figure is observed rather than predicted.
* **The socket** is not covered by that middleware -- uvicorn negotiates
  permessage-deflate with context takeover instead -- and `TestClient` does no
  WebSocket framing, so its wire cost is the synthesised `deflate` column.

Comparing the wrong column ranks the reduction levers wrongly: see the halt
breakdown, where a field can be a large share of the payload and nearly free on
the wire because it is identical at every halt.

The HTTP traffic here is not the protocol's own doing -- it is what the client
chooses to fetch. `client/src/app/core/state/inspection.store.ts` refetches an
attention tile whenever `(step, layer, head)` changes, and the selected layer
follows execution, so stepping across a layer boundary costs a tile. That
behaviour is reproduced below; if the client's fetch policy changes, this must
change with it or the numbers stop meaning anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.main import app  # noqa: E402
from app.metering import ByteMeter, meter, reset_all, set_enabled, wire_meter  # noqa: E402
from app.wire import drain, envelope  # noqa: E402

BASELINE = Path(__file__).resolve().parent / "baseline.json"

#: Mirrors CACHE_LIMIT in client/src/app/core/state/inspection.store.ts.
TILE_CACHE_LIMIT = 40

#: The client's default head; it only changes when the user picks another.
DEFAULT_HEAD = 0

#: Endpoints the client implements but never calls. Reported as a note rather
#: than as traffic -- this data reaches the client inside `halted` instead.
UNUSED_ENDPOINTS = [
    "GET /api/sessions/{session_id}/steps/{step}/residual",
    "GET /api/sessions/{session_id}/steps/{step}/kv",
    "GET /api/sessions/{session_id}/steps/{step}/logits",
    "GET /api/sessions",
]


class Profile:
    def __init__(self, name: str, prompt: str, max_new_tokens: int) -> None:
        self.name = name
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens


def profiles(limits: dict[str, int]) -> dict[str, Profile]:
    # A prompt long enough to hit maxPromptTokens. The server truncates to the
    # limit, so overshooting is safe and keeps this independent of the
    # tokenizer's exact word-to-token ratio.
    long_prompt = "The key to happiness is " * 120
    return {
        "typical": Profile("typical", "The key to happiness is", 4),
        "at-cap": Profile("at-cap", long_prompt, limits["maxTotalTokens"] - limits["maxPromptTokens"]),
    }


class TileCache:
    """The client's LRU over `(step, layer, head)`, so we fetch when it would."""

    def __init__(self, limit: int = TILE_CACHE_LIMIT) -> None:
        self.limit = limit
        self.keys: list[tuple[int, int, int]] = []

    def should_fetch(self, key: tuple[int, int, int]) -> bool:
        if key in self.keys:
            return False
        self.keys.append(key)
        if len(self.keys) > self.limit:
            self.keys.pop(0)
        return True


def run_profile(client: TestClient, profile: Profile, byte_meter: ByteMeter = meter) -> dict[str, Any]:
    """`byte_meter` is the payload-layer meter; `wire_meter` is read alongside."""
    reset_all()
    set_enabled(True)
    try:
        return _script(client, profile, byte_meter)
    finally:
        set_enabled(False)


def _script(client: TestClient, profile: Profile, byte_meter: ByteMeter) -> dict[str, Any]:
    """One session, start to finish, as the client would drive it."""
    model = client.get("/api/model").json()
    limits = model["limits"]

    # The prompt box tokenizes as you type, debounced at 150 ms
    # (prompt-box.component.ts). One call here; a real session issues several,
    # which is called out in the report rather than guessed at.
    prompt_tokens = len(client.post("/api/tokenize", json={"text": profile.prompt}).json())

    session_id = client.post(
        "/api/sessions",
        json={
            "config": {
                "prompt": profile.prompt,
                "maxNewTokens": profile.max_new_tokens,
                "samplingMode": "greedy",
            }
        },
    ).json()["id"]

    client.put(
        f"/api/sessions/{session_id}/breakpoints",
        json={
            "breakpoints": [
                {"id": "b1", "stageId": "L5.attention", "enabled": True, "oneShot": False, "hitCount": 0},
                {"id": "b2", "stageId": "emit", "enabled": True, "oneShot": False, "hitCount": 0},
            ]
        },
    )

    cache = TileCache()
    selected_layer = 0
    halts = 0
    tiles = 0

    def after_halt(payload: dict[str, Any]) -> None:
        """Reproduce the client's reaction to a halt: maybe fetch a tile."""
        nonlocal selected_layer, halts, tiles
        halts += 1
        stage = payload["position"]["stage"]
        if stage.get("layer") is not None:
            selected_layer = stage["layer"]
        step = payload["position"]["step"]
        t = payload["position"]["sequenceLength"]
        if t == 0 or t > limits["attentionWarnThreshold"]:
            return
        key = (step, selected_layer, DEFAULT_HEAD)
        if cache.should_fetch(key):
            response = client.get(
                f"/api/sessions/{session_id}/steps/{step}/attention",
                params={"layer": selected_layer, "head": DEFAULT_HEAD},
            )
            if response.status_code == 200:
                tiles += 1

    with client.websocket_connect(f"/api/sessions/{session_id}/ws") as ws:
        drain(ws, "session_state")

        ws.send_json(envelope("1", "start"))
        after_halt(drain(ws, "halted")["payload"])

        # Four single steps: L5.ffn, L6.attention, L6.ffn, L7.attention. Two of
        # those cross a layer boundary and so cost a tile.
        for n in range(4):
            ws.send_json(envelope(f"s{n}", "step", {"count": 1}))
            after_halt(drain(ws, "halted")["payload"])

        # Continue to `emit` on this step, then on to the next token's L5.
        ws.send_json(envelope("c1", "continue"))
        after_halt(drain(ws, "halted")["payload"])
        ws.send_json(envelope("c2", "continue"))
        after_halt(drain(ws, "halted")["payload"])

        for n in range(2):
            ws.send_json(envelope(f"t{n}", "step", {"count": 1}))
            after_halt(drain(ws, "halted")["payload"])

        # Clear the breakpoints and run to completion, so the profile includes
        # a free-running stretch and its stage_entered traffic.
        ws.send_json(envelope("bp", "set_breakpoints", {"breakpoints": []}))
        drain(ws, "breakpoints_changed")
        ws.send_json(envelope("c3", "continue"))
        drain(ws, "finished")

    client.delete(f"/api/sessions/{session_id}")

    return {
        "profile": profile.name,
        # The tokenizer's count, not the session's: the server truncates the
        # prompt to maxPromptTokens, which the at-cap profile relies on.
        "promptTokens": min(prompt_tokens, limits["maxPromptTokens"]),
        "maxNewTokens": profile.max_new_tokens,
        "halts": halts,
        "attentionTiles": tiles,
        "meter": byte_meter.as_json(),
        "wire": wire_meter.as_json(),
        "haltBreakdown": halt_breakdown(byte_meter),
    }


# -- the actionable part ----------------------------------------------------


def halt_breakdown(byte_meter: ByteMeter) -> dict[str, Any]:
    """Where the bytes inside `halted` actually go.

    Two numbers per field, because they rank differently. `raw` is its share of
    the serialised JSON. `marginalDeflate` is what the whole run of `halted`
    messages costs on a shared deflate stream minus what it costs with the
    field removed -- i.e. what dropping the field would actually save on the
    socket. A field that is large but identical at every halt has a big raw
    share and an almost-zero marginal cost.
    """
    bodies = byte_meter.halted_bodies
    if not bodies:
        return {}

    fields = [
        ("kv.keyNorms", lambda p: p.get("kv", {}).pop("keyNorms", None)),
        ("kv.occupancy", lambda p: p.get("kv", {}).pop("occupancy", None)),
        ("headSummary", lambda p: p.pop("headSummary", None)),
        ("residual", lambda p: p.pop("residual", None)),
        ("residualStats", lambda p: p.pop("residualStats", None)),
        ("topK", lambda p: p.pop("topK", None)),
        ("sequence", lambda p: p.pop("sequence", None)),
        ("timings", lambda p: p.pop("timings", None)),
        ("position", lambda p: p.pop("position", None)),
    ]

    baseline_deflate = _stream_size(bodies)
    total_raw = sum(len(b) for b in bodies)

    rows = []
    for name, strip in fields:
        raw_share = 0
        stripped: list[bytes] = []
        for body in bodies:
            message = json.loads(body)
            payload = message.get("payload", {})
            removed = strip(payload)
            if removed is not None:
                # The +len(name)+4 approximates the key and JSON punctuation.
                raw_share += len(json.dumps(removed, separators=(",", ":"))) + len(name) + 4
            stripped.append(json.dumps(message, separators=(",", ":")).encode())
        rows.append(
            {
                "field": name,
                "raw": raw_share,
                "marginalDeflate": max(0, baseline_deflate - _stream_size(stripped)),
            }
        )

    accounted = sum(r["raw"] for r in rows)
    rows.append({"field": "envelope + other", "raw": max(0, total_raw - accounted), "marginalDeflate": 0})
    rows.sort(key=lambda r: -r["raw"])
    return {
        "messages": len(bodies),
        "totalRaw": total_raw,
        "totalDeflate": baseline_deflate,
        "largestRaw": max(len(b) for b in bodies),
        "fields": rows,
    }


def _stream_size(bodies: list[bytes]) -> int:
    """Total cost of these messages on one shared deflate stream."""
    import zlib

    stream = zlib.compressobj(9, zlib.DEFLATED, -15)
    return sum(len(stream.compress(b) + stream.flush(zlib.Z_SYNC_FLUSH)) for b in bodies)


# -- reporting --------------------------------------------------------------


#: How each channel is actually compressed, which is why the columns differ.
_MECHANISM = {
    "http/out": "(gzip)",
    "http/in": "(uncompressed)",
    "ws/out": "(deflate+takeover)",
    "ws/in": "(deflate+takeover)",
}


def _wire_bytes(key: str, payload_totals: dict[str, int], http_wire: dict[str, int]) -> int:
    """What this channel actually costs.

    HTTP is observed above `GZipMiddleware`. The socket is not covered by that
    middleware, so its wire cost is the synthesised context-takeover column.
    """
    if key == "http/out":
        return http_wire.get("raw", payload_totals["raw"])
    if key == "http/in":
        # Request bodies are not compressed: the client does not gzip uploads.
        return payload_totals["raw"]
    return payload_totals["deflate"]


def human(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} kB"
    return f"{n / (1024 * 1024):.2f} MB"


def report(result: dict[str, Any]) -> None:
    m = result["meter"]
    totals = m["totals"]
    http = totals.get("http/out", {})
    ws_out = totals.get("ws/out", {})
    # Measured above GZipMiddleware, so this is what the client downloads.
    http_wire = result.get("wire", {}).get("totals", {}).get("http/out", {})

    print()
    print("=" * 84)
    print(
        f"  profile {result['profile']}  |  {result['promptTokens']} prompt tokens, "
        f"maxNewTokens={result['maxNewTokens']}  |  {result['halts']} halts, "
        f"{result['attentionTiles']} tiles"
    )
    print("=" * 84)

    print(f"\n{'channel':<26}{'msgs':>6}{'payload':>12}{'on the wire':>14}{'saved':>8}")
    print("-" * 84)
    for key in ("http/out", "http/in", "ws/out", "ws/in"):
        if key not in totals:
            continue
        t = totals[key]
        wire = _wire_bytes(key, t, http_wire)
        saved = f"{100.0 * (1 - wire / t['raw']):.0f}%" if t["raw"] else "-"
        print(
            f"{key + '  ' + _MECHANISM[key]:<26}{t['count']:>6}{human(t['raw']):>12}"
            f"{human(wire):>14}{saved:>8}"
        )

    http_out_wire = http_wire.get("raw", http.get("raw", 0))
    ws_out_wire = ws_out.get("deflate", 0)
    print("-" * 84)
    print(f"  server -> client, as deployed:  {human(http_out_wire + ws_out_wire)}"
          f"   (payload was {human(http.get('raw', 0) + ws_out.get('raw', 0))})")
    print(f"    HTTP, measured above GZipMiddleware            {human(http_out_wire):>12}")
    print(f"    WebSocket, permessage-deflate w/ takeover      {human(ws_out_wire):>12}")

    wire_rows = {
        (r["channel"], r["direction"], r["label"]): r
        for r in result.get("wire", {}).get("rows", [])
    }
    print(f"\n{'message / endpoint':<52}{'n':>5}{'payload':>13}{'wire':>13}")
    print("-" * 84)
    for row in m["rows"]:
        direction = "<-" if row["direction"] == "out" else "->"
        label = f"{direction} {row['label']}"
        if row["channel"] == "http":
            key = (row["channel"], row["direction"], row["label"])
            wire = wire_rows.get(key, {}).get("raw", row["raw"])
        else:
            wire = row["deflate"]
        print(f"{label[:51]:<52}{row['count']:>5}{human(row['raw']):>13}{human(wire):>13}")

    hb = result.get("haltBreakdown") or {}
    if hb:
        print(f"\ninside the {hb['messages']} `halted` payloads   "
              f"(largest single payload {human(hb['largestRaw'])})")
        print(f"{'field':<32}{'raw':>14}{'share':>9}{'marginal deflate':>20}")
        print("-" * 84)
        for row in hb["fields"]:
            share = 100.0 * row["raw"] / hb["totalRaw"] if hb["totalRaw"] else 0
            print(
                f"{row['field']:<32}{human(row['raw']):>14}{share:>8.1f}%"
                f"{human(row['marginalDeflate']):>20}"
            )
        print("-" * 84)
        print(f"{'total':<32}{human(hb['totalRaw']):>14}{'':>9}{human(hb['totalDeflate']):>20}")

    print("\nnot reproducible run to run, and excluded from the baseline:")
    print("  - `stage_entered` is coalesced to 30/s against the wall clock, so how")
    print("    many survive depends on machine speed. Its volume is not a protocol")
    print("    property; its per-message size is.")
    print("  - `halted` carries `timings` rounded to 3 decimals, which moves the")
    print("    payload by a byte or two.")

    print("\nimplemented on the client but never called, so not counted here:")
    for endpoint in UNUSED_ENDPOINTS:
        print(f"  - {endpoint}")
    print("`/api/tokenize` is counted once; the prompt box debounces one call "
          "per 150 ms of typing.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["typical", "at-cap", "all"], default="all")
    parser.add_argument("--write-baseline", action="store_true", help=f"write {BASELINE.name}")
    parser.add_argument("--json", action="store_true", help="print JSON instead of the tables")
    args = parser.parse_args()

    results: dict[str, Any] = {}
    with TestClient(app) as client:
        limits = client.get("/api/model").json()["limits"]
        available = profiles(limits)
        wanted = list(available) if args.profile == "all" else [args.profile]
        for name in wanted:
            results[name] = run_profile(client, available[name], meter)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for name in results:
            report(results[name])

    if args.write_baseline:
        BASELINE.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {BASELINE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
