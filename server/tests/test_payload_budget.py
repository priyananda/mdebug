"""Budgets for what the server puts on the wire.

`test_api.py` checks that the protocol is *correct*. This checks that it is not
getting *more expensive*, which is a separate thing to regress.

It generalises the one size assertion the suite used to carry, which asserted a
halt payload under 25 kB against a five-token prompt -- the payload is
dominated by terms in `numLayers x T`, so at five tokens that assertion passed
by a factor of forty and could never have caught the ceiling being breached.
Here the budget is a function of T, and the at-cap case is exercised directly.

Read `app/metering.py` before changing a number here: the socket ships
permessage-deflate with context takeover, so the raw JSON size and the cost on
the wire are different quantities and a field can be huge in one and free in
the other.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.main import app  # noqa: E402
from app.metering import meter, wire_meter  # noqa: E402
from app.runner import TOP_K_STORED  # noqa: E402
from app.session import TOP_K  # noqa: E402
from app.wire import drain, envelope  # noqa: E402
from bench.session_bytes import profiles, run_profile  # noqa: E402

PROMPT = "The key to happiness is"

#: Measured on the `typical` profile, with headroom for noise. These exist to
#: fail on a regression, not to pin an exact number: see bench/baseline.json
#: for what was actually measured.
#:
#: `RAW` is the payload the application built, below GZipMiddleware; `WIRE` is
#: what leaves the process. Both are budgeted, because shrinking a payload and
#: shrinking the wire are different wins and either can regress alone.
TYPICAL_HTTP_RAW_MAX = 12_000
TYPICAL_HTTP_WIRE_MAX = 7_000
TYPICAL_WS_DEFLATE_MAX = 27_000
TYPICAL_WS_RAW_MAX = 66_000


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def b64_len(raw_bytes: int) -> int:
    return 4 * math.ceil(raw_bytes / 3)


class TestHaltPayloadBudget:
    """The halt payload is what makes stepping feel instant, so it is bounded."""

    def _halt(self, client, prompt: str, stage: str = "L5.attention") -> dict:
        session_id = client.post(
            "/api/sessions",
            json={"config": {"prompt": prompt, "maxNewTokens": 2, "samplingMode": "greedy"}},
        ).json()["id"]
        client.put(
            f"/api/sessions/{session_id}/breakpoints",
            json={"breakpoints": [{"id": "b1", "stageId": stage, "enabled": True,
                                   "oneShot": False, "hitCount": 0}]},
        )
        with client.websocket_connect(f"/api/sessions/{session_id}/ws") as ws:
            drain(ws, "session_state")
            ws.send_json(envelope("1", "start"))
            payload = drain(ws, "halted")["payload"]
            ws.send_json(envelope("2", "stop"))
        client.delete(f"/api/sessions/{session_id}")
        return payload

    def test_short_prompt_halt_stays_small(self, client):
        payload = self._halt(client, PROMPT)
        size = len(json.dumps(payload).encode())
        assert size < 25_000, f"halt payload is {size} bytes"

    def test_the_encoded_arrays_are_exactly_their_arithmetic_size(self, client):
        """If these drift, the wire format changed and every budget below moved.

        Checking the sizes rather than the values is deliberate: `test_api.py`
        already checks the values decode correctly.
        """
        payload = self._halt(client, PROMPT)
        t = payload["position"]["sequenceLength"]
        layers = payload["kv"]["numLayers"]
        heads = client.get("/api/model").json()["numHeads"]
        hidden = client.get("/api/model").json()["hiddenSize"]

        assert len(payload["kv"]["keyNorms"]["data"]) == b64_len(layers * t * 4)
        assert len(payload["kv"]["occupancy"]["data"]) == b64_len(layers * t)
        assert len(payload["headSummary"]["data"]) == b64_len(layers * heads * 4)
        assert len(payload["residual"]["data"]) == b64_len(hidden * 4)

    def test_halt_payload_scales_as_documented(self, client):
        """The payload grows linearly in T, with terms of `numLayers x T`.

        This is the property the 25 kB budget depends on; it holding is why a
        cap on sequence length is a cap on payload size.
        """
        short = self._halt(client, PROMPT)
        long = self._halt(client, "The key to happiness is " * 40)

        t_short = short["position"]["sequenceLength"]
        t_long = long["position"]["sequenceLength"]
        assert t_long > t_short * 4, "the long prompt must actually be longer"

        # keyNorms is the dominant O(numLayers * T) term.
        ratio = len(long["kv"]["keyNorms"]["data"]) / len(short["kv"]["keyNorms"]["data"])
        assert ratio == pytest.approx(t_long / t_short, rel=0.02)

    def test_at_cap_halt_exceeds_the_documented_budget(self, client):
        """A known, deliberate failure of the contract's stated 25 kB budget.

        Asserted rather than ignored so that fixing it is visible: when the
        halt payload is reduced this test fails and gets inverted, which is the
        point. See docs/api-contract.md section 7.
        """
        limits = client.get("/api/model").json()["limits"]
        payload = self._halt(client, "The key to happiness is " * 120)
        assert payload["position"]["sequenceLength"] == limits["maxPromptTokens"]
        size = len(json.dumps(payload).encode())
        assert size > 25_000, (
            f"the at-cap halt payload is now {size} bytes, within the documented "
            "budget -- update docs/api-contract.md section 7 and invert this test"
        )


class TestSessionBudget:
    """Totals for a whole scripted session, the number we are trying to cut.

    Only the `typical` profile runs here; `at-cap` is slow and stays a manual
    `python -m bench.session_bytes --profile at-cap`.
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def measured(client):
        limits = client.get("/api/model").json()["limits"]
        return run_profile(client, profiles(limits)["typical"], meter)

    def test_http_is_within_budget(self, measured):
        payload = measured["meter"]["totals"]["http/out"]
        wire = measured["wire"]["totals"]["http/out"]
        assert payload["raw"] < TYPICAL_HTTP_RAW_MAX, f"HTTP payload is {payload['raw']} bytes"
        assert wire["raw"] < TYPICAL_HTTP_WIRE_MAX, f"HTTP wire is {wire['raw']} bytes"

    def test_gzip_actually_shrinks_the_http_channel(self, measured):
        """The outer meter is above GZipMiddleware, so this is observed.

        Guards against the middleware being dropped, reordered below the outer
        meter, or silently skipping everything because responses fall under
        `minimum_size`.
        """
        payload = measured["meter"]["totals"]["http/out"]["raw"]
        wire = measured["wire"]["totals"]["http/out"]["raw"]
        assert wire < payload * 0.8, f"gzip saved only {payload - wire} of {payload} bytes"

    def test_gzip_leaves_the_socket_alone(self, measured):
        """GZipMiddleware ignores non-HTTP scopes; permessage-deflate covers it.

        If this ever fails the socket is being compressed twice, which costs
        CPU and makes the `deflate` column fiction.
        """
        assert (
            measured["wire"]["totals"]["ws/out"]["raw"]
            == measured["meter"]["totals"]["ws/out"]["raw"]
        )

    def test_socket_is_within_budget(self, measured):
        ws = measured["meter"]["totals"]["ws/out"]
        assert ws["deflate"] < TYPICAL_WS_DEFLATE_MAX, f"socket is {ws['deflate']} deflated bytes"
        assert ws["raw"] < TYPICAL_WS_RAW_MAX, f"socket is {ws['raw']} raw bytes"

    def test_compression_columns_are_ordered(self, measured):
        """deflate-with-takeover <= independent gzip <= raw, on the socket.

        If this inverts, the shared compressor is being reset per message and
        the whole benchmark is measuring the wrong thing.
        """
        ws = measured["meter"]["totals"]["ws/out"]
        assert ws["deflate"] <= ws["gzip"] <= ws["raw"]

    def test_the_run_is_deterministic(self, client, measured):
        """A second run must cost the same, or the baseline is noise.

        Two things legitimately wobble and are excluded rather than papered
        over, because both are clocks rather than payload:

        * `stage_entered` is coalesced to 30/s against the wall clock
          (`session.py:34-35`), so how many survive depends on how fast the
          machine ran the forward passes. Its *volume* is not a property of the
          protocol and must not be baselined.
        * `halted` carries `timings` rounded to three decimals, so the decimal
          width of a float moves the payload by a byte or two.

        Everything else -- every HTTP response, every other event, and the
        number of attention tiles the client would fetch -- is exact.
        """
        limits = client.get("/api/model").json()["limits"]
        again = run_profile(client, profiles(limits)["typical"], meter)

        assert again["attentionTiles"] == measured["attentionTiles"]
        # HTTP carries no timing data at all, so it is exactly reproducible.
        assert again["meter"]["totals"]["http/out"]["raw"] == measured["meter"]["totals"]["http/out"]["raw"]

        def rows(result):
            return {
                r["label"]: r
                for r in result["meter"]["rows"]
                if r["direction"] == "out" and r["channel"] == "ws"
            }

        before, after = rows(measured), rows(again)
        assert set(before) == set(after), "a different set of events was emitted"
        for label in before:
            if label == "stage_entered":
                continue
            assert after[label]["count"] == before[label]["count"], f"{label} count moved"
            if label == "halted":
                # `timings` jitter only; a real regression is orders larger.
                assert abs(after[label]["raw"] - before[label]["raw"]) <= 64, "halted size moved"
            else:
                assert after[label]["raw"] == before[label]["raw"], f"{label} size moved"


class TestMeteringIsInert:
    def test_disabled_meters_record_nothing(self, client):
        """Both middlewares ship enabled=False, and must cost nothing there."""
        for m in (meter, wire_meter):
            assert m.enabled is False
            m.reset()
        client.get("/api/model")
        for m in (meter, wire_meter):
            assert m.rows == {}


class TestCompressionIsInstalled:
    def test_large_responses_are_gzipped(self, client):
        """End to end, through the real header the browser will see."""
        response = client.get("/api/model", headers={"accept-encoding": "gzip"})
        assert response.status_code == 200
        assert response.headers.get("content-encoding") == "gzip"
        # httpx decodes transparently, so the body must still be usable.
        assert response.json()["numLayers"] > 0

    def test_gzip_varies_on_accept_encoding(self, client):
        """Required, or a shared cache can serve gzip to a client without it."""
        response = client.get("/api/model", headers={"accept-encoding": "gzip"})
        assert "accept-encoding" in response.headers.get("vary", "").lower()

    def test_a_client_that_cannot_gzip_still_gets_json(self, client):
        response = client.get("/api/model", headers={"accept-encoding": "identity"})
        assert response.status_code == 200
        assert "content-encoding" not in response.headers
        assert response.json()["numLayers"] > 0


class TestTopKDepth:
    """What rides along on every token, versus what is kept for asking.

    `session.TOP_K` is pushed eagerly in `halted` and `token_emitted`;
    `runner.TOP_K_STORED` is what the server keeps so `GET .../logits?k=` can
    still widen. Confusing the two is how the eager payload silently grows.
    """

    def _run(self, client, **config) -> str:
        base = {"prompt": PROMPT, "maxNewTokens": 3, "samplingMode": "greedy"}
        base.update(config)
        session_id = client.post("/api/sessions", json={"config": base}).json()["id"]
        with client.websocket_connect(f"/api/sessions/{session_id}/ws") as ws:
            drain(ws, "session_state")
            ws.send_json(envelope("1", "start"))
            drain(ws, "finished")
        return session_id

    def test_pushed_payload_carries_only_top_k(self, client):
        """One run, captured as it happens.

        Deliberately not `self._run` then a second `start`: `Session.start`
        no-ops while the previous task is still winding down, so restarting
        right after `finished` is a race.
        """
        session_id = client.post(
            "/api/sessions",
            json={"config": {"prompt": PROMPT, "maxNewTokens": 3, "samplingMode": "greedy"}},
        ).json()["id"]
        with client.websocket_connect(f"/api/sessions/{session_id}/ws") as ws:
            drain(ws, "session_state")
            ws.send_json(envelope("1", "start"))
            emitted = drain(ws, "token_emitted")["payload"]
            ws.send_json(envelope("2", "stop"))
            drain(ws, "finished")
        assert len(emitted["topK"]["entries"]) <= TOP_K
        assert emitted["topK"]["k"] == len(emitted["topK"]["entries"])
        client.delete(f"/api/sessions/{session_id}")

    def test_the_endpoint_can_still_widen_past_what_is_pushed(self, client):
        """Cutting the eager payload must not cap on-demand inspection."""
        session_id = self._run(client)
        for k in (TOP_K, TOP_K * 2, TOP_K_STORED):
            body = client.get(f"/api/sessions/{session_id}/steps/0/logits", params={"k": k}).json()
            assert len(body["entries"]) == k, f"k={k} returned {len(body['entries'])}"
        # Asking beyond the stored depth returns the stored depth, not an error.
        body = client.get(f"/api/sessions/{session_id}/steps/0/logits",
                          params={"k": TOP_K_STORED * 10}).json()
        assert len(body["entries"]) == TOP_K_STORED
        client.delete(f"/api/sessions/{session_id}")

    def test_full_vocabulary_is_still_reachable(self, client):
        session_id = self._run(client)
        body = client.get(f"/api/sessions/{session_id}/steps/0/logits", params={"k": 0}).json()
        vocab = client.get("/api/model").json()["vocabSize"]
        assert body["values"]["shape"] == [vocab]
        client.delete(f"/api/sessions/{session_id}")

    @pytest.mark.parametrize("temperature", [1.0, 2.0, 50.0])
    def test_the_emitted_token_is_always_in_the_list(self, client, temperature):
        """The one row the panel exists to highlight must never be missing.

        At temperature 2 and above the sampled token usually falls outside even
        the stored top-50, so this is the case that actually bites: the server
        keeps it regardless and reports its true rank.
        """
        session_id = self._run(
            client, samplingMode="temperature", temperature=temperature, seed=7, maxNewTokens=8
        )
        for step in range(8):
            response = client.get(
                f"/api/sessions/{session_id}/steps/{step}/logits", params={"k": TOP_K}
            )
            if response.status_code != 200:
                break
            body = response.json()
            ids = [e["tokenId"] for e in body["entries"]]
            assert body["chosenTokenId"] in ids, f"step {step} lost the emitted token"
            probs = [e["prob"] for e in body["entries"]]
            assert probs == sorted(probs, reverse=True), "appending broke the ordering"
        client.delete(f"/api/sessions/{session_id}")
