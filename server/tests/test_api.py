"""End-to-end checks against the contract the client was built to.

These drive the real FastAPI app with the real checkpoint over a real
WebSocket: create a session, set a breakpoint, run, halt, step, continue, and
pull an attention tile back out. If the client works against the mock and these
pass, it should work against this server unchanged.
"""

from __future__ import annotations

import math
import struct
import sys
from base64 import b64decode
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.main import app  # noqa: E402
from app.wire import drain  # noqa: E402,F401

PROMPT = "The key to happiness is"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def decode_array(encoded: dict) -> list[float]:
    """The client's decode, reimplemented so the test checks the wire format."""
    raw = b64decode(encoded["data"])
    dtype = encoded["dtype"]
    if dtype == "f32":
        values = list(struct.unpack(f"<{len(raw) // 4}f", raw))
    elif dtype == "u8":
        values = list(raw)
    else:  # pragma: no cover
        raise AssertionError(f"unexpected dtype {dtype}")

    values = [v * encoded["scale"] + encoded["offset"] for v in values]
    if encoded["transform"] == "sqrt":
        values = [v * v for v in values]

    if encoded["layout"] == "causal_lower":
        size = encoded["shape"][-1]
        dense = [0.0] * (size * size)
        cursor = 0
        for row in range(size):
            for col in range(row + 1):
                dense[row * size + col] = values[cursor]
                cursor += 1
        return dense
    return values


class TestModelEndpoint:
    def test_reports_the_shape_the_client_renders_from(self, client):
        info = client.get("/api/model").json()

        assert info["numLayers"] == 20
        assert info["numHeads"] == 8
        assert info["hiddenSize"] == 512
        assert info["headDim"] == 64
        assert info["ropeDim"] == 16
        assert info["tokenizerVocabSize"] == 6258
        assert len(info["stages"]) == 8
        assert {s["kind"] for s in info["stages"]} == {
            "tokenize", "embed", "attention", "ffn", "final_norm", "lm_head", "sample", "emit",
        }

    def test_claims_a_real_cache_and_real_attention(self, client):
        """Both are true here, which is why the client may drop the badge."""
        info = client.get("/api/model").json()
        assert info["hasKvCache"] is True
        assert info["capturesAttention"] is True

    def test_surfaces_the_models_naming_discrepancies(self, client):
        notes = " ".join(client.get("/api/model").json()["notes"]).lower()
        assert "multi-head attention" in notes
        assert "rmsnorm" in notes


class TestTokenize:
    def test_matches_the_trained_tokenizer(self, client):
        tokens = client.post("/api/tokenize", json={"text": PROMPT}).json()
        assert [t["id"] for t in tokens] == [333, 1273, 281, 1339, 316]
        assert [t["display"] for t in tokens] == ["The", "·key", "·to", "·happiness", "·is"]
        assert [t["position"] for t in tokens] == [0, 1, 2, 3, 4]


class TestTokenDisplay:
    """`display` is the human-readable field; `text` stays the raw piece.

    Must stay in step with `ByteLevelBpeTokenizer.display` in the client.
    """

    def test_decodes_multi_byte_pieces(self, client):
        tokens = client.post("/api/tokenize", json={"text": '\u201cquiet\u201d \u2014 caf\u00e9'}).json()
        displays = [t["display"] for t in tokens]
        # The byte alphabet renders a curly quote as three characters; the
        # display field must not leak that.
        assert not any("\u00c3" in d or "\u00e2" in d for d in displays), displays
        assert "".join(displays).replace("\u00b7", " ") == '\u201cquiet\u201d \u2014 caf\u00e9'

    def test_keeps_the_raw_piece_in_text(self, client):
        tokens = client.post("/api/tokenize", json={"text": "the key"}).json()
        assert tokens[1]["text"].startswith("\u0120")
        assert tokens[1]["display"].startswith("\u00b7")

    def test_renders_whitespace_visibly(self, client):
        tokens = client.post("/api/tokenize", json={"text": "a\nb"}).json()
        assert any("\u23ce" in t["display"] for t in tokens)


class TestSessionLifecycle:
    def test_create_get_list_delete(self, client):
        created = client.post("/api/sessions", json={"config": {"prompt": PROMPT, "maxNewTokens": 2}}).json()
        session_id = created["id"]

        assert created["status"] == "idle"
        assert len(created["promptTokens"]) == 5

        assert client.get(f"/api/sessions/{session_id}").json()["id"] == session_id
        assert any(s["id"] == session_id for s in client.get("/api/sessions").json())

        assert client.delete(f"/api/sessions/{session_id}").status_code == 204
        missing = client.get(f"/api/sessions/{session_id}")
        assert missing.status_code == 404
        assert missing.json()["code"] == "session_not_found"

    def test_unknown_session_is_a_clean_404(self, client):
        response = client.get("/api/sessions/nope")
        assert response.status_code == 404
        assert response.json()["code"] == "session_not_found"


class TestBreakpointValidation:
    def test_rejects_a_condition_the_stage_cannot_evaluate(self, client):
        session_id = client.post("/api/sessions", json={"config": {"prompt": PROMPT}}).json()["id"]
        response = client.put(
            f"/api/sessions/{session_id}/breakpoints",
            json={"breakpoints": [{
                "id": "b1", "stageId": "L3.ffn", "enabled": True, "oneShot": False, "hitCount": 0,
                "condition": {"kind": "top1_prob", "op": "<", "value": 0.3},
            }]},
        )
        assert response.status_code == 422
        assert response.json()["code"] == "invalid_condition"

    def test_accepts_the_same_condition_where_it_makes_sense(self, client):
        session_id = client.post("/api/sessions", json={"config": {"prompt": PROMPT}}).json()["id"]
        response = client.put(
            f"/api/sessions/{session_id}/breakpoints",
            json={"breakpoints": [{
                "id": "b1", "stageId": "sample", "enabled": True, "oneShot": False, "hitCount": 0,
                "condition": {"kind": "top1_prob", "op": "<", "value": 0.3},
            }]},
        )
        assert response.status_code == 200


class TestRunControl:
    def test_halts_at_a_breakpoint_then_steps_then_continues(self, client):
        session_id = client.post(
            "/api/sessions", json={"config": {"prompt": PROMPT, "maxNewTokens": 3, "samplingMode": "greedy"}}
        ).json()["id"]
        client.put(
            f"/api/sessions/{session_id}/breakpoints",
            json={"breakpoints": [{"id": "b1", "stageId": "L7.attention", "enabled": True,
                                   "oneShot": False, "hitCount": 0}]},
        )

        with client.websocket_connect(f"/api/sessions/{session_id}/ws") as ws:
            state = drain(ws, "session_state")
            assert state["payload"]["id"] == session_id

            ws.send_json({"v": 1, "id": "1", "ts": 0, "type": "start", "payload": {}})
            halted = drain(ws, "halted")["payload"]

            assert halted["position"]["stageId"] == "L7.attention"
            assert halted["position"]["step"] == 0
            assert halted["position"]["reason"] == "breakpoint"
            assert halted["position"]["breakpointId"] == "b1"
            # The first halt carries the prompt; later ones do not repeat it.
            assert len(halted["sequence"]["promptTokens"]) == 5

            # One step advances exactly one stage.
            ws.send_json({"v": 1, "id": "2", "ts": 0, "type": "step", "payload": {"count": 1}})
            stepped = drain(ws, "halted")["payload"]
            assert stepped["position"]["stageId"] == "L7.ffn"
            assert stepped["position"]["reason"] == "step"
            assert "promptTokens" not in stepped["sequence"]

            # Continue runs to the same breakpoint on the next token.
            ws.send_json({"v": 1, "id": "3", "ts": 0, "type": "continue", "payload": {}})
            again = drain(ws, "halted")["payload"]
            assert again["position"]["stageId"] == "L7.attention"
            assert again["position"]["step"] == 1
            assert again["position"]["sequenceLength"] == 6

            ws.send_json({"v": 1, "id": "4", "ts": 0, "type": "stop", "payload": {}})
            assert drain(ws, "finished")["payload"]["reason"] == "stopped"

    def test_step_count_advances_that_many_stages(self, client):
        session_id = client.post(
            "/api/sessions", json={"config": {"prompt": PROMPT, "maxNewTokens": 2, "samplingMode": "greedy"}}
        ).json()["id"]
        client.put(
            f"/api/sessions/{session_id}/breakpoints",
            json={"breakpoints": [{"id": "b1", "stageId": "embed", "enabled": True,
                                   "oneShot": False, "hitCount": 0}]},
        )
        with client.websocket_connect(f"/api/sessions/{session_id}/ws") as ws:
            drain(ws, "session_state")
            ws.send_json({"v": 1, "id": "1", "ts": 0, "type": "start", "payload": {}})
            drain(ws, "halted")

            ws.send_json({"v": 1, "id": "2", "ts": 0, "type": "step", "payload": {"count": 4}})
            # embed -> L0.attention, L0.ffn, L1.attention, L1.ffn
            assert drain(ws, "halted")["payload"]["position"]["stageId"] == "L1.ffn"
            ws.send_json({"v": 1, "id": "3", "ts": 0, "type": "stop", "payload": {}})

    def test_a_condition_skips_steps_until_it_holds(self, client):
        session_id = client.post(
            "/api/sessions", json={"config": {"prompt": PROMPT, "maxNewTokens": 6, "samplingMode": "greedy"}}
        ).json()["id"]
        client.put(
            f"/api/sessions/{session_id}/breakpoints",
            json={"breakpoints": [{
                "id": "b1", "stageId": "sample", "enabled": True, "oneShot": False, "hitCount": 0,
                "condition": {"kind": "token_index", "op": "==", "value": 3},
            }]},
        )
        with client.websocket_connect(f"/api/sessions/{session_id}/ws") as ws:
            drain(ws, "session_state")
            ws.send_json({"v": 1, "id": "1", "ts": 0, "type": "start", "payload": {}})
            halted = drain(ws, "halted")["payload"]
            assert halted["position"]["step"] == 3
            assert halted["position"]["stageId"] == "sample"
            ws.send_json({"v": 1, "id": "2", "ts": 0, "type": "stop", "payload": {}})

    def test_runs_to_completion_without_breakpoints(self, client):
        session_id = client.post(
            "/api/sessions", json={"config": {"prompt": PROMPT, "maxNewTokens": 3, "samplingMode": "greedy"}}
        ).json()["id"]
        with client.websocket_connect(f"/api/sessions/{session_id}/ws") as ws:
            drain(ws, "session_state")
            ws.send_json({"v": 1, "id": "1", "ts": 0, "type": "start", "payload": {}})
            finished = drain(ws, "finished")["payload"]
            assert finished["reason"] == "max_tokens"
            assert finished["totalSteps"] == 3
            assert isinstance(finished["text"], str) and finished["text"]


class TestHaltPayload:
    @pytest.fixture(scope="class")
    def halted(self, client):
        session_id = client.post(
            "/api/sessions", json={"config": {"prompt": PROMPT, "maxNewTokens": 2, "samplingMode": "greedy"}}
        ).json()["id"]
        client.put(
            f"/api/sessions/{session_id}/breakpoints",
            json={"breakpoints": [{"id": "b1", "stageId": "L5.attention", "enabled": True,
                                   "oneShot": False, "hitCount": 0}]},
        )
        with client.websocket_connect(f"/api/sessions/{session_id}/ws") as ws:
            drain(ws, "session_state")
            ws.send_json({"v": 1, "id": "1", "ts": 0, "type": "start", "payload": {}})
            payload = drain(ws, "halted")["payload"]
            ws.send_json({"v": 1, "id": "2", "ts": 0, "type": "stop", "payload": {}})
        return session_id, payload

    # The size budget moved to tests/test_payload_budget.py, which checks it as
    # a function of sequence length rather than at one prompt length where it
    # passes by a factor of forty.

    def test_head_summary_covers_every_layer_and_head(self, halted):
        _, payload = halted
        summary = payload["headSummary"]
        assert summary["shape"] == [20, 8]
        values = decode_array(summary)
        assert len(values) == 160
        # Entropy in nats, bounded by log(T) for the executed layers.
        assert all(0 <= v <= math.log(5) + 1e-5 for v in values)

    def test_residual_matches_the_hidden_size_and_its_stats(self, halted):
        _, payload = halted
        values = decode_array(payload["residual"])
        assert len(values) == 512

        stats = payload["residualStats"]
        expected_l2 = math.sqrt(sum(v * v for v in values))
        assert stats["l2"] == pytest.approx(expected_l2, rel=1e-4)
        assert stats["min"] == pytest.approx(min(values), rel=1e-5)
        assert stats["max"] == pytest.approx(max(values), rel=1e-5)

    def test_kv_grid_reflects_where_execution_has_reached(self, halted):
        """Halted at L5.attention: layers 0-5 hold this step's column, 6-19 do not."""
        _, payload = halted
        kv = payload["kv"]
        assert kv["simulated"] is False
        assert kv["numLayers"] == 20

        t = kv["sequenceLength"]
        occupancy = decode_array(kv["occupancy"])
        newest = t - 1
        for layer in range(20):
            cell = occupancy[layer * t + newest]
            assert cell == (1 if layer <= 5 else 0), f"layer {layer} newest cell = {cell}"

    def test_reports_real_cache_bytes(self, halted):
        _, payload = halted
        # 6 layers executed x 2 tensors x 8 heads x 5 positions x 64 dims x 4 bytes.
        assert payload["kv"]["bytesResident"] == 6 * 2 * 8 * 5 * 64 * 4


class TestAttentionTiles:
    @pytest.fixture(scope="class")
    def ran(self, client):
        session_id = client.post(
            "/api/sessions", json={"config": {"prompt": PROMPT, "maxNewTokens": 2, "samplingMode": "greedy"}}
        ).json()["id"]
        with client.websocket_connect(f"/api/sessions/{session_id}/ws") as ws:
            drain(ws, "session_state")
            ws.send_json({"v": 1, "id": "1", "ts": 0, "type": "start", "payload": {}})
            drain(ws, "finished")
        return session_id

    def test_tile_is_causal_and_its_rows_are_distributions(self, client, ran):
        response = client.get(f"/api/sessions/{ran}/steps/1/attention", params={"layer": 3, "head": 2})
        assert response.status_code == 200
        assert "immutable" in response.headers.get("cache-control", "")

        tile = response.json()
        t = tile["sequenceLength"]
        assert t == 6  # 5 prompt tokens + 1 generated

        assert tile["weights"]["dtype"] == "u8"
        assert tile["weights"]["transform"] == "sqrt"
        assert tile["weights"]["layout"] == "causal_lower"
        assert len(b64decode(tile["weights"]["data"])) == t * (t + 1) // 2

        dense = decode_array(tile["weights"])
        for row in range(t):
            for col in range(row + 1, t):
                assert dense[row * t + col] == 0.0, f"({row},{col}) is in the future"
            total = sum(dense[row * t : row * t + row + 1])
            assert abs(total - 1.0) < 0.02, f"row {row} sums to {total}"

    def test_stats_describe_the_tile(self, client, ran):
        tile = client.get(f"/api/sessions/{ran}/steps/1/attention", params={"layer": 3, "head": 2}).json()
        stats = tile["stats"]
        assert 0 < stats["maxWeight"] <= 1.0
        assert stats["meanEntropy"] >= 0
        assert 0 <= stats["sinkMass"] <= 1.0

    def test_missing_capture_is_a_clean_404(self, client, ran):
        response = client.get(f"/api/sessions/{ran}/steps/99/attention", params={"layer": 0, "head": 0})
        assert response.status_code == 404
        assert response.json()["code"] == "not_captured"


class TestOnDemandTensors:
    @pytest.fixture(scope="class")
    def ran(self, client):
        session_id = client.post(
            "/api/sessions", json={"config": {"prompt": PROMPT, "maxNewTokens": 2, "samplingMode": "greedy"}}
        ).json()["id"]
        with client.websocket_connect(f"/api/sessions/{session_id}/ws") as ws:
            drain(ws, "session_state")
            ws.send_json({"v": 1, "id": "1", "ts": 0, "type": "start", "payload": {}})
            drain(ws, "finished")
        return session_id

    def test_residual_for_one_position(self, client, ran):
        response = client.get(
            f"/api/sessions/{ran}/steps/1/residual",
            params={"layer": 2, "stage": "post_ffn", "position": 0},
        )
        assert response.status_code == 200
        body = response.json()
        assert len(decode_array(body["values"])) == 512
        assert body["stats"]["l2"] > 0

    def test_residual_block_for_all_positions(self, client, ran):
        body = client.get(
            f"/api/sessions/{ran}/steps/1/residual", params={"layer": 2, "stage": "post_ffn"}
        ).json()
        assert body["values"]["shape"] == [6, 512]
        assert len(decode_array(body["values"])) == 6 * 512

    def test_top_k_logits(self, client, ran):
        body = client.get(f"/api/sessions/{ran}/steps/0/logits", params={"k": 5}).json()
        assert body["k"] == 5
        assert len(body["entries"]) == 5
        probs = [e["prob"] for e in body["entries"]]
        assert probs == sorted(probs, reverse=True)

        # The invariant the client actually relies on: it locates the emitted
        # token by id, not by indexing with `chosenRank`. `chosenRank` is the
        # token's rank in the whole distribution, which can exceed the length
        # of a truncated list -- see TestTopKDepth.
        ids = [e["tokenId"] for e in body["entries"]]
        assert body["chosenTokenId"] in ids
        assert body["entries"][0]["tokenId"] == body["chosenTokenId"]  # greedy
        assert body["chosenRank"] == 0

    def test_full_logits_vector(self, client, ran):
        body = client.get(f"/api/sessions/{ran}/steps/0/logits", params={"k": 0}).json()
        assert len(decode_array(body["values"])) == 30000

    def test_kv_snapshot(self, client, ran):
        body = client.get(f"/api/sessions/{ran}/steps/1/kv").json()
        assert body["numLayers"] == 20
        assert body["simulated"] is False
