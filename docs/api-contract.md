# mdebug — client/server API contract

**Status:** v1 draft. The client implements this in full against a mock engine
(`client/src/app/core/api/mock/`). The Python server does not exist yet; this
document is what it should be built against.

The normative TypeScript lives in `client/src/app/core/models/`. Where this
document and that code disagree, the code wins — but they should not disagree,
and a PR changing one should change the other.

---

## 1. Design stance

Three rules, and everything else follows from them.

**1. Halt payloads are bounded by a constant. Unbounded tensors are fetched on
demand.** A halt must feel instantaneous, so the `halted` event carries a fixed
budget of state (~25 kB) and nothing that grows with sequence length beyond
`O(numLayers · T)`.

**2. Numeric arrays never travel as JSON arrays of numbers.** They travel as
base64 typed arrays with an explicit dtype/shape/transform header (§6).

**3. Control flow is WebSocket; data retrieval is HTTP.** The socket carries
commands and events. Tensors come over plain `GET` requests, so they are
cacheable, independently cancellable, and unaffected by socket health.

### Why rule 1 is not negotiable

Attention for this model, at various sequence lengths:

| shape | f32 | u8 | u8 + causal_lower |
|---|---|---|---|
| 20×8×1024×1024 (whole run at `seq_len`) | 671 MB | 168 MB | 84 MB |
| 20×8×192×192 (recommended cap) | 23.6 MB | 5.9 MB | 3.0 MB |
| 1×1×192×192 (one tile) | 147 kB | 37 kB | **18 kB** |
| 20×8 head-summary scalars | 640 B | — | — |

A single `(layer, head)` tile is the unit of transfer. The full tensor is never
sent, at any `T`. By contrast the KV occupancy grid (20×192 u8 ≈ 4 kB) and one
residual vector (512 f32 = 2 kB) are free, so they ship eagerly.

---

## 2. Model description

`GET /api/model` → `ModelInfo`. Fetched once at startup.

**The client renders the pipeline graph, the layer and head pickers, and every
label from this response.** It hardcodes no model constants. If the server's
config changes, the UI follows; and the UI never asserts something about the
model that the server did not say.

```ts
interface ModelInfo {
  name: string;
  numLayers: number;            // 20
  numHeads: number;             // 8
  hiddenSize: number;           // 512
  headDim: number;              // 64
  intermediateSize: number;     // 1536
  vocabSize: number;            // 30000 — the output projection
  tokenizerVocabSize: number;   // 6258  — what the tokenizer actually has
  maxPositionEmbeddings: number;// 4096
  ropePct: number;              // 0.25
  ropeDim: number;              // 16
  hasKvCache: boolean;
  capturesAttention: boolean;
  limits: ModelLimits;
  stages: StageDescriptor[];
  notes?: string[];             // caveats to surface in the UI
}

interface ModelLimits {
  maxPromptTokens: number;        // recommended 128
  maxNewTokens: number;           // recommended 64
  maxTotalTokens: number;         // recommended 192
  attentionWarnThreshold: number; // recommended 512
}
```

```python
class ModelLimits(BaseModel):
    maxPromptTokens: int = 128
    maxNewTokens: int = 64
    maxTotalTokens: int = 192
    attentionWarnThreshold: int = 512

class ModelInfo(BaseModel):
    name: str
    numLayers: int
    numHeads: int
    hiddenSize: int
    headDim: int
    intermediateSize: int
    vocabSize: int
    tokenizerVocabSize: int
    maxPositionEmbeddings: int
    ropePct: float
    ropeDim: int
    hasKvCache: bool
    capturesAttention: bool
    limits: ModelLimits
    stages: list[StageDescriptor]
    notes: list[str] | None = None
```

### `notes` is for honesty, and it is expected to be non-empty

The checked-in model has several discrepancies between what things are called
and what they do. The client surfaces `notes` verbatim. At minimum:

- `GroupedQueryAttention` is plain multi-head attention — there is no KV-head
  grouping.
- `RMSNorm` divides by the L2 norm, not by `sqrt(mean(x²))` — off by a constant
  factor of `sqrt(512) ≈ 22.6`, absorbed by the learned scale.
- The model adds **learned absolute position embeddings and RoPE**.
- `vocab_size` is 30000 but the trained tokenizer has 6258 entries, so roughly
  24k logits are unreachable.

---

## 3. Pipeline and halt positions

```ts
type StageKind = 'tokenize' | 'embed' | 'attention' | 'ffn'
               | 'final_norm' | 'lm_head' | 'sample' | 'emit';

interface StageRef { kind: StageKind; layer?: number }  // layer iff per-layer

type StageId = string;  // 'tokenize' | 'embed' | 'L7.attention' | 'L7.ffn' | …
```

`StageId` is the canonical string form and is used as a map key everywhere —
breakpoints, graph geometry, the program counter. Format: `kind` for global
stages, `L{layer}.{kind}` for per-layer ones. No other spelling is valid.

### Stage order within one decode step

```
tokenize (step 0 only)
embed
L0.attention → L0.ffn → L1.attention → … → L19.ffn
final_norm → lm_head → sample → emit
```

That is `2·numLayers + 6` halt points on step 0 and `2·numLayers + 5`
thereafter — **46 and 45** for this model.

`attention` covers `norm1 → MHA → residual add`; `ffn` covers
`norm2 → SiLU FFN → residual add`. The halt happens *after* the stage's work,
so the state reported at `L7.attention` is the residual stream with layer 7's
attention output already added.

### Heads are not halt points

All 8 heads are computed in one batched matmul, so there is no instant at which
head 3 is running and head 4 is not. A head-level breakpoint would be fiction.
Heads are addressable for *inspection* (`GET …/attention?head=`) and inside
conditions (`attention_max` takes an optional `head`), but never as a position.

```ts
interface HaltPosition {
  step: number;            // decode step; 0 = generating the first new token
  stage: StageRef;
  stageId: StageId;
  sequenceLength: number;  // T at this instant
  reason: 'breakpoint' | 'step' | 'start' | 'run_to_cursor' | 'finished';
  breakpointId?: string;   // set iff reason === 'breakpoint'
}
```

---

## 4. Sessions

```ts
type SessionStatus = 'idle' | 'running' | 'halted' | 'finished' | 'error' | 'closed';

interface SessionConfig {
  prompt: string;
  maxNewTokens: number;
  samplingMode: 'greedy' | 'temperature' | 'top_k';
  temperature?: number;
  topK?: number;
  seed?: number;
  captureAttention: boolean;   // false → skip storing attention entirely
}

interface Token {
  id: number;
  text: string;      // raw piece, e.g. 'Ġhappiness'
  display: string;   // Ġ→·, newline→⏎, tab→⇥
  position: number;  // absolute sequence position
  isSpecial: boolean;
  origin: 'prompt' | 'generated';
}
```

`SessionInfo` is the full rehydration payload: config, `ModelInfo`, prompt and
generated tokens, `currentStep`, `halt`, and the breakpoint list.

### HTTP surface

All under `/api`. JSON in, JSON out. CORS must allow the GitHub Pages origin.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/model` | `ModelInfo` |
| `POST` | `/api/sessions` | `{config}` → `SessionInfo` |
| `GET` | `/api/sessions` | `SessionSummary[]` |
| `GET` | `/api/sessions/{id}` | `SessionInfo` — full rehydration |
| `DELETE` | `/api/sessions/{id}` | close, free tensors |
| `POST` | `/api/tokenize` | `{text}` → `Token[]`; session-free, for the live prompt preview |
| `PUT` | `/api/sessions/{id}/breakpoints` | replace the set → `Breakpoint[]` |
| `GET` | `/api/sessions/{id}/steps/{step}/attention?layer=&head=` | `AttentionTile` |
| `GET` | `/api/sessions/{id}/steps/{step}/residual?layer=&stage=[&position=]` | `ResidualVector`, or `ResidualBlock` when `position` is omitted |
| `GET` | `/api/sessions/{id}/steps/{step}/kv` | `KvSnapshot` |
| `GET` | `/api/sessions/{id}/steps/{step}/logits?k=` | `TopKLogits`; `k=0` → full f32 `EncodedArray` |

Tensor `GET`s must be safe to call concurrently and cheap to abort — the client
fires them on hover and selection changes and cancels in-flight ones. A
completed step's tensors never change, so they should carry
`Cache-Control: public, max-age=31536000, immutable`.

A request for a session the server no longer has must return **404** with
`{"code": "session_not_found"}`. The client has a specific recovery path for
this (offer a fresh session) and must not be left retrying.

---

## 5. Breakpoints

```ts
interface Breakpoint {
  id: string;
  stageId: StageId;
  enabled: boolean;
  condition?: BreakpointCondition;
  oneShot: boolean;   // "run to here": server removes it after it fires
  hitCount: number;   // server-maintained
}

type ComparisonOp = '==' | '!=' | '<' | '<=' | '>' | '>=';

type BreakpointCondition =
  | { kind: 'token_index';        op: ComparisonOp; value: number }
  | { kind: 'sequence_length';    op: ComparisonOp; value: number }
  | { kind: 'hit_count';          op: ComparisonOp; value: number }
  | { kind: 'top1_prob';          op: ComparisonOp; value: number }
  | { kind: 'logit_entropy';      op: ComparisonOp; value: number }
  | { kind: 'residual_norm';      op: ComparisonOp; value: number }
  | { kind: 'attention_max'; head?: number; op: ComparisonOp; value: number }
  | { kind: 'emitted_token_text'; op: 'equals' | 'contains'; value: string }
  | { kind: 'emitted_token_id';   op: '=='; value: number };
```

A closed union, deliberately — not an expression string. The server implements
it as dict dispatch in about twenty lines, with no parser and no `eval`; the
client builds a real form instead of a text box; and there is no sandbox-escape
surface. An `{kind: 'expr'}` escape hatch is intentionally absent from v1.

### Condition compatibility

A condition that references a value the stage does not have must be **rejected
at set time** with a validation error, not silently never fire.

| Condition | Valid at |
|---|---|
| `token_index`, `sequence_length`, `hit_count` | every stage |
| `residual_norm` | `embed`, `attention`, `ffn`, `final_norm` |
| `attention_max` | `attention` |
| `top1_prob`, `logit_entropy` | `lm_head`, `sample`, `emit` |
| `emitted_token_text`, `emitted_token_id` | `emit` |

Semantics: `residual_norm` is the L2 norm of the hidden state at the last
position after the stage; `attention_max` is the maximum weight over the whole
matrix (or over `head` when given); `top1_prob` is the post-temperature
probability of the argmax token.

### Ownership

Breakpoints are **server-side truth** — conditions need server-side values. The
client keeps an optimistic local copy so gutter clicks feel instant, sends
`set_breakpoints`, and reconciles against `session_state`.

---

## 6. Numeric array encoding (normative)

```ts
interface EncodedArray {
  dtype: 'f32' | 'u8' | 'u16' | 'i32';
  shape: number[];                    // ALWAYS the dense shape
  layout: 'dense' | 'causal_lower';
  transform: 'linear' | 'sqrt';
  scale: number;
  offset: number;
  encoding: 'base64';
  data: string;                       // little-endian, tightly packed, row-major
}
```

**Decode:** base64 → bytes → read as `dtype` (little-endian) →
`v = raw * scale + offset` → if `transform === 'sqrt'`, `v = v * v` → if
`layout === 'causal_lower'`, scatter row `i`'s `i + 1` values into the first
`i + 1` columns of a dense row and leave the rest zero.

**Encode** is the exact inverse. Reference implementation, with tests:
`client/src/app/core/util/encoding.ts`.

### Attention must be `u8` + `sqrt` + `causal_lower`

This is a requirement, not an optimization.

Attention distributions are extremely peaked. Linear `u8` quantization has a
step of `1/255 ≈ 0.0039`, so every weight below that — the broad background
attention, the induction tails, everything except the few bright cells —
quantizes to zero. That is precisely the structure the heatmap exists to show.

Storing `round(255 · sqrt(p))` and squaring on decode gives roughly 16× better
resolution near zero at identical byte cost. `causal_lower` then halves the
payload for free, since the upper triangle is structurally zero.

Residuals and logits stay `f32` / `linear` / `dense`: they are small and
sign-bearing.

```python
def encode_attention(w: torch.Tensor) -> dict:   # w: [T, T], rows sum to 1
    T = w.shape[-1]
    tril = w[torch.tril_indices(T, T).unbind()]          # packed, row-major
    q = (tril.clamp_min(0).sqrt() * 255).round().to(torch.uint8)
    return {
        "dtype": "u8", "shape": [T, T], "layout": "causal_lower",
        "transform": "sqrt", "scale": 1 / 255, "offset": 0.0,
        "encoding": "base64",
        "data": base64.b64encode(q.numpy().tobytes()).decode(),
    }
```

---

## 7. Halt payload

Sent eagerly with every `halted` event. **Budget: under 25 kB.**

```ts
interface HaltPayload {
  position: HaltPosition;
  sequence: {
    promptTokens?: Token[];    // first halt only; the client caches
    generatedTokens: Token[];  // incremental — only what hasn't been sent
  };
  headSummary?: EncodedArray;  // f32 [numLayers, numHeads], see below
  residual?: EncodedArray;     // f32 [hiddenSize], current layer, last position
  residualStats?: VectorStats; // l2, mean, std, min, max
  kv?: KvSnapshot;
  topK?: TopKLogits;           // only at lm_head | sample | emit
  timings?: { stageMs: number; stepMs: number };
}
```

`headSummary` is the entropy, in nats, of the **current query row** of each
head's attention — that is, the row for the token being generated. Not the mean
over all rows: the current row is the one that matters at a halt, and it costs
`O(numLayers · numHeads · T)` instead of `O(numLayers · numHeads · T²)`.

It is what colours the per-head density marks in the pipeline graph, turning the
layer ladder into a live 20×8 view of head behaviour. At 640 bytes, send it on
every halt.

### KV snapshot, and the honesty requirement

```ts
interface KvSnapshot {
  numLayers: number;
  sequenceLength: number;
  occupancy: EncodedArray;   // u8 [numLayers, T]: 0 empty, 1 this step, 2 earlier
  keyNorms?: EncodedArray;   // f32 [numLayers, T] of ||k||, if available
  bytesResident: number;
  simulated: boolean;
}
```

**The checked-in model has no KV cache.** `server/infer.py` re-runs the full
forward pass over the entire prefix for every token, so there is nothing to
measure. Two acceptable responses:

1. **Add a real cache** — the right answer. It is a contained change to
   `GroupedQueryAttention.forward` and `QwenModel.forward`, and it speeds
   generation up by roughly a factor of `T`. Then `hasKvCache: true` and
   `simulated: false`.
2. Report `hasKvCache: false` and `simulated: true`. The client renders a
   "simulated" badge explaining that this model recomputes the prefix each step.

What is **not** acceptable is `simulated: false` on reconstructed data. For this
audience, presenting fiction as measurement is the fastest possible way to lose
trust in everything else on screen.

---

## 8. WebSocket protocol

One socket per session: `wss://{host}/api/sessions/{id}/ws`.

```ts
interface WsEnvelope<TType extends string, TPayload> {
  v: 1;
  id: string;        // unique per message
  ts: number;        // epoch ms at the sender
  type: TType;
  payload: TPayload;
  replyTo?: string;  // echoes a command's id
}
```

### Client → server

| `type` | payload | meaning |
|---|---|---|
| `start` | `{config?}` | begin a run from `idle` |
| `continue` | `{}` | from `halted`: run to the next breakpoint hit or completion |
| `step` | `{count?}` | from `halted`: advance exactly `count` (default 1) halt points |
| `stop` | `{}` | abort; session → `idle`, captured state stays inspectable |
| `set_breakpoints` | `{breakpoints}` | full replace; legal mid-run |
| `run_to_cursor` | `{stageId, step?}` | install a one-shot breakpoint, then continue |
| `patch_config` | `Partial<SessionConfig>` | live parameter change; legal while halted |
| `ping` | `{}` | heartbeat, 20 s |

### Server → client

| `type` | payload |
|---|---|
| `session_state` | `SessionInfo` — sent immediately on connect and on any resync |
| `run_started` | `{step}` |
| `stage_entered` | `{step, stageId, sequenceLength}` |
| `token_emitted` | `{step, token, topK?}` |
| `halted` | `HaltPayload` |
| `resumed` | `{from: HaltPosition}` |
| `finished` | `{reason: 'max_tokens'\|'eos'\|'stopped', totalSteps, text}` |
| `breakpoints_changed` | `{breakpoints}` |
| `error` | `{code, message, fatal, detail?}` |
| `pong` | `{}` |

### `stage_entered` throttling — a required guarantee

At ~45 stages per token and 64 tokens, a full run is ~2,900 of these. Each is
tiny, but they are pure animation data.

**The server must coalesce `stage_entered` to at most 30 per second**, dropping
intermediate stages, subject to two guarantees:

1. A stage that causes a halt is **never** dropped.
2. The last stage before an idle gap is **always** delivered.

The client uses these only to move the program counter, so dropped frames are
invisible. Implemented naively (one event per stage) the socket becomes the
bottleneck over a real network.

### Reconnection

On socket drop the client reconnects with exponential backoff, receives
`session_state`, and re-fetches whatever tensors the current view needs. No
state is lost, because the run lives on the server. If the session is gone, the
server returns `session_not_found` and the client offers a fresh one.

---

## 9. What the server has to grow

In rough order of difficulty:

1. **A web layer.** There is none — no FastAPI, no WebSocket, no ASGI server.
2. **Stage-level execution control.** `QwenModel.forward` runs start to finish.
   It needs to become steppable — a generator over stages, or hooks plus a
   condition variable — so a breakpoint can halt between layer 7 and layer 8.
3. **Attention capture.** `GroupedQueryAttention.forward` computes `attn` and
   discards it (`server/src/model.py:90`). It needs to be retained, quantized on
   capture (u8 — see §6), and bounded: at T=192 keeping every layer/head/step is
   ~24 MB per session as f32, ~120 MB at T=512. Honour
   `SessionConfig.captureAttention` and LRU-evict old steps.
4. **A KV cache** (§7), if the KV visualization is to show a real one.
5. **Sequence-length limits.** `seq_len` is 1024 in `src/config.py`; a 1024×1024
   attention matrix is neither transferable nor legible. Enforce
   `ModelLimits` server-side.

### Deployment notes

- GitHub Pages is HTTPS-only, so the deployed client can only reach `https://`
  and `wss://`. There is no "deployed client, local server" configuration —
  local development runs both on localhost.
- CORS must allow the Pages origin, including for the WebSocket upgrade's
  `Origin` check.
- Sessions are server-side state. On a scale-to-zero platform an instance can be
  reclaimed mid-run, taking the session with it. Deploy with `min-instances: 1`,
  and return `session_not_found` cleanly when it happens anyway.
