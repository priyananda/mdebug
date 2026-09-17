import { StageId, StageRef } from './pipeline.model';
import { Token } from './session.model';
import {
  EncodedArray,
  KvSnapshot,
  TopKLogits,
  VectorStats,
} from './tensors.model';

/** Client-side run status. Narrower than SessionStatus: no 'closed'. */
export type RunStatus =
  | 'idle'
  | 'starting'
  | 'running'
  | 'halted'
  | 'finished'
  | 'error';

export type HaltReason =
  | 'breakpoint'
  | 'step'
  | 'start'
  | 'run_to_cursor'
  | 'finished';

/** Where execution is stopped. The debugger's program counter. */
export interface HaltPosition {
  /** Decode step index; 0 is the first generated token. */
  step: number;
  stage: StageRef;
  stageId: StageId;
  /** T at this instant: prompt tokens plus tokens generated so far. */
  sequenceLength: number;
  reason: HaltReason;
  breakpointId?: string;
}

/**
 * Sent eagerly with every `halted` event. Budget: under ~25kB, so that a halt
 * feels instantaneous. Anything unbounded is fetched on demand instead.
 */
export interface HaltPayload {
  position: HaltPosition;
  sequence: {
    /** Sent on the first halt only; the client caches thereafter. */
    promptTokens?: Token[];
    /** Incremental: only tokens the client has not been sent yet. */
    generatedTokens: Token[];
  };
  /** f32 [numLayers, numHeads] of per-head mean attention entropy. */
  headSummary?: EncodedArray;
  /** f32 [hiddenSize] — the hidden state at the current layer, last position. */
  residual?: EncodedArray;
  residualStats?: VectorStats;
  kv?: KvSnapshot;
  /** Present only at `lm_head`, `sample` and `emit`. */
  topK?: TopKLogits;
  timings?: { stageMs: number; stepMs: number };
}

/** One point on the layer-over-layer residual norm trend. */
export interface ResidualTrendPoint {
  stageId: StageId;
  /** The decode step this point belongs to; the trend resets when it changes. */
  step: number;
  layer: number;
  l2: number;
}
