import { Injectable, computed, inject, signal } from '@angular/core';

import { ApiError } from '../api/api-errors';
import { InferenceApi } from '../api/inference-api';
import { ServerEvent } from '../models/events.model';
import { StageId } from '../models/pipeline.model';
import {
  HaltPosition,
  ResidualTrendPoint,
  RunStatus,
} from '../models/run-state.model';
import { SessionConfig, Token } from '../models/session.model';
import { TopKLogits, VectorStats } from '../models/tensors.model';
import { decodeArray } from '../util/encoding';
import { SessionStore } from './session.store';

export interface DecodedKv {
  numLayers: number;
  sequenceLength: number;
  /** [numLayers, T]: 0 empty, 1 written this step, 2 written earlier. */
  occupancy: Float32Array;
  keyNorms?: Float32Array;
  bytesResident: number;
  simulated: boolean;
}

/**
 * The centre of gravity: run status, the program counter, and the state
 * captured at the current halt.
 *
 * Everything arrives through one reducer over `ServerEvent`. That single switch
 * is what makes the mock engine and the real transport interchangeable by
 * construction — neither one gets to have its own path into the UI's state.
 *
 * Decoding from the wire format happens here, once. Components only ever see
 * typed arrays.
 */
@Injectable({ providedIn: 'root' })
export class DebuggerStore {
  private readonly api = inject(InferenceApi);
  private readonly sessions = inject(SessionStore);

  private readonly _status = signal<RunStatus>('idle');
  private readonly _halt = signal<HaltPosition | null>(null);
  private readonly _programCounter = signal<StageId | null>(null);
  private readonly _promptTokens = signal<Token[]>([]);
  private readonly _generatedTokens = signal<Token[]>([]);
  private readonly _currentStep = signal(0);
  private readonly _sequenceLength = signal(0);
  private readonly _error = signal<ApiError | null>(null);

  private readonly _headSummary = signal<Float32Array | null>(null);
  private readonly _residual = signal<Float32Array | null>(null);
  private readonly _residualStats = signal<VectorStats | null>(null);
  private readonly _residualTrend = signal<ResidualTrendPoint[]>([]);
  private readonly _kv = signal<DecodedKv | null>(null);
  private readonly _topK = signal<TopKLogits | null>(null);
  private readonly _finishedText = signal<string | null>(null);

  readonly status = this._status.asReadonly();
  readonly halt = this._halt.asReadonly();
  readonly programCounter = this._programCounter.asReadonly();
  readonly promptTokens = this._promptTokens.asReadonly();
  readonly generatedTokens = this._generatedTokens.asReadonly();
  readonly currentStep = this._currentStep.asReadonly();
  readonly sequenceLength = this._sequenceLength.asReadonly();
  readonly error = this._error.asReadonly();

  readonly headSummary = this._headSummary.asReadonly();
  readonly residual = this._residual.asReadonly();
  readonly residualStats = this._residualStats.asReadonly();
  readonly residualTrend = this._residualTrend.asReadonly();
  readonly kv = this._kv.asReadonly();
  readonly topK = this._topK.asReadonly();
  readonly finishedText = this._finishedText.asReadonly();

  /**
   * The control bar binds `[disabled]` to these rather than to `status`, so the
   * rules about what is legal when live in exactly one place.
   */
  readonly canStart = computed(() => {
    const s = this._status();
    return s === 'idle' || s === 'finished' || s === 'error';
  });
  readonly canContinue = computed(() => this._status() === 'halted');
  readonly canStep = computed(() => this._status() === 'halted');
  readonly canStop = computed(() => {
    const s = this._status();
    return s === 'running' || s === 'halted' || s === 'starting';
  });
  readonly isBusy = computed(() => this._status() === 'running' || this._status() === 'starting');

  constructor() {
    this.api.events$.subscribe((event) => this.reduce(event));
  }

  // --- commands -------------------------------------------------------------

  start(): void {
    this._status.set('starting');
    this._error.set(null);
    this._finishedText.set(null);
    this._generatedTokens.set([]);
    this._residualTrend.set([]);
    this.api.send({ type: 'start', payload: { config: this.sessions.draft() } });
  }

  continue_(): void {
    this.api.send({ type: 'continue', payload: {} });
  }

  step(count = 1): void {
    this.api.send({ type: 'step', payload: { count } });
  }

  stop(): void {
    this.api.send({ type: 'stop', payload: {} });
  }

  runToCursor(stageId: StageId): void {
    this.api.send({ type: 'run_to_cursor', payload: { stageId } });
  }

  patchConfig(patch: Partial<SessionConfig>): void {
    this.api.send({ type: 'patch_config', payload: patch });
  }

  // --- the one reducer ------------------------------------------------------

  private reduce(event: ServerEvent): void {
    switch (event.type) {
      case 'session_state': {
        const s = event.payload;
        this.sessions.sync(s);
        this._promptTokens.set(s.promptTokens);
        this._generatedTokens.set(s.generatedTokens);
        this._currentStep.set(s.currentStep);
        this._halt.set(s.halt);
        this._programCounter.set(s.halt?.stageId ?? null);
        this._status.set(toRunStatus(s.status));
        this._sequenceLength.set(s.promptTokens.length + s.generatedTokens.length);
        break;
      }

      case 'run_started':
        this._status.set('running');
        this._currentStep.set(event.payload.step);
        this._residualTrend.set([]);
        break;

      case 'stage_entered':
        this._programCounter.set(event.payload.stageId);
        this._currentStep.set(event.payload.step);
        this._sequenceLength.set(event.payload.sequenceLength);
        break;

      case 'token_emitted':
        this._generatedTokens.update((tokens) => [...tokens, event.payload.token]);
        if (event.payload.topK) this._topK.set(event.payload.topK);
        break;

      case 'halted': {
        const p = event.payload;
        this._status.set('halted');
        this._halt.set(p.position);
        this._programCounter.set(p.position.stageId);
        this._currentStep.set(p.position.step);
        this._sequenceLength.set(p.position.sequenceLength);

        if (p.sequence.promptTokens) this._promptTokens.set(p.sequence.promptTokens);
        if (p.sequence.generatedTokens.length) {
          this._generatedTokens.update((tokens) => mergeTokens(tokens, p.sequence.generatedTokens));
        }

        this._headSummary.set(p.headSummary ? decodeArray(p.headSummary) : null);
        this._residual.set(p.residual ? decodeArray(p.residual) : null);
        this._residualStats.set(p.residualStats ?? null);
        this._kv.set(
          p.kv
            ? {
                numLayers: p.kv.numLayers,
                sequenceLength: p.kv.sequenceLength,
                occupancy: decodeArray(p.kv.occupancy),
                keyNorms: p.kv.keyNorms ? decodeArray(p.kv.keyNorms) : undefined,
                bytesResident: p.kv.bytesResident,
                simulated: p.kv.simulated,
              }
            : null,
        );
        if (p.topK) this._topK.set(p.topK);
        this.recordTrend(p.position, p.residualStats?.l2);
        break;
      }

      case 'resumed':
        this._status.set('running');
        this._halt.set(null);
        break;

      case 'finished':
        this._status.set(event.payload.reason === 'stopped' ? 'idle' : 'finished');
        this._halt.set(null);
        this._programCounter.set(null);
        this._finishedText.set(event.payload.text);
        break;

      case 'breakpoints_changed':
        // BreakpointStore subscribes to the same stream and owns this.
        break;

      case 'error':
        this._error.set(new ApiError('server', event.payload.message, event.payload.code));
        if (event.payload.fatal) this._status.set('error');
        break;

      case 'pong':
        break;
    }
  }

  /**
   * The layer-over-layer residual norm trend, accumulated as execution walks
   * down the stack. Reset whenever the step changes, so the trend always shows
   * one token's journey through the model.
   */
  private recordTrend(position: HaltPosition, l2: number | undefined): void {
    if (l2 === undefined || position.stage.layer === undefined) return;
    this._residualTrend.update((points) => {
      const sameStep = points.length && points[0].step === position.step;
      const kept = sameStep ? points.filter((p) => p.stageId !== position.stageId) : [];
      return [...kept, { stageId: position.stageId, layer: position.stage.layer!, l2, step: position.step }].sort(
        (a, b) => a.layer - b.layer,
      );
    });
  }
}

function toRunStatus(status: string): RunStatus {
  switch (status) {
    case 'running':
      return 'running';
    case 'halted':
      return 'halted';
    case 'finished':
      return 'finished';
    case 'error':
      return 'error';
    default:
      return 'idle';
  }
}

/** Appends only tokens we have not already got, by position. */
function mergeTokens(existing: Token[], incoming: Token[]): Token[] {
  const seen = new Set(existing.map((t) => t.position));
  return [...existing, ...incoming.filter((t) => !seen.has(t.position))];
}
