import { Injectable, computed, effect, inject, signal, untracked } from '@angular/core';

import { ApiError, isCancelled } from '../api/api-errors';
import { InferenceApi } from '../api/inference-api';
import { Token } from '../models/session.model';
import { decodeArray } from '../util/encoding';
import { DebuggerStore } from './debugger.store';
import { SessionStore } from './session.store';

export interface DecodedAttention {
  step: number;
  layer: number;
  head: number;
  sequenceLength: number;
  /** Dense T x T, row-major. */
  weights: Float32Array;
  maxWeight: number;
  meanEntropy: number;
  sinkMass: number;
}

/** Roughly 40 tiles at T=192 is about 6 MB decoded. */
const CACHE_LIMIT = 40;

/**
 * What the right column is currently looking at, plus a lazy cache of the
 * tensors that backs it.
 *
 * Deliberately not using Angular 19's `resource()`: it is experimental in this
 * version and its API changed in the next one. An effect plus a Map is boring,
 * stable and easy to reason about.
 */
@Injectable({ providedIn: 'root' })
export class InspectionStore {
  private readonly api = inject(InferenceApi);
  private readonly sessions = inject(SessionStore);
  private readonly debug = inject(DebuggerStore);

  readonly selectedLayer = signal(0);
  readonly selectedHead = signal(0);
  /** When false, the selected layer follows the program counter. */
  readonly layerPinned = signal(false);
  /** The query row locked in the heatmap, or null when following the cursor. */
  readonly lockedQuery = signal<number | null>(null);
  readonly attentionScale = signal<'sqrt' | 'linear'>('sqrt');

  private readonly _attention = signal<DecodedAttention | null>(null);
  private readonly _loading = signal(false);
  private readonly _error = signal<ApiError | null>(null);
  /** Set when the matrix is big enough that we ask before loading it. */
  private readonly _oversized = signal<number | null>(null);

  readonly attention = this._attention.asReadonly();
  readonly loading = this._loading.asReadonly();
  readonly error = this._error.asReadonly();
  readonly oversized = this._oversized.asReadonly();

  private readonly cache = new Map<string, DecodedAttention>();
  private inFlight: AbortController | null = null;
  private inFlightKey: string | null = null;
  private confirmedOversize = new Set<string>();

  /** Tokens labelling the axes of the current tile. */
  readonly axisTokens = computed<Token[]>(() => {
    const t = this._attention()?.sequenceLength ?? 0;
    return [...this.debug.promptTokens(), ...this.debug.generatedTokens()].slice(0, t);
  });

  constructor() {
    // The selected layer follows execution unless the user has pinned it. This
    // is the difference between the right column feeling alive and feeling inert.
    effect(() => {
      const halt = this.debug.halt();
      if (!halt || untracked(() => this.layerPinned())) return;
      if (halt.stage.layer !== undefined) this.selectedLayer.set(halt.stage.layer);
    });

    effect(() => {
      const sessionId = this.sessions.sessionId();
      const step = this.debug.currentStep();
      const layer = this.selectedLayer();
      const head = this.selectedHead();
      const halted = this.debug.status() === 'halted';
      const t = this.debug.sequenceLength();

      if (!sessionId || !halted || t === 0) return;
      void this.loadTile(sessionId, step, layer, head, t);
    });
  }

  /** Accepts a tile the size guard had refused. */
  confirmOversized(): void {
    const key = this._oversized();
    if (key === null) return;
    this._oversized.set(null);
    const sessionId = this.sessions.sessionId();
    if (!sessionId) return;
    this.confirmedOversize.add(
      tileKey(sessionId, this.debug.currentStep(), this.selectedLayer(), this.selectedHead()),
    );
    void this.loadTile(
      sessionId,
      this.debug.currentStep(),
      this.selectedLayer(),
      this.selectedHead(),
      this.debug.sequenceLength(),
    );
  }

  private async loadTile(
    sessionId: string,
    step: number,
    layer: number,
    head: number,
    t: number,
  ): Promise<void> {
    const key = tileKey(sessionId, step, layer, head);
    if (this.inFlightKey === key) return;

    const cached = this.cache.get(key);
    if (cached) {
      this._attention.set(cached);
      this._error.set(null);
      this._oversized.set(null);
      return;
    }

    const threshold = this.sessions.model()?.limits.attentionWarnThreshold ?? Infinity;
    if (t > threshold && !this.confirmedOversize.has(key)) {
      this._attention.set(null);
      this._oversized.set(t);
      return;
    }

    // Abandon whatever we were fetching: the selection has moved on.
    this.inFlight?.abort();
    const controller = new AbortController();
    this.inFlight = controller;
    this.inFlightKey = key;
    this._loading.set(true);
    this._error.set(null);
    this._oversized.set(null);

    try {
      const tile = await this.api.getAttentionTile(sessionId, step, layer, head, controller.signal);
      const decoded: DecodedAttention = {
        step: tile.step,
        layer: tile.layer,
        head: tile.head,
        sequenceLength: tile.sequenceLength,
        weights: decodeArray(tile.weights),
        maxWeight: tile.stats.maxWeight,
        meanEntropy: tile.stats.meanEntropy,
        sinkMass: tile.stats.sinkMass,
      };
      this.remember(key, decoded);
      this._attention.set(decoded);
    } catch (e) {
      if (!isCancelled(e)) {
        this._error.set(e instanceof ApiError ? e : ApiError.transport('Could not load attention', e));
      }
    } finally {
      if (this.inFlightKey === key) {
        this.inFlightKey = null;
        this.inFlight = null;
        this._loading.set(false);
      }
    }
  }

  private remember(key: string, tile: DecodedAttention): void {
    if (this.cache.size >= CACHE_LIMIT) {
      // Map iterates in insertion order, so the first key is the oldest.
      const oldest = this.cache.keys().next().value;
      if (oldest !== undefined) this.cache.delete(oldest);
    }
    this.cache.set(key, tile);
  }

  clearCache(): void {
    this.cache.clear();
    this._attention.set(null);
  }
}

function tileKey(sessionId: string, step: number, layer: number, head: number): string {
  return `${sessionId}|${step}|${layer}|${head}`;
}
