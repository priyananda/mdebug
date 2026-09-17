import { Injectable, computed, inject, signal } from '@angular/core';

import { InferenceApi } from '../api/inference-api';
import {
  Breakpoint,
  BreakpointCondition,
} from '../models/breakpoint.model';
import { StageId } from '../models/pipeline.model';

/**
 * Breakpoints are server-side truth, because conditions are evaluated against
 * values only the server has. The client keeps an optimistic copy so that
 * clicking the gutter feels instant, sends the full set, and reconciles when
 * the server echoes back.
 */
@Injectable({ providedIn: 'root' })
export class BreakpointStore {
  private readonly api = inject(InferenceApi);

  private readonly _all = signal<Breakpoint[]>([]);
  readonly all = this._all.asReadonly();

  /** Computed map so the pipeline graph can do O(1) lookups per node. */
  readonly byStage = computed(() => {
    const map = new Map<StageId, Breakpoint>();
    for (const bp of this._all()) map.set(bp.stageId, bp);
    return map as ReadonlyMap<StageId, Breakpoint>;
  });

  readonly count = computed(() => this._all().length);
  readonly enabledCount = computed(() => this._all().filter((b) => b.enabled).length);

  constructor() {
    this.api.events$.subscribe((event) => {
      if (event.type === 'breakpoints_changed') {
        this._all.set(event.payload.breakpoints);
      } else if (event.type === 'session_state') {
        this._all.set(event.payload.breakpoints);
      }
    });
  }

  toggle(stageId: StageId): void {
    const existing = this.byStage().get(stageId);
    this.commit(
      existing
        ? this._all().filter((b) => b.id !== existing.id)
        : [...this._all(), newBreakpoint(stageId)],
    );
  }

  setEnabled(id: string, enabled: boolean): void {
    this.commit(this._all().map((b) => (b.id === id ? { ...b, enabled } : b)));
  }

  setCondition(id: string, condition: BreakpointCondition | undefined): void {
    this.commit(this._all().map((b) => (b.id === id ? { ...b, condition } : b)));
  }

  remove(id: string): void {
    this.commit(this._all().filter((b) => b.id !== id));
  }

  clearAll(): void {
    this.commit([]);
  }

  /** Used by the left column's scenarios to install a whole set at once. */
  applyPreset(preset: { stageId: StageId; condition?: BreakpointCondition }[]): void {
    this.commit(preset.map((p) => ({ ...newBreakpoint(p.stageId), condition: p.condition })));
  }

  private commit(next: Breakpoint[]): void {
    this._all.set(next);
    this.api.send({ type: 'set_breakpoints', payload: { breakpoints: next } });
  }
}

let counter = 0;

function newBreakpoint(stageId: StageId): Breakpoint {
  return {
    id: `bp-${++counter}`,
    stageId,
    enabled: true,
    oneShot: false,
    hitCount: 0,
  };
}
