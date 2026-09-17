import { ChangeDetectionStrategy, Component, computed, inject, signal } from '@angular/core';

import {
  Breakpoint,
  BreakpointCondition,
  ComparisonOp,
  ConditionKind,
  conditionsForStage,
  describeCondition,
} from '../../../core/models/breakpoint.model';
import { parseStageId } from '../../../core/models/pipeline.model';
import { BreakpointStore } from '../../../core/state/breakpoint.store';
import { formatStageId } from '../../../core/util/format';

const NUMERIC_OPS: ComparisonOp[] = ['==', '!=', '<', '<=', '>', '>='];

/** Sensible starting values, so the editor opens on something meaningful. */
const DEFAULTS: Record<ConditionKind, { op: string; value: number | string }> = {
  token_index: { op: '==', value: 5 },
  sequence_length: { op: '>=', value: 16 },
  hit_count: { op: '>=', value: 3 },
  top1_prob: { op: '<', value: 0.3 },
  logit_entropy: { op: '>', value: 2 },
  residual_norm: { op: '>', value: 50 },
  attention_max: { op: '>', value: 0.9 },
  emitted_token_text: { op: 'contains', value: 'the' },
  emitted_token_id: { op: '==', value: 100 },
};

const LABELS: Record<ConditionKind, string> = {
  token_index: 'token index',
  sequence_length: 'sequence length',
  hit_count: 'hit count',
  top1_prob: 'top-1 probability',
  logit_entropy: 'logit entropy',
  residual_norm: 'residual norm',
  attention_max: 'max attention',
  emitted_token_text: 'emitted token text',
  emitted_token_id: 'emitted token id',
};

@Component({
  selector: 'mdbg-breakpoint-list',
  templateUrl: './breakpoint-list.component.html',
  styleUrl: './breakpoint-list.component.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class BreakpointListComponent {
  protected readonly store = inject(BreakpointStore);

  /** Id of the breakpoint whose condition editor is open. */
  protected readonly editing = signal<string | null>(null);
  protected readonly draftKind = signal<ConditionKind>('top1_prob');
  protected readonly draftOp = signal<string>('<');
  protected readonly draftValue = signal<string>('0.3');

  protected readonly breakpoints = computed(() =>
    [...this.store.all()].sort((a, b) => a.stageId.localeCompare(b.stageId)),
  );

  /**
   * Only conditions the stage can actually evaluate are offered. A
   * `top1_prob` test on an FFN stage would be rejected by the server, so the
   * UI should never let it be built in the first place.
   */
  protected readonly availableKinds = computed<ConditionKind[]>(() => {
    const id = this.editing();
    const bp = id ? this.store.all().find((b) => b.id === id) : undefined;
    return bp ? conditionsForStage(parseStageId(bp.stageId).kind) : [];
  });

  protected readonly ops = computed(() =>
    this.draftKind() === 'emitted_token_text' ? ['equals', 'contains'] : NUMERIC_OPS,
  );

  protected readonly isText = computed(() => this.draftKind() === 'emitted_token_text');

  protected label(kind: ConditionKind): string {
    return LABELS[kind];
  }

  protected stageLabel(bp: Breakpoint): string {
    return formatStageId(bp.stageId);
  }

  protected conditionLabel(bp: Breakpoint): string | null {
    return bp.condition ? describeCondition(bp.condition) : null;
  }

  protected openEditor(bp: Breakpoint): void {
    this.editing.set(bp.id);
    const kinds = conditionsForStage(parseStageId(bp.stageId).kind);
    const kind = bp.condition?.kind ?? kinds[kinds.length - 1] ?? 'token_index';
    this.setKind(kind);
    if (bp.condition) {
      this.draftOp.set(bp.condition.op);
      this.draftValue.set(String(bp.condition.value));
    }
  }

  protected onKind(event: Event): void {
    this.setKind((event.target as HTMLSelectElement).value as ConditionKind);
  }

  private setKind(kind: ConditionKind): void {
    this.draftKind.set(kind);
    const preset = DEFAULTS[kind];
    this.draftOp.set(preset.op);
    this.draftValue.set(String(preset.value));
  }

  protected onOp(event: Event): void {
    this.draftOp.set((event.target as HTMLSelectElement).value);
  }

  protected onValue(event: Event): void {
    this.draftValue.set((event.target as HTMLInputElement).value);
  }

  protected commit(bp: Breakpoint): void {
    const kind = this.draftKind();
    const op = this.draftOp();
    const raw = this.draftValue();
    const value = this.isText() ? raw : Number(raw);
    if (!this.isText() && !Number.isFinite(value as number)) return;

    this.store.setCondition(bp.id, { kind, op, value } as BreakpointCondition);
    this.editing.set(null);
  }

  protected clearCondition(bp: Breakpoint): void {
    this.store.setCondition(bp.id, undefined);
    this.editing.set(null);
  }
}
