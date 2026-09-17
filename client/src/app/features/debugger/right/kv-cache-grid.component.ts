import { ChangeDetectionStrategy, Component, computed, inject, signal } from '@angular/core';

import { DebuggerStore } from '../../../core/state/debugger.store';
import { InspectionStore } from '../../../core/state/inspection.store';
import { SessionStore } from '../../../core/state/session.store';
import { formatBytes } from '../../../core/util/format';
import { CellHover, MatrixCanvasComponent } from '../../../shared/matrix-canvas.component';

@Component({
  selector: 'mdbg-kv-cache-grid',
  imports: [MatrixCanvasComponent],
  templateUrl: './kv-cache-grid.component.html',
  styleUrl: './kv-cache-grid.component.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class KvCacheGridComponent {
  private readonly sessions = inject(SessionStore);
  protected readonly debug = inject(DebuggerStore);
  protected readonly inspect = inject(InspectionStore);

  protected readonly kv = this.debug.kv;
  protected readonly hover = signal<CellHover | null>(null);
  /** Occupancy states, or the per-cell key norm when the server provides it. */
  protected readonly mode = signal<'occupancy' | 'norms'>('occupancy');

  protected readonly hasNorms = computed(() => !!this.kv()?.keyNorms);

  protected readonly values = computed(() => {
    const kv = this.kv();
    if (!kv) return null;
    return this.mode() === 'norms' && kv.keyNorms ? kv.keyNorms : kv.occupancy;
  });

  protected readonly colormap = computed(() =>
    this.mode() === 'norms' ? ('sequential' as const) : ('occupancy' as const),
  );

  /** Occupancy values are the literal LUT indices 0/1/2, so they are not scaled. */
  protected readonly max = computed(() => (this.mode() === 'occupancy' ? 255 : null));

  protected readonly hoverLabel = computed(() => {
    const cell = this.hover();
    const kv = this.kv();
    if (!cell || !kv) return null;
    const tokens = [...this.debug.promptTokens(), ...this.debug.generatedTokens()];
    const token = tokens[cell.col]?.display ?? `#${cell.col}`;
    if (this.mode() === 'norms') {
      return `L${cell.row} · pos ${cell.col} (${token}) · ||k||=${cell.value.toFixed(2)}`;
    }
    const state = cell.value >= 2 ? 'resident' : cell.value >= 1 ? 'written this step' : 'not written';
    return `L${cell.row} · pos ${cell.col} (${token}) · ${state}`;
  });

  protected readonly residentLabel = computed(() => {
    const kv = this.kv();
    return kv ? formatBytes(kv.bytesResident) : '';
  });

  protected readonly gridHeight = computed(() => {
    const layers = this.sessions.model()?.numLayers ?? 20;
    return Math.max(80, layers * 5);
  });

  protected selectLayer(cell: CellHover): void {
    this.inspect.selectedLayer.set(cell.row);
    this.inspect.layerPinned.set(true);
  }
}
