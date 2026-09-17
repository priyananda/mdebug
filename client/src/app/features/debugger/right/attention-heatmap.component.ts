import { ChangeDetectionStrategy, Component, computed, inject, signal } from '@angular/core';

import { DebuggerStore } from '../../../core/state/debugger.store';
import { InspectionStore } from '../../../core/state/inspection.store';
import { SessionStore } from '../../../core/state/session.store';
import { formatFixed, formatProb } from '../../../core/util/format';
import { CellHover, MatrixCanvasComponent } from '../../../shared/matrix-canvas.component';

interface RowBar {
  col: number;
  label: string;
  weight: number;
  /** Percentage width of the bar, relative to the row's maximum. */
  width: number;
}

/** Bars below the heatmap are only readable for so many keys. */
const MAX_ROW_BARS = 40;

@Component({
  selector: 'mdbg-attention-heatmap',
  imports: [MatrixCanvasComponent],
  templateUrl: './attention-heatmap.component.html',
  styleUrl: './attention-heatmap.component.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class AttentionHeatmapComponent {
  private readonly sessions = inject(SessionStore);
  protected readonly debug = inject(DebuggerStore);
  protected readonly inspect = inject(InspectionStore);

  protected readonly model = this.sessions.model;
  protected readonly tile = this.inspect.attention;
  protected readonly hover = signal<CellHover | null>(null);

  protected readonly layers = computed(() =>
    Array.from({ length: this.model()?.numLayers ?? 0 }, (_, i) => i),
  );
  protected readonly heads = computed(() =>
    Array.from({ length: this.model()?.numHeads ?? 0 }, (_, i) => i),
  );

  /**
   * The query row on show beneath the heatmap: whatever is locked, else
   * whatever is hovered, else the row for the token being generated — which is
   * the one that actually decides the next token.
   */
  protected readonly focusRow = computed(() => {
    const tile = this.tile();
    if (!tile) return null;
    const locked = this.inspect.lockedQuery();
    if (locked !== null && locked < tile.sequenceLength) return locked;
    const hovered = this.hover()?.row;
    if (hovered !== undefined) return hovered;
    return tile.sequenceLength - 1;
  });

  protected readonly rowBars = computed<RowBar[]>(() => {
    const tile = this.tile();
    const row = this.focusRow();
    if (!tile || row === null) return [];

    const t = tile.sequenceLength;
    const tokens = this.inspect.axisTokens();
    const bars: RowBar[] = [];
    let max = 0;
    for (let col = 0; col <= row; col++) {
      const weight = tile.weights[row * t + col];
      if (weight > max) max = weight;
      bars.push({ col, label: tokens[col]?.display ?? `#${col}`, weight, width: 0 });
    }
    for (const bar of bars) bar.width = max > 0 ? (bar.weight / max) * 100 : 0;

    // Keep the heaviest keys when the row is longer than we can draw.
    if (bars.length > MAX_ROW_BARS) {
      return [...bars].sort((a, b) => b.weight - a.weight).slice(0, MAX_ROW_BARS);
    }
    return bars;
  });

  /** Row 0 is a legitimate value, so this must not be expressed as a truthiness test. */
  protected readonly rowMarkTop = computed(() => {
    const tile = this.tile();
    const row = this.focusRow();
    if (!tile || row === null) return 0;
    return ((row + 0.5) / tile.sequenceLength) * 100;
  });

  /**
   * Colour ceiling for the map.
   *
   * The first rows are degenerate — the query at position 0 has exactly one key,
   * so its weight is 1.0 by construction. Letting that set the ceiling
   * compresses every genuinely interesting value into the pale end of the ramp.
   * Normalizing against rows that have a real distribution to spread over fixes
   * it without touching the data.
   */
  protected readonly displayMax = computed(() => {
    const tile = this.tile();
    if (!tile) return null;
    const t = tile.sequenceLength;
    const from = Math.min(3, t - 1);
    let max = 0;
    for (let row = from; row < t; row++) {
      for (let col = 0; col <= row; col++) {
        const v = tile.weights[row * t + col];
        if (v > max) max = v;
      }
    }
    return max > 0 ? max : tile.maxWeight;
  });

  protected readonly cellCount = computed(() => {
    const t = this.inspect.oversized() ?? 0;
    return `${Math.round((t * t) / 1000)}k`;
  });

  protected readonly hoverLabel = computed(() => {
    const cell = this.hover();
    const tokens = this.inspect.axisTokens();
    if (!cell) return null;
    const query = tokens[cell.row]?.display ?? `#${cell.row}`;
    const key = tokens[cell.col]?.display ?? `#${cell.col}`;
    if (cell.col > cell.row) return `${query} cannot attend to ${key} (future)`;
    return `${query} → ${key} · p=${formatProb(cell.value)}`;
  });

  protected readonly focusLabel = computed(() => {
    const row = this.focusRow();
    const tokens = this.inspect.axisTokens();
    if (row === null) return '';
    return tokens[row]?.display ?? `#${row}`;
  });

  protected onLayer(event: Event): void {
    this.inspect.selectedLayer.set(Number((event.target as HTMLSelectElement).value));
    this.inspect.layerPinned.set(true);
  }

  protected onHead(event: Event): void {
    this.inspect.selectedHead.set(Number((event.target as HTMLSelectElement).value));
  }

  protected toggleFollow(): void {
    this.inspect.layerPinned.set(!this.inspect.layerPinned());
  }

  protected toggleScale(): void {
    this.inspect.attentionScale.set(this.inspect.attentionScale() === 'sqrt' ? 'linear' : 'sqrt');
  }

  protected lockRow(cell: CellHover): void {
    this.inspect.lockedQuery.set(this.inspect.lockedQuery() === cell.row ? null : cell.row);
  }

  protected readonly fmt = formatFixed;
  protected readonly prob = formatProb;
}
