import { ChangeDetectionStrategy, Component, computed, inject, signal } from '@angular/core';

import { DebuggerStore } from '../../../core/state/debugger.store';
import { SessionStore } from '../../../core/state/session.store';
import { formatFixed } from '../../../core/util/format';
import { CellHover, MatrixCanvasComponent } from '../../../shared/matrix-canvas.component';

interface OutlierDim {
  dim: number;
  value: number;
}

const SPARK_W = 300;
const SPARK_H = 44;
const TREND_W = 300;
const TREND_H = 34;
const OUTLIER_COUNT = 8;

@Component({
  selector: 'mdbg-residual-strip',
  imports: [MatrixCanvasComponent],
  templateUrl: './residual-strip.component.html',
  styleUrl: './residual-strip.component.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ResidualStripComponent {
  private readonly sessions = inject(SessionStore);
  protected readonly debug = inject(DebuggerStore);

  protected readonly values = this.debug.residual;
  protected readonly stats = this.debug.residualStats;
  protected readonly hover = signal<CellHover | null>(null);

  protected readonly hiddenSize = computed(() => this.sessions.model()?.hiddenSize ?? 512);

  /** Symmetric colour ceiling so sign is readable at a glance. */
  protected readonly absMax = computed(() => {
    const s = this.stats();
    return s ? Math.max(Math.abs(s.min), Math.abs(s.max)) : null;
  });

  protected readonly sparkPath = computed(() => {
    const v = this.values();
    const ceiling = this.absMax();
    if (!v || !ceiling) return '';
    const n = v.length;
    let d = '';
    for (let i = 0; i < n; i++) {
      const x = (i / (n - 1)) * SPARK_W;
      const y = SPARK_H / 2 - (v[i] / ceiling) * (SPARK_H / 2 - 1);
      d += `${i === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`;
    }
    return d;
  });

  /**
   * Residual norm layer by layer, accumulated as execution walks down the
   * stack. Growth through depth is a real phenomenon, and watching the line
   * climb as you step is the clearest thing in the panel.
   */
  protected readonly trend = this.debug.residualTrend;

  protected readonly trendPath = computed(() => {
    const points = this.trend();
    const layers = this.sessions.model()?.numLayers ?? 20;
    if (points.length < 2) return '';
    const max = Math.max(...points.map((p) => p.l2));
    return points
      .map((p, i) => {
        const x = (p.layer / Math.max(1, layers - 1)) * TREND_W;
        const y = TREND_H - (p.l2 / max) * (TREND_H - 2) - 1;
        return `${i === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`;
      })
      .join('');
  });

  protected readonly trendDots = computed(() => {
    const points = this.trend();
    const layers = this.sessions.model()?.numLayers ?? 20;
    if (!points.length) return [];
    const max = Math.max(...points.map((p) => p.l2));
    return points.map((p) => ({
      ...p,
      x: (p.layer / Math.max(1, layers - 1)) * TREND_W,
      y: TREND_H - (p.l2 / max) * (TREND_H - 2) - 1,
    }));
  });

  /**
   * The largest-magnitude dimensions. Transformer residual streams have
   * persistent outlier dimensions; noticing the same index recur as you step
   * down the stack is a genuine insight, so these are worth calling out.
   */
  protected readonly outliers = computed<OutlierDim[]>(() => {
    const v = this.values();
    if (!v) return [];
    const indices = Array.from({ length: v.length }, (_, i) => i);
    indices.sort((a, b) => Math.abs(v[b]) - Math.abs(v[a]));
    return indices.slice(0, OUTLIER_COUNT).map((dim) => ({ dim, value: v[dim] }));
  });

  protected readonly hoverLabel = computed(() => {
    const cell = this.hover();
    return cell ? `dim ${cell.col} = ${formatFixed(cell.value, 3)}` : null;
  });

  protected readonly layerLabel = computed(() => {
    const halt = this.debug.halt();
    return halt?.stage.layer !== undefined ? `L${halt.stage.layer}` : (halt?.stageId ?? '');
  });

  protected readonly sparkW = SPARK_W;
  protected readonly sparkH = SPARK_H;
  protected readonly trendW = TREND_W;
  protected readonly trendH = TREND_H;
  protected readonly fmt = formatFixed;
}
