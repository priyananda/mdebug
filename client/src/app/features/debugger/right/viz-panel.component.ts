import { ChangeDetectionStrategy, Component } from '@angular/core';

import { AttentionHeatmapComponent } from './attention-heatmap.component';
import { KvCacheGridComponent } from './kv-cache-grid.component';
import { ResidualStripComponent } from './residual-strip.component';
import { TopKLogitsComponent } from './top-k-logits.component';

/**
 * Hosts the visualizations, stacked rather than tabbed. Tabs would hide the
 * relationships between them, and watching the KV cache fill while the residual
 * norm climbs is most of the point.
 */
@Component({
  selector: 'mdbg-viz-panel',
  imports: [
    AttentionHeatmapComponent,
    TopKLogitsComponent,
    KvCacheGridComponent,
    ResidualStripComponent,
  ],
  template: `
    <div class="stack">
      <mdbg-attention-heatmap />
      <mdbg-top-k-logits />
      <mdbg-kv-cache-grid />
      <mdbg-residual-strip />
    </div>
  `,
  styles: [
    `
      :host {
        display: block;
        min-width: 0;
      }
      .stack {
        display: grid;
        gap: var(--s-3);
        min-width: 0;
      }
    `,
  ],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class VizPanelComponent {}
