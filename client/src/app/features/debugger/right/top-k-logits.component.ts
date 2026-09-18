import { ChangeDetectionStrategy, Component, computed, inject, signal } from '@angular/core';

import { DebuggerStore } from '../../../core/state/debugger.store';
import { formatFixed, formatProb } from '../../../core/util/format';

interface Row {
  rank: number;
  tokenId: number;
  display: string;
  prob: number;
  logit: number;
  /** Bar width as a percentage of the most likely token. */
  width: number;
  chosen: boolean;
}

type Scale = 'linear' | 'sqrt' | 'log';

/** Mass threshold for the "n tokens cover x%" readout. */
const COVERAGE = 0.9;

@Component({
  selector: 'mdbg-top-k-logits',
  templateUrl: './top-k-logits.component.html',
  styleUrl: './top-k-logits.component.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class TopKLogitsComponent {
  protected readonly debug = inject(DebuggerStore);

  protected readonly topK = this.debug.topK;
  /**
   * Peaked distributions hide their tail under a linear bar, and this model is
   * very peaked: the runner-up is four orders of magnitude down, which even a
   * square root cannot lift off the axis. Log is the only scale that shows the
   * shape of the tail, so all three are offered.
   */
  protected readonly scale = signal<Scale>('linear');

  /**
   * The distribution survives past the stage that produced it, so that stepping
   * through the next token's layers still shows the last decision. That is
   * useful, but only if it says which step it came from -- otherwise it reads
   * as the current one.
   */
  protected readonly stale = computed(() => {
    const top = this.topK();
    return top !== null && top.step !== this.debug.currentStep();
  });

  protected readonly rows = computed<Row[]>(() => {
    const top = this.topK();
    if (!top) return [];
    const max = top.entries[0]?.prob ?? 0;
    const smallest = top.entries[top.entries.length - 1]?.prob ?? 0;
    const scale = this.scale();

    // Log needs a floor to measure against; the least likely token shown is the
    // natural one, and it keeps the axis honest about what is on screen.
    const logMin = Math.log(Math.max(smallest, 1e-12));
    const logSpan = Math.log(Math.max(max, 1e-12)) - logMin;

    return top.entries.map((entry, rank) => {
      const ratio = max > 0 ? entry.prob / max : 0;
      let fraction: number;
      if (scale === 'sqrt') {
        fraction = Math.sqrt(ratio);
      } else if (scale === 'log') {
        fraction =
          logSpan > 0 ? (Math.log(Math.max(entry.prob, 1e-12)) - logMin) / logSpan : ratio;
      } else {
        fraction = ratio;
      }
      return {
        rank,
        tokenId: entry.tokenId,
        display: entry.display,
        prob: entry.prob,
        logit: entry.logit,
        width: Math.max(0, Math.min(1, fraction)) * 100,
        chosen: entry.tokenId === top.chosenTokenId,
      };
    });
  });

  protected readonly chosen = computed(() => {
    const top = this.topK();
    if (!top) return null;
    return top.entries.find((e) => e.tokenId === top.chosenTokenId) ?? null;
  });

  /** How many tokens it takes to cover most of the mass. */
  protected readonly coverage = computed(() => {
    const top = this.topK();
    if (!top) return null;
    let mass = 0;
    for (let i = 0; i < top.entries.length; i++) {
      mass += top.entries[i].prob;
      if (mass >= COVERAGE) return { count: i + 1, mass };
    }
    return { count: top.entries.length, mass };
  });

  protected readonly coveragePercent = Math.round(COVERAGE * 100);

  protected cycleScale(): void {
    const order: Scale[] = ['linear', 'sqrt', 'log'];
    this.scale.set(order[(order.indexOf(this.scale()) + 1) % order.length]);
  }

  protected readonly prob = formatProb;
  protected readonly fmt = formatFixed;
}
