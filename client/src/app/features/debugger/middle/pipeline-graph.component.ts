import { ChangeDetectionStrategy, Component, computed, inject, signal } from '@angular/core';

import { Breakpoint } from '../../../core/models/breakpoint.model';
import { StageId } from '../../../core/models/pipeline.model';
import { BreakpointStore } from '../../../core/state/breakpoint.store';
import { DebuggerStore } from '../../../core/state/debugger.store';
import { InspectionStore } from '../../../core/state/inspection.store';
import { SessionStore } from '../../../core/state/session.store';
import { cssColor, normIndex } from '../../../core/util/colormap';
import { describeCondition } from '../../../core/models/breakpoint.model';
import { GraphLayout, layoutPipeline } from './pipeline-layout';

/**
 * The pipeline graph: the primary interactive surface of the app.
 *
 * Inline SVG rather than canvas. It is about 230 elements, comfortably inside
 * SVG's range, and canvas would mean hand-rolling hit testing, focus order,
 * tooltips and the accessibility story for the one surface where the platform's
 * versions of all four actually matter.
 */
@Component({
  selector: 'mdbg-pipeline-graph',
  templateUrl: './pipeline-graph.component.html',
  styleUrl: './pipeline-graph.component.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class PipelineGraphComponent {
  private readonly sessions = inject(SessionStore);
  protected readonly debug = inject(DebuggerStore);
  protected readonly breakpoints = inject(BreakpointStore);
  protected readonly inspect = inject(InspectionStore);

  protected readonly compact = signal(false);

  protected readonly layout = computed<GraphLayout | null>(() => {
    const model = this.sessions.model();
    return model ? layoutPipeline(model, { compact: this.compact() }) : null;
  });

  /** The row containing the program counter, for the highlight band. */
  protected readonly activeRow = computed(() => {
    const pc = this.debug.programCounter();
    const layout = this.layout();
    return pc && layout ? (layout.rowByStage.get(pc) ?? null) : null;
  });

  protected readonly haltedHere = computed(() => this.debug.status() === 'halted');

  /**
   * Stages already executed in the current step, so you can see how far through
   * a step you are at a glance.
   */
  protected readonly completed = computed(() => {
    const layout = this.layout();
    const pc = this.debug.programCounter();
    if (!layout || !pc) return new Set<StageId>();
    const order = layout.nodes.map((n) => n.stageId);
    const upTo = order.indexOf(pc);
    return new Set(upTo < 0 ? [] : order.slice(0, upTo));
  });

  /**
   * Per-head colour, from the entropy of the current query row. Shown as
   * *focus* — the inverse of normalized entropy — so that a head concentrating
   * its attention reads as dark and a diffuse one fades out.
   */
  protected headFill(layer: number, head: number): string {
    const summary = this.debug.headSummary();
    const model = this.sessions.model();
    if (!summary || !model) return 'var(--bg-raised)';
    const entropy = summary[layer * model.numHeads + head];
    const maxEntropy = Math.log(Math.max(2, this.debug.sequenceLength()));
    const focus = 1 - Math.min(1, entropy / maxEntropy);
    return cssColor('sequential', normIndex(focus));
  }

  protected headTitle(layer: number, head: number): string {
    const summary = this.debug.headSummary();
    const model = this.sessions.model();
    if (!summary || !model) return `layer ${layer}, head ${head}`;
    const entropy = summary[layer * model.numHeads + head];
    return `L${layer} head ${head} · entropy ${entropy.toFixed(2)} nats — click to inspect`;
  }

  protected breakpointAt(stageId: StageId): Breakpoint | undefined {
    return this.breakpoints.byStage().get(stageId);
  }

  protected breakpointTitle(stageId: StageId): string {
    const bp = this.breakpointAt(stageId);
    if (!bp) return `Set a breakpoint at ${stageId}`;
    const condition = bp.condition ? ` when ${describeCondition(bp.condition)}` : '';
    const state = bp.enabled ? 'Breakpoint' : 'Disabled breakpoint';
    return `${state} at ${stageId}${condition} · ${bp.hitCount} hits — click to remove`;
  }

  protected toggle(stageId: StageId): void {
    this.breakpoints.toggle(stageId);
  }

  protected runToHere(event: Event, stageId: StageId): void {
    event.stopPropagation();
    this.debug.runToCursor(stageId);
  }

  protected selectHead(layer: number, head: number): void {
    this.inspect.selectedLayer.set(layer);
    this.inspect.selectedHead.set(head);
    this.inspect.layerPinned.set(true);
  }

  protected isSelectedHead(layer: number, head: number): boolean {
    return this.inspect.selectedLayer() === layer && this.inspect.selectedHead() === head;
  }
}
