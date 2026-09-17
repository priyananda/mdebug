import { ChangeDetectionStrategy, Component, computed, inject } from '@angular/core';

import { formatStageId } from '../../../core/util/format';
import { DebuggerStore } from '../../../core/state/debugger.store';
import { SessionStore } from '../../../core/state/session.store';

/**
 * Start / Continue / Step / Stop.
 *
 * There is no step-into and no step-over: with breakpoints as the only
 * execution model there are no nested granularities to step into or over.
 * `Step` advances exactly one pipeline stage, and the step-over case ("skip the
 * next thirty stages, stop me at L15") is served by "run to here" on a graph
 * node, which is where the target actually is.
 */
@Component({
  selector: 'mdbg-control-bar',
  templateUrl: './control-bar.component.html',
  styleUrl: './control-bar.component.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ControlBarComponent {
  protected readonly debug = inject(DebuggerStore);
  private readonly sessions = inject(SessionStore);

  protected readonly maxSteps = computed(() => this.sessions.draft().maxNewTokens);

  protected readonly position = computed(() => {
    const halt = this.debug.halt();
    const pc = this.debug.programCounter();
    const stage = halt?.stageId ?? pc;
    return stage ? formatStageId(stage) : '—';
  });

  protected readonly statusLabel = computed(() => {
    const status = this.debug.status();
    return status === 'starting' ? 'running' : status;
  });

  /** Shift-click jumps ten stages: faster than clicking Step forty times. */
  protected step(event: MouseEvent): void {
    this.debug.step(event.shiftKey ? 10 : 1);
  }
}
