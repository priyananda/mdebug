import { ChangeDetectionStrategy, Component, computed, inject, signal } from '@angular/core';

import { BreakpointStore } from '../../../core/state/breakpoint.store';
import { DebuggerStore } from '../../../core/state/debugger.store';
import { InspectionStore } from '../../../core/state/inspection.store';
import { SessionStore } from '../../../core/state/session.store';
import { SCENARIOS, Scenario } from './scenarios.data';

@Component({
  selector: 'mdbg-guide-panel',
  templateUrl: './guide-panel.component.html',
  styleUrl: './guide-panel.component.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class GuidePanelComponent {
  private readonly sessions = inject(SessionStore);
  private readonly breakpoints = inject(BreakpointStore);
  private readonly inspect = inject(InspectionStore);
  protected readonly debug = inject(DebuggerStore);

  protected readonly scenarios = SCENARIOS;
  protected readonly active = signal<string | null>(null);
  protected readonly showNotes = signal(false);

  protected readonly notes = computed(() => this.sessions.model()?.notes ?? []);
  protected readonly breakpointCount = this.breakpoints.count;

  protected clearBreakpoints(): void {
    this.breakpoints.clearAll();
    this.active.set(null);
  }

  protected apply(scenario: Scenario): void {
    this.active.set(scenario.id);
    if (scenario.prompt) this.sessions.patchDraft({ prompt: scenario.prompt });
    this.breakpoints.applyPreset(scenario.breakpoints);
    if (scenario.focus) {
      this.inspect.selectedLayer.set(scenario.focus.layer);
      this.inspect.layerPinned.set(true);
      if (scenario.focus.head !== undefined) this.inspect.selectedHead.set(scenario.focus.head);
    }
  }
}
