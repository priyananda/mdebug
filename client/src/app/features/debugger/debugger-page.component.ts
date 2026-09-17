import {
  ChangeDetectionStrategy,
  Component,
  effect,
  inject,
  input,
  signal,
} from '@angular/core';
import { RouterLink } from '@angular/router';

import { DebuggerStore } from '../../core/state/debugger.store';
import { SessionStore } from '../../core/state/session.store';
import { GuidePanelComponent } from './left/guide-panel.component';
import { BreakpointListComponent } from './middle/breakpoint-list.component';
import { ControlBarComponent } from './middle/control-bar.component';
import { PipelineGraphComponent } from './middle/pipeline-graph.component';
import { PromptBoxComponent } from './middle/prompt-box.component';
import { VizPanelComponent } from './right/viz-panel.component';

@Component({
  selector: 'mdbg-debugger-page',
  imports: [
    RouterLink,
    GuidePanelComponent,
    PromptBoxComponent,
    ControlBarComponent,
    PipelineGraphComponent,
    BreakpointListComponent,
    VizPanelComponent,
  ],
  templateUrl: './debugger-page.component.html',
  styleUrl: './debugger-page.component.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class DebuggerPageComponent {
  /** Route parameter, bound by `withComponentInputBinding()`. */
  readonly sessionId = input.required<string>();

  protected readonly sessions = inject(SessionStore);
  protected readonly debug = inject(DebuggerStore);
  protected readonly openError = signal<string | null>(null);

  constructor() {
    effect(() => {
      const id = this.sessionId();
      if (!id) return;
      this.openError.set(null);
      this.sessions.open(id).catch(() => {
        this.openError.set(
          `Session ${id} is not open and could not be rebuilt. Start a new one.`,
        );
      });
    });
  }
}
