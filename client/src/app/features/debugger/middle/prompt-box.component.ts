import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  computed,
  effect,
  inject,
  signal,
} from '@angular/core';

import { InferenceApi } from '../../../core/api/inference-api';
import { SamplingMode, Token } from '../../../core/models/session.model';
import { DebuggerStore } from '../../../core/state/debugger.store';
import { SessionStore } from '../../../core/state/session.store';

/** Long enough that typing is not a fetch storm, short enough to feel live. */
const TOKENIZE_DEBOUNCE_MS = 150;

@Component({
  selector: 'mdbg-prompt-box',
  templateUrl: './prompt-box.component.html',
  styleUrl: './prompt-box.component.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class PromptBoxComponent {
  private readonly api = inject(InferenceApi);
  private readonly sessions = inject(SessionStore);
  protected readonly debug = inject(DebuggerStore);

  protected readonly draft = this.sessions.draft;
  protected readonly model = this.sessions.model;

  protected readonly preview = signal<Token[]>([]);
  protected readonly showParams = signal(false);

  protected readonly overLimit = computed(() => {
    const max = this.model()?.limits.maxPromptTokens ?? Infinity;
    return this.preview().length > max;
  });

  /** Locked while the engine is mid-run: changing the prompt then is meaningless. */
  protected readonly locked = computed(() => this.debug.status() !== 'idle' && this.debug.status() !== 'finished');

  constructor() {
    const destroy = inject(DestroyRef);
    let timer: ReturnType<typeof setTimeout> | undefined;
    let generation = 0;

    effect(() => {
      const text = this.draft().prompt;
      clearTimeout(timer);
      const mine = ++generation;
      timer = setTimeout(async () => {
        const tokens = await this.api.tokenize(text);
        // Drop the result if the user has typed again since.
        if (mine === generation) this.preview.set(tokens);
      }, TOKENIZE_DEBOUNCE_MS);
    });

    destroy.onDestroy(() => clearTimeout(timer));
  }

  protected onPrompt(event: Event): void {
    this.sessions.patchDraft({ prompt: (event.target as HTMLTextAreaElement).value });
  }

  protected onMaxTokens(event: Event): void {
    const value = Number((event.target as HTMLInputElement).value);
    if (Number.isFinite(value)) this.sessions.patchDraft({ maxNewTokens: Math.max(1, value) });
  }

  protected onSamplingMode(event: Event): void {
    this.sessions.patchDraft({
      samplingMode: (event.target as HTMLSelectElement).value as SamplingMode,
    });
  }

  protected onTemperature(event: Event): void {
    const temperature = Number((event.target as HTMLInputElement).value);
    this.sessions.patchDraft({ temperature });
    // Sampling parameters are legal to change while halted, so push them live.
    if (this.debug.status() === 'halted') this.debug.patchConfig({ temperature });
  }
}
