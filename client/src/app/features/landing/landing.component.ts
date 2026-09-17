import { ChangeDetectionStrategy, Component, inject, signal } from '@angular/core';
import { Router } from '@angular/router';

import { environment } from '../../../environments/environment';
import { DEFAULT_SESSION_CONFIG } from '../../core/models/session.model';
import { RecentSessionsStore } from '../../core/state/recent-sessions.store';
import { SessionStore } from '../../core/state/session.store';

/** Something in the training corpus's register, so the first run reads sensibly. */
const STARTER_PROMPT = 'The key to happiness is';

@Component({
  selector: 'mdbg-landing',
  templateUrl: './landing.component.html',
  styleUrl: './landing.component.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class LandingComponent {
  private readonly router = inject(Router);
  private readonly sessions = inject(SessionStore);
  protected readonly recents = inject(RecentSessionsStore);

  protected readonly backend = environment.useMock
    ? 'simulated engine, real tokenizer'
    : environment.apiBase;
  protected readonly creating = signal(false);
  protected readonly error = signal<string | null>(null);

  protected async newSession(): Promise<void> {
    if (this.creating()) return;
    this.creating.set(true);
    this.error.set(null);
    try {
      const session = await this.sessions.create({
        ...DEFAULT_SESSION_CONFIG,
        prompt: STARTER_PROMPT,
      });
      await this.router.navigate(['/debug', session.id]);
    } catch (e) {
      this.error.set(e instanceof Error ? e.message : 'Could not start a session');
      this.creating.set(false);
    }
  }

  protected resume(id: string): void {
    void this.router.navigate(['/debug', id]);
  }

  protected forget(event: Event, id: string): void {
    event.stopPropagation();
    this.recents.forget(id);
  }

  protected when(iso: string): string {
    const delta = Date.now() - new Date(iso).getTime();
    const minutes = Math.round(delta / 60000);
    if (minutes < 1) return 'just now';
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.round(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    return `${Math.round(hours / 24)}d ago`;
  }
}
