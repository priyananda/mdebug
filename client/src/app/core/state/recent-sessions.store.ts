import { Injectable, signal } from '@angular/core';

import { SessionConfig } from '../models/session.model';

export interface RecentSession {
  id: string;
  prompt: string;
  createdAt: string;
  config: SessionConfig;
}

const KEY = 'mdebug.recent-sessions';
const MAX = 12;

/**
 * The landing page's "continue an existing session" list, and the fallback used
 * to rehydrate a session after a reload.
 *
 * localStorage is per-viewer and can throw (private browsing, blocked storage),
 * so every access is guarded and the app renders correctly when it comes back
 * empty.
 */
@Injectable({ providedIn: 'root' })
export class RecentSessionsStore {
  private readonly _all = signal<RecentSession[]>(read());
  readonly all = this._all.asReadonly();

  remember(entry: RecentSession): void {
    const next = [entry, ...this._all().filter((s) => s.id !== entry.id)].slice(0, MAX);
    this._all.set(next);
    write(next);
  }

  find(id: string): RecentSession | undefined {
    return this._all().find((s) => s.id === id);
  }

  forget(id: string): void {
    const next = this._all().filter((s) => s.id !== id);
    this._all.set(next);
    write(next);
  }

  clear(): void {
    this._all.set([]);
    write([]);
  }
}

function read(): RecentSession[] {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as RecentSession[]) : [];
  } catch {
    return [];
  }
}

function write(entries: RecentSession[]): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(entries));
  } catch {
    // Storage is unavailable or full. The list is a convenience, not state we
    // depend on, so carrying on is the right behaviour.
  }
}
