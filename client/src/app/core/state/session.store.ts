import { Injectable, computed, inject, signal } from '@angular/core';

import { ApiError } from '../api/api-errors';
import { InferenceApi } from '../api/inference-api';
import { MockInferenceApi } from '../api/mock/mock-inference-api';
import { ModelInfo } from '../models/model-info.model';
import { DEFAULT_SESSION_CONFIG, SessionConfig, SessionInfo } from '../models/session.model';
import { RecentSessionsStore } from './recent-sessions.store';

export type ConnectionState = 'disconnected' | 'connecting' | 'open' | 'reconnecting';

/**
 * Owns the model description and the current session's identity. Run state
 * lives in DebuggerStore; this store is only about which session we are looking
 * at and whether we can talk to it.
 */
@Injectable({ providedIn: 'root' })
export class SessionStore {
  private readonly api = inject(InferenceApi);
  private readonly recents = inject(RecentSessionsStore);

  private readonly _model = signal<ModelInfo | null>(null);
  private readonly _session = signal<SessionInfo | null>(null);
  private readonly _connection = signal<ConnectionState>('disconnected');
  private readonly _error = signal<ApiError | null>(null);

  readonly model = this._model.asReadonly();
  readonly session = this._session.asReadonly();
  readonly connection = this._connection.asReadonly();
  readonly error = this._error.asReadonly();

  /**
   * The run configuration as currently edited in the prompt box. Kept separate
   * from the session's committed config: the user can retype a prompt while the
   * previous run's state is still on screen, and nothing changes until Start.
   */
  private readonly _draft = signal<SessionConfig>(DEFAULT_SESSION_CONFIG);
  readonly draft = this._draft.asReadonly();

  readonly sessionId = computed(() => this._session()?.id ?? null);
  readonly ready = computed(
    () => this._model() !== null && this._session() !== null && this._connection() === 'open',
  );

  async loadModel(): Promise<ModelInfo> {
    const existing = this._model();
    if (existing) return existing;
    const model = await this.api.getModel();
    this._model.set(model);
    return model;
  }

  async create(config: SessionConfig): Promise<SessionInfo> {
    await this.loadModel();
    const session = await this.api.createSession({ config });
    this.recents.remember({
      id: session.id,
      prompt: config.prompt,
      createdAt: session.createdAt,
      config,
    });
    this.adopt(session);
    return session;
  }

  /**
   * Opens an existing session by id.
   *
   * After a page reload the id survives in the URL but the mock's in-memory
   * engine does not, so we fall back to rebuilding it from the config we
   * remembered locally. Because the session id seeds every generated tensor,
   * the rebuilt session is identical to the original.
   */
  async open(sessionId: string): Promise<SessionInfo> {
    this._connection.set('connecting');
    this._error.set(null);

    try {
      await this.loadModel();
      const session = await this.api.getSession(sessionId);
      this.adopt(session);
      return session;
    } catch (e) {
      const remembered = this.recents.find(sessionId);
      if (remembered && this.api instanceof MockInferenceApi) {
        const session = await this.api.adoptSession(sessionId, remembered.config);
        this.adopt(session);
        return session;
      }
      this._connection.set('disconnected');
      this._error.set(
        e instanceof ApiError ? e : ApiError.transport('Could not open the session', e),
      );
      throw e;
    }
  }

  patchDraft(patch: Partial<SessionConfig>): void {
    this._draft.update((d) => ({ ...d, ...patch }));
  }

  private adopt(session: SessionInfo): void {
    this._session.set(session);
    this._draft.set(session.config);
    this.api.connect(session.id);
    this._connection.set('open');
  }

  /** Called by DebuggerStore when a `session_state` event arrives. */
  sync(session: SessionInfo): void {
    this._session.set(session);
  }

  async close(): Promise<void> {
    const id = this.sessionId();
    if (id) await this.api.closeSession(id);
    this._session.set(null);
    this._connection.set('disconnected');
  }
}
