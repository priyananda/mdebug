import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

import { environment } from '../../../../environments/environment';
import { ClientCommand, ServerEvent } from '../../models/events.model';
import { ModelInfo } from '../../models/model-info.model';
import {
  CreateSessionRequest,
  SessionInfo,
  SessionSummary,
  Token,
} from '../../models/session.model';
import {
  AttentionTile,
  KvSnapshot,
  ResidualBlock,
  ResidualStage,
  ResidualVector,
  TopKLogits,
} from '../../models/tensors.model';
import { ApiError } from '../api-errors';
import { InferenceApi } from '../inference-api';
import { WsTransport } from './ws-transport';

/**
 * The real backend, talking to the Python server described in
 * docs/api-contract.md.
 *
 * The server does not exist yet. This is written against the contract so the
 * seam is demonstrably real: swapping `environment.useMock` should change
 * nothing about how the UI behaves, only where the events come from.
 */
@Injectable()
export class HttpInferenceApi extends InferenceApi {
  private readonly transport = new WsTransport();
  private readonly base = environment.apiBase.replace(/\/$/, '');

  readonly events$: Observable<ServerEvent> = this.transport.events$;
  readonly phase$ = this.transport.phase$;

  async getModel(): Promise<ModelInfo> {
    return this.get<ModelInfo>('/api/model');
  }

  async createSession(request: CreateSessionRequest): Promise<SessionInfo> {
    return this.post<SessionInfo>('/api/sessions', request);
  }

  async listSessions(): Promise<SessionSummary[]> {
    return this.get<SessionSummary[]>('/api/sessions');
  }

  async getSession(sessionId: string): Promise<SessionInfo> {
    return this.get<SessionInfo>(`/api/sessions/${encodeURIComponent(sessionId)}`);
  }

  async closeSession(sessionId: string): Promise<void> {
    await this.request(`/api/sessions/${encodeURIComponent(sessionId)}`, { method: 'DELETE' });
  }

  async tokenize(text: string): Promise<Token[]> {
    return this.post<Token[]>('/api/tokenize', { text });
  }

  connect(sessionId: string): void {
    const ws = this.base.replace(/^http/, 'ws');
    this.transport.connect(`${ws}/api/sessions/${encodeURIComponent(sessionId)}/ws`);
  }

  disconnect(): void {
    this.transport.disconnect();
  }

  send(command: ClientCommand): void {
    this.transport.send(command);
  }

  getAttentionTile(
    sessionId: string,
    step: number,
    layer: number,
    head: number,
    signal?: AbortSignal,
  ): Promise<AttentionTile> {
    return this.get<AttentionTile>(
      `${this.steps(sessionId, step)}/attention?layer=${layer}&head=${head}`,
      signal,
    );
  }

  getResidual(
    sessionId: string,
    step: number,
    layer: number,
    stage: ResidualStage,
    position: number,
    signal?: AbortSignal,
  ): Promise<ResidualVector> {
    return this.get<ResidualVector>(
      `${this.steps(sessionId, step)}/residual?layer=${layer}&stage=${stage}&position=${position}`,
      signal,
    );
  }

  getResidualBlock(
    sessionId: string,
    step: number,
    layer: number,
    stage: ResidualStage,
    signal?: AbortSignal,
  ): Promise<ResidualBlock> {
    return this.get<ResidualBlock>(
      `${this.steps(sessionId, step)}/residual?layer=${layer}&stage=${stage}`,
      signal,
    );
  }

  getKv(sessionId: string, step: number, signal?: AbortSignal): Promise<KvSnapshot> {
    return this.get<KvSnapshot>(`${this.steps(sessionId, step)}/kv`, signal);
  }

  getLogits(
    sessionId: string,
    step: number,
    k: number,
    signal?: AbortSignal,
  ): Promise<TopKLogits> {
    return this.get<TopKLogits>(`${this.steps(sessionId, step)}/logits?k=${k}`, signal);
  }

  private steps(sessionId: string, step: number): string {
    return `/api/sessions/${encodeURIComponent(sessionId)}/steps/${step}`;
  }

  private get<T>(path: string, signal?: AbortSignal): Promise<T> {
    return this.request<T>(path, { signal });
  }

  private post<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
    return this.request<T>(path, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
      signal,
    });
  }

  private async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    let response: Response;
    try {
      response = await fetch(`${this.base}${path}`, init);
    } catch (e) {
      if (init.signal?.aborted) throw ApiError.cancelled();
      throw ApiError.transport(`Cannot reach the inference server at ${this.base}`, e);
    }

    if (response.status === 404) {
      const match = /\/api\/sessions\/([^/?]+)/.exec(path);
      if (match) throw ApiError.notFound(decodeURIComponent(match[1]));
    }

    if (!response.ok) {
      const detail = await response.text().catch(() => '');
      throw new ApiError(
        'server',
        `${response.status} ${response.statusText}`,
        String(response.status),
        detail,
      );
    }

    if (response.status === 204) return undefined as T;

    try {
      return (await response.json()) as T;
    } catch (e) {
      throw new ApiError('protocol', 'The server sent a response we could not parse', undefined, e);
    }
  }
}
