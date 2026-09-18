import { Injectable } from '@angular/core';
import { Observable, Subject } from 'rxjs';

import { ClientCommand, ServerEvent } from '../../models/events.model';
import { ModelInfo } from '../../models/model-info.model';
import { DEFAULT_STAGE_DESCRIPTORS } from '../../models/pipeline.model';
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
import { encodeArray } from '../../util/encoding';
import { ApiError } from '../api-errors';
import { InferenceApi } from '../inference-api';
import { ByteLevelBpeTokenizer, loadTokenizer } from './byte-bpe';
import { attentionTile, residualVector, statsOf } from './fake-tensors';
import { MockEngine } from './mock-engine';
import { rngFor } from './rng';

/**
 * Describes the checked-in model in server/src/config.py, including the places
 * where it does not do what its names suggest. These notes are surfaced in the
 * UI: the audience reads labels literally, so the labels have to be honest.
 */
const MOCK_MODEL: ModelInfo = {
  name: 'QwenModel (from scratch, 20L/8H/512d)',
  numLayers: 20,
  numHeads: 8,
  hiddenSize: 512,
  headDim: 64,
  intermediateSize: 1536,
  vocabSize: 30000,
  tokenizerVocabSize: 6258,
  maxPositionEmbeddings: 4096,
  ropePct: 0.25,
  ropeDim: 16,
  // server/infer.py re-runs the full forward pass over the whole prefix for
  // every token. There is no cache, so the KV panel badges itself.
  hasKvCache: false,
  capturesAttention: true,
  limits: {
    maxPromptTokens: 128,
    maxNewTokens: 64,
    maxTotalTokens: 192,
    attentionWarnThreshold: 512,
  },
  stages: [...DEFAULT_STAGE_DESCRIPTORS],
  notes: [
    'GroupedQueryAttention is plain multi-head attention - there is no KV-head grouping.',
    'RMSNorm divides by the L2 norm rather than sqrt(mean(x^2)), off by a factor of sqrt(512).',
    'The model adds learned absolute position embeddings AND applies RoPE inside attention.',
    'vocab_size is 30000 but the trained tokenizer has 6258 entries, so ~24k logits are unreachable.',
    'There is no KV cache: the full prefix is recomputed every step.',
  ],
};

/** Simulated network latency for on-demand tensor fetches. */
const FETCH_LATENCY_MS: [number, number] = [80, 200];

/**
 * The mock backend.
 *
 * Not a stub — this is what the whole UI is demoed on, so it produces data that
 * looks like transformer data and behaves with realistic timing, including
 * latency on tensor fetches so that loading and cancellation paths are actually
 * exercised rather than merely written.
 */
@Injectable()
export class MockInferenceApi extends InferenceApi {
  private readonly relay = new Subject<ServerEvent>();
  readonly events$: Observable<ServerEvent> = this.relay.asObservable();

  private tokenizer: ByteLevelBpeTokenizer | null = null;
  private readonly sessions = new Map<string, MockEngine>();
  private connected: string | null = null;
  private subscription: { unsubscribe(): void } | null = null;

  /** Dev knobs, surfaced in the UI's debug panel. */
  failureRate = 0;

  async getModel(): Promise<ModelInfo> {
    return MOCK_MODEL;
  }

  private async tok(): Promise<ByteLevelBpeTokenizer> {
    this.tokenizer ??= await loadTokenizer();
    return this.tokenizer;
  }

  async tokenize(text: string): Promise<Token[]> {
    const tok = await this.tok();
    return tok.encode(text).map((t, i) => ({
      id: t.id,
      text: t.text,
      display: tok.display(t.text),
      position: i,
      isSpecial: tok.isSpecial(t.id),
      origin: 'prompt' as const,
    }));
  }

  async createSession(request: CreateSessionRequest): Promise<SessionInfo> {
    const tok = await this.tok();
    const id = newSessionId();
    const now = new Date().toISOString();
    const session: SessionInfo = {
      id,
      createdAt: now,
      updatedAt: now,
      status: 'idle',
      config: request.config,
      model: MOCK_MODEL,
      promptTokens: await this.tokenize(request.config.prompt),
      generatedTokens: [],
      currentStep: 0,
      halt: null,
      breakpoints: [],
    };
    this.sessions.set(id, new MockEngine(session, MOCK_MODEL, tok));
    return session;
  }

  async listSessions(): Promise<SessionSummary[]> {
    return [...this.sessions.values()].map((engine) => {
      const s = engine.state;
      return {
        id: s.id,
        createdAt: s.createdAt,
        status: s.status,
        promptPreview: s.config.prompt.slice(0, 60),
        tokensGenerated: s.generatedTokens.length,
      };
    });
  }

  async getSession(sessionId: string): Promise<SessionInfo> {
    const engine = this.sessions.get(sessionId);
    if (!engine) throw ApiError.notFound(sessionId);
    return engine.state;
  }

  /**
   * Adopts a session id the client already has — after a reload, the id lives
   * in the URL but the in-memory engine is gone. A real server would rehydrate
   * from its own store; the mock rebuilds an idle session with the same id so
   * that seeds, and therefore all generated tensors, stay identical.
   */
  async adoptSession(sessionId: string, config: CreateSessionRequest['config']): Promise<SessionInfo> {
    const tok = await this.tok();
    const now = new Date().toISOString();
    const session: SessionInfo = {
      id: sessionId,
      createdAt: now,
      updatedAt: now,
      status: 'idle',
      config,
      model: MOCK_MODEL,
      promptTokens: await this.tokenize(config.prompt),
      generatedTokens: [],
      currentStep: 0,
      halt: null,
      breakpoints: [],
    };
    this.sessions.set(sessionId, new MockEngine(session, MOCK_MODEL, tok));
    return session;
  }

  async closeSession(sessionId: string): Promise<void> {
    this.sessions.get(sessionId)?.stop();
    this.sessions.delete(sessionId);
    if (this.connected === sessionId) this.disconnect();
  }

  // --- control channel ------------------------------------------------------

  connect(sessionId: string): void {
    if (this.connected === sessionId) return;
    this.disconnect();

    const engine = this.sessions.get(sessionId);
    if (!engine) {
      this.relay.next({
        type: 'error',
        payload: {
          code: 'session_not_found',
          message: `Session ${sessionId} is not open`,
          fatal: true,
        },
      });
      return;
    }

    this.connected = sessionId;
    this.subscription = engine.events$.subscribe((e) => this.relay.next(e));
    engine.emitSessionState();
  }

  disconnect(): void {
    this.subscription?.unsubscribe();
    this.subscription = null;
    this.connected = null;
  }

  send(command: ClientCommand): void {
    const engine = this.connected ? this.sessions.get(this.connected) : undefined;
    if (!engine) return;

    switch (command.type) {
      case 'start':
        engine.start(command.payload.config);
        break;
      case 'continue':
        engine.continueRun();
        break;
      case 'step':
        engine.stepOver(command.payload.count ?? 1);
        break;
      case 'stop':
        engine.stop();
        break;
      case 'set_breakpoints':
        engine.setBreakpoints(command.payload.breakpoints);
        break;
      case 'run_to_cursor':
        engine.runToCursor(command.payload.stageId, command.payload.step);
        break;
      case 'patch_config':
        engine.patchConfig(command.payload);
        break;
      case 'ping':
        this.relay.next({ type: 'pong', payload: {} });
        break;
    }
  }

  /** Engine handle for the session currently connected, if any. */
  get engine(): MockEngine | undefined {
    return this.connected ? this.sessions.get(this.connected) : undefined;
  }

  // --- on-demand tensors ----------------------------------------------------

  async getAttentionTile(
    sessionId: string,
    step: number,
    layer: number,
    head: number,
    signal?: AbortSignal,
  ): Promise<AttentionTile> {
    await this.latency(signal);
    const session = await this.getSession(sessionId);
    const t = session.promptTokens.length + step;

    const tile = attentionTile(sessionId, step, layer, head, MOCK_MODEL.numLayers, t);
    return {
      step,
      layer,
      head,
      sequenceLength: t,
      weights: encodeArray(tile.weights, {
        shape: [t, t],
        dtype: 'u8',
        transform: 'sqrt',
        layout: 'causal_lower',
      }),
      stats: {
        maxWeight: tile.maxWeight,
        meanEntropy: tile.meanEntropy,
        sinkMass: tile.sinkMass,
      },
    };
  }

  async getResidual(
    sessionId: string,
    step: number,
    layer: number,
    stage: ResidualStage,
    position: number,
    signal?: AbortSignal,
  ): Promise<ResidualVector> {
    await this.latency(signal);
    const values = residualVector(sessionId, position, layer, MOCK_MODEL.hiddenSize);
    return {
      step,
      layer,
      position,
      stage,
      values: encodeArray(values, { shape: [MOCK_MODEL.hiddenSize] }),
      stats: statsOf(values),
    };
  }

  async getResidualBlock(
    sessionId: string,
    step: number,
    layer: number,
    stage: ResidualStage,
    signal?: AbortSignal,
  ): Promise<ResidualBlock> {
    await this.latency(signal);
    const session = await this.getSession(sessionId);
    const t = session.promptTokens.length + step;
    const block = new Float32Array(t * MOCK_MODEL.hiddenSize);
    for (let p = 0; p < t; p++) {
      block.set(residualVector(sessionId, p, layer, MOCK_MODEL.hiddenSize), p * MOCK_MODEL.hiddenSize);
    }
    return {
      step,
      layer,
      stage,
      values: encodeArray(block, { shape: [t, MOCK_MODEL.hiddenSize] }),
    };
  }

  /**
   * The KV snapshot and the top-k logits both ride along with every `halted`
   * event, so the client never needs to pull them. They exist on the interface
   * for the real transport, which does need them after a reconnect.
   */
  async getKv(_sessionId: string, _step: number, signal?: AbortSignal): Promise<KvSnapshot> {
    await this.latency(signal);
    throw new ApiError('protocol', 'KV snapshots arrive with the halt payload in mock mode');
  }

  async getLogits(
    _sessionId: string,
    _step: number,
    _k: number,
    signal?: AbortSignal,
  ): Promise<TopKLogits> {
    await this.latency(signal);
    throw new ApiError('protocol', 'Logits arrive with the halt payload in mock mode');
  }

  /**
   * Simulated latency, plus optional injected failure. Both exist so that the
   * loading states and request cancellation are exercised for real rather than
   * being theoretical code paths.
   */
  private latency(signal?: AbortSignal): Promise<void> {
    const [lo, hi] = FETCH_LATENCY_MS;
    const ms = lo + Math.random() * (hi - lo);
    return new Promise((resolve, reject) => {
      if (signal?.aborted) return reject(ApiError.cancelled());
      const id = setTimeout(() => {
        signal?.removeEventListener('abort', onAbort);
        if (this.failureRate > 0 && Math.random() < this.failureRate) {
          reject(ApiError.transport('Simulated fetch failure'));
        } else {
          resolve();
        }
      }, ms);
      const onAbort = () => {
        clearTimeout(id);
        reject(ApiError.cancelled());
      };
      signal?.addEventListener('abort', onAbort, { once: true });
    });
  }
}

function newSessionId(): string {
  const rng = rngFor('session', Date.now(), Math.random());
  let out = '';
  const alphabet = 'abcdefghijkmnpqrstuvwxyz23456789';
  for (let i = 0; i < 8; i++) out += alphabet[Math.floor(rng() * alphabet.length)];
  return out;
}
