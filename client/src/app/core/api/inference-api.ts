import { Observable } from 'rxjs';

import { ClientCommand, ServerEvent } from '../models/events.model';
import { ModelInfo } from '../models/model-info.model';
import {
  CreateSessionRequest,
  SessionInfo,
  SessionSummary,
  Token,
} from '../models/session.model';
import {
  AttentionTile,
  KvSnapshot,
  ResidualBlock,
  ResidualStage,
  ResidualVector,
  TopKLogits,
} from '../models/tensors.model';

/**
 * The seam between the UI and whatever is actually running inference.
 *
 * Two implementations: `MockInferenceApi`, which simulates the whole thing in
 * the browser, and `HttpInferenceApi`, which talks to the Python server. The UI
 * depends only on this class, and `DebuggerStore` reduces `events$` the same way
 * regardless of which one is provided.
 *
 * An abstract class rather than an interface + InjectionToken: it is its own DI
 * token, which is less ceremony for the same result.
 */
export abstract class InferenceApi {
  /** Fetched once at startup. Drives the pipeline graph and every label. */
  abstract getModel(): Promise<ModelInfo>;

  // --- session lifecycle ---------------------------------------------------

  abstract createSession(request: CreateSessionRequest): Promise<SessionInfo>;
  abstract listSessions(): Promise<SessionSummary[]>;
  abstract getSession(sessionId: string): Promise<SessionInfo>;
  abstract closeSession(sessionId: string): Promise<void>;

  /** Live token preview for the prompt box. Does not need a running session. */
  abstract tokenize(text: string): Promise<Token[]>;

  // --- control flow --------------------------------------------------------

  /**
   * Opens the control channel for a session. Idempotent: connecting to the
   * session that is already connected is a no-op.
   */
  abstract connect(sessionId: string): void;
  abstract disconnect(): void;

  /** Every event for the connected session. Never completes, never errors. */
  abstract readonly events$: Observable<ServerEvent>;

  abstract send(command: ClientCommand): void;

  // --- on-demand tensor retrieval ------------------------------------------
  //
  // These are fired on hover and selection changes, so they must be safe to
  // call concurrently and cheap to abort.

  abstract getAttentionTile(
    sessionId: string,
    step: number,
    layer: number,
    head: number,
    signal?: AbortSignal,
  ): Promise<AttentionTile>;

  abstract getResidual(
    sessionId: string,
    step: number,
    layer: number,
    stage: ResidualStage,
    position: number,
    signal?: AbortSignal,
  ): Promise<ResidualVector>;

  abstract getResidualBlock(
    sessionId: string,
    step: number,
    layer: number,
    stage: ResidualStage,
    signal?: AbortSignal,
  ): Promise<ResidualBlock>;

  abstract getKv(sessionId: string, step: number, signal?: AbortSignal): Promise<KvSnapshot>;

  abstract getLogits(
    sessionId: string,
    step: number,
    k: number,
    signal?: AbortSignal,
  ): Promise<TopKLogits>;
}
