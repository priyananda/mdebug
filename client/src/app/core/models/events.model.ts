import { Breakpoint } from './breakpoint.model';
import { SessionConfig, SessionInfo, Token } from './session.model';
import { StageId } from './pipeline.model';
import { HaltPayload, HaltPosition } from './run-state.model';
import { TopKLogits } from './tensors.model';

/**
 * The WebSocket protocol.
 *
 * This file is the single point where the mock engine and the real transport
 * must agree. DebuggerStore reduces `ServerEvent` in one switch, so anything
 * that produces this union drives the UI identically.
 */

export interface WsEnvelope<TType extends string, TPayload> {
  v: 1;
  /** Unique per message; a command's id comes back as the reply's `replyTo`. */
  id: string;
  /** Epoch milliseconds at the sender. */
  ts: number;
  type: TType;
  payload: TPayload;
  replyTo?: string;
}

// --- client -> server -------------------------------------------------------

export type ClientCommand =
  | { type: 'start'; payload: { config?: Partial<SessionConfig> } }
  | { type: 'continue'; payload: Record<string, never> }
  | { type: 'step'; payload: { count?: number } }
  | { type: 'stop'; payload: Record<string, never> }
  | { type: 'set_breakpoints'; payload: { breakpoints: Breakpoint[] } }
  | { type: 'run_to_cursor'; payload: { stageId: StageId; step?: number } }
  | { type: 'patch_config'; payload: Partial<SessionConfig> }
  | { type: 'ping'; payload: Record<string, never> };

export type ClientCommandType = ClientCommand['type'];

// --- server -> client -------------------------------------------------------

export type ServerEvent =
  /** Sent immediately on connect, and after any resync. Full state. */
  | { type: 'session_state'; payload: SessionInfo }
  | { type: 'run_started'; payload: { step: number } }
  /**
   * High frequency (~45 per token). Coalesced by the server to <= 30/sec.
   * Used only to animate the program counter, so dropped frames are invisible.
   * A stage that causes a halt is never dropped.
   */
  | { type: 'stage_entered'; payload: { step: number; stageId: StageId; sequenceLength: number } }
  | { type: 'token_emitted'; payload: { step: number; token: Token; topK?: TopKLogits } }
  | { type: 'halted'; payload: HaltPayload }
  | { type: 'resumed'; payload: { from: HaltPosition } }
  | {
      type: 'finished';
      payload: { reason: 'max_tokens' | 'eos' | 'stopped'; totalSteps: number; text: string };
    }
  | { type: 'breakpoints_changed'; payload: { breakpoints: Breakpoint[] } }
  | { type: 'error'; payload: { code: string; message: string; fatal: boolean; detail?: unknown } }
  | { type: 'pong'; payload: Record<string, never> };

export type ServerEventType = ServerEvent['type'];

export type ServerEventOf<T extends ServerEventType> = Extract<ServerEvent, { type: T }>;

/** Maximum `stage_entered` events per second the server may emit. */
export const STAGE_EVENT_RATE_LIMIT = 30;
