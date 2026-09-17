import { Observable, Subject } from 'rxjs';

import { ClientCommand, ServerEvent, WsEnvelope } from '../../models/events.model';

export type ConnectionPhase = 'closed' | 'connecting' | 'open' | 'reconnecting';

const HEARTBEAT_MS = 20_000;
const BACKOFF_START_MS = 500;
const BACKOFF_MAX_MS = 15_000;

/**
 * The control channel: one WebSocket per session, carrying commands out and
 * events in.
 *
 * Data never travels over this socket — tensors come over plain HTTP so they
 * stay cacheable and independently cancellable. That split is also why a
 * dropped socket is survivable: the run lives on the server, so reconnecting
 * and asking for `session_state` restores everything.
 */
export class WsTransport {
  private socket: WebSocket | null = null;
  private url: string | null = null;
  private heartbeat: ReturnType<typeof setInterval> | null = null;
  private retry: ReturnType<typeof setTimeout> | null = null;
  private backoff = BACKOFF_START_MS;
  private closedByUs = false;
  /** Commands issued before the socket opened, replayed on connect. */
  private queue: ClientCommand[] = [];

  private readonly events = new Subject<ServerEvent>();
  private readonly phases = new Subject<ConnectionPhase>();

  readonly events$: Observable<ServerEvent> = this.events.asObservable();
  readonly phase$: Observable<ConnectionPhase> = this.phases.asObservable();

  connect(url: string): void {
    if (this.url === url && this.socket?.readyState === WebSocket.OPEN) return;
    this.disconnect();
    this.url = url;
    this.closedByUs = false;
    this.open('connecting');
  }

  private open(phase: ConnectionPhase): void {
    if (!this.url) return;
    this.phases.next(phase);

    const socket = new WebSocket(this.url);
    this.socket = socket;

    socket.onopen = () => {
      this.backoff = BACKOFF_START_MS;
      this.phases.next('open');
      for (const command of this.queue.splice(0)) this.send(command);
      this.heartbeat = setInterval(() => this.send({ type: 'ping', payload: {} }), HEARTBEAT_MS);
    };

    socket.onmessage = (message) => {
      const event = parseEvent(message.data);
      if (event) this.events.next(event);
    };

    socket.onerror = () => {
      // `onclose` always follows, and that is where reconnection is handled.
    };

    socket.onclose = () => {
      this.stopHeartbeat();
      this.socket = null;
      if (this.closedByUs) {
        this.phases.next('closed');
        return;
      }
      this.phases.next('reconnecting');
      this.retry = setTimeout(() => this.open('reconnecting'), this.backoff);
      this.backoff = Math.min(BACKOFF_MAX_MS, this.backoff * 2);
    };
  }

  send(command: ClientCommand): void {
    if (this.socket?.readyState !== WebSocket.OPEN) {
      // Heartbeats are not worth replaying; anything else is.
      if (command.type !== 'ping') this.queue.push(command);
      return;
    }
    const envelope: WsEnvelope<string, unknown> = {
      v: 1,
      id: crypto.randomUUID(),
      ts: Date.now(),
      type: command.type,
      payload: command.payload,
    };
    this.socket.send(JSON.stringify(envelope));
  }

  disconnect(): void {
    this.closedByUs = true;
    this.stopHeartbeat();
    if (this.retry) {
      clearTimeout(this.retry);
      this.retry = null;
    }
    this.queue = [];
    this.socket?.close();
    this.socket = null;
    this.url = null;
  }

  private stopHeartbeat(): void {
    if (this.heartbeat) {
      clearInterval(this.heartbeat);
      this.heartbeat = null;
    }
  }
}

/**
 * Turns a raw frame into a `ServerEvent`, or null if it is not one we
 * understand. A malformed frame must not take down the stream.
 */
function parseEvent(data: unknown): ServerEvent | null {
  if (typeof data !== 'string') return null;
  try {
    const envelope = JSON.parse(data) as Partial<WsEnvelope<string, unknown>>;
    if (envelope.v !== 1 || typeof envelope.type !== 'string') return null;
    return { type: envelope.type, payload: envelope.payload } as ServerEvent;
  } catch {
    return null;
  }
}
