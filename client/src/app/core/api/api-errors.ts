export type ApiErrorKind =
  /** The request never reached the server. */
  | 'transport'
  /** The server answered, but not in a shape we understand. */
  | 'protocol'
  /** The server answered with an error. */
  | 'server'
  /** We abandoned the request — a stale tensor fetch, usually. */
  | 'cancelled'
  /** The session id is not known to the server. */
  | 'session_not_found';

export class ApiError extends Error {
  constructor(
    readonly kind: ApiErrorKind,
    message: string,
    readonly code?: string,
    readonly detail?: unknown,
  ) {
    super(message);
    this.name = 'ApiError';
  }

  static transport(message: string, detail?: unknown): ApiError {
    return new ApiError('transport', message, undefined, detail);
  }

  static cancelled(): ApiError {
    return new ApiError('cancelled', 'Request cancelled');
  }

  static notFound(sessionId: string): ApiError {
    return new ApiError('session_not_found', `Session ${sessionId} is no longer on the server`);
  }

  /** Whether the UI should offer a retry rather than a fresh start. */
  get retryable(): boolean {
    return this.kind === 'transport' || this.kind === 'server';
  }
}

export function isCancelled(e: unknown): boolean {
  return e instanceof ApiError && e.kind === 'cancelled';
}
