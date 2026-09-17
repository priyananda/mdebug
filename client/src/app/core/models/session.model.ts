import { ModelInfo } from './model-info.model';
import { Breakpoint } from './breakpoint.model';
import { HaltPosition } from './run-state.model';

export type SessionStatus =
  | 'idle'
  | 'running'
  | 'halted'
  | 'finished'
  | 'error'
  | 'closed';

export type SamplingMode = 'greedy' | 'temperature' | 'top_k';

export interface SessionConfig {
  prompt: string;
  maxNewTokens: number;
  samplingMode: SamplingMode;
  temperature?: number;
  topK?: number;
  /** Makes sampling reproducible; the mock engine also derives tensors from it. */
  seed?: number;
  /** When false the server skips storing attention, which is the memory hog. */
  captureAttention: boolean;
}

export interface Token {
  id: number;
  /** The raw piece as the tokenizer produced it, e.g. 'Ġhappiness'. */
  text: string;
  /** Display-safe: leading space as a middot, newline as a return glyph. */
  display: string;
  /** Absolute position in the sequence. */
  position: number;
  isSpecial: boolean;
  origin: 'prompt' | 'generated';
}

export interface SessionInfo {
  id: string;
  createdAt: string;
  updatedAt: string;
  status: SessionStatus;
  config: SessionConfig;
  model: ModelInfo;
  promptTokens: Token[];
  generatedTokens: Token[];
  /** Index of the decode step in progress; -1 before the first. */
  currentStep: number;
  halt: HaltPosition | null;
  breakpoints: Breakpoint[];
}

/** The lightweight form listed on the landing page. */
export interface SessionSummary {
  id: string;
  createdAt: string;
  status: SessionStatus;
  promptPreview: string;
  tokensGenerated: number;
}

export interface CreateSessionRequest {
  config: SessionConfig;
}

export const DEFAULT_SESSION_CONFIG: SessionConfig = {
  prompt: '',
  maxNewTokens: 32,
  samplingMode: 'temperature',
  temperature: 0.8,
  topK: 40,
  captureAttention: true,
};
