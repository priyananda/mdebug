import { StageId, StageKind } from './pipeline.model';

/**
 * Breakpoint conditions are a closed, declarative union rather than an
 * expression string. Three reasons:
 *   - the server implements it as dict dispatch, with no parser and no eval;
 *   - the client can build a real form instead of a text box;
 *   - there is no sandbox-escape surface.
 */

export type ComparisonOp = '==' | '!=' | '<' | '<=' | '>' | '>=';

export type BreakpointCondition =
  | { kind: 'token_index'; op: ComparisonOp; value: number }
  | { kind: 'sequence_length'; op: ComparisonOp; value: number }
  | { kind: 'hit_count'; op: ComparisonOp; value: number }
  | { kind: 'top1_prob'; op: ComparisonOp; value: number }
  | { kind: 'logit_entropy'; op: ComparisonOp; value: number }
  | { kind: 'residual_norm'; op: ComparisonOp; value: number }
  | { kind: 'attention_max'; head?: number; op: ComparisonOp; value: number }
  | { kind: 'emitted_token_text'; op: 'equals' | 'contains'; value: string }
  | { kind: 'emitted_token_id'; op: '=='; value: number };

export type ConditionKind = BreakpointCondition['kind'];

export interface Breakpoint {
  id: string;
  stageId: StageId;
  enabled: boolean;
  condition?: BreakpointCondition;
  /** True for "run to here": removed by the server the first time it fires. */
  oneShot: boolean;
  /** Maintained server-side; survives reconnects. */
  hitCount: number;
}

/**
 * Which conditions make sense at which stages. A condition referencing a value
 * a stage does not have (top1_prob at L3.ffn) is rejected when it is set,
 * rather than silently never firing.
 *
 * Conditions valid everywhere are listed in ALWAYS_VALID_CONDITIONS.
 */
export const ALWAYS_VALID_CONDITIONS: readonly ConditionKind[] = [
  'token_index',
  'sequence_length',
  'hit_count',
] as const;

export const CONDITIONS_BY_STAGE: Readonly<Record<StageKind, readonly ConditionKind[]>> = {
  tokenize: [],
  embed: ['residual_norm'],
  attention: ['residual_norm', 'attention_max'],
  ffn: ['residual_norm'],
  final_norm: ['residual_norm'],
  lm_head: ['top1_prob', 'logit_entropy'],
  sample: ['top1_prob', 'logit_entropy'],
  emit: ['top1_prob', 'logit_entropy', 'emitted_token_text', 'emitted_token_id'],
};

export function conditionsForStage(kind: StageKind): ConditionKind[] {
  return [...ALWAYS_VALID_CONDITIONS, ...CONDITIONS_BY_STAGE[kind]];
}

export function isConditionValidAt(kind: StageKind, condition: ConditionKind): boolean {
  return conditionsForStage(kind).includes(condition);
}

const OP_LABELS: Record<string, string> = {
  '==': '==',
  '!=': '!=',
  '<': '<',
  '<=': '<=',
  '>': '>',
  '>=': '>=',
  equals: '==',
  contains: 'contains',
};

const CONDITION_LABELS: Record<ConditionKind, string> = {
  token_index: 'token index',
  sequence_length: 'seq len',
  hit_count: 'hits',
  top1_prob: 'top-1 p',
  logit_entropy: 'entropy',
  residual_norm: '||x||',
  attention_max: 'max attn',
  emitted_token_text: 'token text',
  emitted_token_id: 'token id',
};

/** Compact one-line rendering, e.g. `top-1 p < 0.3` or `max attn[h3] > 0.9`. */
export function describeCondition(c: BreakpointCondition): string {
  const name =
    c.kind === 'attention_max' && c.head !== undefined
      ? `${CONDITION_LABELS[c.kind]}[h${c.head}]`
      : CONDITION_LABELS[c.kind];
  const value = typeof c.value === 'string' ? `"${c.value}"` : c.value;
  return `${name} ${OP_LABELS[c.op]} ${value}`;
}

export function compare(op: ComparisonOp, actual: number, expected: number): boolean {
  switch (op) {
    case '==':
      return actual === expected;
    case '!=':
      return actual !== expected;
    case '<':
      return actual < expected;
    case '<=':
      return actual <= expected;
    case '>':
      return actual > expected;
    case '>=':
      return actual >= expected;
  }
}
