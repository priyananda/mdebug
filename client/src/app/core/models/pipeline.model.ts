/**
 * The pipeline: the sequence of stages a decode step walks through, and the
 * identifiers used to name a point in it.
 *
 * This mirrors `QwenModel.forward` in server/src/model.py:
 *   token_emb + pos_emb -> 20 x (norm1 -> attn -> add ; norm2 -> ffn -> add)
 *   -> final RMSNorm -> lm head -> sample.
 */

export type StageKind =
  | 'tokenize'
  | 'embed'
  | 'attention'
  | 'ffn'
  | 'final_norm'
  | 'lm_head'
  | 'sample'
  | 'emit';

export interface StageRef {
  kind: StageKind;
  /** Present iff the stage is per-layer (`attention`, `ffn`). */
  layer?: number;
}

/**
 * Canonical string form of a StageRef: 'tokenize', 'embed', 'L7.attention',
 * 'L7.ffn', 'final_norm', 'lm_head', 'sample', 'emit'.
 *
 * Used as a map key throughout — breakpoints, graph geometry, the program
 * counter — so that comparing positions is a string compare.
 */
export type StageId = string;

export interface StageDescriptor {
  kind: StageKind;
  label: string;
  perLayer: boolean;
  breakpointable: boolean;
  /** Shown as a tooltip on the graph node. */
  description: string;
}

/** Per-layer stages, in execution order within a layer. */
export const PER_LAYER_STAGES: readonly StageKind[] = ['attention', 'ffn'] as const;

/** Stages that run once per decode step, before the layer ladder. */
export const PRE_LAYER_STAGES: readonly StageKind[] = ['embed'] as const;

/** Stages that run once per decode step, after the layer ladder. */
export const POST_LAYER_STAGES: readonly StageKind[] = [
  'final_norm',
  'lm_head',
  'sample',
  'emit',
] as const;

export function stageId(ref: StageRef): StageId {
  return ref.layer === undefined ? ref.kind : `L${ref.layer}.${ref.kind}`;
}

export function parseStageId(id: StageId): StageRef {
  const dot = id.indexOf('.');
  if (dot < 0) return { kind: id as StageKind };
  return {
    kind: id.slice(dot + 1) as StageKind,
    layer: Number(id.slice(1, dot)),
  };
}

export function isPerLayer(kind: StageKind): boolean {
  return PER_LAYER_STAGES.includes(kind);
}

/**
 * The full halt-point sequence for one decode step.
 *
 * `tokenize` only exists on step 0 (the prompt is tokenized once), so step 0
 * has 2*numLayers + 6 stages and every later step has 2*numLayers + 5.
 * At numLayers=20 that is 46 and 45.
 */
export function stageSequence(numLayers: number, step: number): StageRef[] {
  const out: StageRef[] = [];
  if (step === 0) out.push({ kind: 'tokenize' });
  for (const kind of PRE_LAYER_STAGES) out.push({ kind });
  for (let layer = 0; layer < numLayers; layer++) {
    for (const kind of PER_LAYER_STAGES) out.push({ kind, layer });
  }
  for (const kind of POST_LAYER_STAGES) out.push({ kind });
  return out;
}

/** Every stage id that can carry a breakpoint, in graph order. */
export function allStageIds(numLayers: number): StageId[] {
  return stageSequence(numLayers, 0).map(stageId);
}

/** The stage catalog a server should report when it has nothing custom to say. */
export const DEFAULT_STAGE_DESCRIPTORS: readonly StageDescriptor[] = [
  {
    kind: 'tokenize',
    label: 'Tokenize',
    perLayer: false,
    breakpointable: true,
    description: 'Byte-level BPE splits the prompt into token ids. Runs once.',
  },
  {
    kind: 'embed',
    label: 'Embed',
    perLayer: false,
    breakpointable: true,
    description:
      'Token embedding plus learned absolute position embedding. This model adds both, then applies RoPE inside attention as well.',
  },
  {
    kind: 'attention',
    label: 'Attention',
    perLayer: true,
    breakpointable: true,
    description:
      'RMSNorm, then multi-head self-attention with RoPE on the first 16 dims of each head, then a residual add.',
  },
  {
    kind: 'ffn',
    label: 'FFN',
    perLayer: true,
    breakpointable: true,
    description: 'RMSNorm, then a SiLU feed-forward (512 -> 1536 -> 512), then a residual add.',
  },
  {
    kind: 'final_norm',
    label: 'Final norm',
    perLayer: false,
    breakpointable: true,
    description: 'The last RMSNorm before the output projection.',
  },
  {
    kind: 'lm_head',
    label: 'LM head',
    perLayer: false,
    breakpointable: true,
    description: 'Linear projection from the hidden size to one logit per vocabulary entry.',
  },
  {
    kind: 'sample',
    label: 'Sample',
    perLayer: false,
    breakpointable: true,
    description: 'Apply temperature and top-k, then draw the next token from the distribution.',
  },
  {
    kind: 'emit',
    label: 'Emit',
    perLayer: false,
    breakpointable: true,
    description: 'Append the chosen token to the sequence and begin the next decode step.',
  },
] as const;
