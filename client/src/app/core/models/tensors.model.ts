/**
 * Wire encoding for numeric arrays.
 *
 * Numeric data never travels as a JSON array of numbers. A 192x192 attention
 * matrix is ~37k values; as JSON that is ~400kB of text to parse, versus 18kB
 * of base64 for the same data quantized to u8 and stored lower-triangular.
 */

export type Dtype = 'f32' | 'u8' | 'u16' | 'i32';

/**
 * The value-space the quantization happened in. `sqrt` means the sender stored
 * `sqrt(v)` and the receiver must square after dequantizing.
 */
export type Transform = 'linear' | 'sqrt';

/**
 * `causal_lower` packs row `i` as exactly `i + 1` entries, with the implicit
 * upper triangle being zero. Halves the bytes of any causal attention matrix.
 */
export type Layout = 'dense' | 'causal_lower';

export interface EncodedArray {
  dtype: Dtype;
  /** Logical shape — ALWAYS the dense shape, even when `layout` is packed. */
  shape: number[];
  layout: Layout;
  transform: Transform;
  /** Dequantization: `v = raw * scale + offset`, then invert `transform`. */
  scale: number;
  offset: number;
  encoding: 'base64';
  /** Little-endian, tightly packed, row-major. */
  data: string;
}

/** Summary statistics that ride along with a vector so the UI needn't scan it. */
export interface VectorStats {
  l2: number;
  mean: number;
  std: number;
  min: number;
  max: number;
}

/** One (layer, head) attention matrix. The unit of on-demand transfer. */
export interface AttentionTile {
  step: number;
  layer: number;
  head: number;
  /** T. The dense matrix is T x T. */
  sequenceLength: number;
  /** u8 / sqrt / causal_lower, shape [T, T]. */
  weights: EncodedArray;
  stats: {
    maxWeight: number;
    /** Mean over rows of the per-row attention entropy, in nats. */
    meanEntropy: number;
    /** Fraction of total mass landing on position 0 — the attention sink. */
    sinkMass: number;
  };
}

/** KV cache occupancy at one instant. Cheap enough to send on every halt. */
export interface KvSnapshot {
  numLayers: number;
  sequenceLength: number;
  /** u8 [numLayers, T]: 0 = empty, 1 = written this step, 2 = written earlier. */
  occupancy: EncodedArray;
  /** Optional f32 [numLayers, T] of ||k||, for a richer grid than occupancy alone. */
  keyNorms?: EncodedArray;
  bytesResident: number;
  /**
   * TRUE when the server has no real KV cache and this is a reconstruction.
   * The UI must badge it. See docs/api-contract.md.
   */
  simulated: boolean;
}

/** The residual stream at one layer and position. */
export interface ResidualVector {
  step: number;
  layer: number;
  position: number;
  stage: ResidualStage;
  /** f32 [hiddenSize]. */
  values: EncodedArray;
  stats: VectorStats;
}

export type ResidualStage = 'input' | 'post_attention' | 'post_ffn';

/** On-demand: the residual across all positions at one layer. */
export interface ResidualBlock {
  step: number;
  layer: number;
  stage: ResidualStage;
  /** f32 [T, hiddenSize]. */
  values: EncodedArray;
}

export interface TopKEntry {
  tokenId: number;
  text: string;
  display: string;
  logit: number;
  prob: number;
}

export interface TopKLogits {
  step: number;
  k: number;
  entries: TopKEntry[];
  /** Entropy of the full post-temperature distribution, in nats. */
  entropy: number;
  temperature: number;
  chosenTokenId: number;
  /** Rank of the sampled token in the full ordering. 0 = argmax. */
  chosenRank: number;
}
