import { Rng, gaussian, rngFor, uniform } from './rng';

/**
 * Deterministic generators for the intermediate state the mock reports.
 *
 * The bar here is not "some numbers" — it is data that *looks like transformer
 * data*, because the whole point of the tool is recognizing structure. So the
 * attention generator produces sinks, diagonals and induction stripes, and the
 * residual generator produces persistent outlier dimensions and norm growth
 * with depth. Those are real phenomena; a user who learns to spot them here is
 * learning something true.
 *
 * Everything is regenerated on demand from a seed string. Nothing is stored, so
 * memory is O(1) in the number of steps.
 */

/**
 * What a given head "does". Drawn from the head's seed, then biased by depth so
 * that clicking down through layers shows a gradient rather than reshuffled
 * noise.
 */
export interface HeadPersonality {
  /** Mass on position 0 — the attention sink. Grows with depth. */
  sink: number;
  /** Weight of the exponential recency falloff. Strongest early. */
  recency: number;
  /** Decay rate of that falloff. */
  lambda: number;
  /** Mass on the immediately preceding token. */
  previous: number;
  /** Mass on the query position itself. */
  self: number;
  /** Mass at a fixed offset back — the induction-head signature. Peaks mid-stack. */
  induction: number;
  /** That offset. */
  delta: number;
  /** Uniform background, so the low tail is not identically zero. */
  background: number;
}

export function headPersonality(
  sessionId: string,
  layer: number,
  head: number,
  numLayers: number,
): HeadPersonality {
  const rng = rngFor(sessionId, 'head', layer, head);
  const depth = numLayers > 1 ? layer / (numLayers - 1) : 0;

  // Depth gradient: early layers look local, middle layers do the matching
  // work, late layers dump mass on the sink.
  const localBias = 1 - depth;
  const midBias = Math.exp(-((depth - 0.45) ** 2) / 0.08);

  return {
    sink: (0.05 + 0.9 * depth ** 2) * uniform(rng, 0.4, 1.6),
    recency: (0.2 + 0.9 * localBias) * uniform(rng, 0.5, 1.5),
    lambda: uniform(rng, 0.05, 1.6),
    previous: 0.9 * localBias * uniform(rng, 0, 1.4),
    self: uniform(rng, 0.05, 0.7),
    induction: 1.1 * midBias * uniform(rng, 0, 1.5),
    delta: 2 + Math.floor(uniform(rng, 0, 12)),
    background: uniform(rng, 0.004, 0.02),
  };
}

/**
 * One row of an attention matrix: the distribution over keys `0..i` for the
 * query at position `i`. Writes into `out` and returns it.
 */
export function attentionRow(
  p: HeadPersonality,
  i: number,
  out: Float32Array,
  rng: Rng,
): Float32Array {
  let total = 0;
  for (let j = 0; j <= i; j++) {
    let s = p.background;
    if (j === 0) s += p.sink;
    s += p.recency * Math.exp(-p.lambda * (i - j));
    if (j === i - 1 && i > 0) s += p.previous;
    if (j === i) s += p.self;
    if (j === i - p.delta) s += p.induction;
    s *= 1 + 0.25 * (rng() - 0.5);
    out[j] = s;
    total += s;
  }
  // Normalize to a distribution; the upper triangle stays zero.
  const inv = total > 0 ? 1 / total : 0;
  for (let j = 0; j <= i; j++) out[j] *= inv;
  for (let j = i + 1; j < out.length; j++) out[j] = 0;
  return out;
}

export interface FakeAttention {
  /** Dense, row-major, T x T. Strictly lower-triangular including the diagonal. */
  weights: Float32Array;
  maxWeight: number;
  meanEntropy: number;
  sinkMass: number;
}

/**
 * A full (layer, head) tile. O(T^2) with a handful of flops per cell: about
 * 37k cells at T=192, comfortably under a millisecond.
 */
export function attentionTile(
  sessionId: string,
  step: number,
  layer: number,
  head: number,
  numLayers: number,
  t: number,
): FakeAttention {
  const p = headPersonality(sessionId, layer, head, numLayers);
  const rng = rngFor(sessionId, 'attn', step, layer, head);
  const weights = new Float32Array(t * t);
  const row = new Float32Array(t);

  let maxWeight = 0;
  let entropySum = 0;
  let sinkMass = 0;

  for (let i = 0; i < t; i++) {
    attentionRow(p, i, row, rng);
    let h = 0;
    for (let j = 0; j <= i; j++) {
      const w = row[j];
      weights[i * t + j] = w;
      if (w > maxWeight) maxWeight = w;
      if (w > 0) h -= w * Math.log(w);
    }
    entropySum += h;
    sinkMass += row[0];
  }

  return {
    weights,
    maxWeight,
    meanEntropy: entropySum / t,
    sinkMass: sinkMass / t,
  };
}

/**
 * Entropy of the *current* query row for every (layer, head).
 *
 * This is what colours the per-head marks in the pipeline graph. Computing it
 * from the final row is O(numLayers * numHeads * T) — about 30k operations —
 * instead of the O(T^2) per head a full tile would cost, so it is affordable on
 * every halt.
 */
export function headSummary(
  sessionId: string,
  step: number,
  numLayers: number,
  numHeads: number,
  t: number,
): Float32Array {
  const out = new Float32Array(numLayers * numHeads);
  const row = new Float32Array(t);
  const i = t - 1;

  for (let layer = 0; layer < numLayers; layer++) {
    for (let head = 0; head < numHeads; head++) {
      const p = headPersonality(sessionId, layer, head, numLayers);
      attentionRow(p, i, row, rngFor(sessionId, 'attn', step, layer, head));
      let h = 0;
      for (let j = 0; j <= i; j++) {
        const w = row[j];
        if (w > 0) h -= w * Math.log(w);
      }
      out[layer * numHeads + head] = h;
    }
  }
  return out;
}

/**
 * Dimensions that carry outsized magnitude in every layer of a given session.
 *
 * Real residual streams have persistent outlier dimensions, and noticing the
 * same index recur as you step down the stack is one of the genuine insight
 * moments the tool can offer. Faking them per-layer would destroy that, so they
 * are fixed for the session.
 */
export function outlierDims(sessionId: string, hiddenSize: number): number[] {
  const rng = rngFor(sessionId, 'outliers');
  const count = 4 + Math.floor(rng() * 5);
  const dims = new Set<number>();
  while (dims.size < count) dims.add(Math.floor(rng() * hiddenSize));
  return [...dims].sort((a, b) => a - b);
}

/** Residual norm grows roughly geometrically through the stack. */
const NORM_GROWTH_PER_LAYER = 1.15;

export function residualVector(
  sessionId: string,
  position: number,
  layer: number,
  hiddenSize: number,
): Float32Array {
  const rng = rngFor(sessionId, 'resid', position, layer);
  const shapeRng = rngFor(sessionId, 'resid-shape');

  // A few harmonics give the strip visible texture rather than white noise.
  const harmonics = 3 + Math.floor(shapeRng() * 3);
  const freq: number[] = [];
  const amp: number[] = [];
  const phase: number[] = [];
  for (let k = 0; k < harmonics; k++) {
    freq.push(uniform(shapeRng, 0.01, 0.4));
    amp.push(uniform(shapeRng, 0.2, 1));
    phase.push(uniform(shapeRng, 0, Math.PI * 2));
  }

  // Position-to-position drift: neighbouring tokens differ but stay related.
  const drift = Math.sin(position * 0.37 + layer * 0.11) * 0.25;
  const scale = Math.pow(NORM_GROWTH_PER_LAYER, layer);

  const out = new Float32Array(hiddenSize);
  for (let d = 0; d < hiddenSize; d++) {
    let v = 0;
    for (let k = 0; k < harmonics; k++) v += amp[k] * Math.sin(freq[k] * d + phase[k] + drift);
    v += gaussian(rng, 0, 0.18);
    out[d] = v * scale;
  }

  for (const d of outlierDims(sessionId, hiddenSize)) {
    const magnitude = uniform(rngFor(sessionId, 'outlier-mag', d), 5, 10);
    out[d] = Math.sign(out[d] || 1) * magnitude * scale;
  }

  return out;
}

export interface Stats {
  l2: number;
  mean: number;
  std: number;
  min: number;
  max: number;
}

export function statsOf(values: ArrayLike<number>): Stats {
  let sum = 0;
  let sumSq = 0;
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    sum += v;
    sumSq += v * v;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const n = values.length || 1;
  const mean = sum / n;
  return {
    l2: Math.sqrt(sumSq),
    mean,
    std: Math.sqrt(Math.max(0, sumSq / n - mean * mean)),
    min: isFinite(min) ? min : 0,
    max: isFinite(max) ? max : 0,
  };
}

/**
 * KV occupancy: 0 empty, 1 written on this step, 2 written earlier.
 *
 * `writtenThrough` is the number of positions whose K/V are resident, and
 * `newestPosition` is the column that lit up on the current step.
 */
export function kvOccupancy(
  numLayers: number,
  t: number,
  writtenThrough: number,
  newestPosition: number,
  layersDone: number,
): Uint8Array {
  const out = new Uint8Array(numLayers * t);
  for (let layer = 0; layer < numLayers; layer++) {
    // Layers below the program counter have not been executed yet this step,
    // so their newest column is not resident.
    const resident = layer < layersDone ? writtenThrough : writtenThrough - 1;
    for (let pos = 0; pos < t; pos++) {
      if (pos > resident) continue;
      out[layer * t + pos] = pos === newestPosition && layer < layersDone ? 1 : 2;
    }
  }
  return out;
}

/** Per-cell ||k||, for a richer grid than the three-state occupancy ramp. */
export function kvKeyNorms(sessionId: string, numLayers: number, t: number): Float32Array {
  const out = new Float32Array(numLayers * t);
  for (let layer = 0; layer < numLayers; layer++) {
    const scale = Math.pow(NORM_GROWTH_PER_LAYER, layer * 0.5);
    for (let pos = 0; pos < t; pos++) {
      const rng = rngFor(sessionId, 'kv', layer, pos);
      out[layer * t + pos] = scale * uniform(rng, 0.6, 1.4);
    }
  }
  return out;
}
