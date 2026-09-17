import { allStageIds, parseStageId, stageId, stageSequence } from '../../models/pipeline.model';
import {
  attentionTile,
  headSummary,
  kvOccupancy,
  outlierDims,
  residualVector,
  statsOf,
} from './fake-tensors';

const SESSION = 'test-session';
const LAYERS = 20;
const HEADS = 8;
const HIDDEN = 512;

describe('stage sequence', () => {
  it('has 2L+6 halt points on step 0 and 2L+5 after', () => {
    expect(stageSequence(LAYERS, 0).length).toBe(2 * LAYERS + 6);
    expect(stageSequence(LAYERS, 1).length).toBe(2 * LAYERS + 5);
  });

  it('orders tokenize, embed, the layer ladder, then the head', () => {
    const ids = allStageIds(LAYERS);
    expect(ids[0]).toBe('tokenize');
    expect(ids[1]).toBe('embed');
    expect(ids[2]).toBe('L0.attention');
    expect(ids[3]).toBe('L0.ffn');
    expect(ids.slice(-4)).toEqual(['final_norm', 'lm_head', 'sample', 'emit']);
  });

  it('round-trips stage ids', () => {
    for (const id of allStageIds(LAYERS)) {
      expect(stageId(parseStageId(id))).toBe(id);
    }
  });

  it('never repeats a stage id within a step', () => {
    const ids = allStageIds(LAYERS);
    expect(new Set(ids).size).toBe(ids.length);
  });
});

describe('fake attention', () => {
  const T = 24;

  it('is deterministic for the same seed', () => {
    const a = attentionTile(SESSION, 3, 7, 2, LAYERS, T);
    const b = attentionTile(SESSION, 3, 7, 2, LAYERS, T);
    expect(Array.from(a.weights)).toEqual(Array.from(b.weights));
  });

  it('differs across heads, layers and steps', () => {
    const base = attentionTile(SESSION, 3, 7, 2, LAYERS, T).weights;
    const otherHead = attentionTile(SESSION, 3, 7, 3, LAYERS, T).weights;
    const otherLayer = attentionTile(SESSION, 3, 8, 2, LAYERS, T).weights;
    const otherStep = attentionTile(SESSION, 4, 7, 2, LAYERS, T).weights;
    expect(Array.from(otherHead)).not.toEqual(Array.from(base));
    expect(Array.from(otherLayer)).not.toEqual(Array.from(base));
    expect(Array.from(otherStep)).not.toEqual(Array.from(base));
  });

  it('is strictly causal', () => {
    const { weights } = attentionTile(SESSION, 0, 5, 1, LAYERS, T);
    for (let i = 0; i < T; i++) {
      for (let j = i + 1; j < T; j++) {
        expect(weights[i * T + j]).toBe(0);
      }
    }
  });

  it('has rows that sum to one', () => {
    const { weights } = attentionTile(SESSION, 0, 5, 1, LAYERS, T);
    for (let i = 0; i < T; i++) {
      let sum = 0;
      for (let j = 0; j <= i; j++) sum += weights[i * T + j];
      expect(sum).toBeCloseTo(1, 5);
    }
  });

  it('puts more mass on the sink in late layers than in early ones', () => {
    // Averaged over heads, because any individual head may buck the trend.
    const meanSink = (layer: number) => {
      let total = 0;
      for (let h = 0; h < HEADS; h++) total += attentionTile(SESSION, 0, layer, h, LAYERS, T).sinkMass;
      return total / HEADS;
    };
    expect(meanSink(LAYERS - 1)).toBeGreaterThan(meanSink(0));
  });

  it('summarizes head entropy at one value per (layer, head)', () => {
    const summary = headSummary(SESSION, 0, LAYERS, HEADS, T);
    expect(summary.length).toBe(LAYERS * HEADS);
    summary.forEach((h) => {
      expect(h).toBeGreaterThan(0);
      expect(h).toBeLessThanOrEqual(Math.log(T) + 1e-6); // entropy is bounded by log T
    });
  });
});

describe('fake residuals', () => {
  it('is deterministic', () => {
    expect(Array.from(residualVector(SESSION, 4, 9, HIDDEN))).toEqual(
      Array.from(residualVector(SESSION, 4, 9, HIDDEN)),
    );
  });

  it('grows in norm with depth', () => {
    const early = statsOf(residualVector(SESSION, 4, 0, HIDDEN)).l2;
    const late = statsOf(residualVector(SESSION, 4, LAYERS - 1, HIDDEN)).l2;
    expect(late).toBeGreaterThan(early * 4);
  });

  it('keeps the same outlier dimensions at every layer', () => {
    const dims = outlierDims(SESSION, HIDDEN);
    expect(dims.length).toBeGreaterThanOrEqual(4);

    for (const layer of [0, 7, 19]) {
      const v = residualVector(SESSION, 4, layer, HIDDEN);
      const stats = statsOf(v);
      for (const d of dims) {
        // Outliers should stand well clear of the typical magnitude.
        expect(Math.abs(v[d])).toBeGreaterThan(stats.std * 2);
      }
    }
  });

  it('uses a different session seed for a different session', () => {
    expect(Array.from(residualVector('other', 4, 9, HIDDEN))).not.toEqual(
      Array.from(residualVector(SESSION, 4, 9, HIDDEN)),
    );
  });
});

describe('kv occupancy', () => {
  const T = 10;

  it('marks exactly one newest column, and only in executed layers', () => {
    const layersDone = 6;
    const occ = kvOccupancy(LAYERS, T, T - 1, T - 1, layersDone);
    for (let layer = 0; layer < LAYERS; layer++) {
      const newest = occ[layer * T + (T - 1)];
      expect(newest).toBe(layer < layersDone ? 1 : 0);
    }
  });

  it('leaves nothing resident beyond the current sequence length', () => {
    const occ = kvOccupancy(LAYERS, T, 4, 4, LAYERS);
    for (let layer = 0; layer < LAYERS; layer++) {
      for (let pos = 5; pos < T; pos++) {
        expect(occ[layer * T + pos]).toBe(0);
      }
    }
  });
});
