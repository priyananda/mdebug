/**
 * Deterministic pseudo-randomness.
 *
 * Every tensor the mock produces is derived from a structured seed string, e.g.
 * `${sessionId}|attn|${step}|${layer}|${head}`. Nothing is stored: the same
 * request regenerates byte-identical data, so reloading the page does not
 * reshuffle the world and the mock's memory is O(1) in the number of steps.
 */

/** FNV-1a, 32-bit. Fast, dependency-free, and good enough to spread seeds. */
export function hashString(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

export type Rng = () => number;

/** mulberry32: 32 bits of state, good statistical properties, four lines. */
export function mulberry32(seed: number): Rng {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function rngFor(...parts: (string | number)[]): Rng {
  return mulberry32(hashString(parts.join('|')));
}

/** Uniform in [min, max). */
export function uniform(rng: Rng, min: number, max: number): number {
  return min + rng() * (max - min);
}

export function randInt(rng: Rng, min: number, maxExclusive: number): number {
  return min + Math.floor(rng() * (maxExclusive - min));
}

/** Box-Muller, one sample per call (the second is discarded; it is cheap enough). */
export function gaussian(rng: Rng, mean = 0, std = 1): number {
  const u = Math.max(rng(), Number.MIN_VALUE);
  const v = rng();
  return mean + std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export function pick<T>(rng: Rng, items: readonly T[]): T {
  return items[randInt(rng, 0, items.length)];
}

/** Draws an index from a power-law over rank — a rough stand-in for token frequency. */
export function powerLawIndex(rng: Rng, n: number, exponent = 1.2): number {
  const u = rng();
  const idx = Math.floor(n * Math.pow(u, exponent));
  return Math.min(n - 1, Math.max(0, idx));
}
