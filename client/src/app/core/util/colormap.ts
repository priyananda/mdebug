/**
 * 256-entry RGBA lookup tables.
 *
 * The heatmap and KV grid render by indexing one of these per cell while
 * filling an ImageData buffer, so the LUT must be a flat Uint8ClampedArray and
 * the lookup must be a single array read.
 */

export type ColormapName = 'sequential' | 'diverging' | 'occupancy';

type Stop = readonly [number, number, number];

/** White -> blue. For non-negative magnitudes on a white page. */
const SEQUENTIAL: readonly Stop[] = [
  [255, 255, 255],
  [219, 234, 254],
  [147, 197, 253],
  [59, 130, 246],
  [29, 78, 216],
  [23, 37, 84],
];

/** Blue -> white -> red. For signed values, symmetric about zero. */
const DIVERGING: readonly Stop[] = [
  [30, 64, 175],
  [96, 165, 250],
  [191, 219, 254],
  [255, 255, 255],
  [254, 202, 202],
  [248, 113, 113],
  [185, 28, 28],
];

function buildLut(stops: readonly Stop[]): Uint8ClampedArray {
  const lut = new Uint8ClampedArray(256 * 4);
  const segments = stops.length - 1;
  for (let i = 0; i < 256; i++) {
    const pos = (i / 255) * segments;
    const idx = Math.min(Math.floor(pos), segments - 1);
    const t = pos - idx;
    const a = stops[idx];
    const b = stops[idx + 1];
    lut[i * 4 + 0] = a[0] + (b[0] - a[0]) * t;
    lut[i * 4 + 1] = a[1] + (b[1] - a[1]) * t;
    lut[i * 4 + 2] = a[2] + (b[2] - a[2]) * t;
    lut[i * 4 + 3] = 255;
  }
  return lut;
}

/**
 * Three discrete states rather than a ramp: empty, written on this step, and
 * written earlier. The "this step" state gets the accent colour so the newest
 * KV column is the thing your eye lands on.
 */
function buildOccupancyLut(): Uint8ClampedArray {
  const lut = new Uint8ClampedArray(256 * 4);
  const states: readonly Stop[] = [
    [251, 251, 250], // 0 empty
    [245, 158, 11], // 1 written this step (accent)
    [147, 197, 253], // 2 written earlier
  ];
  for (let i = 0; i < 256; i++) {
    const s = states[Math.min(i, states.length - 1)];
    lut[i * 4 + 0] = s[0];
    lut[i * 4 + 1] = s[1];
    lut[i * 4 + 2] = s[2];
    lut[i * 4 + 3] = 255;
  }
  return lut;
}

const LUTS: Record<ColormapName, Uint8ClampedArray> = {
  sequential: buildLut(SEQUENTIAL),
  diverging: buildLut(DIVERGING),
  occupancy: buildOccupancyLut(),
};

export function lut(name: ColormapName): Uint8ClampedArray {
  return LUTS[name];
}

/** Maps a value in [0, 1] to a LUT index. */
export function normIndex(v: number): number {
  return v <= 0 ? 0 : v >= 1 ? 255 : (v * 255) | 0;
}

/** Maps a signed value to a diverging LUT index, symmetric about zero. */
export function signedIndex(v: number, absMax: number): number {
  if (absMax <= 0) return 128;
  const t = (v / absMax + 1) / 2;
  return normIndex(t);
}

/** `rgb(...)` for one LUT entry — for SVG fills, where ImageData is not in play. */
export function cssColor(name: ColormapName, index: number): string {
  const l = LUTS[name];
  const i = Math.max(0, Math.min(255, index | 0)) * 4;
  return `rgb(${l[i]} ${l[i + 1]} ${l[i + 2]})`;
}
