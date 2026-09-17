import { Dtype, EncodedArray, Layout, Transform } from '../models/tensors.model';

/**
 * base64 <-> typed arrays, with dequantization.
 *
 * Every numeric payload on the wire goes through here, and every component
 * downstream sees a plain dense Float32Array. Decoding happens once, in the
 * event reducer, never in a component.
 */

/**
 * Typed array views read multi-byte values in platform byte order, and the wire
 * format is little-endian. Every platform this runs on is little-endian, so the
 * fast path is the only one that ever executes in practice -- but assuming it
 * silently would be a bug that only ever appears on hardware nobody can debug.
 */
const PLATFORM_IS_LITTLE_ENDIAN =
  new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

const BYTES_PER_ELEMENT: Record<Dtype, number> = { f32: 4, u8: 1, u16: 2, i32: 4 };

/** Serializes a typed array to little-endian bytes, swapping on BE platforms. */
function toLittleEndianBytes(view: ArrayBufferView, width: number): Uint8Array {
  const bytes = new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
  if (PLATFORM_IS_LITTLE_ENDIAN || width === 1) return bytes;
  const out = new Uint8Array(bytes.length);
  for (let i = 0; i < bytes.length; i += width) {
    for (let b = 0; b < width; b++) out[i + b] = bytes[i + width - 1 - b];
  }
  return out;
}

export function base64ToBytes(b64: string): Uint8Array {
  const binary = atob(b64);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
  return out;
}

export function bytesToBase64(bytes: Uint8Array): string {
  // Chunked: String.fromCharCode(...bytes) blows the argument limit past ~64k.
  const CHUNK = 0x8000;
  let binary = '';
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  return btoa(binary);
}

export function elementCount(shape: readonly number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

/** Packed element count for a layout: causal_lower stores T(T+1)/2 of a T x T. */
export function packedCount(shape: readonly number[], layout: Layout): number {
  if (layout === 'dense') return elementCount(shape);
  const t = shape[shape.length - 1];
  return (t * (t + 1)) / 2;
}

/** Reads raw (still-quantized) values out of the base64 payload. */
function readRaw(bytes: Uint8Array, dtype: Dtype, count: number): Float64Array {
  const out = new Float64Array(count);
  const expected = count * BYTES_PER_ELEMENT[dtype];
  if (bytes.byteLength < expected) {
    throw new RangeError(
      `EncodedArray truncated: expected ${expected} bytes for ${count} ${dtype} values, got ${bytes.byteLength}`,
    );
  }

  if (dtype === 'u8') {
    for (let i = 0; i < count; i++) out[i] = bytes[i];
    return out;
  }

  if (PLATFORM_IS_LITTLE_ENDIAN) {
    // `bytes` may be a view into a larger buffer at a non-aligned offset, so
    // copy rather than aliasing when the offset is not a multiple of the width.
    const width = BYTES_PER_ELEMENT[dtype];
    const aligned =
      bytes.byteOffset % width === 0
        ? bytes
        : new Uint8Array(bytes.slice(0, expected));
    const { buffer, byteOffset } = aligned;
    const view =
      dtype === 'f32'
        ? new Float32Array(buffer, byteOffset, count)
        : dtype === 'u16'
          ? new Uint16Array(buffer, byteOffset, count)
          : new Int32Array(buffer, byteOffset, count);
    for (let i = 0; i < count; i++) out[i] = view[i];
    return out;
  }

  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let i = 0; i < count; i++) {
    const at = i * BYTES_PER_ELEMENT[dtype];
    out[i] =
      dtype === 'f32'
        ? dv.getFloat32(at, true)
        : dtype === 'u16'
          ? dv.getUint16(at, true)
          : dv.getInt32(at, true);
  }
  return out;
}

function applyTransform(v: number, transform: Transform): number {
  return transform === 'sqrt' ? v * v : v;
}

/**
 * Decodes to a DENSE Float32Array of `elementCount(shape)` values, with
 * quantization and any packing undone.
 */
export function decodeArray(a: EncodedArray): Float32Array {
  const bytes = base64ToBytes(a.data);
  const packed = packedCount(a.shape, a.layout);
  const raw = readRaw(bytes, a.dtype, packed);

  const dense = new Float32Array(elementCount(a.shape));

  if (a.layout === 'dense') {
    for (let i = 0; i < raw.length; i++) {
      dense[i] = applyTransform(raw[i] * a.scale + a.offset, a.transform);
    }
    return dense;
  }

  // causal_lower: row i holds i+1 entries; the rest of the row stays zero.
  const t = a.shape[a.shape.length - 1];
  let src = 0;
  for (let row = 0; row < t; row++) {
    const base = row * t;
    for (let col = 0; col <= row; col++) {
      dense[base + col] = applyTransform(raw[src++] * a.scale + a.offset, a.transform);
    }
  }
  return dense;
}

export interface EncodeOptions {
  shape: number[];
  dtype?: Dtype;
  layout?: Layout;
  transform?: Transform;
}

/**
 * The inverse of `decodeArray`, used by the mock engine and by tests.
 *
 * For quantized dtypes the scale is derived from the observed range after the
 * transform, so a round-trip is exact at the endpoints and within half a
 * quantization step elsewhere.
 */
export function encodeArray(dense: ArrayLike<number>, opts: EncodeOptions): EncodedArray {
  const dtype = opts.dtype ?? 'f32';
  const layout = opts.layout ?? 'dense';
  const transform = opts.transform ?? 'linear';
  const shape = opts.shape;

  // Gather the values that actually travel, in wire order.
  const count = packedCount(shape, layout);
  const values = new Float64Array(count);
  if (layout === 'dense') {
    for (let i = 0; i < count; i++) values[i] = transformForward(dense[i], transform);
  } else {
    const t = shape[shape.length - 1];
    let dst = 0;
    for (let row = 0; row < t; row++) {
      for (let col = 0; col <= row; col++) {
        values[dst++] = transformForward(dense[row * t + col], transform);
      }
    }
  }

  if (dtype === 'f32') {
    const f32 = new Float32Array(count);
    for (let i = 0; i < count; i++) f32[i] = values[i];
    return {
      dtype,
      shape,
      layout,
      transform,
      scale: 1,
      offset: 0,
      encoding: 'base64',
      data: bytesToBase64(toLittleEndianBytes(f32, 4)),
    };
  }

  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < count; i++) {
    if (values[i] < min) min = values[i];
    if (values[i] > max) max = values[i];
  }
  if (!isFinite(min)) {
    min = 0;
    max = 0;
  }

  const levels = dtype === 'u8' ? 255 : dtype === 'u16' ? 65535 : 2 ** 31 - 1;
  const span = max - min;
  const scale = span === 0 ? 1 : span / levels;
  const offset = min;

  const quantized =
    dtype === 'u8'
      ? new Uint8Array(count)
      : dtype === 'u16'
        ? new Uint16Array(count)
        : new Int32Array(count);
  for (let i = 0; i < count; i++) {
    quantized[i] = Math.round((values[i] - offset) / scale);
  }

  return {
    dtype,
    shape,
    layout,
    transform,
    scale,
    offset,
    encoding: 'base64',
    data: bytesToBase64(toLittleEndianBytes(quantized, BYTES_PER_ELEMENT[dtype])),
  };
}

function transformForward(v: number, transform: Transform): number {
  return transform === 'sqrt' ? Math.sqrt(Math.max(0, v)) : v;
}

/** Row-major index helper for 2-D decoded arrays. */
export function at2d(dense: Float32Array, cols: number, row: number, col: number): number {
  return dense[row * cols + col];
}
