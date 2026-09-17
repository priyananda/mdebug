import { EncodedArray } from '../models/tensors.model';
import {
  base64ToBytes,
  bytesToBase64,
  decodeArray,
  encodeArray,
  packedCount,
} from './encoding';

describe('encoding', () => {
  describe('base64', () => {
    it('round-trips arbitrary bytes', () => {
      const bytes = new Uint8Array(512);
      for (let i = 0; i < bytes.length; i++) bytes[i] = (i * 7 + 13) % 256;
      expect(Array.from(base64ToBytes(bytesToBase64(bytes)))).toEqual(Array.from(bytes));
    });

    it('handles payloads past the fromCharCode argument limit', () => {
      const bytes = new Uint8Array(200_000).fill(0xab);
      expect(base64ToBytes(bytesToBase64(bytes)).length).toBe(200_000);
    });
  });

  describe('f32 dense', () => {
    it('round-trips exactly', () => {
      const values = Float32Array.from([-3.5, 0, 1e-7, 2.25, 1e6]);
      const decoded = decodeArray(encodeArray(values, { shape: [5] }));
      expect(Array.from(decoded)).toEqual(Array.from(values));
    });

    it('is little-endian on the wire', () => {
      const encoded = encodeArray(Float32Array.from([1]), { shape: [1] });
      // 1.0f32 = 0x3F800000, little-endian => 00 00 80 3F
      expect(Array.from(base64ToBytes(encoded.data))).toEqual([0x00, 0x00, 0x80, 0x3f]);
    });
  });

  describe('u8 quantization', () => {
    it('stays within half a step of the original', () => {
      const values = new Float32Array(256);
      for (let i = 0; i < 256; i++) values[i] = i / 255;
      const encoded = encodeArray(values, { shape: [256], dtype: 'u8' });
      const decoded = decodeArray(encoded);
      for (let i = 0; i < 256; i++) {
        expect(Math.abs(decoded[i] - values[i])).toBeLessThan(1 / 255);
      }
    });

    it('survives a constant array without dividing by zero', () => {
      const values = new Float32Array(16).fill(0.25);
      const decoded = decodeArray(encodeArray(values, { shape: [16], dtype: 'u8' }));
      decoded.forEach((v) => expect(v).toBeCloseTo(0.25, 6));
    });
  });

  describe('sqrt transform', () => {
    // The point of sqrt: linear u8 crushes everything below 1/255 to zero,
    // which is exactly the low-probability structure the heatmap exists to show.
    it('resolves small probabilities far better than linear u8', () => {
      const probs = Float32Array.from([1e-4, 5e-4, 1e-3, 5e-3, 0.5, 1]);

      const linear = decodeArray(encodeArray(probs, { shape: [6], dtype: 'u8' }));
      const sqrt = decodeArray(
        encodeArray(probs, { shape: [6], dtype: 'u8', transform: 'sqrt' }),
      );

      // Linear collapses the whole low tail onto one level.
      expect(linear[0]).toBe(linear[1]);
      // sqrt keeps them apart, and to a useful relative accuracy.
      expect(sqrt[1]).toBeGreaterThan(sqrt[0]);
      expect(Math.abs(sqrt[3] - probs[3]) / probs[3]).toBeLessThan(0.05);
    });

    it('never produces negatives', () => {
      const decoded = decodeArray(
        encodeArray(Float32Array.from([0, 0.1, 1]), {
          shape: [3],
          dtype: 'u8',
          transform: 'sqrt',
        }),
      );
      decoded.forEach((v) => expect(v).toBeGreaterThanOrEqual(0));
    });
  });

  describe('causal_lower layout', () => {
    const T = 6;

    function causalMatrix(): Float32Array {
      const m = new Float32Array(T * T);
      for (let r = 0; r < T; r++) {
        for (let c = 0; c <= r; c++) m[r * T + c] = (r + 1) * 0.1 + c * 0.01;
      }
      return m;
    }

    it('packs T(T+1)/2 values', () => {
      expect(packedCount([T, T], 'causal_lower')).toBe((T * (T + 1)) / 2);
      const encoded = encodeArray(causalMatrix(), {
        shape: [T, T],
        dtype: 'f32',
        layout: 'causal_lower',
      });
      expect(base64ToBytes(encoded.data).length).toBe(((T * (T + 1)) / 2) * 4);
    });

    it('scatters back to a dense strictly-causal matrix', () => {
      const original = causalMatrix();
      const decoded = decodeArray(
        encodeArray(original, { shape: [T, T], dtype: 'f32', layout: 'causal_lower' }),
      );
      for (let r = 0; r < T; r++) {
        for (let c = 0; c < T; c++) {
          if (c > r) expect(decoded[r * T + c]).toBe(0);
          else expect(decoded[r * T + c]).toBeCloseTo(original[r * T + c], 6);
        }
      }
    });
  });

  describe('validation', () => {
    it('rejects a truncated payload rather than reading garbage', () => {
      const encoded: EncodedArray = {
        dtype: 'f32',
        shape: [100],
        layout: 'dense',
        transform: 'linear',
        scale: 1,
        offset: 0,
        encoding: 'base64',
        data: bytesToBase64(new Uint8Array(8)),
      };
      expect(() => decodeArray(encoded)).toThrowError(RangeError);
    });
  });
});
