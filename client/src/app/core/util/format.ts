import { Token } from '../models/session.model';

/**
 * Token pieces back to the text they came from.
 *
 * Rendering a piece for a human is the tokenizer's job -- only it holds the
 * byte alphabet -- so that lives on `ByteLevelBpeTokenizer.display`.
 */
export function detokenize(tokens: readonly Token[]): string {
  return tokens
    .map((t) => t.text.replace(/Ġ/g, ' ').replace(/Ċ/g, '\n'))
    .join('');
}

export function formatProb(p: number): string {
  if (p >= 0.0995) return p.toFixed(2);
  if (p >= 0.001) return p.toFixed(3);
  return p.toExponential(1);
}

export function formatFixed(v: number, digits = 2): string {
  if (!isFinite(v)) return '—';
  const abs = Math.abs(v);
  if (abs !== 0 && (abs < 1e-3 || abs >= 1e5)) return v.toExponential(1);
  return v.toFixed(digits);
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} kB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/** 'L7.attention' -> 'L7 attention'; 'final_norm' -> 'final norm'. */
export function formatStageId(id: string): string {
  return id.replace('.', ' ').replace(/_/g, ' ');
}
