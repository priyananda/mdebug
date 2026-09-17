import { Token } from '../models/session.model';

/**
 * Token text is raw byte-level BPE output: leading spaces arrive as 'Ġ' and
 * newlines as 'Ċ'. Rendering those literally makes the token stream unreadable,
 * and rendering them as actual whitespace makes token boundaries invisible.
 * Both problems go away with visible substitutes.
 */
export function displayToken(text: string): string {
  return text
    .replace(/Ġ/g, '·') // Ġ — byte-level BPE's space
    .replace(/Ċ/g, '⏎') // Ċ — newline
    .replace(/ /g, '·')
    .replace(/\n/g, '⏎')
    .replace(/\t/g, '⇥');
}

/** Reverses displayToken for assembling readable output text. */
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
