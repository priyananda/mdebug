/**
 * Byte-level BPE, matching the HuggingFace `ByteLevelBPETokenizer` that
 * produced server/data/tokenizer/{vocab.json,merges.txt}.
 *
 * This is a real implementation, not a fake one. The vocabulary is only 6,258
 * entries (85 kB), so shipping it to the browser is free, and it means the
 * mock's token ids and segmentation boundaries are genuinely correct. A
 * technical user will immediately test whether a long word splits sensibly, and
 * it should.
 */

/**
 * GPT-2's byte <-> unicode table. Bytes that are not printable ASCII are mapped
 * into a private run starting at U+0100, so that any byte sequence becomes a
 * string with no whitespace or control characters in it. This is why a leading
 * space shows up as the glyph at U+0120 (0x20 + 0x100).
 */
function bytesToUnicode(): Map<number, string> {
  const bs: number[] = [];
  for (let i = 0x21; i <= 0x7e; i++) bs.push(i);
  for (let i = 0xa1; i <= 0xac; i++) bs.push(i);
  for (let i = 0xae; i <= 0xff; i++) bs.push(i);

  const printable = new Set(bs);
  const cs = [...bs];
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!printable.has(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }
  return new Map(bs.map((b, i) => [b, String.fromCodePoint(cs[i])]));
}

const BYTE_ENCODER = bytesToUnicode();
const BYTE_DECODER = new Map([...BYTE_ENCODER].map(([b, c]) => [c, b]));

/**
 * GPT-2's pre-tokenizer pattern: contractions, then runs of letters, digits or
 * symbols each optionally preceded by one space, then whitespace.
 */
const PRETOKENIZE =
  /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

/**
 * Merge pairs are joined with a space. Safe because the byte mapping above
 * guarantees no token string ever contains one.
 */
function pairKey(a: string, b: string): string {
  return `${a} ${b}`;
}

export interface BpeFiles {
  vocab: Record<string, number>;
  merges: string;
}

export class ByteLevelBpeTokenizer {
  private readonly vocab: Map<string, number>;
  private readonly inverse: Map<number, string>;
  /** Merge pair -> rank. Lower rank merges first. */
  private readonly ranks: Map<string, number>;
  private readonly cache = new Map<string, string[]>();

  readonly size: number;
  readonly unkId: number;

  constructor(files: BpeFiles) {
    this.vocab = new Map(Object.entries(files.vocab));
    this.inverse = new Map([...this.vocab].map(([t, i]) => [i, t]));
    this.size = this.vocab.size;
    this.unkId = this.vocab.get('<unk>') ?? 3;

    this.ranks = new Map();
    let rank = 0;
    for (const line of files.merges.split('\n')) {
      if (!line || line.startsWith('#version')) continue;
      const sp = line.indexOf(' ');
      if (sp < 0) continue;
      this.ranks.set(pairKey(line.slice(0, sp), line.slice(sp + 1).trim()), rank++);
    }
  }

  /** Applies the merge list to one pre-tokenized piece. */
  private bpe(piece: string): string[] {
    const cached = this.cache.get(piece);
    if (cached) return cached;

    let symbols = [...piece];
    while (symbols.length > 1) {
      let bestRank = Infinity;
      let bestAt = -1;
      for (let i = 0; i < symbols.length - 1; i++) {
        const rank = this.ranks.get(pairKey(symbols[i], symbols[i + 1]));
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestAt = i;
        }
      }
      if (bestAt < 0) break;
      symbols = [
        ...symbols.slice(0, bestAt),
        symbols[bestAt] + symbols[bestAt + 1],
        ...symbols.slice(bestAt + 2),
      ];
    }

    this.cache.set(piece, symbols);
    return symbols;
  }

  /** Text -> token strings, in the tokenizer's own alphabet. */
  encodeToPieces(text: string): string[] {
    const utf8 = new TextEncoder();
    const out: string[] = [];
    for (const match of text.matchAll(PRETOKENIZE)) {
      let mapped = '';
      for (const byte of utf8.encode(match[0])) mapped += BYTE_ENCODER.get(byte)!;
      out.push(...this.bpe(mapped));
    }
    return out;
  }

  encode(text: string): { id: number; text: string }[] {
    return this.encodeToPieces(text).map((piece) => ({
      id: this.vocab.get(piece) ?? this.unkId,
      text: piece,
    }));
  }

  tokenText(id: number): string {
    return this.inverse.get(id) ?? '<unk>';
  }

  /** Token strings -> the original text, undoing the byte mapping. */
  decodePieces(pieces: readonly string[]): string {
    const bytes: number[] = [];
    for (const piece of pieces) {
      for (const ch of piece) {
        const b = BYTE_DECODER.get(ch);
        if (b !== undefined) bytes.push(b);
      }
    }
    return new TextDecoder('utf-8', { fatal: false }).decode(new Uint8Array(bytes));
  }

  decode(ids: readonly number[]): string {
    return this.decodePieces(ids.map((id) => this.tokenText(id)));
  }

  /** True for the five reserved tokens, which should never be sampled. */
  isSpecial(id: number): boolean {
    return id <= 4;
  }
}

let loaded: Promise<ByteLevelBpeTokenizer> | null = null;

/** Loads and caches the tokenizer. Both files together are ~134 kB. */
export function loadTokenizer(base = 'mock/'): Promise<ByteLevelBpeTokenizer> {
  loaded ??= (async () => {
    const [vocabRes, mergesRes] = await Promise.all([
      fetch(`${base}vocab.json`),
      fetch(`${base}merges.txt`),
    ]);
    if (!vocabRes.ok || !mergesRes.ok) {
      throw new Error('Could not load the tokenizer files from public/mock/');
    }
    return new ByteLevelBpeTokenizer({
      vocab: (await vocabRes.json()) as Record<string, number>,
      merges: await mergesRes.text(),
    });
  })();
  return loaded;
}
