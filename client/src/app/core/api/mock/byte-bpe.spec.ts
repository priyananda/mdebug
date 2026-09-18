import { ByteLevelBpeTokenizer } from './byte-bpe';

/**
 * Ground truth for these cases was produced by the real HuggingFace
 * `ByteLevelBPETokenizer` loaded from server/data/tokenizer/. If this file's
 * implementation drifts from the library's, these fail.
 */
const EXPECTED: Record<string, [number, string][]> = {
  'The key to happiness is': [
    [333, 'The'],
    [1273, 'Ġkey'],
    [281, 'Ġto'],
    [1339, 'Ġhappiness'],
    [316, 'Ġis'],
  ],
  unbelievable: [
    [366, 'un'],
    [70, 'b'],
    [426, 'el'],
    [388, 'ie'],
    [90, 'v'],
    [747, 'able'],
  ],
  'Hello, world!': [
    [4215, 'Hello'],
    [16, ','],
    [635, 'Ġworld'],
    [5, '!'],
  ],
  'logotherapy and ikigai': [
    [80, 'l'],
    [1001, 'ogotherapy'],
    [290, 'Ġand'],
    [598, 'Ġikigai'],
  ],
  'café 42 tokens': [
    [71, 'c'],
    [69, 'a'],
    [4585, 'fÃ©'],
    [3710, 'Ġ4'],
    [22, '2'],
    [281, 'Ġto'],
    [79, 'k'],
    [852, 'ens'],
  ],
  'a\nb  c': [
    [69, 'a'],
    [203, 'Ċ'],
    [70, 'b'],
    [225, 'Ġ'],
    [284, 'Ġc'],
  ],
};

describe('ByteLevelBpeTokenizer', () => {
  let tok: ByteLevelBpeTokenizer;

  beforeAll(async () => {
    const [vocab, merges] = await Promise.all([
      fetch('/mock/vocab.json').then((r) => r.json() as Promise<Record<string, number>>),
      fetch('/mock/merges.txt').then((r) => r.text()),
    ]);
    tok = new ByteLevelBpeTokenizer({ vocab, merges });
  });

  it('loads the trained vocabulary', () => {
    expect(tok.size).toBe(6258);
    expect(tok.unkId).toBe(3);
  });

  for (const [text, expected] of Object.entries(EXPECTED)) {
    it(`matches the reference tokenizer for ${JSON.stringify(text)}`, () => {
      expect(tok.encode(text).map((t) => [t.id, t.text])).toEqual(expected);
    });
  }

  it('round-trips text through encode/decode', () => {
    const samples = [
      'The key to happiness is',
      'Hello, world!',
      'café 42 tokens',
      'a\nb  c',
      'Mixed CASE with punctuation -- and "quotes".',
    ];
    for (const s of samples) {
      expect(tok.decode(tok.encode(s).map((t) => t.id))).toBe(s);
    }
  });

  it('never emits unk for ASCII input, since every byte is in the vocab', () => {
    const ids = tok.encode('zzqxj ~!@#$%^&*()_+ 0123456789').map((t) => t.id);
    expect(ids).not.toContain(tok.unkId);
  });

  describe('display', () => {
    // Must stay in step with `display_token` in server/app/runtime.py.
    it('decodes multi-byte pieces back to real characters', () => {
      // A curly quote is three characters in the byte alphabet.
      expect(tok.display('\u0120\u00e2\u0122\u013e')).toBe('\u00b7\u201c');
      expect(tok.display('\u00e2\u0122\u0136')).toBe('\u2014');
    });

    it('makes whitespace visible so token boundaries stay legible', () => {
      expect(tok.display('\u0120key')).toBe('\u00b7key');
      expect(tok.display('\u010a')).toBe('\u23ce');
      expect(tok.display('The')).toBe('The');
    });

    it('shows a byte fragment as hex rather than a replacement character', () => {
      // Byte 0xA1 cannot stand alone as UTF-8; which byte it is matters.
      expect(tok.display('\u00a1')).toBe('\\xA1');
    });

    it('leaves special tokens alone', () => {
      expect(tok.display('<s>')).toBe('<s>');
      expect(tok.display('<unk>')).toBe('<unk>');
    });

    it('renders every vocabulary entry without throwing', () => {
      for (let id = 0; id < tok.size; id++) {
        expect(typeof tok.display(tok.tokenText(id))).toBe('string');
      }
    });
  });

  it('treats the five reserved ids as special', () => {
    expect([0, 1, 2, 3, 4].every((id) => tok.isSpecial(id))).toBeTrue();
    expect(tok.isSpecial(5)).toBeFalse();
  });
});
