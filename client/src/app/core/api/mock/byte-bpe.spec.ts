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

  it('treats the five reserved ids as special', () => {
    expect([0, 1, 2, 3, 4].every((id) => tok.isSpecial(id))).toBeTrue();
    expect(tok.isSpecial(5)).toBeFalse();
  });
});
