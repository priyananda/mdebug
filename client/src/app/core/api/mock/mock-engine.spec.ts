import { firstValueFrom } from 'rxjs';
import { filter, take } from 'rxjs/operators';

import { Breakpoint } from '../../models/breakpoint.model';
import { ServerEvent } from '../../models/events.model';
import { ModelInfo } from '../../models/model-info.model';
import { DEFAULT_STAGE_DESCRIPTORS } from '../../models/pipeline.model';
import { SessionConfig, SessionInfo } from '../../models/session.model';
import { displayToken } from '../../util/format';
import { ByteLevelBpeTokenizer } from './byte-bpe';
import { MockEngine } from './mock-engine';

const MODEL: ModelInfo = {
  name: 'test',
  numLayers: 4,
  numHeads: 4,
  hiddenSize: 64,
  headDim: 16,
  intermediateSize: 128,
  vocabSize: 30000,
  tokenizerVocabSize: 6258,
  maxPositionEmbeddings: 4096,
  ropePct: 0.25,
  ropeDim: 4,
  hasKvCache: false,
  capturesAttention: true,
  limits: {
    maxPromptTokens: 128,
    maxNewTokens: 64,
    maxTotalTokens: 192,
    attentionWarnThreshold: 512,
  },
  stages: [...DEFAULT_STAGE_DESCRIPTORS],
};

/** 4 layers: tokenize, embed, 8 layer stages, final_norm, lm_head, sample, emit. */
const STAGES_STEP_0 = 2 * MODEL.numLayers + 6;

let tokenizer: ByteLevelBpeTokenizer;

function breakpoint(partial: Partial<Breakpoint> & { stageId: string }): Breakpoint {
  return { id: partial.stageId, enabled: true, oneShot: false, hitCount: 0, ...partial };
}

function makeEngine(config: Partial<SessionConfig> = {}, breakpoints: Breakpoint[] = []): MockEngine {
  const full: SessionConfig = {
    prompt: 'The key to happiness is',
    maxNewTokens: 8,
    samplingMode: 'temperature',
    temperature: 0.8,
    captureAttention: true,
    ...config,
  };
  const session: SessionInfo = {
    id: 'spec-session',
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    status: 'idle',
    config: full,
    model: MODEL,
    promptTokens: tokenizer.encode(full.prompt).map((t, i) => ({
      id: t.id,
      text: t.text,
      display: displayToken(t.text),
      position: i,
      isSpecial: false,
      origin: 'prompt' as const,
    })),
    generatedTokens: [],
    currentStep: 0,
    halt: null,
    breakpoints,
  };
  const engine = new MockEngine(session, MODEL, tokenizer);
  // Compress the simulated timings so the suite runs in milliseconds.
  engine.speed = 200;
  return engine;
}

function nextEvent<T extends ServerEvent['type']>(
  engine: MockEngine,
  type: T,
): Promise<Extract<ServerEvent, { type: T }>> {
  return firstValueFrom(
    engine.events$.pipe(
      filter((e): e is Extract<ServerEvent, { type: T }> => e.type === type),
      take(1),
    ),
  );
}

describe('MockEngine', () => {
  beforeAll(async () => {
    const [vocab, merges] = await Promise.all([
      fetch('/mock/vocab.json').then((r) => r.json() as Promise<Record<string, number>>),
      fetch('/mock/merges.txt').then((r) => r.text()),
    ]);
    tokenizer = new ByteLevelBpeTokenizer({ vocab, merges });
  });

  it('halts at the breakpoint it was given', async () => {
    const engine = makeEngine({}, [breakpoint({ stageId: 'L2.attention' })]);
    const halted = nextEvent(engine, 'halted');
    engine.start();
    const event = await halted;

    expect(event.payload.position.stageId).toBe('L2.attention');
    expect(event.payload.position.step).toBe(0);
    expect(event.payload.position.reason).toBe('breakpoint');
    engine.stop();
  });

  it('halts at the same breakpoint again on the next token', async () => {
    const engine = makeEngine({}, [breakpoint({ stageId: 'L2.attention' })]);
    const first = nextEvent(engine, 'halted');
    engine.start();
    await first;

    const second = nextEvent(engine, 'halted');
    engine.continueRun();
    const event = await second;

    expect(event.payload.position.stageId).toBe('L2.attention');
    expect(event.payload.position.step).toBe(1);
    engine.stop();
  });

  it('advances exactly one stage per step command', async () => {
    const engine = makeEngine({}, [breakpoint({ stageId: 'embed' })]);
    const first = nextEvent(engine, 'halted');
    engine.start();
    await first;

    for (const expected of ['L0.attention', 'L0.ffn', 'L1.attention']) {
      const halted = nextEvent(engine, 'halted');
      engine.stepOver(1);
      const event = await halted;
      expect(event.payload.position.stageId).toBe(expected);
      expect(event.payload.position.reason).toBe('step');
    }
    engine.stop();
  });

  it('advances n stages for a step of n', async () => {
    const engine = makeEngine({}, [breakpoint({ stageId: 'embed' })]);
    const first = nextEvent(engine, 'halted');
    engine.start();
    await first;

    const halted = nextEvent(engine, 'halted');
    engine.stepOver(4);
    // embed -> L0.attention, L0.ffn, L1.attention, L1.ffn
    expect((await halted).payload.position.stageId).toBe('L1.ffn');
    engine.stop();
  });

  describe('conditions', () => {
    it('skips steps until token_index matches', async () => {
      const engine = makeEngine({}, [
        breakpoint({
          stageId: 'sample',
          condition: { kind: 'token_index', op: '==', value: 3 },
        }),
      ]);
      const halted = nextEvent(engine, 'halted');
      engine.start();
      const event = await halted;

      expect(event.payload.position.step).toBe(3);
      expect(event.payload.position.stageId).toBe('sample');
      engine.stop();
    });

    it('fires on sequence_length, which grows with each emitted token', async () => {
      const engine = makeEngine({}, [
        breakpoint({
          stageId: 'embed',
          condition: { kind: 'sequence_length', op: '>=', value: 7 },
        }),
      ]);
      const halted = nextEvent(engine, 'halted');
      engine.start();
      const event = await halted;

      expect(event.payload.position.sequenceLength).toBeGreaterThanOrEqual(7);
      engine.stop();
    });

    it('does not halt when the condition never holds', async () => {
      const engine = makeEngine({ maxNewTokens: 3 }, [
        breakpoint({
          stageId: 'sample',
          condition: { kind: 'token_index', op: '==', value: 99 },
        }),
      ]);
      let halts = 0;
      engine.events$.subscribe((e) => {
        if (e.type === 'halted') halts++;
      });

      const finished = nextEvent(engine, 'finished');
      engine.start();
      await finished;

      expect(halts).toBe(0);
    });
  });

  it('produces top-1 probabilities that actually vary between steps', async () => {
    // The point of the low-confidence scenario is that a `top1_prob < 0.3`
    // breakpoint skips confident steps. That only means anything if confidence
    // varies, so this guards the sampling model rather than the plumbing.
    const engine = makeEngine({ maxNewTokens: 12 }, [breakpoint({ stageId: 'sample' })]);
    const probs: number[] = [];
    engine.events$.subscribe((e) => {
      if (e.type === 'halted' && e.payload.topK) {
        probs.push(e.payload.topK.entries[0].prob);
        queueMicrotask(() => engine.continueRun());
      }
    });

    const finished = nextEvent(engine, 'finished');
    engine.start();
    await finished;

    expect(probs.length).toBe(12);
    expect(Math.min(...probs)).toBeLessThan(0.3);
    expect(Math.max(...probs)).toBeGreaterThan(0.5);
  });

  it('is deterministic: the same session id replays the same tokens', async () => {
    const run = async () => {
      const engine = makeEngine({ maxNewTokens: 5 });
      const finished = nextEvent(engine, 'finished');
      engine.start();
      await finished;
      return engine.state.generatedTokens.map((t) => t.id);
    };
    expect(await run()).toEqual(await run());
  });

  it('stops cleanly, leaving captured state inspectable', async () => {
    const engine = makeEngine({}, [breakpoint({ stageId: 'L1.ffn' })]);
    const halted = nextEvent(engine, 'halted');
    engine.start();
    await halted;

    const finished = nextEvent(engine, 'finished');
    engine.stop();
    const event = await finished;

    expect(event.payload.reason).toBe('stopped');
    expect(engine.state.status).toBe('idle');
  });

  it('restarts from scratch rather than resuming', async () => {
    const engine = makeEngine({ maxNewTokens: 4 });
    const firstRun = nextEvent(engine, 'finished');
    engine.start();
    await firstRun;
    expect(engine.state.generatedTokens.length).toBe(4);

    const secondRun = nextEvent(engine, 'finished');
    engine.start();
    await secondRun;
    expect(engine.state.generatedTokens.length).toBe(4);
  });

  it('re-tokenizes when Start carries a new prompt', async () => {
    const engine = makeEngine({ maxNewTokens: 1 });
    const finished = nextEvent(engine, 'finished');
    engine.start({ prompt: 'a much longer prompt than the original one here' });
    await finished;

    expect(engine.state.promptTokens.length).toBeGreaterThan(5);
    expect(engine.state.config.prompt).toContain('much longer');
  });

  it('emits one halt per stage when stepping through a whole token', async () => {
    const engine = makeEngine({}, [breakpoint({ stageId: 'tokenize' })]);
    const first = nextEvent(engine, 'halted');
    engine.start();
    await first;

    const seen: string[] = [];
    engine.events$.subscribe((e) => {
      if (e.type === 'halted') seen.push(e.payload.position.stageId);
    });

    for (let i = 0; i < STAGES_STEP_0 - 1; i++) {
      const halted = nextEvent(engine, 'halted');
      engine.stepOver(1);
      await halted;
    }

    expect(seen).toEqual([
      'embed',
      'L0.attention',
      'L0.ffn',
      'L1.attention',
      'L1.ffn',
      'L2.attention',
      'L2.ffn',
      'L3.attention',
      'L3.ffn',
      'final_norm',
      'lm_head',
      'sample',
      'emit',
    ]);
    engine.stop();
  });
});
