import { Observable, Subject } from 'rxjs';

import {
  Breakpoint,
  BreakpointCondition,
  compare,
} from '../../models/breakpoint.model';
import { ServerEvent, STAGE_EVENT_RATE_LIMIT } from '../../models/events.model';
import { ModelInfo } from '../../models/model-info.model';
import {
  StageId,
  StageKind,
  StageRef,
  stageId,
  stageSequence,
} from '../../models/pipeline.model';
import { HaltPayload, HaltPosition, HaltReason } from '../../models/run-state.model';
import { SessionConfig, SessionInfo, Token } from '../../models/session.model';
import { KvSnapshot, TopKEntry, TopKLogits } from '../../models/tensors.model';
import { encodeArray } from '../../util/encoding';
import { ByteLevelBpeTokenizer } from './byte-bpe';
import {
  attentionRow,
  headPersonality,
  headSummary,
  kvKeyNorms,
  kvOccupancy,
  residualVector,
  statsOf,
} from './fake-tensors';
import { Rng, powerLawIndex, rngFor, uniform } from './rng';

/** How long each kind of stage "takes". A full step lands around 650ms at 1x. */
const STAGE_MS: Record<StageKind, number> = {
  tokenize: 60,
  embed: 18,
  attention: 20,
  ffn: 12,
  final_norm: 10,
  lm_head: 40,
  sample: 25,
  emit: 15,
};

const TOP_K = 50;

class StopSignal extends Error {
  constructor() {
    super('stopped');
    this.name = 'StopSignal';
  }
}

/**
 * The simulated run loop.
 *
 * This mirrors the server's control semantics exactly, because `DebuggerStore`
 * must not be able to tell the difference between this and a real backend. It
 * is also the reference implementation of halt and breakpoint semantics — the
 * Python side should behave identically, and `docs/api-contract.md` describes
 * what this does.
 */
export class MockEngine {
  private readonly events = new Subject<ServerEvent>();
  readonly events$: Observable<ServerEvent> = this.events.asObservable();

  private breakpoints: Breakpoint[] = [];
  private running = false;
  private resume: { resolve: () => void; reject: (e: unknown) => void } | null = null;
  /** Halt points still to pass before stopping. Infinity while `continue`ing. */
  private stepBudget = Infinity;
  private nextHaltReason: HaltReason = 'step';
  private stopped = false;

  /** Rate limiting for `stage_entered`, per the contract's 30/sec guarantee. */
  private lastStageEventAt = 0;

  /** Cached per-step sampling decisions, so a step is stable once taken. */
  private readonly samples = new Map<number, TopKLogits>();
  private promptTokensSent = false;
  private generatedTokensSent = 0;

  /** Wall-clock speed multiplier. 1 = the timings above; higher is faster. */
  speed = 1;
  /** Dev-only: fraction of on-demand tensor fetches that should fail. */
  failureRate = 0;

  constructor(
    private session: SessionInfo,
    private readonly model: ModelInfo,
    private readonly tokenizer: ByteLevelBpeTokenizer,
  ) {
    this.breakpoints = [...session.breakpoints];
  }

  get state(): SessionInfo {
    return this.session;
  }

  private tokenizePrompt(prompt: string): Token[] {
    return this.tokenizer.encode(prompt).map((t, i) => ({
      id: t.id,
      text: t.text,
      display: this.tokenizer.display(t.text),
      position: i,
      isSpecial: this.tokenizer.isSpecial(t.id),
      origin: 'prompt' as const,
    }));
  }

  // --- commands -------------------------------------------------------------

  start(configPatch?: Partial<SessionConfig>): void {
    if (this.running) return;
    const config = { ...this.session.config, ...configPatch };
    const promptTokens =
      config.prompt === this.session.config.prompt && this.session.promptTokens.length
        ? this.session.promptTokens
        : this.tokenizePrompt(config.prompt);
    // Start restarts, the way it does in a real debugger. Continue is what
    // resumes; there is no "start from where I stopped".
    this.session = {
      ...this.session,
      config,
      promptTokens,
      generatedTokens: [],
      currentStep: 0,
      halt: null,
    };
    this.samples.clear();
    this.generatedTokensSent = 0;
    this.promptTokensSent = false;
    this.stopped = false;
    this.stepBudget = Infinity;
    this.nextHaltReason = 'breakpoint';
    void this.drive();
  }

  continueRun(): void {
    this.stepBudget = Infinity;
    this.nextHaltReason = 'breakpoint';
    this.releaseHalt();
  }

  stepOver(count = 1): void {
    this.stepBudget = Math.max(1, count);
    this.nextHaltReason = 'step';
    this.releaseHalt();
  }

  stop(): void {
    this.stopped = true;
    if (this.resume) {
      const { reject } = this.resume;
      this.resume = null;
      reject(new StopSignal());
    }
  }

  setBreakpoints(breakpoints: Breakpoint[]): void {
    // Preserve server-side hit counts across a replace.
    const previous = new Map(this.breakpoints.map((b) => [b.id, b.hitCount]));
    this.breakpoints = breakpoints.map((b) => ({ ...b, hitCount: previous.get(b.id) ?? b.hitCount }));
    this.session = { ...this.session, breakpoints: this.breakpoints };
    this.emit({ type: 'breakpoints_changed', payload: { breakpoints: this.breakpoints } });
  }

  runToCursor(target: StageId, step?: number): void {
    this.breakpoints = [
      ...this.breakpoints,
      {
        id: `oneshot-${Date.now()}`,
        stageId: target,
        enabled: true,
        oneShot: true,
        hitCount: 0,
        condition:
          step === undefined ? undefined : { kind: 'token_index', op: '==', value: step },
      },
    ];
    if (this.running) this.continueRun();
    else this.start();
  }

  patchConfig(patch: Partial<SessionConfig>): void {
    this.session = { ...this.session, config: { ...this.session.config, ...patch } };
    // A changed temperature invalidates decisions not yet taken.
    for (const step of [...this.samples.keys()]) {
      if (step > this.session.currentStep) this.samples.delete(step);
    }
    this.emit({ type: 'session_state', payload: this.session });
  }

  emitSessionState(): void {
    this.emit({ type: 'session_state', payload: this.session });
  }

  private releaseHalt(): void {
    if (!this.resume) return;
    const { resolve } = this.resume;
    this.resume = null;
    resolve();
  }

  // --- the driver -----------------------------------------------------------

  private async drive(): Promise<void> {
    this.running = true;
    this.session = { ...this.session, status: 'running' };

    const startStep = Math.max(0, this.session.currentStep);
    this.emit({ type: 'run_started', payload: { step: startStep } });

    try {
      for (let step = startStep; step < this.session.config.maxNewTokens; step++) {
        this.session = { ...this.session, currentStep: step };
        for (const stage of stageSequence(this.model.numLayers, step)) {
          await this.sleep(STAGE_MS[stage.kind]);
          this.announceStage(step, stage);

          if (stage.kind === 'emit') this.emitToken(step);

          const reason = this.haltReasonFor(step, stage);
          if (reason) await this.haltAt(step, stage, reason);
        }
      }

      this.finish('max_tokens');
    } catch (e) {
      if (e instanceof StopSignal) this.finish('stopped');
      else {
        this.running = false;
        this.session = { ...this.session, status: 'error' };
        this.emit({
          type: 'error',
          payload: {
            code: 'mock_engine_failure',
            message: e instanceof Error ? e.message : String(e),
            fatal: true,
          },
        });
      }
    }
  }

  private finish(reason: 'max_tokens' | 'eos' | 'stopped'): void {
    this.running = false;
    this.session = {
      ...this.session,
      status: reason === 'stopped' ? 'idle' : 'finished',
      halt: null,
    };
    this.emit({
      type: 'finished',
      payload: {
        reason,
        totalSteps: this.session.generatedTokens.length,
        text: this.tokenizer.decodePieces(
          [...this.session.promptTokens, ...this.session.generatedTokens].map((t) => t.text),
        ),
      },
    });
  }

  /**
   * `stage_entered` is pure animation data at ~45 events per token. Coalescing
   * it here rather than in the transport means the mock exercises the same
   * dropped-frame behaviour the real server is required to have.
   */
  private announceStage(step: number, stage: StageRef): void {
    const now = performance.now();
    if (now - this.lastStageEventAt < 1000 / STAGE_EVENT_RATE_LIMIT) return;
    this.lastStageEventAt = now;
    this.emit({
      type: 'stage_entered',
      payload: {
        step,
        stageId: stageId(stage),
        sequenceLength: this.sequenceLength(step),
      },
    });
  }

  private async haltAt(step: number, stage: StageRef, reason: HaltReason): Promise<void> {
    const position: HaltPosition = {
      step,
      stage,
      stageId: stageId(stage),
      sequenceLength: this.sequenceLength(step),
      reason,
      breakpointId: this.lastHitBreakpointId,
    };

    this.session = { ...this.session, status: 'halted', halt: position };
    // A halting stage is never dropped, whatever the rate limiter said.
    this.lastStageEventAt = 0;
    this.emit({ type: 'halted', payload: this.buildHaltPayload(position) });

    await new Promise<void>((resolve, reject) => {
      this.resume = { resolve, reject };
    });

    this.session = { ...this.session, status: 'running', halt: null };
    this.emit({ type: 'resumed', payload: { from: position } });
  }

  private sleep(ms: number): Promise<void> {
    const jittered = (ms * uniform(Math.random, 0.6, 1.4)) / Math.max(0.01, this.speed);
    return new Promise((resolve, reject) => {
      const id = setTimeout(() => {
        if (this.stopped) reject(new StopSignal());
        else resolve();
      }, jittered);
      // Stop must unwind the loop promptly, not after the current delay.
      if (this.stopped) {
        clearTimeout(id);
        reject(new StopSignal());
      }
    });
  }

  // --- breakpoints ----------------------------------------------------------

  private lastHitBreakpointId: string | undefined;

  private haltReasonFor(step: number, stage: StageRef): HaltReason | null {
    this.lastHitBreakpointId = undefined;
    const id = stageId(stage);

    for (const bp of this.breakpoints) {
      if (!bp.enabled || bp.stageId !== id) continue;
      if (bp.condition && !this.evaluate(bp.condition, step, stage)) continue;

      bp.hitCount++;
      this.lastHitBreakpointId = bp.id;
      if (bp.oneShot) {
        this.breakpoints = this.breakpoints.filter((b) => b.id !== bp.id);
        this.emit({ type: 'breakpoints_changed', payload: { breakpoints: this.breakpoints } });
      }
      return bp.oneShot ? 'run_to_cursor' : 'breakpoint';
    }

    if (this.stepBudget !== Infinity) {
      this.stepBudget--;
      if (this.stepBudget <= 0) return this.nextHaltReason;
    }
    return null;
  }

  /**
   * Condition evaluation. The Python server should implement exactly this, as
   * dict dispatch — see docs/api-contract.md section 5.
   */
  private evaluate(c: BreakpointCondition, step: number, stage: StageRef): boolean {
    switch (c.kind) {
      case 'token_index':
        return compare(c.op, step, c.value);
      case 'sequence_length':
        return compare(c.op, this.sequenceLength(step), c.value);
      case 'hit_count': {
        const bp = this.breakpoints.find((b) => b.stageId === stageId(stage));
        return compare(c.op, bp?.hitCount ?? 0, c.value);
      }
      case 'top1_prob': {
        const top = this.sampleFor(step).entries[0];
        return compare(c.op, top?.prob ?? 0, c.value);
      }
      case 'logit_entropy':
        return compare(c.op, this.sampleFor(step).entropy, c.value);
      case 'residual_norm': {
        const layer = stage.layer ?? 0;
        return compare(c.op, statsOf(residualVector(this.session.id, this.sequenceLength(step) - 1, layer, this.model.hiddenSize)).l2, c.value);
      }
      case 'attention_max':
        return compare(c.op, this.attentionMax(step, stage.layer ?? 0, c.head), c.value);
      case 'emitted_token_text': {
        const text = this.tokenizer.display(this.tokenizer.tokenText(this.sampleFor(step).chosenTokenId));
        return c.op === 'equals' ? text === c.value : text.includes(c.value);
      }
      case 'emitted_token_id':
        return this.sampleFor(step).chosenTokenId === c.value;
    }
  }

  /** Max attention weight on the current query row, over one head or all of them. */
  private attentionMax(step: number, layer: number, head?: number): number {
    const t = this.sequenceLength(step);
    const row = new Float32Array(t);
    const heads = head === undefined ? [...Array(this.model.numHeads).keys()] : [head];
    let max = 0;
    for (const h of heads) {
      attentionRow(
        headPersonality(this.session.id, layer, h, this.model.numLayers),
        t - 1,
        row,
        rngFor(this.session.id, 'attn', step, layer, h),
      );
      for (let j = 0; j < t; j++) if (row[j] > max) max = row[j];
    }
    return max;
  }

  // --- state payloads -------------------------------------------------------

  private sequenceLength(step: number): number {
    return this.session.promptTokens.length + step;
  }

  /** How many layers of the current step have executed at this stage. */
  private layersDone(stage: StageRef): number {
    // A layer's K/V are written during its attention stage, so by the time we
    // halt at either per-layer stage that layer is resident.
    if (stage.layer !== undefined) return stage.layer + 1;
    return stage.kind === 'tokenize' || stage.kind === 'embed' ? 0 : this.model.numLayers;
  }

  private buildHaltPayload(position: HaltPosition): HaltPayload {
    const { step, stage } = position;
    const t = position.sequenceLength;
    const layer = stage.layer ?? 0;

    const residual = residualVector(this.session.id, t - 1, layer, this.model.hiddenSize);
    const stats = statsOf(residual);

    const payload: HaltPayload = {
      position,
      sequence: {
        promptTokens: this.promptTokensSent ? undefined : this.session.promptTokens,
        generatedTokens: this.session.generatedTokens.slice(this.generatedTokensSent),
      },
      headSummary: encodeArray(
        headSummary(this.session.id, step, this.model.numLayers, this.model.numHeads, t),
        { shape: [this.model.numLayers, this.model.numHeads] },
      ),
      residual: encodeArray(residual, { shape: [this.model.hiddenSize] }),
      residualStats: stats,
      kv: this.buildKv(step, t, stage),
      timings: { stageMs: STAGE_MS[stage.kind], stepMs: 0 },
    };

    if (stage.kind === 'lm_head' || stage.kind === 'sample' || stage.kind === 'emit') {
      payload.topK = this.sampleFor(step);
    }

    this.promptTokensSent = true;
    this.generatedTokensSent = this.session.generatedTokens.length;
    return payload;
  }

  private buildKv(step: number, t: number, stage: StageRef): KvSnapshot {
    const layersDone = this.layersDone(stage);
    const occupancy = kvOccupancy(this.model.numLayers, t, t - 1, t - 1, layersDone);
    return {
      numLayers: this.model.numLayers,
      sequenceLength: t,
      occupancy: encodeArray(occupancy, {
        shape: [this.model.numLayers, t],
        dtype: 'u8',
      }),
      keyNorms: encodeArray(kvKeyNorms(this.session.id, this.model.numLayers, t), {
        shape: [this.model.numLayers, t],
      }),
      // 2 tensors * layers * positions * headDim * numHeads * 4 bytes
      bytesResident:
        2 * this.model.numLayers * t * this.model.headDim * this.model.numHeads * 4,
      // This model recomputes the whole prefix every step; there is no cache.
      simulated: !this.model.hasKvCache,
    };
  }

  // --- sampling -------------------------------------------------------------

  /**
   * The sampling decision for a step, memoized so that revisiting a step shows
   * the same numbers. Candidates are drawn from the real vocabulary with a
   * power law over token id, which approximates frequency because BPE assigns
   * low ids to common merges.
   */
  private sampleFor(step: number): TopKLogits {
    const cached = this.samples.get(step);
    if (cached) return cached;

    const rng = rngFor(this.session.id, 'sample', step, this.session.config.seed ?? 0);
    const config = this.session.config;
    const temperature =
      config.samplingMode === 'greedy' ? 0 : (config.temperature ?? 1);

    const candidates = this.drawCandidates(rng);

    // Each step has its own intrinsic confidence. Without this every step looks
    // identically uncertain, and a condition like `top1_prob < 0.3` either fires
    // always or never -- which makes it useless as a demonstration and, worse,
    // misrepresents how a real model behaves.
    const confidence = uniform(rng, 0.12, 0.96);
    const top =
      temperature === 0
        ? 0.999
        : Math.min(0.98, Math.max(0.03, Math.pow(confidence, temperature)));

    // The remaining mass decays geometrically down the ranking.
    const tail = candidates.slice(1).map((_, i) => Math.pow(0.72, i));
    const tailTotal = tail.reduce((a, b) => a + b, 0) || 1;
    const probs = [top, ...tail.map((w) => ((1 - top) * w) / tailTotal)];

    const entries: TopKEntry[] = candidates.map((id, i) => {
      const text = this.tokenizer.tokenText(id);
      return {
        tokenId: id,
        text,
        display: this.tokenizer.display(text),
        logit: Math.log(Math.max(probs[i], 1e-12)) * Math.max(0.2, temperature || 0.2) + 6,
        prob: probs[i],
      };
    });

    const chosenRank = temperature === 0 ? 0 : this.drawRank(rng, probs);
    const entropy = -probs.reduce((h, p) => h + (p > 0 ? p * Math.log(p) : 0), 0);

    const result: TopKLogits = {
      step,
      k: TOP_K,
      entries,
      entropy,
      temperature,
      chosenTokenId: entries[chosenRank].tokenId,
      chosenRank,
    };
    this.samples.set(step, result);
    return result;
  }

  private drawCandidates(rng: Rng): number[] {
    const limit = this.model.tokenizerVocabSize;
    const ids = new Set<number>();
    let guard = 0;
    while (ids.size < TOP_K && guard++ < TOP_K * 20) {
      const id = powerLawIndex(rng, limit, 1.6);
      if (!this.tokenizer.isSpecial(id)) ids.add(id);
    }
    return [...ids];
  }

  private drawRank(rng: Rng, probs: number[]): number {
    let u = rng();
    for (let i = 0; i < probs.length; i++) {
      u -= probs[i];
      if (u <= 0) return i;
    }
    return 0;
  }

  private emitToken(step: number): void {
    const sample = this.sampleFor(step);
    const text = this.tokenizer.tokenText(sample.chosenTokenId);
    const token: Token = {
      id: sample.chosenTokenId,
      text,
      display: this.tokenizer.display(text),
      position: this.session.promptTokens.length + step,
      isSpecial: false,
      origin: 'generated',
    };
    this.session = {
      ...this.session,
      generatedTokens: [...this.session.generatedTokens, token],
      updatedAt: new Date().toISOString(),
    };
    this.emit({ type: 'token_emitted', payload: { step, token, topK: sample } });
  }

  private emit(event: ServerEvent): void {
    this.events.next(event);
  }
}
