import { BreakpointCondition } from '../../../core/models/breakpoint.model';
import { StageId } from '../../../core/models/pipeline.model';

export interface ScenarioBreakpoint {
  stageId: StageId;
  condition?: BreakpointCondition;
}

export interface Scenario {
  id: string;
  title: string;
  /** What the user is about to see, in one sentence. */
  blurb: string;
  /** Where to look once it halts. */
  lookFor: string;
  prompt?: string;
  breakpoints: ScenarioBreakpoint[];
  /** Layer and head to preselect in the right column. */
  focus?: { layer: number; head?: number };
}

/**
 * Preloaded investigations.
 *
 * Each one sets a prompt, installs breakpoints and points the right column at
 * the thing worth looking at, so that pressing Start immediately does something
 * interesting rather than leaving the user to guess where to break.
 *
 * The phenomena are real ones — attention sinks, induction heads, residual norm
 * growth through depth — so learning to spot them here transfers.
 */
export const SCENARIOS: readonly Scenario[] = [
  {
    id: 'sink',
    title: 'Watch the attention sink form',
    blurb:
      'Deep layers dump a large share of their attention on position 0, whatever that token is.',
    lookFor:
      'The bright left-hand column in the heatmap, and the sink figure under it. Switch between L1 and L19 to see it grow.',
    prompt: 'The key to happiness is not found in wealth, but in paying attention.',
    breakpoints: [{ stageId: 'L19.attention' }],
    focus: { layer: 19, head: 0 },
  },
  {
    id: 'induction',
    title: 'Find an induction-style head',
    blurb:
      'Some middle-layer heads attend to a fixed offset back rather than to the neighbouring token.',
    lookFor:
      'A stripe running parallel to the diagonal, offset from it. Click through heads 0-7 in the ladder; most look local, one or two do not.',
    prompt: 'Monday Tuesday Wednesday Monday Tuesday Wednesday Monday Tuesday',
    breakpoints: [{ stageId: 'L10.attention' }],
    focus: { layer: 10, head: 0 },
  },
  {
    id: 'residual-growth',
    title: 'See the residual norm grow with depth',
    blurb: 'The hidden state gets geometrically larger as it passes through the stack.',
    lookFor:
      'The by-layer trend line in the residual panel. Press Step repeatedly and watch it climb; the outlier dimensions stay put while everything around them scales.',
    breakpoints: [
      { stageId: 'L0.ffn' },
      { stageId: 'L5.ffn' },
      { stageId: 'L10.ffn' },
      { stageId: 'L15.ffn' },
      { stageId: 'L19.ffn' },
    ],
    focus: { layer: 0 },
  },
  {
    id: 'low-confidence',
    title: 'Catch a low-confidence token',
    blurb: 'Run freely, but halt only when the model is genuinely unsure what comes next.',
    lookFor:
      'A conditional breakpoint on Sample. It skips every confident step and stops at the first one where the top choice is under 30%.',
    breakpoints: [
      { stageId: 'sample', condition: { kind: 'top1_prob', op: '<', value: 0.3 } },
    ],
  },
  {
    id: 'kv-fill',
    title: 'Watch the KV cache fill',
    blurb: 'Every emitted token adds one column of keys and values, in every layer.',
    lookFor:
      'The KV grid. Continue repeatedly: a new amber column appears per token, and during a step you can see it exists only in the layers already executed.',
    breakpoints: [{ stageId: 'emit' }],
  },
  {
    id: 'tokenizer',
    title: 'Inspect the tokenizer',
    blurb:
      'This model has a 6,258-entry vocabulary trained on one book, so it splits ordinary words oddly.',
    lookFor:
      'The token chips under the prompt. "unbelievable" becomes un|b|el|ie|v|able. Hover a chip for its id.',
    prompt: 'It was unbelievable, unquestionably extraordinary, and altogether unrepeatable.',
    breakpoints: [{ stageId: 'tokenize' }],
  },
];
