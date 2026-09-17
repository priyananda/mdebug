import { StageDescriptor } from './pipeline.model';

/**
 * Everything the client knows about the model it is debugging.
 *
 * Fetched once from `GET /api/model`. The pipeline graph, the layer/head
 * pickers and every label are rendered from this — never from constants baked
 * into the client. That keeps the UI honest if the server config changes, and
 * stops the client asserting things about the model that are not true.
 */
export interface ModelInfo {
  name: string;
  numLayers: number;
  numHeads: number;
  hiddenSize: number;
  headDim: number;
  intermediateSize: number;
  /** Size of the output projection — not necessarily the tokenizer's size. */
  vocabSize: number;
  /** Entries actually present in the trained tokenizer. May be far smaller. */
  tokenizerVocabSize: number;
  maxPositionEmbeddings: number;
  ropePct: number;
  ropeDim: number;
  /**
   * False when the server recomputes the whole prefix each step. The KV panel
   * badges itself accordingly rather than presenting a reconstruction as a
   * measurement.
   */
  hasKvCache: boolean;
  /** True when the server can return real attention matrices. */
  capturesAttention: boolean;
  limits: ModelLimits;
  stages: StageDescriptor[];
  /** Free-form caveats the server wants surfaced, e.g. naming discrepancies. */
  notes?: string[];
}

export interface ModelLimits {
  maxPromptTokens: number;
  maxNewTokens: number;
  maxTotalTokens: number;
  /** Above this T, the client asks before loading an attention tile. */
  attentionWarnThreshold: number;
}
