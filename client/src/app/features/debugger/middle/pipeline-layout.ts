import { ModelInfo } from '../../../core/models/model-info.model';
import {
  POST_LAYER_STAGES,
  PRE_LAYER_STAGES,
  StageId,
  StageKind,
  stageId,
} from '../../../core/models/pipeline.model';

/**
 * Pure geometry for the pipeline graph. No Angular imports: this is a function
 * from a model description to rectangles, which makes it unit-testable and
 * keeps the component's template a dumb `@for`.
 */

export interface GraphNode {
  stageId: StageId;
  kind: StageKind;
  label: string;
  /** Right-aligned annotation, e.g. '512 -> 30000'. */
  note: string;
  layer?: number;
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface GraphRow {
  index: number;
  y: number;
  h: number;
  /** 'L7' for layer rows, '' otherwise. */
  label: string;
  /** Which breakpoint the gutter cell toggles for this row. */
  primaryStageId: StageId;
  /** Every stage on this row, for "does this row contain the PC" tests. */
  stageIds: StageId[];
  layer?: number;
}

export interface HeadCell {
  layer: number;
  head: number;
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface GraphLayout {
  width: number;
  height: number;
  gutterWidth: number;
  rows: GraphRow[];
  nodes: GraphNode[];
  headCells: HeadCell[];
  /** Vertical connector segments between consecutive rows. */
  connectors: { x: number; y1: number; y2: number }[];
  nodeById: ReadonlyMap<StageId, GraphNode>;
  rowByStage: ReadonlyMap<StageId, GraphRow>;
}

export interface LayoutOptions {
  /** Collapses the layer ladder so the whole pipeline fits without scrolling. */
  compact?: boolean;
}

const WIDTH = 300;
const GUTTER = 18;
const LABEL_W = 24;
const PAD_TOP = 6;
const GLOBAL_H = 22;
const GLOBAL_GAP = 5;
const LAYER_H = 18;
const LAYER_H_COMPACT = 9;
const LADDER_GAP = 7;

const NODE_X = GUTTER + LABEL_W;
const NODE_W = WIDTH - NODE_X - 6;

/** How wide the attention cell is within a layer row; the FFN cell takes the rest. */
const ATTN_FRACTION = 0.6;

export function layoutPipeline(model: ModelInfo, options: LayoutOptions = {}): GraphLayout {
  const compact = options.compact ?? false;
  const layerH = compact ? LAYER_H_COMPACT : LAYER_H;

  const rows: GraphRow[] = [];
  const nodes: GraphNode[] = [];
  const headCells: HeadCell[] = [];
  const connectors: { x: number; y1: number; y2: number }[] = [];

  const labelFor = (kind: StageKind) =>
    model.stages.find((s) => s.kind === kind)?.label ?? kind;

  let y = PAD_TOP;
  let index = 0;

  const pushGlobal = (kind: StageKind, note: string) => {
    const id = stageId({ kind });
    const node: GraphNode = {
      stageId: id,
      kind,
      label: labelFor(kind),
      note,
      x: NODE_X,
      y,
      w: NODE_W,
      h: GLOBAL_H,
    };
    nodes.push(node);
    rows.push({ index: index++, y, h: GLOBAL_H, label: '', primaryStageId: id, stageIds: [id] });
    y += GLOBAL_H + GLOBAL_GAP;
  };

  pushGlobal('tokenize', 'byte-level BPE');
  for (const kind of PRE_LAYER_STAGES) {
    pushGlobal(kind, `tok + pos, ${model.hiddenSize}d`);
  }

  // --- the layer ladder ------------------------------------------------------

  const ladderTop = y;
  const attnW = Math.round(NODE_W * ATTN_FRACTION);
  const ffnW = NODE_W - attnW - 4;
  const headW = compact ? 0 : Math.max(2, Math.floor((attnW * 0.34) / model.numHeads) - 1);

  for (let layer = 0; layer < model.numLayers; layer++) {
    const attnId = stageId({ kind: 'attention', layer });
    const ffnId = stageId({ kind: 'ffn', layer });
    const h = layerH - 2;

    nodes.push({
      stageId: attnId,
      kind: 'attention',
      label: compact ? '' : 'attn',
      note: '',
      layer,
      x: NODE_X,
      y,
      w: attnW,
      h,
    });
    nodes.push({
      stageId: ffnId,
      kind: 'ffn',
      label: compact ? '' : 'ffn',
      note: '',
      layer,
      x: NODE_X + attnW + 4,
      y,
      w: ffnW,
      h,
    });

    if (!compact) {
      // Eight density marks inside the attention cell. They double as the
      // head selector, which is why heads need no nodes of their own.
      for (let head = 0; head < model.numHeads; head++) {
        headCells.push({
          layer,
          head,
          x: NODE_X + 4 + head * (headW + 1),
          y: y + 3,
          w: headW,
          h: h - 6,
        });
      }
    }

    rows.push({
      index: index++,
      y,
      h,
      label: `L${layer}`,
      primaryStageId: attnId,
      stageIds: [attnId, ffnId],
      layer,
    });
    y += layerH;
  }

  connectors.push({ x: NODE_X - 8, y1: ladderTop, y2: y - (layerH - LAYER_H) });
  y += LADDER_GAP;

  // --- head ------------------------------------------------------------------

  const notes: Partial<Record<StageKind, string>> = {
    lm_head: `${model.hiddenSize} -> ${model.vocabSize}`,
    final_norm: 'RMSNorm',
  };
  for (const kind of POST_LAYER_STAGES) {
    pushGlobal(kind, notes[kind] ?? '');
  }

  const height = y - GLOBAL_GAP + PAD_TOP;

  return {
    width: WIDTH,
    height,
    gutterWidth: GUTTER,
    rows,
    nodes,
    headCells,
    connectors,
    nodeById: new Map(nodes.map((n) => [n.stageId, n])),
    rowByStage: new Map(rows.flatMap((r) => r.stageIds.map((id) => [id, r] as const))),
  };
}
