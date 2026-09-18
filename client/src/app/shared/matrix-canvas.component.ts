import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  ElementRef,
  afterNextRender,
  computed,
  effect,
  inject,
  input,
  output,
  signal,
  viewChild,
} from '@angular/core';

import { ColormapName, lut, normIndex, signedIndex } from '../core/util/colormap';
import { Size, sizeCanvas } from '../core/util/dom';

export interface CellHover {
  row: number;
  col: number;
  value: number;
  /** Position of the cell's centre, in CSS pixels relative to the canvas. */
  x: number;
  y: number;
}

/**
 * Renders an R x C matrix of numbers as an image.
 *
 * Canvas, not SVG, and not negotiable: a 192x192 attention matrix is 37k cells
 * and the model's configured sequence length would make it a million. The
 * technique is to fill an ImageData at exactly C x R, blit it to an offscreen
 * canvas, then draw that scaled up with smoothing off — one GPU operation, and
 * nearest-neighbour keeps the cell edges crisp at any display size.
 *
 * Shared by the attention heatmap and the KV grid so there is one hit-test and
 * one colour path, not two.
 */
@Component({
  selector: 'mdbg-matrix-canvas',
  template: `
    <canvas
      #canvas
      class="matrix"
      (pointermove)="onMove($event)"
      (pointerleave)="onLeave()"
      (click)="onClick($event)"
    ></canvas>
  `,
  styles: [
    `
      :host {
        display: block;
        position: relative;
        min-width: 0;
      }
      .matrix {
        display: block;
        cursor: crosshair;
        image-rendering: pixelated;
      }
    `,
  ],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class MatrixCanvasComponent {
  /** Dense, row-major, length rows*cols. */
  readonly values = input.required<Float32Array | null>();
  readonly rows = input.required<number>();
  readonly cols = input.required<number>();
  readonly colormap = input<ColormapName>('sequential');
  /** Normalization ceiling. When absent, the observed maximum is used. */
  readonly max = input<number | null>(null);
  /** Signed data is coloured symmetrically about zero. */
  readonly signed = input(false);
  /** Non-linear display emphasis for peaked distributions. */
  readonly scale = input<'linear' | 'sqrt'>('linear');
  readonly height = input(240);
  /** Renders cells square by deriving the height from the measured width. */
  readonly square = input(false);

  readonly hovered = output<CellHover | null>();
  readonly picked = output<CellHover>();

  // Not `viewChild.required`: the repaint effect can run before the view is
  // initialised or while it is being torn down, and a required query throws
  // there. A thrown effect is never retried, so the canvas would stay blank
  // for good -- a permanent failure from a transient condition.
  private readonly canvasRef = viewChild<ElementRef<HTMLCanvasElement>>('canvas');
  private readonly host = inject(ElementRef<HTMLElement>);
  private readonly destroyRef = inject(DestroyRef);

  private readonly size = signal<Size>({ width: 0, height: 0 });
  private offscreen: HTMLCanvasElement | null = null;

  /** Effective drawing height: the measured width when square, else the input. */
  private readonly drawHeight = computed(() =>
    this.square() ? Math.max(1, this.size().width) : this.height(),
  );

  constructor() {
    afterNextRender(() => {
      const el = this.host.nativeElement;
      const observer = new ResizeObserver((entries) => {
        const box = entries[0]?.contentRect;
        if (box) this.size.set({ width: Math.round(box.width), height: Math.round(box.height) });
      });
      observer.observe(el);
      this.destroyRef.onDestroy(() => observer.disconnect());
      this.size.set({ width: el.clientWidth, height: this.height() });
    });

    effect(() => {
      // Read everything that should trigger a repaint.
      this.values();
      this.rows();
      this.cols();
      this.colormap();
      this.max();
      this.scale();
      this.signed();
      this.size();
      this.height();
      this.square();
      this.draw();
    });
  }

  private draw(): void {
    const canvas = this.canvasRef()?.nativeElement;
    if (!canvas) return;
    const { width } = this.size();
    const height = this.drawHeight();
    if (width <= 0) return;

    const ctx = sizeCanvas(canvas, width, height);
    if (!ctx) return;

    ctx.clearRect(0, 0, width, height);

    const values = this.values();
    const rows = this.rows();
    const cols = this.cols();
    if (!values || rows <= 0 || cols <= 0) return;

    const table = lut(this.colormap());
    const ceiling = this.max() ?? observedMax(values, this.signed());
    const sqrt = this.scale() === 'sqrt';

    this.offscreen ??= document.createElement('canvas');
    const off = this.offscreen;
    off.width = cols;
    off.height = rows;
    const offCtx = off.getContext('2d');
    if (!offCtx) return;

    const image = offCtx.createImageData(cols, rows);
    const pixels = image.data;

    for (let i = 0; i < rows * cols; i++) {
      const v = values[i];
      let index: number;
      if (this.signed()) {
        index = signedIndex(v, ceiling);
      } else {
        const t = ceiling > 0 ? v / ceiling : 0;
        index = normIndex(sqrt ? Math.sqrt(Math.max(0, t)) : t);
      }
      const at = index * 4;
      const p = i * 4;
      pixels[p] = table[at];
      pixels[p + 1] = table[at + 1];
      pixels[p + 2] = table[at + 2];
      pixels[p + 3] = 255;
    }

    offCtx.putImageData(image, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, cols, rows, 0, 0, width, height);
  }

  private locate(event: PointerEvent | MouseEvent): CellHover | null {
    const values = this.values();
    const rows = this.rows();
    const cols = this.cols();
    const { width } = this.size();
    const height = this.drawHeight();
    if (!values || rows <= 0 || cols <= 0 || width <= 0) return null;

    const element = this.canvasRef()?.nativeElement;
    if (!element) return null;
    const rect = element.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const col = Math.floor((x / width) * cols);
    const row = Math.floor((y / height) * rows);
    if (col < 0 || col >= cols || row < 0 || row >= rows) return null;

    return {
      row,
      col,
      value: values[row * cols + col],
      x: ((col + 0.5) / cols) * width,
      y: ((row + 0.5) / rows) * height,
    };
  }

  protected onMove(event: PointerEvent): void {
    this.hovered.emit(this.locate(event));
  }

  protected onLeave(): void {
    this.hovered.emit(null);
  }

  protected onClick(event: MouseEvent): void {
    const cell = this.locate(event);
    if (cell) this.picked.emit(cell);
  }
}

function observedMax(values: Float32Array, signed: boolean): number {
  let max = 0;
  for (let i = 0; i < values.length; i++) {
    const v = signed ? Math.abs(values[i]) : values[i];
    if (v > max) max = v;
  }
  return max;
}
