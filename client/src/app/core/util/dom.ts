import { DestroyRef, Signal, signal } from '@angular/core';

export interface Size {
  width: number;
  height: number;
}

/**
 * Element size as a signal. Used by the canvas visualizations, which need to
 * re-render at the device pixel ratio whenever their box changes.
 */
export function elementSize(
  el: HTMLElement,
  destroyRef: DestroyRef,
  initial: Size = { width: 0, height: 0 },
): Signal<Size> {
  const size = signal<Size>(initial);
  const observer = new ResizeObserver((entries) => {
    const box = entries[0]?.contentRect;
    if (box) size.set({ width: Math.round(box.width), height: Math.round(box.height) });
  });
  observer.observe(el);
  destroyRef.onDestroy(() => observer.disconnect());
  return size.asReadonly();
}

/**
 * Sizes a canvas for the current device pixel ratio and returns a context whose
 * units are CSS pixels. Without this, everything is blurry on a HiDPI display.
 */
export function sizeCanvas(
  canvas: HTMLCanvasElement,
  width: number,
  height: number,
): CanvasRenderingContext2D | null {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(width * dpr));
  const h = Math.max(1, Math.round(height * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;

  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}
