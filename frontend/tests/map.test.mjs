import { test } from 'node:test';
import assert from 'node:assert/strict';
import { observeMapResize } from '../src/map.ts';

test('resize waits for map movement, coalesces callbacks and cancels on removal', (t) => {
  const events = new Map(), frames = new Map();
  let next = 0, resize, disconnected = false;
  const originals = [globalThis.ResizeObserver, globalThis.requestAnimationFrame, globalThis.cancelAnimationFrame];
  t.after(() => {
    [globalThis.ResizeObserver, globalThis.requestAnimationFrame, globalThis.cancelAnimationFrame] = originals;
  });
  globalThis.requestAnimationFrame = (fn) => { frames.set(++next, fn); return next; };
  globalThis.cancelAnimationFrame = (id) => frames.delete(id);
  globalThis.ResizeObserver = class {
    constructor(fn) { resize = fn; }
    observe() {}
    disconnect() { disconnected = true; }
  };
  const calls = [];
  const map = {
    on(name, fn) { events.set(name, fn); return this; },
    off(name, fn) { assert.equal(events.get(name), fn); events.delete(name); return this; },
    invalidateSize(options) { calls.push(options); },
  };
  const flush = () => { const pending = [...frames.values()]; frames.clear(); pending.forEach(fn => fn()); };
  const cleanup = observeMapResize(map, {});
  resize(); resize();
  assert.equal(frames.size, 1);
  events.get('movestart')();
  flush();
  resize();
  assert.equal(calls.length, 0);
  assert.equal(frames.size, 0);
  events.get('moveend')();
  flush();
  assert.deepEqual(calls, [{ animate: false, pan: false }]);
  resize();
  cleanup();
  flush();
  assert.equal(calls.length, 1);
  assert.equal(events.size, 0);
  assert.ok(disconnected);
});
