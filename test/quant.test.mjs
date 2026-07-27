// 4-bit weights, MXFP4 style (Kimi K3 §4.1.4): blocks of 32 numbers sharing one
// power-of-two scale, each number kept as sign + one of eight magnitudes.

import assert from "node:assert/strict";
import {
  Adam,
  Transformer,
  cloneTransformer,
  evalLoss,
  mulberry32,
  quantizeFfnMxfp4,
  trainStep,
} from "./dist/ml.js";
import { buildTokenizer } from "./dist/corpus.js";

const text =
  "det var en gang en fisker som rodde ut pa fjorden. han sa en sel som lekte i solen.\n";
const tok = buildTokenizer(text);
const data = tok.encode(text);
const cfg = { vocab: tok.vocab, dim: 16, nLayer: 2, nHead: 2, seqLen: 16, ffnMult: 4 };

function trained(config = cfg, steps = 150) {
  const model = new Transformer(config, mulberry32(1337));
  const opt = new Adam(model.params, 0.01);
  const r = mulberry32(42);
  for (let i = 0; i < steps; i++) trainStep(model, opt, data, config.seqLen, 2, r);
  return model;
}

// ---- 1. representable values survive exactly ------------------------------
// The grid is {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6} × a power of two, so those
// values must round-trip bit for bit.
{
  const model = new Transformer(cfg, mulberry32(1337));
  const grid = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.5, -1, -1.5, -2, -3, -4, -6];
  const W = model.blocks[0].W1;
  for (let i = 0; i < W.d.length; i++) W.d[i] = grid[i % grid.length] * 0.0625; // × 2⁻⁴
  const before = Float32Array.from(W.d);
  quantizeFfnMxfp4(model);
  for (let i = 0; i < W.d.length; i++)
    assert.equal(W.d[i], before[i], `exactly representable value changed at ${i}`);
}

// ---- 2. the error is bounded by the block's own scale ---------------------
// The coarsest gap in the grid is 2 (between 4 and 6) at scale s, and s is at
// most maxAbs/4, so no number can move by more than maxAbs/4.
{
  const model = trained();
  const blocks = [];
  for (const blk of model.blocks) blocks.push(blk.W1, blk.W2);
  const before = blocks.map((t) => Float32Array.from(t.d));
  const stats = quantizeFfnMxfp4(model, 32);

  let worstRatio = 0;
  for (let b = 0; b < blocks.length; b++) {
    const d = blocks[b].d;
    const orig = before[b];
    for (let start = 0; start < d.length; start += 32) {
      const end = Math.min(d.length, start + 32);
      let maxAbs = 0;
      for (let i = start; i < end; i++) maxAbs = Math.max(maxAbs, Math.abs(orig[i]));
      if (maxAbs === 0) continue;
      for (let i = start; i < end; i++)
        worstRatio = Math.max(worstRatio, Math.abs(d[i] - orig[i]) / maxAbs);
    }
  }
  assert.ok(worstRatio <= 0.25 + 1e-6, `per-block error ratio ${worstRatio} exceeded 1/4`);
  assert.ok(stats.maxAbsErr > 0, "a trained model should not be exactly representable");
  assert.ok(stats.meanAbsErr < stats.maxAbsErr, "mean error should sit below the worst case");
}

// ---- 3. the accounting ----------------------------------------------------
{
  const model = trained();
  const stats = quantizeFfnMxfp4(model, 32);
  const expectedValues = model.blocks.reduce((n, b) => n + b.W1.d.length + b.W2.d.length, 0);
  assert.equal(stats.values, expectedValues, "every wide-layer weight should be counted");
  // Blocks never straddle two matrices, so they are counted per matrix.
  const expectedBlocks = model.blocks.reduce(
    (n, b) => n + Math.ceil(b.W1.d.length / 32) + Math.ceil(b.W2.d.length / 32),
    0
  );
  assert.equal(stats.blocks, expectedBlocks);
  assert.equal(stats.bytesBefore, stats.values * 4);
  assert.equal(stats.bytesAfter, Math.ceil(stats.values / 2) + stats.blocks);
  const shrink = stats.bytesBefore / stats.bytesAfter;
  assert.ok(shrink > 7 && shrink < 8, `should shrink about 7.5×, got ${shrink}`);

  // Quantizing again is a no-op: the values are already on the grid.
  const again = quantizeFfnMxfp4(model, 32);
  assert.equal(again.maxAbsErr, 0, "re-quantizing should change nothing");
}

// ---- 4. only the wide layer is touched ------------------------------------
// K3 keeps attention, norms and the tables in higher precision; the same split
// is what makes the demo honest.
{
  const model = trained();
  const blk = model.blocks[0];
  const untouched = [blk.Wq, blk.Wk, blk.Wv, blk.Wo, blk.ln1g, blk.b1, model.tokEmb, model.head];
  const before = untouched.map((t) => Float32Array.from(t.d));
  const wide = Float32Array.from(blk.W1.d);
  quantizeFfnMxfp4(model);
  for (let i = 0; i < untouched.length; i++)
    assert.deepEqual(Array.from(untouched[i].d), Array.from(before[i]), "this should be untouched");
  assert.notDeepEqual(Array.from(blk.W1.d), Array.from(wide), "the wide layer should change");
}

// ---- 5. the SiTU-GLU up branch is quantized too ---------------------------
{
  const situCfg = { ...cfg, act: "situ" };
  const model = trained(situCfg);
  const expected = model.blocks.reduce(
    (n, b) => n + b.W1.d.length + b.Wu.d.length + b.W2.d.length,
    0
  );
  const stats = quantizeFfnMxfp4(model);
  assert.equal(stats.values, expected, "all three GLU matrices should be quantized");
}

// ---- 6. the model still works afterwards ----------------------------------
// It gets worse — that is the point of the demo — but it must stay a language
// model, not turn into noise.
{
  const model = trained(cfg, 400);
  const copy = cloneTransformer(model);
  const before = evalLoss(model, data, cfg.seqLen, 8, mulberry32(9));
  quantizeFfnMxfp4(copy);
  const after = evalLoss(copy, data, cfg.seqLen, 8, mulberry32(9));
  assert.ok(Number.isFinite(after), "loss after quantization must be finite");
  assert.ok(after > before - 1e-6, "4 bits cannot beat 32 bits");
  assert.ok(after < before + 1.5, `loss should not collapse: ${before} → ${after}`);

  // The original is untouched: the UI quantizes a copy so nobody loses a model.
  const untouched = evalLoss(model, data, cfg.seqLen, 8, mulberry32(9));
  assert.equal(untouched, before, "quantizing the clone must not touch the original");
}

// ---- 7. evalLoss is deterministic for a given seed ------------------------
{
  const model = trained();
  const a = evalLoss(model, data, cfg.seqLen, 4, mulberry32(2));
  const b = evalLoss(model, data, cfg.seqLen, 4, mulberry32(2));
  assert.equal(a, b);
  assert.throws(() => evalLoss(model, [1], cfg.seqLen, 4, mulberry32(2)), RangeError);
  assert.throws(() => quantizeFfnMxfp4(model, 0), RangeError);
}

console.log("quant: ok");
