// SiTU-GLU (Kimi K3 §2.3.2, eq. 12): the gated activation with a soft cap,
// its gradients, and a model that uses it.

import assert from "node:assert/strict";
import {
  Adam,
  SITU_B1,
  SITU_B2,
  Transformer,
  backward,
  ffnWidth,
  mulberry32,
  seqLogProb,
  situGlu,
  trainStep,
} from "./dist/ml.js";
import { buildTokenizer } from "./dist/corpus.js";

const rng = mulberry32(3);

function leaf(rows, cols, fill) {
  const d = new Float32Array(rows * cols);
  for (let i = 0; i < d.length; i++) d[i] = fill(i);
  return { d, rows, cols, grad: new Float32Array(rows * cols), _prev: [], _back: () => {} };
}

// ---- 1. the soft cap holds -----------------------------------------------
// Both branches are bounded, so the product can never exceed β₁·β₂ = 100.
// That is the whole reason K3 replaced SwiGLU, whose factors are unbounded.
{
  const extremes = [-1e6, -500, -50, -1, 0, 1, 50, 500, 1e6];
  const n = extremes.length;
  const g = leaf(n, n, (i) => extremes[i % n]);
  const u = leaf(n, n, (i) => extremes[Math.floor(i / n)]);
  const out = situGlu(g, u);
  const cap = SITU_B1 * SITU_B2;
  assert.equal(cap, 100);
  for (const v of out.d) {
    assert.ok(Number.isFinite(v), "the cap must not produce NaN or Infinity");
    assert.ok(Math.abs(v) <= cap + 1e-3, `|${v}| exceeded the cap ${cap}`);
  }
}

// near zero it should track SwiGLU: gate ≈ x·σ(x), up ≈ x
{
  const g = leaf(1, 3, () => 0.1);
  const u = leaf(1, 3, () => 0.2);
  const out = situGlu(g, u);
  const swiglu = 0.1 * (1 / (1 + Math.exp(-0.1))) * 0.2;
  assert.ok(
    Math.abs(out.d[0] - swiglu) < 1e-3,
    `near the origin SiTU-GLU should track SwiGLU: ${out.d[0]} vs ${swiglu}`
  );
}

// ---- 2. gradients match finite differences -------------------------------
// The activation is built from mul/tanh/sigmoid/scale, so this checks all four
// of their backward rules at once through a scalar objective.
{
  const rows = 3;
  const cols = 5;
  const gv = Array.from({ length: rows * cols }, () => (rng() * 2 - 1) * 3);
  const uv = Array.from({ length: rows * cols }, () => (rng() * 2 - 1) * 3);
  const targets = [1, 3, 0];

  const value = (gArr, uArr) => {
    const g = leaf(rows, cols, (i) => gArr[i]);
    const u = leaf(rows, cols, (i) => uArr[i]);
    return seqLogProb(situGlu(g, u), 0, targets).d[0];
  };

  const g = leaf(rows, cols, (i) => gv[i]);
  const u = leaf(rows, cols, (i) => uv[i]);
  const loss = seqLogProb(situGlu(g, u), 0, targets);
  backward(loss);

  const h = 1e-2;
  for (let i = 0; i < rows * cols; i++) {
    for (const [name, arr, t] of [
      ["g", gv, g],
      ["u", uv, u],
    ]) {
      const keep = arr[i];
      arr[i] = keep + h;
      const up = value(gv, uv);
      arr[i] = keep - h;
      const dn = value(gv, uv);
      arr[i] = keep;
      const numeric = (up - dn) / (2 * h);
      const analytic = t.grad[i];
      const tol = 2e-2 * Math.max(1, Math.abs(numeric));
      assert.ok(
        Math.abs(numeric - analytic) < tol,
        `${name}[${i}]: analytic ${analytic} vs numeric ${numeric}`
      );
    }
  }
}

// ---- 3. width bookkeeping ------------------------------------------------
// GLU needs three matrices where GELU needs two, so the wide layer shrinks to
// 2/3 and the parameter count stays comparable — otherwise any comparison
// between the two would just be measuring the extra parameters.
{
  const base = { vocab: 30, dim: 48, nLayer: 2, nHead: 2, seqLen: 16, ffnMult: 4 };
  assert.equal(ffnWidth(base), 192);
  assert.equal(ffnWidth({ ...base, act: "gelu" }), 192);
  assert.equal(ffnWidth({ ...base, act: "situ" }), 128);

  const a = new Transformer(base, mulberry32(1));
  const b = new Transformer({ ...base, act: "situ" }, mulberry32(1));
  const drift = Math.abs(a.paramCount() - b.paramCount()) / a.paramCount();
  assert.ok(drift < 0.05, `parameter counts should stay within 5%, drifted ${drift}`);
  assert.equal(a.act, "gelu");
  assert.equal(b.act, "situ");
  assert.throws(() => new Transformer({ ...base, act: "swish" }, mulberry32(1)), RangeError);
}

// ---- 4. a SiTU-GLU model trains ------------------------------------------
{
  const text =
    "det var en gang en fisker som rodde ut pa fjorden. han sa en sel som lekte i solen.\n";
  const tok = buildTokenizer(text);
  const data = tok.encode(text);
  const cfg = { vocab: tok.vocab, dim: 16, nLayer: 2, nHead: 2, seqLen: 16, ffnMult: 4, act: "situ" };
  const model = new Transformer(cfg, mulberry32(1337));
  const opt = new Adam(model.params, 0.01);
  const r = mulberry32(42);
  let first = 0;
  let last = 0;
  for (let i = 0; i < 150; i++) {
    const l = trainStep(model, opt, data, cfg.seqLen, 2, r);
    if (i === 0) first = l;
    last = l;
    assert.ok(Number.isFinite(l), `loss went non-finite at step ${i}`);
  }
  assert.ok(last < first * 0.7, `SiTU-GLU should learn: ${first.toFixed(3)} → ${last.toFixed(3)}`);

  // The extra branch is a real parameter and really receives gradient.
  const blk = model.blocks[0];
  assert.ok(blk.Wu && blk.bu, "the up branch should exist");
  assert.ok(model.params.includes(blk.Wu), "the up branch must be trained");
  let moved = 0;
  for (const v of blk.Wu.grad) moved += Math.abs(v);
  assert.ok(moved > 0, "the up branch should receive gradient");
}

console.log("situ: ok");
