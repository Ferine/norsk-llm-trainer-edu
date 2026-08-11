// Mange små ekspertar (DeepSeekMoE / DeepSeek-V3 §2.1): the three routing
// primitives and their gradients, the parameter bookkeeping that keeps the
// comparison with a dense layer honest, and the auxiliary-loss-free balancing.

import assert from "node:assert/strict";
import {
  Adam,
  MOE_DEFAULT,
  Muon,
  Transformer,
  backward,
  cloneTransformer,
  expertWidth,
  ffnWidth,
  moeActiveFraction,
  mulCol,
  mulberry32,
  scatterRows,
  seqLogProb,
  takeRows,
  trainStep,
} from "./dist/ml.js";
import { buildTokenizer } from "./dist/corpus.js";
import { buildModelWorkbook } from "./dist/excel-model.js";
import { buildModelGguf } from "./dist/gguf.js";

const rng = mulberry32(5);

function leaf(rows, cols, fill) {
  const d = new Float32Array(rows * cols);
  for (let i = 0; i < d.length; i++) d[i] = fill(i);
  return { d, rows, cols, grad: new Float32Array(rows * cols), _prev: [], _back: () => {} };
}

// ---- 1. the three primitives move values where they should ----------------
{
  const x = leaf(4, 2, (i) => i);
  const picked = takeRows(x, [3, 1, 1]);
  assert.deepEqual([...picked.d], [6, 7, 2, 3, 2, 3]);
  assert.equal(picked.rows, 3);

  // A row taken twice must have both contributions land back on it.
  const back = scatterRows(picked, [3, 1, 1], 4);
  assert.deepEqual([...back.d], [0, 0, 4, 6, 0, 0, 6, 7]);

  const scaled = mulCol(leaf(2, 3, () => 2), leaf(2, 1, (i) => i + 1));
  assert.deepEqual([...scaled.d], [2, 2, 2, 4, 4, 4]);

  assert.throws(() => takeRows(x, [4]), RangeError);
  assert.throws(() => takeRows(x, [-1]), RangeError);
}

// ---- 2. gradients match finite differences --------------------------------
// One expression exercises all three backward rules at once, including the
// double-counted row that takeRows/scatterRows have to accumulate correctly.
{
  const T = 4;
  const cols = 3;
  const idx = [3, 1, 1, 0];
  const xv = Array.from({ length: T * cols }, () => (rng() * 2 - 1) * 2);
  const wv = Array.from({ length: idx.length }, () => rng() * 2 - 1);
  const targets = [1, 2, 0, 1];

  const build = (xArr, wArr) => {
    const x = leaf(T, cols, (i) => xArr[i]);
    const w = leaf(idx.length, 1, (i) => wArr[i]);
    const out = scatterRows(mulCol(takeRows(x, idx), w), idx, T);
    return { x, w, loss: seqLogProb(out, 0, targets) };
  };

  const { x, w, loss } = build(xv, wv);
  backward(loss);

  const h = 1e-2;
  for (const [name, arr, t] of [
    ["x", xv, x],
    ["w", wv, w],
  ]) {
    for (let i = 0; i < arr.length; i++) {
      const keep = arr[i];
      arr[i] = keep + h;
      const up = build(xv, wv).loss.d[0];
      arr[i] = keep - h;
      const dn = build(xv, wv).loss.d[0];
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

// ---- 3. the router learns -------------------------------------------------
// Selection is a step function, so finite differences only make sense where
// nothing can flip: with topK = experts every expert is always chosen and the
// whole path is smooth. What is being checked is that the gate weight carries
// gradient back to the router at all — without it the router never learns.
{
  const cfg = {
    vocab: 12, dim: 8, nLayer: 1, nHead: 2, seqLen: 6, ffnMult: 4,
    moe: { experts: 3, topK: 3, bias: 0 },
  };
  const model = new Transformer(cfg, mulberry32(11));
  const ids = [1, 4, 2, 7];
  const targets = [4, 2, 7, 3];
  const Wr = model.blocks[0].router.W;

  const value = () => seqLogProb(model.forward(ids), 0, targets).d[0];
  for (const p of model.params) p.grad.fill(0);
  backward(seqLogProb(model.forward(ids), 0, targets));

  const h = 1e-3;
  let checked = 0;
  for (let i = 0; i < Wr.d.length; i++) {
    const keep = Wr.d[i];
    Wr.d[i] = keep + h;
    const up = value();
    Wr.d[i] = keep - h;
    const dn = value();
    Wr.d[i] = keep;
    const numeric = (up - dn) / (2 * h);
    const tol = 5e-2 * Math.max(1, Math.abs(numeric));
    assert.ok(
      Math.abs(numeric - Wr.grad[i]) < tol,
      `router[${i}]: analytic ${Wr.grad[i]} vs numeric ${numeric}`
    );
    checked++;
  }
  assert.ok(checked === 24, `expected a dim×experts router, checked ${checked}`);
}

// ---- 4. the wide layer is split, not widened ------------------------------
// The whole comparison rests on this: experts must move screws around, never
// add them, or a better loss would just be measuring the extra parameters.
{
  const base = { vocab: 30, dim: 48, nLayer: 2, nHead: 2, seqLen: 16, ffnMult: 4, act: "situ" };
  const moe = { ...base, moe: { experts: 4, topK: 2, bias: 0.001 } };

  assert.equal(ffnWidth(base), 128);
  assert.equal(expertWidth(base), 128); // no experts: the expert is the whole layer
  assert.equal(expertWidth(moe), 26); // 128 / (4 + 1), rounded
  assert.equal(moeActiveFraction(base), 1);
  assert.equal(moeActiveFraction(moe), 3 / 5);

  const dense = new Transformer(base, mulberry32(1));
  const sparse = new Transformer(moe, mulberry32(1));
  const drift = Math.abs(dense.paramCount() - sparse.paramCount()) / dense.paramCount();
  assert.ok(drift < 0.05, `parameter counts should stay within 5%, drifted ${drift}`);

  // Every expert is trained, and the router with them.
  const blk = sparse.blocks[0];
  assert.equal(blk.routed.length, 4);
  for (const e of blk.routed) {
    assert.ok(sparse.params.includes(e.W1), "each expert must be trained");
    assert.ok(sparse.params.includes(e.Wu), "each expert's up branch must be trained");
  }
  assert.ok(sparse.params.includes(blk.router.W), "the router must be trained");
  assert.ok(!sparse.params.includes(blk.router.bias), "the balancing nudge is not a parameter");

  // Muon orthogonalizes experts but leaves the router to Adam — pushing a
  // router's columns apart would move the choice, not just the step.
  const g = sparse.optimGroups();
  assert.ok(g.matrix.some((m) => m.p === blk.routed[0].W1), "experts belong to Muon");
  assert.ok(g.scalar.includes(blk.router.W), "the router belongs to Adam");
  assert.ok(!g.matrix.some((m) => m.p === blk.router.W), "the router must not be orthogonalized");

  assert.throws(() => new Transformer({ ...base, moe: { experts: 0, topK: 1, bias: 0 } }, mulberry32(1)), RangeError);
  assert.throws(() => new Transformer({ ...base, moe: { experts: 4, topK: 5, bias: 0 } }, mulberry32(1)), RangeError);
  assert.throws(() => new Transformer({ ...base, moe: { experts: 4, topK: 2, bias: -1 } }, mulberry32(1)), RangeError);
}

// ---- 5. routing is sparse, and visible ------------------------------------
{
  const cfg = {
    vocab: 20, dim: 16, nLayer: 2, nHead: 2, seqLen: 8, ffnMult: 4,
    moe: { experts: 4, topK: 2, bias: 0.001 },
  };
  const model = new Transformer(cfg, mulberry32(3));
  const ids = [1, 5, 9, 2, 7];
  const { routes } = model.inspect(ids);

  assert.equal(routes.length, 2, "one view per layer");
  for (const r of routes) {
    assert.equal(r.T, ids.length);
    assert.equal(r.experts, 4);
    assert.equal(r.chosen.length, ids.length * 2);
    for (let t = 0; t < r.T; t++) {
      let sum = 0;
      for (let e = 0; e < r.experts; e++) sum += r.gates[t * r.experts + e];
      assert.ok(Math.abs(sum - 1) < 1e-4, `the router splits one whole portion: got ${sum}`);
      const picked = [r.chosen[t * 2], r.chosen[t * 2 + 1]];
      assert.notEqual(picked[0], picked[1], "a character cannot wake the same expert twice");
      // Chosen by score + bias; with equal bias that is simply the top two.
      const ranked = [...Array(4).keys()].sort(
        (a, b) => r.gates[t * 4 + b] - r.gates[t * 4 + a]
      );
      assert.deepEqual(picked.slice().sort(), ranked.slice(0, 2).sort());
    }
  }
}

// ---- 6. the balancing nudge does its job ---------------------------------
{
  const cfg = {
    vocab: 20, dim: 16, nLayer: 1, nHead: 2, seqLen: 8, ffnMult: 4,
    moe: { experts: 4, topK: 1, bias: 0.01 },
  };
  const model = new Transformer(cfg, mulberry32(3));
  const r = model.blocks[0].router;

  // Busy experts get less attractive, idle ones more. Nothing else moves.
  r.load.set([10, 10, 0, 20]);
  model.rebalanceRouters();
  assert.ok(r.bias[3] < 0, "the busiest expert should be pushed down");
  assert.ok(r.bias[2] > 0, "the idlest expert should be pulled up");
  assert.equal(r.bias[0], 0, "an expert at the mean should not move");
  assert.deepEqual([...r.load], [0, 0, 0, 0], "the counters reset each step");

  // Measuring must not disturb the balance — only training counts load.
  model.forward([1, 2, 3]);
  assert.deepEqual([...r.load], [0, 0, 0, 0], "a plain forward pass must not count");
}

// ---- 7. a routed model trains, and no expert starves ---------------------
{
  const text =
    "det var en gang en fisker som rodde ut pa fjorden. han sa en sel som lekte i solen.\n";
  const tok = buildTokenizer(text);
  const data = tok.encode(text);
  const cfg = {
    vocab: tok.vocab, dim: 16, nLayer: 2, nHead: 2, seqLen: 16, ffnMult: 4,
    act: "situ", moe: { experts: 4, topK: 2, bias: 0.001 },
  };
  const model = new Transformer(cfg, mulberry32(1337));
  const opt = new Adam(model.params, 0.01);
  const r = mulberry32(42);
  let first = 0;
  let last = 0;
  for (let i = 0; i < 200; i++) {
    const l = trainStep(model, opt, data, cfg.seqLen, 2, r);
    if (i === 0) first = l;
    last = l;
    assert.ok(Number.isFinite(l), `loss went non-finite at step ${i}`);
  }
  assert.ok(last < first * 0.7, `a routed model should learn: ${first.toFixed(3)} → ${last.toFixed(3)}`);

  // Count one more step's worth of routing and check nobody is starved. Without
  // the nudge the router is free to collapse onto one expert, which is the
  // failure this whole mechanism exists to prevent.
  model.countRouting = true;
  for (let i = 0; i < 20; i++) model.forward(data.slice(i, i + 16));
  model.countRouting = false;
  for (const { load } of model.routerLoad()) {
    const total = load.reduce((a, b) => a + b, 0);
    assert.ok(total > 0, "routing should have been counted");
    for (let e = 0; e < load.length; e++)
      assert.ok(load[e] > 0, `expert ${e} was never used: ${[...load]}`);
  }

  // Muon has to cope with the narrower expert matrices too.
  const m2 = new Transformer(cfg, mulberry32(1337));
  const muon = new Muon(m2.optimGroups(), 0.01);
  const r2 = mulberry32(42);
  for (let i = 0; i < 20; i++)
    assert.ok(Number.isFinite(trainStep(m2, muon, data, cfg.seqLen, 2, r2)), "Muon step went bad");
}

// ---- 8. a clone routes identically ---------------------------------------
// The DPO reference is a frozen copy. If the balancing nudge did not come with
// it, the reference would route differently from the model it is measuring.
{
  const cfg = {
    vocab: 20, dim: 16, nLayer: 2, nHead: 2, seqLen: 8, ffnMult: 4,
    moe: { experts: 4, topK: 1, bias: 0.05 },
  };
  const model = new Transformer(cfg, mulberry32(3));
  const r = model.blocks[0].router;
  r.load.set([40, 0, 0, 0]);
  model.rebalanceRouters(); // give the biases something to say
  assert.ok(r.bias[0] < 0);

  const copy = cloneTransformer(model);
  assert.deepEqual([...copy.blocks[0].router.bias], [...r.bias], "the nudge must travel with the copy");

  const ids = [1, 5, 9, 2, 7];
  const a = model.inspect(ids);
  const b = copy.inspect(ids);
  assert.deepEqual([...b.routes[0].chosen], [...a.routes[0].chosen], "the copy must route identically");
  for (let i = 0; i < a.logits.d.length; i++)
    assert.ok(Math.abs(a.logits.d[i] - b.logits.d[i]) < 1e-6, "the copy must predict identically");
}

// ---- 9. the exports tell the truth about what they hold ------------------
{
  const text = "det var en gang en fisker som rodde ut pa fjorden i solen.\n";
  const tok = buildTokenizer(text);
  const cfg = {
    vocab: tok.vocab, dim: 16, nLayer: 2, nHead: 2, seqLen: 16, ffnMult: 4,
    act: "situ", moe: { experts: 4, topK: 2, bias: 0.001 },
  };
  const model = new Transformer(cfg, mulberry32(1337));

  // The workbook computes one wide layer per block. Rather than hand out a file
  // that quietly computes a different model, it refuses.
  assert.throws(
    () =>
      buildModelWorkbook({
        model, tokenizer: tok, prompt: "det", nGen: 2, step: 10, loss: 1,
        presetName: "test", lang: "nn", includeQuant: false,
      }),
    /MoE|ekspertar/,
    "the workbook must refuse a routed model, not export a dense one"
  );

  // GGUF carries every expert, and says so in the header.
  const built = buildModelGguf({
    model, tokenizer: tok, step: 10, loss: 1, presetName: "liten", lang: "bm",
  });
  assert.equal(built.arch, "sprakmodell-moe", "a routed model is not a gpt2");

  const dense = buildModelGguf({
    model: new Transformer({ ...cfg, moe: undefined }, mulberry32(1337)),
    tokenizer: tok, step: 10, loss: 1, presetName: "liten", lang: "bm",
  });
  // Per block: 4 routed experts × 4 tensors (gate/up/down + biases → 6 each for
  // SiTU) plus the router. Whatever the arithmetic, it must be strictly more.
  assert.ok(
    built.tensorCount > dense.tensorCount,
    `experts must reach the file: ${built.tensorCount} vs ${dense.tensorCount}`
  );
  assert.ok(built.kvCount > dense.kvCount, "the header must describe the experts");
}

// ---- 10. the defaults are the ones the app ships -------------------------
{
  assert.equal(MOE_DEFAULT.experts, 4);
  assert.equal(MOE_DEFAULT.topK, 1);
  assert.ok(MOE_DEFAULT.bias > 0, "shipping with balancing off would let the router collapse");
}

console.log("moe: ok");
