import { Transformer, mulberry32, rowProbs } from "./dist/ml.js";
import assert from "node:assert/strict";

const cfg = { vocab: 7, dim: 8, nLayer: 2, nHead: 2, seqLen: 6, ffnMult: 2 };
const m = new Transformer(cfg, mulberry32(3));
const ids = [1, 2, 3, 0, 4];
const T = ids.length;
const logits = m.forward(ids);

// --- rowProbs ---
const probs = rowProbs(logits, T - 1);
assert.equal(probs.length, cfg.vocab, "length == vocab");
let sum = 0;
for (const p of probs) {
  assert.ok(p >= 0, "probabilities are non-negative");
  sum += p;
}
assert.ok(Math.abs(sum - 1) < 1e-5, "probabilities sum to 1");

// matches a hand-computed softmax of the same logits row
const off = (T - 1) * cfg.vocab;
let mx = -Infinity;
for (let c = 0; c < cfg.vocab; c++) mx = Math.max(mx, logits.d[off + c]);
let z = 0;
const manual = [];
for (let c = 0; c < cfg.vocab; c++) {
  const e = Math.exp(logits.d[off + c] - mx);
  manual.push(e);
  z += e;
}
for (let c = 0; c < cfg.vocab; c++)
  assert.ok(Math.abs(probs[c] - manual[c] / z) < 1e-6, "matches manual softmax");

// out-of-range row throws
assert.throws(() => rowProbs(logits, T), RangeError, "row >= rows throws");
assert.throws(() => rowProbs(logits, -1), RangeError, "negative row throws");

console.log("rowProbs: PASS");

// --- inspect ---
const { logits: l2, attn } = m.inspect(ids);

// logits identical to a plain forward() (the sink does not change the math)
assert.equal(l2.d.length, logits.d.length, "inspect logits length matches forward");
for (let i = 0; i < l2.d.length; i++)
  assert.ok(Math.abs(l2.d[i] - logits.d[i]) < 1e-6, "inspect logits == forward logits");

// one attention view per (layer, head)
assert.equal(attn.length, cfg.nLayer * cfg.nHead, "one view per (layer, head)");

for (const v of attn) {
  assert.equal(v.T, T, "view.T == sequence length");
  assert.equal(v.weights.length, T * T, "weights length == T*T");
  for (let r = 0; r < T; r++) {
    let rowSum = 0;
    for (let c = 0; c < T; c++) {
      const w = v.weights[r * T + c];
      if (c > r) assert.ok(Math.abs(w) < 1e-7, "causal mask: no attention to the future");
      else rowSum += w;
    }
    assert.ok(Math.abs(rowSum - 1) < 1e-5, "each query row sums to 1 over allowed keys");
  }
}

// every (layer, head) pair is present exactly once
const seen = new Set(attn.map((v) => `${v.layer}:${v.head}`));
assert.equal(seen.size, cfg.nLayer * cfg.nHead, "all (layer, head) pairs distinct");

console.log("inspect: PASS");
