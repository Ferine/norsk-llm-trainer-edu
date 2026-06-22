import { mulberry32, seqLogProb, seqLogProbValue, backward } from "./dist/ml.js";
import assert from "node:assert/strict";

const rng = mulberry32(11);
function rt(rows, cols) {
  const t = { d: new Float32Array(rows * cols), rows, cols, grad: new Float32Array(rows * cols), _prev: [], _back: () => {} };
  for (let i = 0; i < t.d.length; i++) t.d[i] = rng() * 2 - 1;
  return t;
}

const V = 5;
const logits = rt(4, V);
const r0 = 1, targets = [2, 0, 4]; // scores rows 1,2,3

// 1) value matches a manual log-softmax sum
let manual = 0;
for (let i = 0; i < targets.length; i++) {
  const r = r0 + i;
  let mx = -Infinity;
  for (let c = 0; c < V; c++) mx = Math.max(mx, logits.d[r * V + c]);
  let s = 0;
  for (let c = 0; c < V; c++) s += Math.exp(logits.d[r * V + c] - mx);
  manual += logits.d[r * V + targets[i]] - mx - Math.log(s);
}
assert.ok(Math.abs(seqLogProbValue(logits, r0, targets) - manual) < 1e-5, "seqLogProbValue matches manual");

// 2) autograd matches central differences
logits.grad.fill(0);
backward(seqLogProb(logits, r0, targets));
const g = Float32Array.from(logits.grad);
const eps = 1e-2;
let maxAbs = 0;
for (let i = 0; i < logits.d.length; i++) {
  const o = logits.d[i];
  logits.d[i] = o + eps; const lp = seqLogProb(logits, r0, targets).d[0];
  logits.d[i] = o - eps; const lm = seqLogProb(logits, r0, targets).d[0];
  logits.d[i] = o;
  const num = (lp - lm) / (2 * eps);
  maxAbs = Math.max(maxAbs, Math.abs(num - g[i]));
}
assert.ok(maxAbs < 1e-3, `seqLogProb grad maxAbs=${maxAbs}`);
console.log("seq-logprob: PASS");
