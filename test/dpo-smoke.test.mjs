import { Transformer, Adam, trainStep, cloneTransformer, dpoStep, seqLogProbValue, mulberry32 } from "./dist/ml.js";
import { buildTokenizer, corpus } from "./dist/corpus.js";
import assert from "node:assert/strict";

const tok = buildTokenizer(corpus);
const data = tok.encode(corpus);
const cfg = { vocab: tok.vocab, dim: 48, nLayer: 2, nHead: 2, seqLen: 32, ffnMult: 4 };
const model = new Transformer(cfg, mulberry32(1337));
const sft = new Adam(model.params, 8e-4);
const rng = mulberry32(42);
for (let s = 0; s < 200; s++) trainStep(model, sft, data, 32, 4, rng);

const ref = cloneTransformer(model);

// synthetic preferences: chosen = real corpus continuation, rejected = random tokens
const prng = mulberry32(7);
const pairs = [];
for (let k = 0; k < 12; k++) {
  const start = Math.floor(prng() * (data.length - 40));
  const promptIds = data.slice(start, start + 8);
  const chosenIds = data.slice(start + 8, start + 24);
  const rejectedIds = [];
  for (let i = 0; i < 16; i++) rejectedIds.push(Math.floor(prng() * tok.vocab));
  pairs.push({ promptIds, chosenIds, rejectedIds });
}

// These short fixtures fit in one context. Long answers and sliding-window
// scoring are exercised independently in dpo-context.test.mjs.
function score(model, prompt, continuation) {
  const seq = prompt.concat(continuation).slice(0, -1);
  assert.ok(seq.length <= model.seqLen);
  return seqLogProbValue(model.forward(seq), prompt.length - 1, continuation);
}
function evalMargin(policy, reference, pairs) {
  let total = 0, wins = 0;
  for (const p of pairs) {
    const lpW = score(policy, p.promptIds, p.chosenIds);
    const lpL = score(policy, p.promptIds, p.rejectedIds);
    const rW = score(reference, p.promptIds, p.chosenIds);
    const rL = score(reference, p.promptIds, p.rejectedIds);
    const m = (lpW - rW) - (lpL - rL);
    total += m; if (m > 0) wins++;
  }
  return { margin: total / pairs.length, winRate: wins / pairs.length };
}

const before = evalMargin(model, ref, pairs);
assert.ok(Math.abs(before.margin) < 1e-4, `margin starts ~0, got ${before.margin}`);

const dpoOpt = new Adam(model.params, 1e-3);
const trng = mulberry32(99);
for (let s = 0; s < 100; s++) dpoStep(model, ref, dpoOpt, pairs, 4, 0.1, trng);

const after = evalMargin(model, ref, pairs);
assert.ok(after.margin > 0.1, `margin should grow, got ${after.margin}`);
assert.ok(after.winRate >= 0.8, `winRate should be high, got ${after.winRate}`);
let bad = 0;
for (const p of model.params) for (const x of p.d) if (!Number.isFinite(x)) bad++;
assert.equal(bad, 0, "no non-finite params");
console.log(`dpo-smoke: PASS (margin ${before.margin.toFixed(3)} -> ${after.margin.toFixed(3)}, winRate ${(after.winRate * 100).toFixed(0)}%)`);
