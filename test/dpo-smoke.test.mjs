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

// eval helper mirrors dpoStep capping
function capSeq(promptIds, contIds, seqLen) {
  let prompt = promptIds.length ? promptIds : [0];
  let cont = contIds.slice();
  if (cont.length < 1) cont = [prompt[prompt.length - 1]];
  if (cont.length >= seqLen) cont = cont.slice(0, seqLen - 1);
  let P = prompt.length;
  if (P + cont.length > seqLen) { P = seqLen - cont.length; prompt = prompt.slice(prompt.length - P); }
  return { seq: prompt.concat(cont), P };
}
function evalMargin(policy, reference, pairs) {
  let total = 0, wins = 0;
  for (const p of pairs) {
    const w = capSeq(p.promptIds, p.chosenIds, policy.seqLen);
    const l = capSeq(p.promptIds, p.rejectedIds, policy.seqLen);
    const tw = w.seq.slice(w.P), tl = l.seq.slice(l.P);
    const lpW = seqLogProbValue(policy.forward(w.seq), w.P - 1, tw);
    const lpL = seqLogProbValue(policy.forward(l.seq), l.P - 1, tl);
    const rW = seqLogProbValue(reference.forward(w.seq), w.P - 1, tw);
    const rL = seqLogProbValue(reference.forward(l.seq), l.P - 1, tl);
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
