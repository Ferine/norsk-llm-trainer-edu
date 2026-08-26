#!/usr/bin/env node

import { runNgramAblation } from "../test/dist/ablation.js";
import { corpora } from "../test/dist/corpus.js";

function positiveInt(name, fallback) {
  const raw = process.env[name];
  if (raw === undefined) return fallback;
  const n = Number(raw);
  if (!Number.isInteger(n) || n < 1) throw new RangeError(`${name} must be a positive integer`);
  return n;
}

function splitCorpus(text) {
  const paragraphs = text.trim().split(/\n\n+/);
  if (paragraphs.length < 3) throw new Error("corpus needs at least three paragraphs");
  return {
    trainText: paragraphs.slice(0, -1).join("\n\n"),
    holdoutText: paragraphs.at(-1),
  };
}

function f(n, digits = 4) {
  return Number.isFinite(n) ? n.toFixed(digits) : String(n);
}

function signed(n, digits = 1) {
  return `${n >= 0 ? "+" : ""}${f(n, digits)}%`;
}

const steps = positiveInt("ABLATION_STEPS", 800);
const seedCount = positiveInt("ABLATION_SEEDS", 3);
const batchSize = positiveInt("ABLATION_BATCH", 4);
const evalBatches = positiveInt("ABLATION_EVAL_BATCHES", 32);
const seeds = Array.from({ length: seedCount }, (_, i) => 20260826 + i * 9973);
const baseConfig = {
  dim: 48,
  nLayer: 2,
  nHead: 2,
  seqLen: 32,
  ffnMult: 4,
  act: "situ",
};

const reports = {};
for (const lang of ["bm", "nn"]) {
  const split = splitCorpus(corpora[lang]);
  reports[lang] = runNgramAblation({
    ...split,
    baseConfig,
    steps,
    batchSize,
    learningRate: 0.0008,
    seeds,
    evalBatches,
  });
}

if (process.argv.includes("--json")) {
  console.log(JSON.stringify(reports, null, 2));
  process.exit(0);
}

console.log(`# Paired trigram-memory ablation`);
console.log(``);
console.log(
  `${steps} steps × ${seedCount} seeds per language; Adam, SiTU, dim 48, 2 layers, ` +
    `batch ${batchSize}, context 32. The last paragraph is held out.`
);
console.log(``);
console.log(`| Language | Variant | Params | Tail train loss | Held-out loss | ms/step | grad p99 | clip rate | max activation |`);
console.log(`|---|---:|---:|---:|---:|---:|---:|---:|---:|`);
for (const lang of ["bm", "nn"]) {
  const r = reports[lang];
  for (const v of [r.baseline, r.trigram])
    console.log(
      `| ${lang} | ${v.variant} | ${v.params} | ${f(v.tailTrainLoss)} | ` +
        `${f(v.heldoutLoss)} | ${f(v.msPerStep, 2)} | ${f(v.gradP99, 3)} | ` +
        `${f(v.clipRate * 100, 1)}% | ${f(v.maxActivation, 3)} |`
    );
  console.log(
    `| ${lang} | trigram Δ | +${r.delta.params} (${signed(r.delta.paramsPct)}) | — | ` +
      `${signed(r.delta.heldoutLossPct)} | ${signed(r.delta.msPerStepPct)} | ` +
      `${signed(r.delta.gradP99Pct)} | ${(r.delta.clipRate * 100).toFixed(1)} pp | ` +
      `${signed(r.delta.maxActivationPct)} |`
  );
}
console.log(``);
for (const lang of ["bm", "nn"]) {
  const r = reports[lang];
  console.log(
    `- ${lang}: ${r.tokenizer.trainCharacters} training characters = ` +
      `${r.tokenizer.trainTokens} tokens; ${r.tokenizer.holdoutCharacters} held-out characters = ` +
      `${r.tokenizer.holdoutTokens} tokens; shared initialization max diff ${r.sharedInitMaxDiff}.`
  );
}
