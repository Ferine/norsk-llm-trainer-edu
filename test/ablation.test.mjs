import assert from "node:assert/strict";
import { runNgramAblation } from "./dist/ablation.js";

const report = runNgramAblation({
  trainText: "æra er her. æra er nær. 🫐 er blå.\n".repeat(3),
  holdoutText: "æra er blå, og 🫐 er her.",
  baseConfig: {
    dim: 8,
    nLayer: 2,
    nHead: 2,
    seqLen: 8,
    ffnMult: 4,
    act: "situ",
  },
  steps: 3,
  batchSize: 1,
  learningRate: 0.001,
  seeds: [7, 19],
  evalBatches: 2,
  ngram: { size: 3, slots: 16, layer: 1 },
});

assert.equal(report.tokenizer.oneCharacterPerToken, true);
assert.equal(report.tokenizer.trainTokens, report.tokenizer.trainCharacters);
assert.equal(report.tokenizer.holdoutTokens, report.tokenizer.holdoutCharacters);
assert.equal(report.sharedInitMaxDiff, 0, "paired shared weights should be exactly identical");
assert.equal(report.runs.length, 4);
assert.equal(report.baseline.runs, 2);
assert.equal(report.trigram.runs, 2);
assert.equal(report.delta.params, 16 * 8);
assert.equal(report.baseline.ngramParams, 0);
assert.equal(report.trigram.ngramParams, 16 * 8);
for (const run of report.runs) {
  for (const key of ["tailTrainLoss", "heldoutLoss", "msPerStep", "gradP99", "clipRate", "maxActivation"])
    assert.ok(Number.isFinite(run[key]), `${run.variant}.${key} should be finite`);
  assert.ok(run.msPerStep >= 0);
  assert.ok(run.clipRate >= 0 && run.clipRate <= 1);
}

assert.throws(
  () => runNgramAblation({
    trainText: "x",
    holdoutText: "yy",
    baseConfig: { dim: 8, nLayer: 2, nHead: 2, seqLen: 4, ffnMult: 4 },
    steps: 1,
    batchSize: 1,
    learningRate: 0.001,
    seeds: [1],
  }),
  /at least two characters/
);

console.log("paired ablation harness: metrics and character-token invariant ok");
