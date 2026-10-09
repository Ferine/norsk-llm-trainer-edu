import assert from "node:assert/strict";
import {
  Adam, Transformer, cloneTransformer, dpoStep, dpoStepChunks, mulberry32, seqLogProbValue,
} from "./dist/ml.js";

// Independent oracle: teacher-force one token at a time exactly as generation
// does, with a fresh sliding context and positional indices on each prediction.
function score(model, prompt, continuation) {
  const context = prompt.length ? prompt.slice() : [0];
  let value = 0;
  for (const token of continuation) {
    const window = context.slice(-model.seqLen);
    value += seqLogProbValue(model.forward(window), window.length - 1, [token]);
    context.push(token);
  }
  return value;
}

function margin(model, reference, pair) {
  return (score(model, pair.promptIds, pair.chosenIds) - score(reference, pair.promptIds, pair.chosenIds)) -
    (score(model, pair.promptIds, pair.rejectedIds) - score(reference, pair.promptIds, pair.rejectedIds));
}

function makeModel(seqLen, extra = {}) {
  return new Transformer({vocab: 5, dim: 8, nLayer: 1, nHead: 2, seqLen, ffnMult: 2, ...extra}, mulberry32(1337));
}

// Every shipped context size used to discard the last character of the 48-token
// answers. A preference differing only there must now produce a learning signal.
for (const seqLen of [32, 40, 48]) {
  const model = makeModel(seqLen);
  const reference = cloneTransformer(model);
  const pair = {
    promptIds: [2, 3, 4, 0],
    chosenIds: [...Array(47).fill(1), 2],
    rejectedIds: [...Array(47).fill(1), 3],
  };
  const opt = new Adam(model.params, 1e-3);
  for (let i = 0; i < 3; i++) dpoStep(model, reference, opt, [pair], 1, 0.1, mulberry32(7));
  assert.ok(margin(model, reference, pair) > 0.01, `seqLen ${seqLen}: learn the preferred final token`);
}

// Check full likelihoods for short/long/empty prompts, unequal answer lengths,
// and the transition from a growing context to a sliding one.
for (const extra of [{}, {act: "situ"}, {moe: {experts: 2, topK: 1, bias: 0.001}}, {ngram: {size: 3, slots: 16, layer: 0}}]) {
  for (const promptIds of [[], [1, 2], [0, 1, 2, 3, 4, 0, 2]]) {
    const model = makeModel(4, extra);
    const reference = cloneTransformer(model);
    model.head.d[0] += 0.2;
    const pair = {promptIds, chosenIds: [2, 3, 1, 4, 2, 0, 1], rejectedIds: [1, 4, 3, 0, 2]};
    const expected = margin(model, reference, pair);
    const result = dpoStep(model, reference, new Adam(model.params, 0), [pair], 1, 0.1, mulberry32(7));
    assert.ok(Math.abs(result.margin - expected) < 1e-5, `generation/scoring context parity: ${promptIds}, ${JSON.stringify(extra)}`);
    assert.ok(Math.abs(result.loss - Math.log1p(Math.exp(-0.1 * expected))) < 1e-6);
  }
}

// Verify the streamed gradients and minibatch averaging against finite
// differences of the independent objective, including tokens after rollover.
{
  const model = makeModel(4);
  const reference = cloneTransformer(model);
  model.head.d[0] += 0.1;
  const pairs = [
    {promptIds: [1, 2], chosenIds: [2, 3, 1, 4, 2, 0], rejectedIds: [1, 4, 3, 0, 2]},
    {promptIds: [4, 3, 2, 1, 0], chosenIds: [1, 2, 3, 4, 0], rejectedIds: [3, 2, 0, 1]},
  ];
  const beta = 0.2;
  const opt = new Adam(model.params, 0);
  opt.clipGradNorm = () => ({norm: 0, clipped: false});
  const samples = [0, 0.75];
  dpoStep(model, reference, opt, pairs, 2, beta, () => samples.shift());
  const objective = () => pairs.reduce((sum, pair) => sum + Math.log1p(Math.exp(-beta * margin(model, reference, pair))), 0) / pairs.length;
  const eps = 1e-3;
  for (const p of [model.head, model.tokEmb, model.posEmb]) {
    for (const index of [0, 3, p.d.length - 1]) {
      const original = p.d[index];
      const analytical = p.grad[index];
      p.d[index] = original + eps;
      const plus = objective();
      p.d[index] = original - eps;
      const minus = objective();
      p.d[index] = original;
      const numerical = (plus - minus) / (2 * eps);
      assert.ok(Math.abs(analytical - numerical) < 2e-3, `gradient ${analytical} vs ${numerical}`);
    }
  }
}

// Cancelling between windows must not commit a partial optimizer update or
// leak the unfinished pair's gradients into the next training request.
{
  const model = makeModel(4);
  const reference = cloneTransformer(model);
  const baseline = cloneTransformer(model);
  const pair = {promptIds: [1, 2], chosenIds: [2, 3, 1, 4, 2, 0], rejectedIds: [1, 4, 3, 0, 2]};
  const opt = new Adam(model.params, 1e-3);
  const work = dpoStepChunks(model, reference, opt, [pair], 1, 0.1, mulberry32(7));
  assert.equal(work.next().done, false);
  assert.equal(work.next().done, false);
  model.params.forEach((p, i) => assert.deepEqual(p.d, reference.params[i].d));
  assert.equal(opt.t, 0, "no optimizer step before the minibatch finishes");
  work.return();
  dpoStep(model, reference, opt, [pair], 1, 0.1, mulberry32(7));
  dpoStep(baseline, reference, new Adam(baseline.params, 1e-3), [pair], 1, 0.1, mulberry32(7));
  model.params.forEach((p, i) => assert.deepEqual(p.d, baseline.params[i].d, "no cancelled gradients leak"));
}

console.log("dpo context: full answers, generation parity, streamed gradients and cancellation ok");
