import { dpoLoss, backward } from "./dist/ml.js";
import assert from "node:assert/strict";

function scalar(v) {
  return { d: Float32Array.from([v]), rows: 1, cols: 1, grad: new Float32Array(1), _prev: [], _back: () => {} };
}
const beta = 0.1, refW = 0.3, refL = -0.2;
function lossVal(lw, ll) {
  const z = beta * ((lw - refW) - (ll - refL));
  return z > 0 ? Math.log1p(Math.exp(-z)) : -z + Math.log1p(Math.exp(z));
}

const lpW = scalar(0.5), lpL = scalar(-0.1);
lpW.grad.fill(0); lpL.grad.fill(0);
backward(dpoLoss(lpW, lpL, refW, refL, beta));
const gW = lpW.grad[0], gL = lpL.grad[0];

const eps = 1e-4;
const numW = (lossVal(0.5 + eps, -0.1) - lossVal(0.5 - eps, -0.1)) / (2 * eps);
const numL = (lossVal(0.5, -0.1 + eps) - lossVal(0.5, -0.1 - eps)) / (2 * eps);
assert.ok(Math.abs(numW - gW) < 1e-4, `dLoss/dlpW ${gW} vs ${numW}`);
assert.ok(Math.abs(numL - gL) < 1e-4, `dLoss/dlpL ${gL} vs ${numL}`);

// loss is smaller when chosen is favoured (z large positive) than when reversed
const favored = dpoLoss(scalar(5), scalar(-5), 0, 0, beta).d[0];
const reversed = dpoLoss(scalar(-5), scalar(5), 0, 0, beta).d[0];
assert.ok(favored < reversed, "favoured pair has lower loss");
console.log("dpo-loss: PASS");
