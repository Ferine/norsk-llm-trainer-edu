import { Transformer, cloneTransformer, mulberry32 } from "./dist/ml.js";
import assert from "node:assert/strict";

const cfg = { vocab: 7, dim: 8, nLayer: 2, nHead: 2, seqLen: 6, ffnMult: 2 };
const m = new Transformer(cfg, mulberry32(3));
const c = cloneTransformer(m);

// identical parameters
for (let i = 0; i < m.params.length; i++)
  for (let j = 0; j < m.params[i].d.length; j++)
    assert.equal(c.params[i].d[j], m.params[i].d[j]);

// identical forward output
const ids = [1, 2, 3, 0];
const lm = m.forward(ids), lc = c.forward(ids);
assert.equal(lm.d.length, lc.d.length);
for (let i = 0; i < lm.d.length; i++) assert.ok(Math.abs(lm.d[i] - lc.d[i]) < 1e-6);

// independence: mutating the source must not change the clone
m.params[0].d[0] += 1;
assert.notEqual(c.params[0].d[0], m.params[0].d[0]);
console.log("clone: PASS");
