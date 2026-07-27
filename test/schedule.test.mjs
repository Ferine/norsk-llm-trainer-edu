// Cosine learning-rate schedule with a short warmup (Kimi K3 §3.3).

import assert from "node:assert/strict";
import { cosineLr } from "./dist/ml.js";

const peak = 0.001;
const total = 3500;
const o = { peak, total };
const warm = Math.floor(total * 0.01); // 35 steps

// ---- warmup ramps linearly up to the peak --------------------------------
assert.ok(cosineLr(0, o) > 0, "warmup must not start at zero");
assert.ok(cosineLr(0, o) < peak / 10, "the first step should be a small fraction of the peak");
assert.ok(Math.abs(cosineLr(warm - 1, o) - peak) < 1e-12, "warmup should end exactly at the peak");
for (let i = 1; i < warm; i++)
  assert.ok(cosineLr(i, o) > cosineLr(i - 1, o), `warmup should rise at step ${i}`);

// ---- then a monotone cosine decay down to the floor ----------------------
const floor = peak * 0.1;
assert.ok(Math.abs(cosineLr(total, o) - floor) < 1e-12, "should land on the floor");
assert.ok(Math.abs(cosineLr(warm, o) - peak) < 1e-9, "decay starts at the peak");
for (let i = warm + 1; i <= total; i++)
  assert.ok(cosineLr(i, o) <= cosineLr(i - 1, o) + 1e-12, `decay should not rise at step ${i}`);

// Halfway through the decay a cosine sits at the midpoint of peak and floor.
const mid = cosineLr(warm + Math.round((total - warm) / 2), o);
assert.ok(Math.abs(mid - (peak + floor) / 2) < peak * 0.01, `midpoint off: ${mid}`);

// ---- clamping and configuration ------------------------------------------
assert.equal(cosineLr(total + 1000, o), cosineLr(total, o), "past the end it holds the floor");
assert.equal(cosineLr(-5, o), cosineLr(0, o), "negative steps clamp to the start");
assert.ok(
  Math.abs(cosineLr(total, { peak, total, minFrac: 0 })) < 1e-12,
  "minFrac 0 should decay to zero"
);
assert.ok(cosineLr(0, { peak, total, warmupFrac: 0 }) === peak, "no warmup starts at the peak");

assert.throws(() => cosineLr(0, { peak: 0, total }), RangeError);
assert.throws(() => cosineLr(0, { peak, total: 0 }), RangeError);
assert.throws(() => cosineLr(0, { peak, total: 1.5 }), RangeError);

console.log("schedule: ok");
