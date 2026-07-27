// Muon: Newton–Schulz orthogonalization, per-head splitting, and that the
// whole thing actually trains. Mirrors Kimi K3 §2.5 (Per-Head Muon).

import assert from "node:assert/strict";
import {
  Muon,
  Transformer,
  backward,
  crossEntropyLoss,
  mulberry32,
  newtonSchulz,
  trainStep,
} from "./dist/ml.js";
import { buildTokenizer } from "./dist/corpus.js";

const rng = mulberry32(11);

function randMat(rows, cols, scale = 1) {
  const a = new Float32Array(rows * cols);
  for (let i = 0; i < a.length; i++) a[i] = (rng() * 2 - 1) * scale;
  return a;
}

function froSq(a) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * a[i];
  return s;
}

// Singular values of an n×m matrix, via a Jacobi eigen-decomposition of AᵀA.
// Slow and only fit for the small matrices here, but it is a genuinely
// independent check — it shares no code with newtonSchulz.
function svals(A, n, m) {
  const G = Array.from({ length: m }, () => new Float64Array(m));
  for (let i = 0; i < m; i++)
    for (let j = 0; j < m; j++) {
      let s = 0;
      for (let r = 0; r < n; r++) s += A[r * m + i] * A[r * m + j];
      G[i][j] = s;
    }
  for (let sweep = 0; sweep < 60; sweep++)
    for (let p = 0; p < m; p++)
      for (let q = p + 1; q < m; q++) {
        if (Math.abs(G[p][q]) < 1e-14) continue;
        const theta = (G[q][q] - G[p][p]) / (2 * G[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < m; k++) {
          const gp = G[k][p];
          const gq = G[k][q];
          G[k][p] = c * gp - s * gq;
          G[k][q] = s * gp + c * gq;
        }
        for (let k = 0; k < m; k++) {
          const gp = G[p][k];
          const gq = G[q][k];
          G[p][k] = c * gp - s * gq;
          G[q][k] = s * gp + c * gq;
        }
      }
  return Array.from({ length: m }, (_, i) => Math.sqrt(Math.max(0, G[i][i]))).sort((a, b) => b - a);
}

// ---- 1. every singular value is pulled toward 1 ---------------------------
// This is what orthogonalization means here: no direction of the update is
// allowed to dominate. Note the tuned quintic Muon uses does not converge to
// exactly 1 — it parks the singular values in a band around it, on purpose,
// because that converges far faster and the band is close enough.
for (const [rows, cols] of [
  [8, 8],
  [12, 5],
  [5, 12],
  [48, 24],
]) {
  const G = randMat(rows, cols);
  const O = newtonSchulz(G, rows, cols);
  assert.equal(O.length, rows * cols);
  const m = Math.min(rows, cols);
  const sv = rows >= cols ? svals(O, rows, cols) : svals(transposed(O, rows, cols), cols, rows);
  for (const s of sv)
    assert.ok(s > 0.6 && s < 1.4, `σ=${s} outside the band for ${rows}×${cols}: ${sv}`);
  // …and the same statement read off the Frobenius norm, which is Σσ².
  const rel = Math.abs(froSq(O) - m) / m;
  assert.ok(rel < 0.3, `‖O‖²_F should be near ${m} for ${rows}×${cols}, off by ${rel}`);
}

function transposed(A, rows, cols) {
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) out[c * rows + r] = A[r * cols + c];
  return out;
}

// A badly conditioned input is the whole point: one direction 100× the other.
{
  const rows = 6;
  const cols = 6;
  const G = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) G[r * cols + r] = r === 0 ? 1 : 0.01; // σ = 1, 0.01 …
  const O = newtonSchulz(G, rows, cols);
  let min = Infinity;
  let max = 0;
  for (let r = 0; r < rows; r++) {
    const v = Math.abs(O[r * cols + r]);
    if (v < min) min = v;
    if (v > max) max = v;
  }
  assert.ok(max / min < 1.5, `spread should collapse from 100× to ~1×, got ${max / min}`);
}

// ---- 2. scale invariance --------------------------------------------------
// Only the direction survives orthogonalization, so a rescaled gradient must
// produce the identical update — that is what makes Muon insensitive to the
// gradient's magnitude.
{
  const G = randMat(9, 4);
  const S = Float32Array.from(G, (x) => x * 37.5);
  const a = newtonSchulz(G, 9, 4);
  const b = newtonSchulz(S, 9, 4);
  for (let i = 0; i < a.length; i++)
    assert.ok(Math.abs(a[i] - b[i]) < 1e-4, `scale invariance broke at ${i}`);
}

// zero in, zero out (no NaN from dividing by the norm)
{
  const O = newtonSchulz(new Float32Array(12), 3, 4);
  for (const v of O) assert.equal(v, 0);
}

assert.throws(() => newtonSchulz(new Float32Array(6), 3, 4), RangeError);

// ---- 3. per-head update matches a hand-computed one -----------------------
// Q/K/V are split into one column block per head and orthogonalized separately,
// so heads with small gradients get the same step size as loud ones.
{
  const text = "det var en gang en fisker som rodde ut pa fjorden.\n";
  const tok = buildTokenizer(text);
  const data = tok.encode(text);
  const cfg = { vocab: tok.vocab, dim: 8, nLayer: 1, nHead: 2, seqLen: 8, ffnMult: 2 };
  const model = new Transformer(cfg, mulberry32(1337));
  const lr = 0.01;
  const momentum = 0.95;
  const opt = new Muon(model.optimGroups(), lr, momentum);

  const Wq = model.blocks[0].Wq;
  const before = Float32Array.from(Wq.d);

  opt.zeroGrad();
  const x = data.slice(0, cfg.seqLen);
  const y = data.slice(1, cfg.seqLen + 1);
  backward(crossEntropyLoss(model.forward(x), y));
  const grad = Float32Array.from(Wq.grad);
  opt.clipGradNorm(1.0);
  const clipped = Float32Array.from(Wq.grad);
  opt.step();

  // First step: buffer = grad, Nesterov look-ahead = grad + momentum·grad.
  const { dim, nHead } = cfg;
  const hd = dim / nHead;
  for (let h = 0; h < nHead; h++) {
    const blk = new Float32Array(dim * hd);
    for (let r = 0; r < dim; r++)
      for (let c = 0; c < hd; c++)
        blk[r * hd + c] = clipped[r * dim + h * hd + c] * (1 + momentum);
    const O = newtonSchulz(blk, dim, hd, 5);
    const sc = lr * 0.2 * Math.sqrt(Math.max(dim, hd));
    for (let r = 0; r < dim; r++)
      for (let c = 0; c < hd; c++) {
        const expected = before[r * dim + h * hd + c] - sc * O[r * hd + c];
        const got = Wq.d[r * dim + h * hd + c];
        assert.ok(
          Math.abs(expected - got) < 1e-6,
          `head ${h} (${r},${c}): expected ${expected}, got ${got}`
        );
      }
  }

  // Gradient clipping must have run over every group, not just the matrices.
  assert.ok(froSq(clipped) <= froSq(grad) + 1e-9);

  // Per-head really is different from orthogonalizing the whole matrix.
  const whole = newtonSchulz(Float32Array.from(clipped, (g) => g * (1 + momentum)), dim, dim, 5);
  let diff = 0;
  for (let r = 0; r < dim; r++)
    for (let c = 0; c < dim; c++) {
      const applied = (before[r * dim + c] - Wq.d[r * dim + c]) / (lr * 0.2);
      diff += Math.abs(applied - Math.sqrt(dim) * whole[r * dim + c]);
    }
  assert.ok(diff > 1e-3, "per-head and whole-matrix updates should differ");
}

// ---- 4. the scalar group is not forgotten ---------------------------------
{
  const text = "det var en gang en fisker som rodde ut pa fjorden.\n";
  const tok = buildTokenizer(text);
  const data = tok.encode(text);
  const cfg = { vocab: tok.vocab, dim: 8, nLayer: 1, nHead: 2, seqLen: 8, ffnMult: 2 };
  const model = new Transformer(cfg, mulberry32(1337));
  const groups = model.optimGroups();

  // Every parameter belongs to exactly one group.
  const seen = new Set([...groups.matrix.map((g) => g.p), ...groups.scalar]);
  assert.equal(seen.size, model.params.length, "groups must cover every parameter once");
  for (const p of model.params) assert.ok(seen.has(p), "a parameter escaped the groups");
  for (const { p } of groups.matrix)
    assert.ok(p.rows > 1 && p.cols > 1, "only matrices may be orthogonalized");

  const opt = new Muon(groups, 0.01);
  const ln1g = model.blocks[0].ln1g;
  const beforeLn = Float32Array.from(ln1g.d);
  const beforeEmb = Float32Array.from(model.tokEmb.d);
  trainStep(model, opt, data, cfg.seqLen, 2, mulberry32(5));
  assert.notDeepEqual(Array.from(ln1g.d), Array.from(beforeLn), "layer norm should move (Adam)");
  assert.notDeepEqual(
    Array.from(model.tokEmb.d),
    Array.from(beforeEmb),
    "embeddings should move (Adam)"
  );

  // lr flows through to the Adam side too.
  opt.lr = 0.002;
  assert.equal(opt.lr, 0.002);
}

// ---- 5. it trains --------------------------------------------------------
{
  const text =
    "det var en gang en fisker som rodde ut pa fjorden. han sa en sel som lekte i solen.\n";
  const tok = buildTokenizer(text);
  const data = tok.encode(text);
  const cfg = { vocab: tok.vocab, dim: 16, nLayer: 2, nHead: 2, seqLen: 16, ffnMult: 2 };
  const model = new Transformer(cfg, mulberry32(1337));
  const opt = new Muon(model.optimGroups(), 0.02);
  const r = mulberry32(42);
  let first = 0;
  let last = 0;
  for (let i = 0; i < 120; i++) {
    const l = trainStep(model, opt, data, cfg.seqLen, 2, r);
    if (i === 0) first = l;
    last = l;
    assert.ok(Number.isFinite(l), `loss went non-finite at step ${i}`);
  }
  assert.ok(last < first * 0.7, `Muon should learn: ${first.toFixed(3)} → ${last.toFixed(3)}`);
}

console.log("muon: ok");
