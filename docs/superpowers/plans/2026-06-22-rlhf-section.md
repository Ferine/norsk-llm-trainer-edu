# RLHF Section Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an interactive DPO-based RLHF section to the in-browser Nynorsk transformer trainer, where a visitor picks the better of two generated continuations and the policy is fine-tuned toward it against a frozen reference.

**Architecture:** Extend the hand-written autograd engine (`src/lib/ml.ts`) with per-sequence log-probability, a DPO loss, a reference-model clone, a token-level sampler, and a DPO training step. Encapsulate UI orchestration in a `useRlhf` hook and a presentational `Rlhf` component, wired into `App.tsx` as a new step-5 section that operates on the Section-3 model.

**Tech Stack:** React 19 + TypeScript (strict) + Vite 7 + Tailwind 4. Tests are standalone Node ESM scripts run against `tsc`-transpiled output (no new dependencies), mirroring the project's existing verification style.

## Global Constraints

- Node `>=20.19.0`, pnpm `>=10` (see `package.json` `engines`). Run `pnpm`, never `npm`/`yarn`.
- `.npmrc` enforces `save-exact=true`, `save-prefix=""`, `minimum-release-age=10080` (7-day cooldown). **Add no new runtime or dev dependencies** — tests use only `tsc` (already present) + Node's built-in `node:assert`.
- TypeScript is `strict` with `noUnusedLocals` and `noUnusedParameters` — every declared local/param must be used.
- Production build is a single inlined HTML file via `vite-plugin-singlefile`; keep everything client-side, no network/storage.
- UI copy is Norwegian **Nynorsk**, matching the existing tone in `App.tsx`.
- Follow existing Tailwind idioms and the `cn()` helper (`src/utils/cn.ts`).
- DPO defaults (locked in spec): `β = 0.1`, DPO learning rate `1e-4`.
- Commit steps assume git. The project is **not yet a git repo** — run Task 0 first to initialize, or skip the commit step of each task if you prefer not to use git.
- Every git commit message must end with the trailer:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

## File Structure

- `src/lib/ml.ts` — **modify**: add `seqLogProb`, `seqLogProbValue`, `dpoLoss`, `cloneTransformer`, `sampleTokens`, `dpoStep`, `PrefPair`; refactor `generate` onto `sampleTokens`.
- `src/lib/useRlhf.ts` — **create**: RLHF state/refs/handlers hook.
- `src/components/ui.tsx` — **create**: `Section` and `Card` extracted from `App.tsx` (shared by App + Rlhf).
- `src/components/Rlhf.tsx` — **create**: presentational RLHF section UI.
- `src/App.tsx` — **modify**: import shared `Section`/`Card`, render RLHF section (step 5), renumber "eigen tekst" to step 6, invalidate RLHF on engine rebuild and base-train start, wire mutual-exclusion flags.
- `test/*.test.mjs` — **create**: gradient/behavior/smoke tests.
- `package.json` — **modify**: add `test:build` and `test` scripts.
- `.gitignore` — **modify**: ignore `test/dist/`.

---

## Task 0: (Optional) Initialize git baseline

Skip this task and all "Commit" steps if you do not want version control.

- [ ] **Step 1: Initialize repo**

Run:
```bash
cd /Users/x/dev/develop-nynorsk-llm-trainer
git init
```

- [ ] **Step 2: Commit the current working tree as baseline**

```bash
git add -A
git commit -m "chore: baseline before RLHF section

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```
Expected: a commit is created with the existing app (including the two earlier bug fixes).

---

## Task 1: Test harness + `seqLogProb` / `seqLogProbValue`

**Files:**
- Modify: `package.json` (scripts)
- Modify: `.gitignore`
- Modify: `src/lib/ml.ts` (add two functions)
- Test: `test/seq-logprob.test.mjs`

**Interfaces:**
- Consumes: existing `Tensor`, `tensor()`, `backward()`, `mulberry32()` from `ml.ts`.
- Produces:
  - `seqLogProb(logits: Tensor, r0: number, targets: number[]): Tensor` — scalar `[1,1]`, autograd into `logits`.
  - `seqLogProbValue(logits: Tensor, r0: number, targets: number[]): number` — numeric only, no graph.

- [ ] **Step 1: Add test scripts to `package.json`**

In `package.json`, add to the `"scripts"` object (after `"preview"`):
```json
    "test:build": "tsc src/lib/ml.ts src/lib/corpus.ts --rootDir src/lib --outDir test/dist --target ES2020 --module ESNext --moduleResolution bundler --skipLibCheck",
    "test": "pnpm run test:build && node test/seq-logprob.test.mjs && node test/dpo-loss.test.mjs && node test/clone.test.mjs && node test/generate-parity.test.mjs && node test/dpo-smoke.test.mjs"
```
(Add a trailing comma after `"preview": "vite preview"` so JSON stays valid.)

- [ ] **Step 2: Ignore the test build output**

Append to `.gitignore`:
```
test/dist/
```

- [ ] **Step 3: Write the failing test** — `test/seq-logprob.test.mjs`

```js
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
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `pnpm run test:build && node test/seq-logprob.test.mjs`
Expected: FAIL — `SyntaxError: The requested module './dist/ml.js' does not provide an export named 'seqLogProb'`.

- [ ] **Step 5: Implement both functions in `src/lib/ml.ts`**

Add after `crossEntropyLoss` (before the `backward` function):
```ts
// Sum of log-probabilities log softmax(logits[r0+i])[targets[i]] for i in [0, targets.length).
// Backward: d(log softmax)/d logit = onehot(target) − softmax. (autograd)
export function seqLogProb(logits: Tensor, r0: number, targets: number[]): Tensor {
  const V = logits.cols;
  const len = targets.length;
  const out = tensor(1, 1, [logits]);
  const probs = new Float32Array(len * V);
  let lp = 0;
  for (let i = 0; i < len; i++) {
    const r = r0 + i;
    let mx = -Infinity;
    for (let c = 0; c < V; c++) {
      const v = logits.d[r * V + c];
      if (v > mx) mx = v;
    }
    let sum = 0;
    for (let c = 0; c < V; c++) {
      const e = Math.exp(logits.d[r * V + c] - mx);
      probs[i * V + c] = e;
      sum += e;
    }
    for (let c = 0; c < V; c++) probs[i * V + c] /= sum;
    lp += Math.log(probs[i * V + targets[i]] + 1e-12);
  }
  out.d[0] = lp;
  out._back = () => {
    const g = out.grad[0];
    for (let i = 0; i < len; i++) {
      const r = r0 + i;
      for (let c = 0; c < V; c++)
        logits.grad[r * V + c] += g * ((c === targets[i] ? 1 : 0) - probs[i * V + c]);
    }
  };
  return out;
}

// Numeric-only version (no autograd graph) for the frozen reference model.
export function seqLogProbValue(logits: Tensor, r0: number, targets: number[]): number {
  const V = logits.cols;
  let lp = 0;
  for (let i = 0; i < targets.length; i++) {
    const r = r0 + i;
    let mx = -Infinity;
    for (let c = 0; c < V; c++) {
      const v = logits.d[r * V + c];
      if (v > mx) mx = v;
    }
    let sum = 0;
    for (let c = 0; c < V; c++) sum += Math.exp(logits.d[r * V + c] - mx);
    lp += logits.d[r * V + targets[i]] - mx - Math.log(sum);
  }
  return lp;
}
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `pnpm run test:build && node test/seq-logprob.test.mjs`
Expected: `seq-logprob: PASS`

- [ ] **Step 7: Commit**

```bash
git add package.json .gitignore src/lib/ml.ts test/seq-logprob.test.mjs
git commit -m "feat(ml): seqLogProb + seqLogProbValue with gradient-checked test harness

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `dpoLoss`

**Files:**
- Modify: `src/lib/ml.ts`
- Test: `test/dpo-loss.test.mjs`

**Interfaces:**
- Consumes: `Tensor`, `tensor()`, `backward()`.
- Produces: `dpoLoss(lpW: Tensor, lpL: Tensor, refW: number, refL: number, beta: number): Tensor` — scalar `[1,1]`; backward writes into `lpW.grad[0]` and `lpL.grad[0]`.

- [ ] **Step 1: Write the failing test** — `test/dpo-loss.test.mjs`

```js
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pnpm run test:build && node test/dpo-loss.test.mjs`
Expected: FAIL — module does not provide export `dpoLoss`.

- [ ] **Step 3: Implement `dpoLoss` in `src/lib/ml.ts`** (add directly after `seqLogProbValue`)

```ts
// DPO loss for one preference pair.
// z = beta * ((lpW − refW) − (lpL − refL));  loss = softplus(−z) = −log sigmoid(z).
// d loss/dz = −sigmoid(−z);  chain: dz/dlpW = +beta, dz/dlpL = −beta.
export function dpoLoss(lpW: Tensor, lpL: Tensor, refW: number, refL: number, beta: number): Tensor {
  const out = tensor(1, 1, [lpW, lpL]);
  const z = beta * ((lpW.d[0] - refW) - (lpL.d[0] - refL));
  out.d[0] = z > 0 ? Math.log1p(Math.exp(-z)) : -z + Math.log1p(Math.exp(z));
  const sigNegZ = z > 0 ? Math.exp(-z) / (1 + Math.exp(-z)) : 1 / (1 + Math.exp(z)); // sigmoid(−z)
  out._back = () => {
    const g = out.grad[0];
    const dz = -sigNegZ; // d loss/dz
    lpW.grad[0] += g * dz * beta;
    lpL.grad[0] += g * dz * -beta;
  };
  return out;
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pnpm run test:build && node test/dpo-loss.test.mjs`
Expected: `dpo-loss: PASS`

- [ ] **Step 5: Commit**

```bash
git add src/lib/ml.ts test/dpo-loss.test.mjs
git commit -m "feat(ml): DPO loss with gradient-checked backward

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `cloneTransformer`

**Files:**
- Modify: `src/lib/ml.ts`
- Test: `test/clone.test.mjs`

**Interfaces:**
- Consumes: `Transformer`, `mulberry32()`. Relies on `Transformer.cfg` (public) and `Transformer.params` (public, deterministic order).
- Produces: `cloneTransformer(src: Transformer): Transformer`.

- [ ] **Step 1: Write the failing test** — `test/clone.test.mjs`

```js
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pnpm run test:build && node test/clone.test.mjs`
Expected: FAIL — module does not provide export `cloneTransformer`.

- [ ] **Step 3: Implement `cloneTransformer` in `src/lib/ml.ts`** (add after the `Transformer` class, before the `Adam` section)

```ts
// Deep-copy a model's parameters into a new Transformer with the same cfg.
// Used to freeze a reference policy for DPO; only forward passes run on the copy.
export function cloneTransformer(src: Transformer): Transformer {
  const dst = new Transformer(src.cfg, mulberry32(0));
  for (let i = 0; i < src.params.length; i++) dst.params[i].d.set(src.params[i].d);
  return dst;
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pnpm run test:build && node test/clone.test.mjs`
Expected: `clone: PASS`

- [ ] **Step 5: Commit**

```bash
git add src/lib/ml.ts test/clone.test.mjs
git commit -m "feat(ml): cloneTransformer for frozen DPO reference

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `sampleTokens` + refactor `generate`

**Files:**
- Modify: `src/lib/ml.ts` (extract sampling core, rewrite `generate` as a wrapper)
- Test: `test/generate-parity.test.mjs`

**Interfaces:**
- Consumes: `Transformer`, existing `SampleOpts`.
- Produces: `sampleTokens(model, encode, prompt, opts, rng): { promptIds: number[]; contIds: number[] }`. `generate(...)` keeps its current signature and returns `prompt + decode(contIds)`.

- [ ] **Step 1: Write the failing test** — `test/generate-parity.test.mjs`

```js
import { Transformer, generate, sampleTokens, mulberry32 } from "./dist/ml.js";
import { buildTokenizer, corpus } from "./dist/corpus.js";
import assert from "node:assert/strict";

const tok = buildTokenizer(corpus);
const cfg = { vocab: tok.vocab, dim: 16, nLayer: 1, nHead: 2, seqLen: 16, ffnMult: 2 };
const m = new Transformer(cfg, mulberry32(5));
const prompt = "Det var";

// greedy: deterministic, and generate is exactly prompt + decode(sampleTokens.contIds)
const og = { temperature: 0, topK: 5, length: 10 };
const g = generate(m, tok.decode, tok.encode, prompt, og, mulberry32(1));
const st = sampleTokens(m, tok.encode, prompt, og, mulberry32(1));
assert.equal(g, prompt + tok.decode(st.contIds));
assert.ok(g.startsWith(prompt));
assert.equal(st.contIds.length, 10);

// sampling: identical seed yields identical result through both entry points
const os = { temperature: 0.9, topK: 5, length: 12 };
const g2 = generate(m, tok.decode, tok.encode, prompt, os, mulberry32(9));
const st2 = sampleTokens(m, tok.encode, prompt, os, mulberry32(9));
assert.equal(g2, prompt + tok.decode(st2.contIds));
console.log("generate-parity: PASS");
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pnpm run test:build && node test/generate-parity.test.mjs`
Expected: FAIL — module does not provide export `sampleTokens`.

- [ ] **Step 3: Refactor `src/lib/ml.ts`**

Replace the entire existing `generate` function (the `export function generate(...) { ... }` block at the end of the file) with the two functions below:
```ts
// Sample a continuation token-by-token. Shared core for generate() and the RLHF arena.
export function sampleTokens(
  model: Transformer,
  encode: (s: string) => number[],
  prompt: string,
  opts: SampleOpts,
  rng: () => number
): { promptIds: number[]; contIds: number[] } {
  let ctx = encode(prompt);
  if (ctx.length === 0) ctx = [0];
  const promptIds = ctx.slice();
  const contIds: number[] = [];
  const maxCtx = model.seqLen;
  const greedy = opts.temperature <= 0;
  const topK = Math.max(1, Math.min(opts.topK, model.vocab));
  for (let step = 0; step < opts.length; step++) {
    const window = ctx.length > maxCtx ? ctx.slice(ctx.length - maxCtx) : ctx;
    const logits = model.forward(window);
    const V = model.vocab;
    const off = (window.length - 1) * V;
    if (greedy) {
      let best = 0;
      let bestv = -Infinity;
      for (let c = 0; c < V; c++) {
        const val = logits.d[off + c];
        if (val > bestv) {
          bestv = val;
          best = c;
        }
      }
      ctx.push(best);
      contIds.push(best);
      continue;
    }
    const scaled = new Float32Array(V);
    let mx = -Infinity;
    for (let c = 0; c < V; c++) {
      const val = logits.d[off + c] / opts.temperature;
      scaled[c] = val;
      if (val > mx) mx = val;
    }
    const idx: number[] = [];
    for (let c = 0; c < V; c++) idx.push(c);
    idx.sort((a, b) => scaled[b] - scaled[a]);
    const top = idx.slice(0, topK);
    let sum = 0;
    const probs = new Float32Array(top.length);
    for (let i = 0; i < top.length; i++) {
      const e = Math.exp(scaled[top[i]] - mx);
      probs[i] = e;
      sum += e;
    }
    const r = rng();
    let acc = 0;
    let chosen = top[top.length - 1];
    for (let i = 0; i < top.length; i++) {
      acc += probs[i] / sum;
      if (r <= acc) {
        chosen = top[i];
        break;
      }
    }
    ctx.push(chosen);
    contIds.push(chosen);
  }
  return { promptIds, contIds };
}

// Generer tekst: gje ein starttekst, så lat modellen predikere teikn for teikn.
export function generate(
  model: Transformer,
  decode: (ids: number[]) => string,
  encode: (s: string) => number[],
  prompt: string,
  opts: SampleOpts,
  rng: () => number
): string {
  const { contIds } = sampleTokens(model, encode, prompt, opts, rng);
  return prompt + decode(contIds);
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pnpm run test:build && node test/generate-parity.test.mjs`
Expected: `generate-parity: PASS`

- [ ] **Step 5: Verify the app still typechecks (generate is used by App)**

Run: `pnpm typecheck`
Expected: exit 0, no output errors.

- [ ] **Step 6: Commit**

```bash
git add src/lib/ml.ts test/generate-parity.test.mjs
git commit -m "refactor(ml): extract sampleTokens core from generate

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: `PrefPair` + `dpoStep`

**Files:**
- Modify: `src/lib/ml.ts`
- Test: `test/dpo-smoke.test.mjs`

**Interfaces:**
- Consumes: `Transformer`, `Adam`, `seqLogProb`, `seqLogProbValue`, `dpoLoss`, `backward`.
- Produces:
  - `interface PrefPair { promptIds: number[]; chosenIds: number[]; rejectedIds: number[]; }`
  - `dpoStep(policy, reference, opt, pairs, batch, beta, rng): { loss: number; margin: number; winRate: number }`

- [ ] **Step 1: Write the failing test** — `test/dpo-smoke.test.mjs`

```js
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pnpm run test:build && node test/dpo-smoke.test.mjs`
Expected: FAIL — module does not provide export `dpoStep`.

- [ ] **Step 3: Implement `PrefPair`, capping helper, and `dpoStep` in `src/lib/ml.ts`**

Add after the `trainStep` function:
```ts
export interface PrefPair {
  promptIds: number[];
  chosenIds: number[];
  rejectedIds: number[];
}

// Build a full sequence (prompt + continuation) capped to seqLen, truncating the
// prompt from the left first so the continuation is preserved. Returns the sequence
// and P = prompt length within the cap (>= 1). Continuation = seq.slice(P) (>= 1 token).
function capSeq(promptIds: number[], contIds: number[], seqLen: number): { seq: number[]; P: number } {
  let prompt = promptIds.length ? promptIds : [0];
  let cont = contIds.slice();
  if (cont.length < 1) cont = [prompt[prompt.length - 1]];
  if (cont.length >= seqLen) cont = cont.slice(0, seqLen - 1);
  let P = prompt.length;
  if (P + cont.length > seqLen) {
    P = seqLen - cont.length;
    prompt = prompt.slice(prompt.length - P);
  }
  return { seq: prompt.concat(cont), P };
}

// One DPO update over a sampled minibatch of preference pairs.
export function dpoStep(
  policy: Transformer,
  reference: Transformer,
  opt: Adam,
  pairs: PrefPair[],
  batch: number,
  beta: number,
  rng: () => number
): { loss: number; margin: number; winRate: number } {
  if (pairs.length === 0) return { loss: 0, margin: 0, winRate: 0 };
  opt.zeroGrad();
  const seqLen = policy.seqLen;
  const n = Math.min(batch, pairs.length);
  let totalLoss = 0;
  let totalMargin = 0;
  let wins = 0;
  for (let b = 0; b < n; b++) {
    const pair = pairs[Math.min(pairs.length - 1, Math.floor(rng() * pairs.length))];
    const w = capSeq(pair.promptIds, pair.chosenIds, seqLen);
    const l = capSeq(pair.promptIds, pair.rejectedIds, seqLen);
    const tgtW = w.seq.slice(w.P);
    const tgtL = l.seq.slice(l.P);

    const lpW = seqLogProb(policy.forward(w.seq), w.P - 1, tgtW);
    const lpL = seqLogProb(policy.forward(l.seq), l.P - 1, tgtL);
    const refW = seqLogProbValue(reference.forward(w.seq), w.P - 1, tgtW);
    const refL = seqLogProbValue(reference.forward(l.seq), l.P - 1, tgtL);

    const loss = dpoLoss(lpW, lpL, refW, refL, beta);
    backward(loss);
    totalLoss += loss.d[0];
    const margin = (lpW.d[0] - refW) - (lpL.d[0] - refL);
    totalMargin += margin;
    if (margin > 0) wins++;
  }
  if (n > 1)
    for (const p of policy.params)
      for (let i = 0; i < p.grad.length; i++) p.grad[i] /= n;
  opt.clipGradNorm(1.0);
  opt.step();
  return { loss: totalLoss / n, margin: totalMargin / n, winRate: wins / n };
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pnpm run test:build && node test/dpo-smoke.test.mjs`
Expected: `dpo-smoke: PASS (margin 0.000 -> <positive>, winRate 100%)`
(If `after.margin` is borderline below `0.1`, the algorithm is still correct — increase the DPO step count in the test from `100` to `150` rather than weakening the assertion.)

- [ ] **Step 5: Run the full suite**

Run: `pnpm test`
Expected: all five test files print `PASS`.

- [ ] **Step 6: Commit**

```bash
git add src/lib/ml.ts test/dpo-smoke.test.mjs
git commit -m "feat(ml): dpoStep preference update + smoke test

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Extract `Section` and `Card` to `src/components/ui.tsx`

This is a pure refactor so the new RLHF component can reuse the layout primitives without importing from `App.tsx`.

**Files:**
- Create: `src/components/ui.tsx`
- Modify: `src/App.tsx` (remove local `Section`/`Card`, import them)

**Interfaces:**
- Produces: `export function Section({ id, step, title, intro, children })` and `export function Card({ className, children })` — identical props/markup to the current `App.tsx` definitions.

- [ ] **Step 1: Create `src/components/ui.tsx`**

```tsx
import type { ReactNode } from "react";
import { cn } from "@/utils/cn";

export function Section({
  id,
  step,
  title,
  intro,
  children,
}: {
  id: string;
  step: number;
  title: string;
  intro?: string;
  children: ReactNode;
}) {
  return (
    <section id={id} className="scroll-mt-20">
      <div className="mb-5 flex items-start gap-4">
        <div className="flex h-10 w-10 flex-none items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500 to-violet-600 text-sm font-bold text-white shadow-md shadow-indigo-200">
          {step}
        </div>
        <div>
          <h2 className="text-2xl font-bold tracking-tight text-slate-900">{title}</h2>
          {intro && <p className="mt-1 max-w-2xl text-slate-600">{intro}</p>}
        </div>
      </div>
      {children}
    </section>
  );
}

export function Card({ className, children }: { className?: string; children: ReactNode }) {
  return (
    <div className={cn("rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6", className)}>
      {children}
    </div>
  );
}
```

- [ ] **Step 2: Update `src/App.tsx` imports**

After the existing `import Architecture from "@/components/Architecture";` line, add:
```tsx
import { Section, Card } from "@/components/ui";
```

- [ ] **Step 3: Delete the local definitions in `src/App.tsx`**

Remove the entire local `function Section({ ... }) { ... }` block and the entire local `function Card({ ... }) { ... }` block (the two helper functions defined between `charLabel` and `export default function App`). Leave `charLabel` intact.

- [ ] **Step 4: Verify typecheck and build**

Run: `pnpm typecheck && pnpm build`
Expected: both exit 0. (`noUnusedLocals` will flag a leftover local `Section`/`Card` if either block was not fully removed.)

- [ ] **Step 5: Commit**

```bash
git add src/components/ui.tsx src/App.tsx
git commit -m "refactor(ui): extract Section and Card into shared module

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: `useRlhf` hook

**Files:**
- Create: `src/lib/useRlhf.ts`

**Interfaces:**
- Consumes: `Adam`, `Transformer`, `cloneTransformer`, `dpoStep`, `mulberry32`, `sampleTokens`, `PrefPair` (from `ml.ts`); `Tokenizer` (type, from `corpus.ts`).
- Produces: `useRlhf(args): RlhfState`, where
  - `args = { getModel: () => Transformer | null; getTokenizer: () => Tokenizer | null; isTrained: () => boolean; baseRunning: boolean }`
  - `RlhfState` exposes: `started, dpoRunning, baseRunning, untrainedHint, prompt, setPrompt, temp, setTemp, pairA, pairB, generating, losses, metrics, start, generatePair, choose, skip, trainMore, stopTrainMore, resetTuning, reset`.
  - `pairA`/`pairB`: `{ text: string; promptIds: number[]; contIds: number[] } | null`.
  - `metrics`: `{ loss: number; margin: number; winRate: number; count: number }`.
  - `choose(winner: "A" | "B"): void`.

- [ ] **Step 1: Create `src/lib/useRlhf.ts`**

```ts
import { useCallback, useEffect, useRef, useState } from "react";
import {
  Adam,
  Transformer,
  cloneTransformer,
  dpoStep,
  mulberry32,
  sampleTokens,
  type PrefPair,
} from "@/lib/ml";
import type { Tokenizer } from "@/lib/corpus";

const BETA = 0.1;
const DPO_LR = 1e-4;
const PAIR_LEN = 48;
const LIVE_BURST = 5;
const LIVE_MINIBATCH = 4;
const TRAIN_MORE_STEPS = 60;
const TRAIN_CHUNK = 6;
const TRAIN_MINIBATCH = 4;
const PAIR_RETRY = 4;

export interface RlhfPair {
  text: string;
  promptIds: number[];
  contIds: number[];
}
export interface RlhfMetrics {
  loss: number;
  margin: number;
  winRate: number;
  count: number;
}

interface Args {
  getModel: () => Transformer | null;
  getTokenizer: () => Tokenizer | null;
  isTrained: () => boolean;
  baseRunning: boolean;
}

export function useRlhf({ getModel, getTokenizer, isTrained, baseRunning }: Args) {
  const referenceRef = useRef<Transformer | null>(null);
  const dpoOptRef = useRef<Adam | null>(null);
  const bufferRef = useRef<PrefPair[]>([]);
  const lossesRef = useRef<number[]>([]);
  const pairRngRef = useRef<() => number>(mulberry32(2027));
  const timerRef = useRef<number | null>(null);
  const dpoStepCountRef = useRef(0);
  const trainStartRef = useRef(0);
  const runningRef = useRef(false);

  const [started, setStarted] = useState(false);
  const [dpoRunning, setDpoRunning] = useState(false);
  const [untrainedHint, setUntrainedHint] = useState(false);
  const [prompt, setPrompt] = useState("Det var ein gong");
  const [temp, setTemp] = useState(1.0);
  const [pairA, setPairA] = useState<RlhfPair | null>(null);
  const [pairB, setPairB] = useState<RlhfPair | null>(null);
  const [generating, setGenerating] = useState(false);
  const [losses, setLosses] = useState<number[]>([]);
  const [metrics, setMetrics] = useState<RlhfMetrics>({ loss: 0, margin: 0, winRate: 0, count: 0 });

  const samplePair = useCallback((): { a: RlhfPair; b: RlhfPair } | null => {
    const model = getModel();
    const tok = getTokenizer();
    if (!model || !tok) return null;
    const opts = { temperature: Math.max(0.1, temp), topK: model.vocab, length: PAIR_LEN };
    const mk = (): RlhfPair => {
      const { promptIds, contIds } = sampleTokens(model, tok.encode, prompt, opts, pairRngRef.current);
      return { text: tok.decode(contIds), promptIds, contIds };
    };
    const a = mk();
    let b = mk();
    for (let r = 0; r < PAIR_RETRY && b.text === a.text; r++) b = mk();
    return { a, b };
  }, [getModel, getTokenizer, prompt, temp]);

  const generatePair = useCallback(() => {
    setGenerating(true);
    // yield to the browser so the "lagar par…" state can paint before heavy sampling
    window.setTimeout(() => {
      const pair = samplePair();
      if (pair) {
        setPairA(pair.a);
        setPairB(pair.b);
      }
      setGenerating(false);
    }, 10);
  }, [samplePair]);

  const start = useCallback(() => {
    const model = getModel();
    if (!model) return;
    referenceRef.current = cloneTransformer(model);
    dpoOptRef.current = new Adam(model.params, DPO_LR);
    bufferRef.current = [];
    lossesRef.current = [];
    dpoStepCountRef.current = 0;
    pairRngRef.current = mulberry32(2027);
    setUntrainedHint(!isTrained());
    setLosses([]);
    setMetrics({ loss: 0, margin: 0, winRate: 0, count: 0 });
    setStarted(true);
    generatePair();
  }, [getModel, isTrained, generatePair]);

  const runBurst = useCallback(
    (steps: number, minibatch: number) => {
      const model = getModel();
      const ref = referenceRef.current;
      const opt = dpoOptRef.current;
      if (!model || !ref || !opt || bufferRef.current.length === 0) return;
      let last = { loss: 0, margin: 0, winRate: 0 };
      for (let i = 0; i < steps; i++) {
        last = dpoStep(model, ref, opt, bufferRef.current, minibatch, BETA, pairRngRef.current);
        lossesRef.current.push(last.loss);
        dpoStepCountRef.current++;
      }
      setLosses(lossesRef.current.slice());
      setMetrics({ loss: last.loss, margin: last.margin, winRate: last.winRate, count: bufferRef.current.length });
    },
    [getModel]
  );

  const choose = useCallback(
    (winner: "A" | "B") => {
      const a = pairA;
      const b = pairB;
      if (!a || !b) return;
      const chosen = winner === "A" ? a : b;
      const rejected = winner === "A" ? b : a;
      bufferRef.current.push({
        promptIds: chosen.promptIds,
        chosenIds: chosen.contIds,
        rejectedIds: rejected.contIds,
      });
      runBurst(LIVE_BURST, LIVE_MINIBATCH);
      generatePair();
    },
    [pairA, pairB, runBurst, generatePair]
  );

  const skip = useCallback(() => generatePair(), [generatePair]);

  const trainLoop = useCallback(() => {
    timerRef.current = null;
    if (!runningRef.current) return;
    const model = getModel();
    const ref = referenceRef.current;
    const opt = dpoOptRef.current;
    if (!model || !ref || !opt || bufferRef.current.length === 0) {
      runningRef.current = false;
      setDpoRunning(false);
      return;
    }
    const done = dpoStepCountRef.current - trainStartRef.current;
    const chunk = Math.min(TRAIN_CHUNK, TRAIN_MORE_STEPS - done);
    let last = { loss: 0, margin: 0, winRate: 0 };
    for (let i = 0; i < chunk; i++) {
      last = dpoStep(model, ref, opt, bufferRef.current, TRAIN_MINIBATCH, BETA, pairRngRef.current);
      lossesRef.current.push(last.loss);
      dpoStepCountRef.current++;
    }
    setLosses(lossesRef.current.slice());
    setMetrics({ loss: last.loss, margin: last.margin, winRate: last.winRate, count: bufferRef.current.length });
    if (dpoStepCountRef.current - trainStartRef.current >= TRAIN_MORE_STEPS) {
      runningRef.current = false;
      setDpoRunning(false);
      return;
    }
    timerRef.current = window.setTimeout(trainLoop, 0);
  }, [getModel]);

  const trainMore = useCallback(() => {
    if (bufferRef.current.length === 0 || runningRef.current) return;
    trainStartRef.current = dpoStepCountRef.current;
    runningRef.current = true;
    setDpoRunning(true);
    trainLoop();
  }, [trainLoop]);

  const stopTrainMore = useCallback(() => {
    runningRef.current = false;
    setDpoRunning(false);
    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const resetTuning = useCallback(() => {
    const model = getModel();
    const ref = referenceRef.current;
    if (model && ref) for (let i = 0; i < model.params.length; i++) model.params[i].d.set(ref.params[i].d);
    if (model) dpoOptRef.current = new Adam(model.params, DPO_LR);
    bufferRef.current = [];
    lossesRef.current = [];
    dpoStepCountRef.current = 0;
    setLosses([]);
    setMetrics({ loss: 0, margin: 0, winRate: 0, count: 0 });
    generatePair();
  }, [getModel, generatePair]);

  const reset = useCallback(() => {
    runningRef.current = false;
    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    referenceRef.current = null;
    dpoOptRef.current = null;
    bufferRef.current = [];
    lossesRef.current = [];
    dpoStepCountRef.current = 0;
    setStarted(false);
    setDpoRunning(false);
    setUntrainedHint(false);
    setPairA(null);
    setPairB(null);
    setGenerating(false);
    setLosses([]);
    setMetrics({ loss: 0, margin: 0, winRate: 0, count: 0 });
  }, []);

  useEffect(
    () => () => {
      runningRef.current = false;
      if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    },
    []
  );

  return {
    started,
    dpoRunning,
    baseRunning,
    untrainedHint,
    prompt,
    setPrompt,
    temp,
    setTemp,
    pairA,
    pairB,
    generating,
    losses,
    metrics,
    start,
    generatePair,
    choose,
    skip,
    trainMore,
    stopTrainMore,
    resetTuning,
    reset,
  };
}
```

- [ ] **Step 2: Verify typecheck**

Run: `pnpm typecheck`
Expected: exit 0. The hook is not yet imported anywhere; `tsc` still typechecks it via the `src` include.

- [ ] **Step 3: Commit**

```bash
git add src/lib/useRlhf.ts
git commit -m "feat(rlhf): useRlhf hook for DPO preference tuning state

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: `Rlhf` presentational component

**Files:**
- Create: `src/components/Rlhf.tsx`

**Interfaces:**
- Consumes: `useRlhf` return type, `Card` (from `ui.tsx`), `LossChart` (existing component).
- Produces: `export default function Rlhf({ rlhf, examples }: { rlhf: ReturnType<typeof useRlhf>; examples: string[] })`.

- [ ] **Step 1: Create `src/components/Rlhf.tsx`**

```tsx
import LossChart from "@/components/LossChart";
import { Card } from "@/components/ui";
import type { useRlhf } from "@/lib/useRlhf";

type RlhfApi = ReturnType<typeof useRlhf>;

function PrefCard({
  label,
  text,
  onPick,
  disabled,
}: {
  label: string;
  text: string;
  onPick: () => void;
  disabled: boolean;
}) {
  return (
    <div className="flex flex-col rounded-xl border border-slate-200 bg-white p-4">
      <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Svar {label}</div>
      <p className="min-h-16 flex-1 whitespace-pre-wrap font-mono text-sm text-slate-700">{text || "…"}</p>
      <button
        onClick={onPick}
        disabled={disabled}
        className="mt-3 rounded-lg bg-indigo-600 px-3 py-2 text-sm font-semibold text-white transition hover:bg-indigo-500 disabled:opacity-50"
      >
        👍 {label} er betre
      </button>
    </div>
  );
}

export default function Rlhf({ rlhf, examples }: { rlhf: RlhfApi; examples: string[] }) {
  const busy = rlhf.baseRunning || rlhf.dpoRunning || rlhf.generating;

  if (!rlhf.started) {
    return (
      <Card className="space-y-4">
        <p className="text-sm text-slate-600">
          RLHF («Reinforcement Learning from Human Feedback») lærer modellen kva slags svar vi
          menneske føretrekkjer. Vi viser deg to framhald, du vel det beste, og modellen blir
          justert mot valet ditt – forankra til ein frosen referansemodell (DPO).
        </p>
        <button
          onClick={rlhf.start}
          disabled={rlhf.baseRunning}
          className="rounded-xl bg-violet-600 px-5 py-2.5 text-sm font-semibold text-white shadow-md shadow-violet-200 transition hover:bg-violet-500 disabled:opacity-50"
        >
          Start preferanse-trening
        </button>
      </Card>
    );
  }

  return (
    <Card className="space-y-5">
      {rlhf.untrainedHint && (
        <div className="rounded-xl border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
          Tips: tren modellen først i steg 3 – då blir framhalda meir meiningsfulle.
        </div>
      )}

      <div>
        <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-slate-500">Starttekst</label>
        <textarea
          value={rlhf.prompt}
          onChange={(e) => rlhf.setPrompt(e.target.value)}
          rows={2}
          className="w-full resize-none rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm focus:border-indigo-400 focus:outline-none focus:ring-2 focus:ring-indigo-100"
        />
        <div className="mt-2 flex flex-wrap gap-2">
          {examples.map((ex) => (
            <button
              key={ex}
              onClick={() => rlhf.setPrompt(ex)}
              className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs text-slate-600 transition hover:border-indigo-300 hover:text-indigo-600"
            >
              {ex}
            </button>
          ))}
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-[1fr_auto] sm:items-end">
        <div>
          <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-slate-500">
            Kreativitet: {rlhf.temp.toFixed(2)}
          </label>
          <input
            type="range"
            min={0.3}
            max={1.5}
            step={0.05}
            value={rlhf.temp}
            onChange={(e) => rlhf.setTemp(Number(e.target.value))}
            className="w-full accent-indigo-600"
          />
        </div>
        <button
          onClick={rlhf.generatePair}
          disabled={busy}
          className="rounded-xl border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-700 transition hover:bg-slate-50 disabled:opacity-50"
        >
          {rlhf.generating ? "Lagar par…" : "↻ Generer eit par"}
        </button>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <PrefCard label="A" text={rlhf.pairA?.text ?? ""} onPick={() => rlhf.choose("A")} disabled={busy} />
        <PrefCard label="B" text={rlhf.pairB?.text ?? ""} onPick={() => rlhf.choose("B")} disabled={busy} />
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <button
          onClick={rlhf.skip}
          disabled={busy}
          className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs font-semibold text-slate-600 transition hover:bg-slate-50 disabled:opacity-50"
        >
          Hopp over (likeverdige)
        </button>
        {!rlhf.dpoRunning ? (
          <button
            onClick={rlhf.trainMore}
            disabled={rlhf.baseRunning || rlhf.metrics.count === 0}
            className="rounded-lg bg-indigo-600 px-3 py-1.5 text-xs font-semibold text-white transition hover:bg-indigo-500 disabled:opacity-50"
          >
            Tren meir på preferansane
          </button>
        ) : (
          <button
            onClick={rlhf.stopTrainMore}
            className="rounded-lg bg-rose-600 px-3 py-1.5 text-xs font-semibold text-white transition hover:bg-rose-500"
          >
            ⏸ Stopp
          </button>
        )}
        <button
          onClick={rlhf.resetTuning}
          disabled={busy}
          className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs font-semibold text-slate-600 transition hover:bg-slate-50 disabled:opacity-50"
        >
          ↺ Nullstill justering
        </button>
        <div className="ml-auto flex gap-3 text-xs text-slate-500">
          <span>
            Preferansar: <b className="text-slate-800">{rlhf.metrics.count}</b>
          </span>
          <span>
            Margin: <b className="text-slate-800">{rlhf.metrics.margin.toFixed(3)}</b>
          </span>
          <span>
            Vinnar-rate: <b className="text-slate-800">{(rlhf.metrics.winRate * 100).toFixed(0)}%</b>
          </span>
        </div>
      </div>

      <div>
        <h3 className="mb-2 font-semibold text-slate-900">DPO-tap over tid</h3>
        <LossChart data={rlhf.losses} />
        <p className="mt-2 text-xs text-slate-500">
          Margin = kor mykje meir sannsynleg det valde framhaldet er enn det avviste, samanlikna med
          referansemodellen. Høgare margin og vinnar-rate = modellen følgjer preferansane dine.
        </p>
      </div>
    </Card>
  );
}
```

- [ ] **Step 2: Verify typecheck and build**

Run: `pnpm typecheck && pnpm build`
Expected: both exit 0. (Component is not yet rendered; this only confirms it compiles.)

- [ ] **Step 3: Commit**

```bash
git add src/components/Rlhf.tsx
git commit -m "feat(rlhf): preference-arena UI component

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Wire RLHF into `App.tsx`

**Files:**
- Modify: `src/App.tsx`

**Interfaces:**
- Consumes: `useRlhf` (hook), `Rlhf` (component).
- Produces: a rendered step-5 RLHF section; RLHF state invalidated on engine rebuild and on base-train start; mutual-exclusion between base training and DPO.

- [ ] **Step 1: Add imports**

After `import { Section, Card } from "@/components/ui";` (added in Task 6), add:
```tsx
import Rlhf from "@/components/Rlhf";
import { useRlhf } from "@/lib/useRlhf";
```

- [ ] **Step 2: Instantiate the hook**

In `App`, immediately after the `activeExtraTextRef` declaration and before the `const [running, setRunning] = useState(false);` line, insert:
```tsx
  const rlhf = useRlhf({
    getModel: () => engineRef.current?.model ?? null,
    getTokenizer: () => engineRef.current?.tokenizer ?? null,
    isTrained: () => stepRef.current > 0,
    baseRunning: running,
  });
```
Note: `running` is declared on the next line. Move this hook call to be **after** the `const [running, setRunning] = useState(false);` line instead, so `running` is in scope. Place it directly after the `const [paramCount, setParamCount] = useState(0);` line.

- [ ] **Step 3: Invalidate RLHF when the engine is rebuilt**

In `buildEngine`, add `rlhf.reset();` as the final statement inside the callback body (after `setParamCount(model.paramCount());`), and add `rlhf.reset` to its dependency array:
```tsx
    setParamCount(model.paramCount());
    rlhf.reset();
    // berre arkitektur (preset) tvingar fram ein ny modell – ikkje lr/batch
  }, [preset, rlhf.reset]);
```

- [ ] **Step 4: Invalidate RLHF when base training (re)starts**

In `start`, add `rlhf.reset();` as the first statement, and add `rlhf.reset` to its dependency array:
```tsx
  const start = useCallback(() => {
    rlhf.reset();
    if (!engineRef.current || stepRef.current >= MAX_STEPS) buildEngine();
    runningRef.current = true;
    setRunning(true);
    loop();
  }, [buildEngine, loop, rlhf.reset]);
```

- [ ] **Step 5: Block hyperparameter changes and base-start while DPO runs**

In the three training control inputs, change `disabled={running}` to `disabled={running || rlhf.dpoRunning}` for: the preset `<select>`, the "Minibatch" `<input type="range">`, and the "Læringsrate" `<input type="range">`.

On the base "▶ Start trening" `<button>`, add `disabled={rlhf.dpoRunning}`:
```tsx
                <button
                  onClick={start}
                  disabled={rlhf.dpoRunning}
                  className="rounded-xl bg-indigo-600 px-5 py-2.5 text-sm font-semibold text-white shadow-md shadow-indigo-200 transition hover:bg-indigo-500 disabled:opacity-50"
                >
                  ▶ Start trening
                </button>
```

- [ ] **Step 6: Render the RLHF section**

Find the end of the chat section — the `</Section>` that closes `id="chat"`, immediately before the `{/* Ærlig note */}` comment. Insert this new section between them:
```tsx
        {/* RLHF */}
        <Section
          id="rlhf"
          step={5}
          title="RLHF – lær modellen kva vi føretrekkjer"
          intro="Etter grunntreninga kan vi finjustere modellen med menneskeleg tilbakemelding. Du vel kva for eit av to framhald som er best, og modellen blir dytta mot valet ditt med DPO – forankra til ein frosen kopi av modellen."
        >
          <Rlhf rlhf={rlhf} examples={examples} />
        </Section>
```

- [ ] **Step 7: Renumber the "eigen tekst" section**

On the `<Section id="eigentekst" ...>` element, change `step={5}` to `step={6}`.

- [ ] **Step 8: Verify typecheck and build**

Run: `pnpm typecheck && pnpm build`
Expected: both exit 0.

- [ ] **Step 9: Manual verification**

Run: `pnpm dev` and open the served URL. Confirm:
- Section 5 "RLHF" appears after the chat section; "Legg til eigen tekst" now shows step 6.
- In Section 3, train the model for a few hundred steps and Stop.
- In Section 5, click "Start preferanse-trening" → two continuations A/B appear (no undertrained hint).
- Click "👍 A er betre" / "👍 B er betre" a few times → the DPO-tap chart updates, "Preferansar" count rises, a new pair appears each time.
- Click "Tren meir på preferansane" → chart continues, button toggles to "⏸ Stopp", base "Start trening" is disabled meanwhile.
- Click "↺ Nullstill justering" → buffer/metrics reset, a fresh pair appears.
- Change the preset → the RLHF section returns to its "Start preferanse-trening" state (invalidated).
- Open Section 4 chat after some DPO and confirm generation still works (reflects the aligned model).

- [ ] **Step 10: Run the full test suite once more**

Run: `pnpm test`
Expected: all five tests `PASS`.

- [ ] **Step 11: Commit**

```bash
git add src/App.tsx
git commit -m "feat(rlhf): wire RLHF section into the app

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review (completed during planning)

**Spec coverage:**
- DPO math (`seqLogProb`, `seqLogProbValue`, `dpoLoss`) → Tasks 1–2.
- Frozen reference (`cloneTransformer`) → Task 3.
- Continuation token ids (`sampleTokens` refactor) → Task 4.
- `PrefPair` + `dpoStep` + capping rule (prompt truncated left) → Task 5.
- Live nudge + "train more" chunked loop → Task 7 (`runBurst`, `trainLoop`/`trainMore`).
- Reuse Section-3 model + reference snapshot on start → Task 7 (`start`), Task 9 wiring.
- UI section (step 5), renumber eigen tekst → 6, intro/metrics/chart → Tasks 8–9.
- Invalidation on rebuild + base-start; mutual exclusion → Task 9.
- Reset that reverts DPO but keeps SFT (`resetTuning`) → Task 7.
- Testing mirrors bug-review style (tsc + node) → Tasks 1–5.

**Placeholder scan:** No TBD/TODO; every code/test step contains complete content.

**Type consistency:** `PrefPair { promptIds, chosenIds, rejectedIds }`, `dpoStep(...)→{loss,margin,winRate}`, `sampleTokens(...)→{promptIds,contIds}`, `RlhfPair {text,promptIds,contIds}`, and `choose("A"|"B")` are used identically across `ml.ts`, `useRlhf.ts`, `Rlhf.tsx`, and the smoke test. β=0.1 and DPO lr=1e-4 are consistent in the hook and the smoke test's training optimizer (the smoke test uses lr 1e-3 deliberately to converge faster within 100 steps — noted inline).
