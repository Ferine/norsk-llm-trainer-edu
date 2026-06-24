# Model Inspector ("Se inni modellen") Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a unified "Look inside the model" section that links the model's attention (what a position looks back at) to its next-character probability distribution (what it therefore predicts), driven by one real forward pass.

**Architecture:** Two small pure additions to the autograd core surface state that already exists but is currently discarded — `rowProbs()` (softmax of one logits row) and `Transformer.inspect()` (forward pass that also captures every head's post-softmax attention via an optional sink). A new presentational `Inspector.tsx` component runs `inspect()` once per `(text, model, step)` and renders a clickable input, a layer/head attention heatmap, and sorted probability bars, all linked by a selected query position. A new `<Section>` in `App.tsx` mounts it between Training and Try-the-model.

**Tech Stack:** React 19 + TypeScript 5.9, Tailwind CSS 4, Vite 7 single-file build, pnpm. Tests are plain Node `node:assert` `.mjs` files run against `tsc`-compiled `src/lib/*.ts`. Zero ML dependencies.

## Global Constraints

- **Node ≥ 20.19, pnpm ≥ 10.** Use `pnpm`, never `npm`/`yarn`.
- **Zero ML/runtime dependencies added.** Only React, `clsx`, `tailwind-merge` are allowed at runtime. No charting libraries — draw with SVG/Tailwind divs, matching `LossChart.tsx` / `Architecture.tsx`.
- **Do not alter the training/generation hot path.** All new capture in `ml.ts` is gated behind an optional, defaulted `sink` parameter; when absent, `forward()` behavior is byte-for-byte unchanged.
- **Bilingual parity is enforced.** Every UI string exists in BOTH `bm` and `nn` bundles in `src/lib/i18n.ts` with identical key shape and matching string/function types. The `i18n-parity` test fails otherwise.
- **Single-file build must keep working** (`pnpm build` inlines everything into `dist/index.html`). No dynamic imports, no asset URLs.
- **Character display convention:** space → `␣`, newline → `⏎` (matches `charLabel` in `App.tsx`).

---

## File Structure

- **Modify** `src/lib/ml.ts` — add `rowProbs()`, `AttnView` interface, optional `sink` threaded through `attention()`/`blockForward()`/`forward()`, and the `inspect()` method. (Task 1, Task 2)
- **Create** `test/inspect.test.mjs` — Node tests for `rowProbs` and `inspect`. (Task 1, Task 2)
- **Modify** `package.json` — add the new test file to the `test` script. (Task 1)
- **Modify** `src/lib/i18n.ts` — add the `inspect` block to the `Strings` interface and both `bm`/`nn` bundles. (Task 3)
- **Create** `src/components/Inspector.tsx` — the presentational inspector component. (Task 4)
- **Modify** `src/App.tsx` — mount the new section, renumber later sections. (Task 5)

---

### Task 1: `rowProbs` — softmax of one logits row

**Files:**
- Modify: `src/lib/ml.ts` (add `rowProbs` near the other loss/softmax helpers, e.g. after `crossEntropyLoss`)
- Create: `test/inspect.test.mjs`
- Modify: `package.json` (the `test` script)

**Interfaces:**
- Consumes: existing exports `Transformer`, `mulberry32`, and the `Tensor` shape (`.d: Float32Array`, `.rows`, `.cols`).
- Produces: `export function rowProbs(logits: Tensor, pos: number): Float32Array` — returns a length-`vocab` probability distribution (non-negative, sums to 1) for sequence row `pos`. Throws `RangeError` if `pos` is out of `[0, logits.rows)`.

- [ ] **Step 1: Write the failing test**

Create `test/inspect.test.mjs`:

```js
import { Transformer, mulberry32, rowProbs } from "./dist/ml.js";
import assert from "node:assert/strict";

const cfg = { vocab: 7, dim: 8, nLayer: 2, nHead: 2, seqLen: 6, ffnMult: 2 };
const m = new Transformer(cfg, mulberry32(3));
const ids = [1, 2, 3, 0, 4];
const T = ids.length;
const logits = m.forward(ids);

// --- rowProbs ---
const probs = rowProbs(logits, T - 1);
assert.equal(probs.length, cfg.vocab, "length == vocab");
let sum = 0;
for (const p of probs) {
  assert.ok(p >= 0, "probabilities are non-negative");
  sum += p;
}
assert.ok(Math.abs(sum - 1) < 1e-5, "probabilities sum to 1");

// matches a hand-computed softmax of the same logits row
const off = (T - 1) * cfg.vocab;
let mx = -Infinity;
for (let c = 0; c < cfg.vocab; c++) mx = Math.max(mx, logits.d[off + c]);
let z = 0;
const manual = [];
for (let c = 0; c < cfg.vocab; c++) {
  const e = Math.exp(logits.d[off + c] - mx);
  manual.push(e);
  z += e;
}
for (let c = 0; c < cfg.vocab; c++)
  assert.ok(Math.abs(probs[c] - manual[c] / z) < 1e-6, "matches manual softmax");

// out-of-range row throws
assert.throws(() => rowProbs(logits, T), RangeError, "row >= rows throws");
assert.throws(() => rowProbs(logits, -1), RangeError, "negative row throws");

console.log("rowProbs: PASS");
```

- [ ] **Step 2: Register the test file in `package.json`**

In `package.json`, append ` && node test/inspect.test.mjs` to the end of the `test` script value, so it reads:

```json
"test": "pnpm run test:build && node test/seq-logprob.test.mjs && node test/dpo-loss.test.mjs && node test/clone.test.mjs && node test/corpus-lang.test.mjs && node test/i18n-parity.test.mjs && node test/generate-parity.test.mjs && node test/dpo-smoke.test.mjs && node test/inspect.test.mjs"
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `pnpm run test:build && node test/inspect.test.mjs`
Expected: FAIL — `SyntaxError: ... does not provide an export named 'rowProbs'` (the function does not exist yet).

- [ ] **Step 4: Implement `rowProbs`**

In `src/lib/ml.ts`, add after the `crossEntropyLoss` function:

```ts
// Softmax of a single logits row → a probability distribution over the vocabulary.
// `pos` selects the sequence row; the result has length = vocab. Pure (no autograd).
export function rowProbs(logits: Tensor, pos: number): Float32Array {
  if (!Number.isInteger(pos) || pos < 0 || pos >= logits.rows)
    throw new RangeError(`row ${pos} is outside [0, ${logits.rows})`);
  const V = logits.cols;
  const off = pos * V;
  let mx = -Infinity;
  for (let c = 0; c < V; c++) {
    const v = logits.d[off + c];
    if (v > mx) mx = v;
  }
  let sum = 0;
  const out = new Float32Array(V);
  for (let c = 0; c < V; c++) {
    const e = Math.exp(logits.d[off + c] - mx);
    out[c] = e;
    sum += e;
  }
  for (let c = 0; c < V; c++) out[c] /= sum;
  return out;
}
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `pnpm run test:build && node test/inspect.test.mjs`
Expected: PASS — prints `rowProbs: PASS`.

- [ ] **Step 6: Commit**

```bash
git add src/lib/ml.ts test/inspect.test.mjs package.json
git commit -m "feat(ml): add rowProbs softmax-of-one-row helper"
```

---

### Task 2: `inspect()` — forward pass that captures attention

**Files:**
- Modify: `src/lib/ml.ts` (add `AttnView`; thread optional `sink` through `attention`, `blockForward`, `forward`; add `inspect` method)
- Modify: `test/inspect.test.mjs` (append the `inspect` assertions)

**Interfaces:**
- Consumes: `rowProbs` (Task 1), `Transformer`, `mulberry32`.
- Produces:
  - `export interface AttnView { layer: number; head: number; T: number; weights: Float32Array }` — `weights` is length `T*T`, row-major, row = query position, col = key position; values are post-softmax attention.
  - `forward(ids: number[], sink?: AttnView[]): Tensor` — unchanged behavior when `sink` is omitted; when provided, pushes one `AttnView` per (layer, head).
  - `inspect(ids: number[]): { logits: Tensor; attn: AttnView[] }` — method on `Transformer`; `attn.length === nLayer * nHead`.

- [ ] **Step 1: Write the failing test**

Append to `test/inspect.test.mjs` (before the final nothing — just add at the end of the file):

```js
// --- inspect ---
const { logits: l2, attn } = m.inspect(ids);

// logits identical to a plain forward() (the sink does not change the math)
assert.equal(l2.d.length, logits.d.length, "inspect logits length matches forward");
for (let i = 0; i < l2.d.length; i++)
  assert.ok(Math.abs(l2.d[i] - logits.d[i]) < 1e-6, "inspect logits == forward logits");

// one attention view per (layer, head)
assert.equal(attn.length, cfg.nLayer * cfg.nHead, "one view per (layer, head)");

for (const v of attn) {
  assert.equal(v.T, T, "view.T == sequence length");
  assert.equal(v.weights.length, T * T, "weights length == T*T");
  for (let r = 0; r < T; r++) {
    let rowSum = 0;
    for (let c = 0; c < T; c++) {
      const w = v.weights[r * T + c];
      if (c > r) assert.ok(Math.abs(w) < 1e-7, "causal mask: no attention to the future");
      else rowSum += w;
    }
    assert.ok(Math.abs(rowSum - 1) < 1e-5, "each query row sums to 1 over allowed keys");
  }
}

// every (layer, head) pair is present exactly once
const seen = new Set(attn.map((v) => `${v.layer}:${v.head}`));
assert.equal(seen.size, cfg.nLayer * cfg.nHead, "all (layer, head) pairs distinct");

console.log("inspect: PASS");
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pnpm run test:build && node test/inspect.test.mjs`
Expected: FAIL — `TypeError: m.inspect is not a function`.

- [ ] **Step 3: Add the `AttnView` interface**

In `src/lib/ml.ts`, just above `export class Transformer {`, add:

```ts
// A single head's post-softmax attention matrix, captured for visualization.
// weights is length T*T, row-major: row = query position, col = key position.
export interface AttnView {
  layer: number;
  head: number;
  T: number;
  weights: Float32Array;
}
```

- [ ] **Step 4: Thread the optional sink through `attention`**

In `src/lib/ml.ts`, change the `attention` method signature and capture each head's softmax. Replace the existing method:

```ts
  private attention(blk: Block, x: Tensor): Tensor {
    const q = matmul(x, blk.Wq);
    const k = matmul(x, blk.Wk);
    const v = matmul(x, blk.Wv);
    const hd = this.cfg.dim / this.cfg.nHead;
    const sc = 1 / Math.sqrt(hd);
    const heads: Tensor[] = [];
    for (let h = 0; h < this.cfg.nHead; h++) {
      const qh = sliceCols(q, h * hd, (h + 1) * hd);
      const kh = sliceCols(k, h * hd, (h + 1) * hd);
      const vh = sliceCols(v, h * hd, (h + 1) * hd);
      let scores = matmul(qh, transpose(kh));
      scores = scale(scores, sc);
      scores = causalMask(scores);
      const sm = softmaxRow(scores);
      heads.push(matmul(sm, vh));
    }
    return matmul(concatCols(heads), blk.Wo);
  }
```

with:

```ts
  private attention(blk: Block, x: Tensor, layer = 0, sink?: AttnView[]): Tensor {
    const q = matmul(x, blk.Wq);
    const k = matmul(x, blk.Wk);
    const v = matmul(x, blk.Wv);
    const hd = this.cfg.dim / this.cfg.nHead;
    const sc = 1 / Math.sqrt(hd);
    const heads: Tensor[] = [];
    for (let h = 0; h < this.cfg.nHead; h++) {
      const qh = sliceCols(q, h * hd, (h + 1) * hd);
      const kh = sliceCols(k, h * hd, (h + 1) * hd);
      const vh = sliceCols(v, h * hd, (h + 1) * hd);
      let scores = matmul(qh, transpose(kh));
      scores = scale(scores, sc);
      scores = causalMask(scores);
      const sm = softmaxRow(scores);
      if (sink) sink.push({ layer, head: h, T: sm.rows, weights: sm.d.slice() });
      heads.push(matmul(sm, vh));
    }
    return matmul(concatCols(heads), blk.Wo);
  }
```

- [ ] **Step 5: Thread the sink through `blockForward`**

Replace the existing `blockForward`:

```ts
  private blockForward(blk: Block, x: Tensor): Tensor {
    const a = this.attention(blk, layernorm(x, blk.ln1g, blk.ln1b));
    x = add(x, a);
    const f = this.ffn(blk, layernorm(x, blk.ln2g, blk.ln2b));
    return add(x, f);
  }
```

with:

```ts
  private blockForward(blk: Block, x: Tensor, layer = 0, sink?: AttnView[]): Tensor {
    const a = this.attention(blk, layernorm(x, blk.ln1g, blk.ln1b), layer, sink);
    x = add(x, a);
    const f = this.ffn(blk, layernorm(x, blk.ln2g, blk.ln2b));
    return add(x, f);
  }
```

- [ ] **Step 6: Thread the sink through `forward` and add `inspect`**

Replace the existing `forward` method body's block loop and add `inspect` after `forward`. Change the loop line inside `forward` from:

```ts
    for (const blk of this.blocks) x = this.blockForward(blk, x);
```

to (also add the `sink` parameter to the signature):

```ts
  forward(ids: number[], sink?: AttnView[]): Tensor {
    const Tt = ids.length;
    if (Tt < 1 || Tt > this.seqLen)
      throw new RangeError(`Expected between 1 and ${this.seqLen} token IDs, got ${Tt}`);
    for (const id of ids)
      if (!Number.isInteger(id) || id < 0 || id >= this.vocab)
        throw new RangeError(`Token ID ${id} is outside the vocabulary`);

    const x0 = gatherRows(this.tokEmb, ids);
    const posIdx: number[] = [];
    for (let i = 0; i < Tt; i++) posIdx[i] = i;
    let x = add(x0, gatherRows(this.posEmb, posIdx));
    for (let l = 0; l < this.blocks.length; l++) x = this.blockForward(this.blocks[l], x, l, sink);
    x = layernorm(x, this.lnFg, this.lnFb);
    return matmul(x, this.head);
  }

  // Forward pass that also records every head's post-softmax attention.
  // For visualization only — no backward pass is run on the result.
  inspect(ids: number[]): { logits: Tensor; attn: AttnView[] } {
    const attn: AttnView[] = [];
    const logits = this.forward(ids, attn);
    return { logits, attn };
  }
```

(Only the loop line and the signature change inside `forward`; the validation and embedding lines above are unchanged — shown for context.)

- [ ] **Step 7: Run the new test to verify it passes**

Run: `pnpm run test:build && node test/inspect.test.mjs`
Expected: PASS — prints `rowProbs: PASS` then `inspect: PASS`.

- [ ] **Step 8: Run the full suite to confirm nothing regressed**

Run: `pnpm test`
Expected: every line prints PASS/OK, including `generate-parity`, `dpo-smoke`, and `clone` (which exercise the unchanged `forward()` path).

- [ ] **Step 9: Commit**

```bash
git add src/lib/ml.ts test/inspect.test.mjs
git commit -m "feat(ml): add Transformer.inspect() capturing per-head attention"
```

---

### Task 3: Bilingual `inspect` UI strings

**Files:**
- Modify: `src/lib/i18n.ts` (add `inspect` to the `Strings` interface and to both `bm` and `nn` bundles)

**Interfaces:**
- Produces: `Strings["inspect"]` with keys: `title, intro, inputLabel, clickHint, attnHeading, attnHelp, layerLabel, headLabel, probHeading, probHelp, fasitLabel, fasitNext, correct, wrong, noNext, untrainedHint, notReady`. All are `string` except `fasitNext: (ch: string) => string`. Consumed by `Inspector.tsx` (Task 4) and `App.tsx` (Task 5).
- Note: no `Seeds` change — the inspector reuses the existing `SEEDS[lang].sampleSentence`.

- [ ] **Step 1: Add the `inspect` block to the `Strings` interface**

In `src/lib/i18n.ts`, inside `export interface Strings { ... }`, add this block immediately after the `chat: { ... };` member:

```ts
  inspect: {
    title: string;
    intro: string;
    inputLabel: string;
    clickHint: string;
    attnHeading: string;
    attnHelp: string;
    layerLabel: string;
    headLabel: string;
    probHeading: string;
    probHelp: string;
    fasitLabel: string;
    fasitNext: (ch: string) => string;
    correct: string;
    wrong: string;
    noNext: string;
    untrainedHint: string;
    notReady: string;
  };
```

- [ ] **Step 2: Add the Bokmål `inspect` strings**

In `src/lib/i18n.ts`, inside `const bm: Strings = { ... }`, add immediately after the `chat: { ... },` block:

```ts
  inspect: {
    title: "Se inni modellen",
    intro:
      "Nå har modellen lært litt. La oss se hva som skjer inni den for ett enkelt tegn: hva ser den på, og hva tror den kommer neste?",
    inputLabel: "Tekst å granske",
    clickHint: "Klikk på et tegn under for å velge hvor i teksten du vil se nærmere.",
    attnHeading: "Hva ser modellen på?",
    attnHelp:
      "Hver rad er ett tegn som «ser» bakover. Mørkere rute = mer oppmerksomhet. Det grå feltet er framtiden – den får modellen ikke se.",
    layerLabel: "Lag",
    headLabel: "Hode",
    probHeading: "Hva tror modellen kommer neste?",
    probHelp: "Lengre søyle = mer sikker. Dette er det modellen faktisk gjetter på.",
    fasitLabel: "Fasit:",
    fasitNext: (ch) => `det virkelige neste tegnet er «${ch}».`,
    correct: "✓ modellen gjettet riktig!",
    wrong: "✗ modellen bommet denne gangen.",
    noNext: "Dette er det siste tegnet – det finnes ingen fasit å sammenligne med.",
    untrainedHint:
      "Modellen er ikke trent ennå, så den gjetter nesten tilfeldig. Tren den i steget over og kom tilbake hit for å se forskjellen!",
    notReady: "Modellen er ikke klar ennå …",
  },
```

- [ ] **Step 3: Add the Nynorsk `inspect` strings**

In `src/lib/i18n.ts`, inside `const nn: Strings = { ... }`, add immediately after the `chat: { ... },` block:

```ts
  inspect: {
    title: "Sjå inni modellen",
    intro:
      "No har modellen lært litt. Lat oss sjå kva som skjer inni han for eitt enkelt teikn: kva ser han på, og kva trur han kjem neste?",
    inputLabel: "Tekst å granske",
    clickHint: "Klikk på eit teikn under for å velje kvar i teksten du vil sjå nærare.",
    attnHeading: "Kva ser modellen på?",
    attnHelp:
      "Kvar rad er eitt teikn som «ser» bakover. Mørkare rute = meir merksemd. Det grå feltet er framtida – den får modellen ikkje sjå.",
    layerLabel: "Lag",
    headLabel: "Hovud",
    probHeading: "Kva trur modellen kjem neste?",
    probHelp: "Lengre søyle = meir sikker. Dette er det modellen faktisk gjettar på.",
    fasitLabel: "Fasit:",
    fasitNext: (ch) => `det verkelege neste teiknet er «${ch}».`,
    correct: "✓ modellen gjetta rett!",
    wrong: "✗ modellen bomma denne gongen.",
    noNext: "Dette er det siste teiknet – det finst ingen fasit å samanlikne med.",
    untrainedHint:
      "Modellen er ikkje trena enno, så han gjettar nesten tilfeldig. Tren han i steget over og kom attende hit for å sjå skilnaden!",
    notReady: "Modellen er ikkje klar enno …",
  },
```

- [ ] **Step 4: Verify parity and types**

Run: `pnpm run test:build && node test/i18n-parity.test.mjs`
Expected: PASS — prints `i18n-parity: OK` (both bundles share the same key shape; `fasitNext` is a function in both).

Run: `pnpm typecheck`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add src/lib/i18n.ts
git commit -m "feat(i18n): add bilingual strings for the model inspector"
```

---

### Task 4: `Inspector.tsx` component

**Files:**
- Create: `src/components/Inspector.tsx`

**Interfaces:**
- Consumes: `rowProbs`, `Transformer`, `AttnView` (Task 1/2) from `@/lib/ml`; `Tokenizer` from `@/lib/corpus`; `Strings` from `@/lib/i18n`.
- Produces: `export default function Inspector(props: { model: Transformer | null; tokenizer: Tokenizer | null; step: number; defaultText: string; s: Strings["inspect"] })`. Consumed by `App.tsx` (Task 5).
- No unit test — consistent with the other presentational components (`LossChart`, `Architecture`, `Rlhf`), which have none. Verified by `pnpm typecheck` and `pnpm build`.

- [ ] **Step 1: Create the component file**

Create `src/components/Inspector.tsx`:

```tsx
import { Fragment, useMemo, useState } from "react";
import { rowProbs, type Transformer } from "@/lib/ml";
import type { Tokenizer } from "@/lib/corpus";
import type { Strings } from "@/lib/i18n";

interface Props {
  model: Transformer | null;
  tokenizer: Tokenizer | null;
  step: number;
  defaultText: string;
  s: Strings["inspect"];
}

const MAX_BARS = 12;

function charLabel(itos: string[], id: number): string {
  const c = itos[id] ?? "";
  if (c === " ") return "␣";
  if (c === "\n") return "⏎";
  return c;
}

export default function Inspector({ model, tokenizer, step, defaultText, s }: Props) {
  const [text, setText] = useState(defaultText);
  const [layer, setLayer] = useState(0);
  const [head, setHead] = useState(0);
  const [pos, setPos] = useState<number | null>(null);

  // One forward pass. Recomputes only when the text, the model instance, or the
  // training step changes (step is an intentional dependency so the panels refresh
  // after a training run). Changing layer/head/pos is a pure re-render off this memo.
  const result = useMemo(() => {
    if (!model || !tokenizer) return null;
    let ids = tokenizer.encode(text);
    if (ids.length === 0) ids = [0];
    if (ids.length > model.seqLen) ids = ids.slice(ids.length - model.seqLen);
    const { logits, attn } = model.inspect(ids);
    return { ids, logits, attn };
  }, [text, model, step, tokenizer]);

  if (!model || !tokenizer || !result) {
    return <p className="text-sm text-slate-500">{s.notReady}</p>;
  }

  const itos = tokenizer.itos;
  const T = result.ids.length;
  const sel = Math.min(pos ?? T - 1, T - 1);
  const nLayer = model.cfg.nLayer;
  const nHead = model.cfg.nHead;

  const view =
    result.attn.find((v) => v.layer === layer && v.head === head) ?? result.attn[0];

  const probs = rowProbs(result.logits, sel);
  const ranking = Array.from(probs, (p, id) => ({ id, p })).sort((a, b) => b.p - a.p);
  const top = ranking.slice(0, MAX_BARS);
  const guess = ranking[0]?.id;
  const actualNext = sel + 1 < T ? result.ids[sel + 1] : null;

  const tabBtn = (active: boolean) =>
    `rounded px-2 py-0.5 text-xs font-semibold transition ${
      active ? "bg-indigo-600 text-white" : "bg-slate-100 text-slate-600 hover:bg-slate-200"
    }`;

  return (
    <div className="space-y-6">
      {step === 0 && (
        <p className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-700">
          {s.untrainedHint}
        </p>
      )}

      {/* input */}
      <div>
        <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-slate-500">
          {s.inputLabel}
        </label>
        <input
          value={text}
          onChange={(e) => {
            setText(e.target.value);
            setPos(null);
          }}
          className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm focus:border-indigo-400 focus:outline-none focus:ring-2 focus:ring-indigo-100"
        />
        <p className="mt-2 text-[11px] text-slate-400">{s.clickHint}</p>
        <div className="mt-2 flex flex-wrap gap-1">
          {result.ids.map((id, i) => (
            <button
              key={i}
              onClick={() => setPos(i)}
              className={`rounded px-1.5 py-1 font-mono text-sm transition ${
                i === sel
                  ? "bg-indigo-600 text-white"
                  : "bg-slate-100 text-slate-700 hover:bg-slate-200"
              }`}
            >
              {charLabel(itos, id)}
            </button>
          ))}
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* attention heatmap */}
        <div>
          <h3 className="text-sm font-semibold text-slate-800">{s.attnHeading}</h3>
          <p className="mb-3 text-[11px] text-slate-400">{s.attnHelp}</p>
          <div className="mb-3 flex flex-wrap items-center gap-1">
            <span className="mr-1 text-xs text-slate-500">{s.layerLabel}</span>
            {Array.from({ length: nLayer }, (_, i) => (
              <button key={i} onClick={() => setLayer(i)} className={tabBtn(layer === i)}>
                {i + 1}
              </button>
            ))}
            <span className="ml-3 mr-1 text-xs text-slate-500">{s.headLabel}</span>
            {Array.from({ length: nHead }, (_, i) => (
              <button key={i} onClick={() => setHead(i)} className={tabBtn(head === i)}>
                {i + 1}
              </button>
            ))}
          </div>
          <div className="overflow-x-auto">
            <div
              className="inline-grid gap-0.5"
              style={{ gridTemplateColumns: `auto repeat(${T}, 1.1rem)` }}
            >
              <div />
              {result.ids.map((id, c) => (
                <div key={`h${c}`} className="text-center font-mono text-[10px] text-slate-400">
                  {charLabel(itos, id)}
                </div>
              ))}
              {result.ids.map((rid, r) => (
                <Fragment key={`r${r}`}>
                  <button
                    onClick={() => setPos(r)}
                    className={`pr-1 text-right font-mono text-[10px] ${
                      r === sel ? "font-bold text-indigo-600" : "text-slate-400"
                    }`}
                  >
                    {charLabel(itos, rid)}
                  </button>
                  {result.ids.map((_, c) => {
                    const future = c > r;
                    const w = future ? 0 : view.weights[r * T + c];
                    return (
                      <div
                        key={`c${r}-${c}`}
                        title={`${charLabel(itos, rid)} → ${charLabel(itos, result.ids[c])}: ${(
                          w * 100
                        ).toFixed(0)}%`}
                        className={`h-[1.1rem] w-[1.1rem] rounded-sm ${
                          r === sel ? "ring-1 ring-indigo-400" : ""
                        }`}
                        style={{
                          backgroundColor: future
                            ? "#f1f5f9"
                            : `rgba(79,70,229,${(0.08 + 0.92 * w).toFixed(3)})`,
                        }}
                      />
                    );
                  })}
                </Fragment>
              ))}
            </div>
          </div>
        </div>

        {/* next-character probabilities */}
        <div>
          <h3 className="text-sm font-semibold text-slate-800">{s.probHeading}</h3>
          <p className="mb-3 text-[11px] text-slate-400">{s.probHelp}</p>
          <div className="space-y-1">
            {top.map(({ id, p }, rank) => (
              <div key={id} className="flex items-center gap-2">
                <span className="w-6 text-right font-mono text-xs text-slate-500">
                  {charLabel(itos, id)}
                </span>
                <div className="h-4 flex-1 overflow-hidden rounded bg-slate-100">
                  <div
                    className={`h-4 rounded ${rank === 0 ? "bg-indigo-600" : "bg-indigo-400"}`}
                    style={{ width: `${(p * 100).toFixed(1)}%` }}
                  />
                </div>
                <span className="w-10 text-right text-xs tabular-nums text-slate-500">
                  {(p * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
          <p className="mt-3 text-sm text-slate-600">
            {actualNext === null ? (
              s.noNext
            ) : (
              <>
                {s.fasitLabel} {s.fasitNext(charLabel(itos, actualNext))}{" "}
                {guess === actualNext ? (
                  <span className="font-semibold text-emerald-600">{s.correct}</span>
                ) : (
                  <span className="font-semibold text-rose-500">{s.wrong}</span>
                )}
              </>
            )}
          </p>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Verify it type-checks**

Run: `pnpm typecheck`
Expected: no errors. (If `rowProbs`/`Transformer`/`AttnView` are reported missing, Task 1/2 were not completed.)

- [ ] **Step 3: Verify it builds (single-file)**

Run: `pnpm build`
Expected: build succeeds and writes `dist/index.html`. (The component is not yet mounted; this only confirms it compiles and bundles.)

- [ ] **Step 4: Commit**

```bash
git add src/components/Inspector.tsx
git commit -m "feat(ui): add Inspector component (attention heatmap + next-char bars)"
```

---

### Task 5: Mount the section in `App.tsx` and renumber

**Files:**
- Modify: `src/App.tsx` (import Inspector; insert the new `<Section>`; bump later `step` badges)

**Interfaces:**
- Consumes: `Inspector` (Task 4), `s.inspect` (Task 3), `engineRef.current.model` / `engineRef.current.tokenizer`, the existing `step` state, and `seed.sampleSentence`.

- [ ] **Step 1: Import the Inspector component**

In `src/App.tsx`, add to the imports near `import { Section, Card } from "@/components/ui";`:

```tsx
import Inspector from "@/components/Inspector";
```

- [ ] **Step 2: Insert the new section before the chat section**

In `src/App.tsx`, immediately before the `{/* Chat / generering */}` comment and its `<Section id="chat" step={4} ...>`, insert:

```tsx
        {/* Se inni modellen */}
        <Section
          id="inspect"
          step={4}
          title={s.inspect.title}
          intro={s.inspect.intro}
        >
          <Card>
            <Inspector
              model={engineRef.current?.model ?? null}
              tokenizer={engineRef.current?.tokenizer ?? null}
              step={step}
              defaultText={seed.sampleSentence}
              s={s.inspect}
            />
          </Card>
        </Section>

```

- [ ] **Step 3: Renumber the three later section badges**

In `src/App.tsx`, change the `step` props on the existing sections so the badges stay sequential:
- The chat section: `step={4}` → `step={5}`
- The RLHF section: `step={5}` → `step={6}`
- The "Add your own text" (extra) section: `step={6}` → `step={7}`

(These are the `<Section>` elements at the chat/RLHF/extra blocks — the only remaining `step={4}`, `step={5}`, `step={6}` on `<Section>` tags. Do not touch the `step={0.0001}`, `step={0.05}`, `step={10}` values — those are `<input type="range">` props.)

- [ ] **Step 4: Verify type-check and build**

Run: `pnpm typecheck && pnpm build`
Expected: both succeed; `dist/index.html` is regenerated.

- [ ] **Step 5: Manual smoke test**

Run: `pnpm dev`, open the printed URL, then:
1. Click **Start trening** and let the loss drop for a few seconds, then stop.
2. Scroll to section **4 — Se inni modellen**.
3. Confirm the input shows the default sentence as clickable character chips, the attention heatmap renders with a blank upper triangle, and the probability bars are populated.
4. Click different characters → the highlighted heatmap row and the probability bars + Fasit line update.
5. Switch **Lag**/**Hode** buttons → the heatmap changes.
6. Toggle the language switch (Bokmål ↔ Nynorsk) → all inspector labels translate.

Expected: all of the above behave as described; no console errors.

- [ ] **Step 6: Run the full test suite**

Run: `pnpm test`
Expected: every test passes (including `i18n-parity` and the unchanged `forward()`-path tests).

- [ ] **Step 7: Commit**

```bash
git add src/App.tsx
git commit -m "feat(app): mount Se inni modellen inspector section"
```

---

## Self-Review

**Spec coverage:**
- Probability bars → Task 1 (`rowProbs`) + Task 4 (bars render). ✓
- Attention heatmap with captured weights → Task 2 (`inspect`/`AttnView`) + Task 4 (heatmap). ✓
- Linked interaction (click char → highlight row + drive bars) → Task 4 (`sel`/`pos` state). ✓
- Layer/head selectors → Task 4. ✓
- Causal upper-triangle blank → Task 4 (`future` cells) + Task 2 test (causal mask asserted). ✓
- Fasit ✓/✗ line, `noNext` for last char → Task 4 + Task 3 strings. ✓
- Untrained vs trained (`step === 0` hint; refresh on `step` change) → Task 4 memo dep + hint. ✓
- Placement between Training and Try-the-model, renumber → Task 5. ✓
- Bilingual parity, reuse `sampleSentence` → Task 3 + Task 5. ✓
- Don't touch training hot path → Task 2 (optional defaulted sink) + Task 2 Step 8 regression run. ✓
- Edge cases: empty input → `[0]`; over-length → tail slice; `selectedPos` clamp on shorten (`Math.min(pos ?? T-1, T-1)`); last-char Fasit (`actualNext === null`); single-char input (1×1 grid). All in Task 4. ✓

**Deviation from spec (documented):** the spec mentioned a ~150 ms debounce on text edits; dropped as YAGNI — a forward pass on these tiny models (small `dim`/`seqLen`) is sub-millisecond, so per-keystroke recompute is imperceptible. The probability bars show the top `MAX_BARS = 12` characters rather than the full vocabulary, for readability (argmax always included since the list is sorted).

**Placeholder scan:** none — every code step contains complete code.

**Type consistency:** `rowProbs(logits, pos)`, `inspect(ids): { logits, attn }`, `AttnView { layer, head, T, weights }`, `forward(ids, sink?)` are used identically across Tasks 1, 2, 4. `Strings["inspect"]` keys in Task 3 match every `s.*` reference in Task 4. Props passed in Task 5 (`model`, `tokenizer`, `step`, `defaultText`, `s`) match the `Inspector` `Props` in Task 4.
