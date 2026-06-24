# BPE Tokenizer Demo ("Fra tegn til ord-biter") Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone, interactive "Fra tegn til ord-biter" section that visualizes Byte-Pair Encoding learning its merges live on the app's Norwegian corpus — step/scrub through merges, watch a sample sentence re-tokenize and its token count shrink, and see why the most frequent pair wins.

**Architecture:** A new pure module `src/lib/bpe.ts` implements BPE (`learnBpe`/`encodeWord`/`tokenizeSentence`), fully isolated from the model's character tokenizer. A presentational `BpeLab.tsx` component precomputes the merges once and renders four panels driven by a single "merges applied" count. A new `<Section>` mounts it after the tokenization section. No change to the model, training, or generation.

**Tech Stack:** React 19 + TypeScript 5.9, Tailwind CSS 4, Vite 7 single-file build, pnpm. Tests are plain Node `node:assert` `.mjs` files run against `tsc`-compiled `src/lib/*.ts`. Zero ML/runtime dependencies.

## Global Constraints

- **Node ≥ 20.19, pnpm ≥ 10.** Use `pnpm`, never `npm`/`yarn`.
- **Zero dependencies added.** Only React, `clsx`, `tailwind-merge` at runtime. No tokenizer or chart libraries — draw with Tailwind divs, matching `LossChart.tsx`/`Architecture.tsx`/`Inspector.tsx`.
- **Standalone — do NOT touch the model.** `bpe.ts` must not be imported by `ml.ts`, the training loop, generation, or the inspector. The model keeps using `buildTokenizer` (character-level) from `corpus.ts`.
- **Determinism:** `learnBpe` must be deterministic — among max-count pairs, choose the lexicographically smallest `(a, then b)`. `NUM_MERGES = 80`, with early stop when the best pair count < 2.
- **Bilingual parity is enforced.** Every UI string exists in BOTH `bm` and `nn` bundles in `src/lib/i18n.ts` with identical key shape and matching string/function types. The `i18n-parity` test fails otherwise.
- **Single-file build must keep working** (`pnpm build` inlines everything into `dist/index.html`). No dynamic imports, no asset URLs.

---

## File Structure

- **Create** `src/lib/bpe.ts` — pure BPE algorithm: `BpeMerge`/`BpeResult` types, `learnBpe`, `encodeWord`, `tokenizeSentence`, and private helpers. (Task 1)
- **Create** `test/bpe.test.mjs` — Node tests for the BPE module. (Task 1)
- **Modify** `package.json` — add `src/lib/bpe.ts` to `test:build` and `test/bpe.test.mjs` to the `test` script. (Task 1)
- **Modify** `src/lib/i18n.ts` — add the `bpe` block to the `Strings` interface and both `bm`/`nn` bundles. (Task 2)
- **Create** `src/components/BpeLab.tsx` — the presentational demo component. (Task 3)
- **Modify** `src/App.tsx` — mount the new section after the tokenization section, renumber later sections. (Task 4)

---

### Task 1: `bpe.ts` — the BPE algorithm

**Files:**
- Create: `src/lib/bpe.ts`
- Create: `test/bpe.test.mjs`
- Modify: `package.json` (`test:build` and `test` scripts)

**Interfaces:**
- Produces:
  - `export interface BpeMerge { a: string; b: string; merged: string; count: number; rank: number; rivals: { pair: string; count: number }[] }`
  - `export interface BpeResult { baseVocab: string[]; merges: BpeMerge[] }`
  - `export function learnBpe(text: string, numMerges: number): BpeResult` — deterministic; merges at most `numMerges`; stops early when the best pair count < 2.
  - `export function encodeWord(word: string, merges: BpeMerge[], k: number): string[]` — applies the first `k` merges greedily in rank order.
  - `export function tokenizeSentence(sentence: string, merges: BpeMerge[], k: number): string[]` — splits on whitespace, encodes each word, concatenates.
- Consumed by `BpeLab.tsx` (Task 3): `learnBpe`, `tokenizeSentence`.

- [ ] **Step 1: Write the failing test**

Create `test/bpe.test.mjs`:

```js
import { learnBpe, encodeWord, tokenizeSentence } from "./dist/bpe.js";
import assert from "node:assert/strict";

// most-frequent pair is merged first, with the right count and rank
{
  const { merges } = learnBpe("abab abab cd", 10);
  assert.equal(merges[0].a + merges[0].b, "ab", "most frequent pair 'a'+'b' merged first");
  assert.equal(merges[0].merged, "ab");
  assert.equal(merges[0].count, 4, "'a'-'b' counted 4 (2 per 'abab' x freq 2)");
  assert.equal(merges[0].rank, 0);
}

// deterministic
{
  const a = learnBpe("det er en gang det er to og det", 20);
  const b = learnBpe("det er en gang det er to og det", 20);
  assert.deepEqual(a, b, "learnBpe is deterministic");
}

// vocab grows by exactly one per merge; merged === a+b; ranks sequential
{
  const { baseVocab, merges } = learnBpe("abab abab cd", 10);
  const vocab = new Set(baseVocab);
  merges.forEach((m, i) => {
    assert.equal(m.rank, i, "ranks are sequential");
    assert.equal(m.merged, m.a + m.b, "merged === a + b");
    assert.ok(!vocab.has(m.merged), "merged token is new to the vocabulary");
    vocab.add(m.merged);
  });
  assert.equal(vocab.size, baseVocab.length + merges.length, "vocab grows by 1 per merge");
}

// ties broken lexicographically: 'a'+'b' and 'c'+'d' both occur twice → 'ab' wins
{
  const { merges } = learnBpe("ab ab cd cd", 5);
  assert.equal(merges[0].a + merges[0].b, "ab", "tie broken lexicographically → 'ab'");
}

// encodeWord: k=0 → characters; all merges → fewest tokens; monotonic non-increasing
{
  const { merges } = learnBpe("abab abab cd", 10);
  assert.deepEqual(encodeWord("abab", merges, 0), ["a", "b", "a", "b"], "k=0 → characters");
  assert.deepEqual(encodeWord("abab", merges, merges.length), ["abab"], "all merges → one token");
  let prev = Infinity;
  for (let k = 0; k <= merges.length; k++) {
    const len = encodeWord("abab", merges, k).length;
    assert.ok(len <= prev, "token count is non-increasing in k");
    prev = len;
  }
}

// tokenizeSentence: token count shrinks (or stays equal) as merges accumulate
{
  const text = "det er en gang det er to og det er";
  const { merges } = learnBpe(text, 30);
  const base = tokenizeSentence(text, merges, 0).length;
  const full = tokenizeSentence(text, merges, merges.length).length;
  assert.ok(full <= base, "token count shrinks with merges");
  assert.ok(base > 0, "base tokenization is non-empty");
}

// early stop: no pair occurs >= 2 → no merges
{
  const { merges } = learnBpe("abc", 10);
  assert.equal(merges.length, 0, "no repeated pair → no merges");
}

// empty corpus → empty vocab and no merges
{
  const { merges, baseVocab } = learnBpe("   ", 10);
  assert.equal(merges.length, 0);
  assert.deepEqual(baseVocab, []);
}

console.log("bpe: PASS");
```

- [ ] **Step 2: Register the module and test in `package.json`**

In `package.json`, add `src/lib/bpe.ts` to the `test:build` compile list and append the new test to the `test` script. The two scripts become exactly:

```json
"test:build": "tsc src/lib/ml.ts src/lib/corpus.ts src/lib/i18n.ts src/lib/bpe.ts --rootDir src/lib --outDir test/dist --target ES2020 --module ESNext --moduleResolution bundler --skipLibCheck",
"test": "pnpm run test:build && node test/seq-logprob.test.mjs && node test/dpo-loss.test.mjs && node test/clone.test.mjs && node test/corpus-lang.test.mjs && node test/i18n-parity.test.mjs && node test/generate-parity.test.mjs && node test/dpo-smoke.test.mjs && node test/inspect.test.mjs && node test/bpe.test.mjs"
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `pnpm run test:build && node test/bpe.test.mjs`
Expected: FAIL — `SyntaxError: ... does not provide an export named 'learnBpe'` (the module does not exist yet; `test:build` produces no `bpe.js`).

- [ ] **Step 4: Implement `src/lib/bpe.ts`**

Create `src/lib/bpe.ts`:

```ts
// Byte-Pair Encoding (BPE) — a small, dependency-free implementation for the
// "Fra tegn til ord-biter" teaching section. Pure: no React, no DOM.
// STANDALONE: this does NOT tokenize for the model — the model uses the
// character tokenizer in corpus.ts. This module only powers the BPE demo.

export interface BpeMerge {
  a: string; // left symbol
  b: string; // right symbol
  merged: string; // a + b
  count: number; // how many times the pair occurred when it was chosen
  rank: number; // 0-based order in which this merge was learned
  rivals: { pair: string; count: number }[]; // up to 2 runner-up pairs this step
}

export interface BpeResult {
  baseVocab: string[]; // sorted unique characters across the corpus words
  merges: BpeMerge[]; // in the order they were learned
}

// NUL separates the two symbols of a pair key; corpus text never contains NUL.
const SEP = "\u0000";

// Split text into words on whitespace → a frequency map of words.
function wordFreqs(text: string): Map<string, number> {
  const freqs = new Map<string, number>();
  for (const w of text.split(/\s+/)) {
    if (w.length === 0) continue;
    freqs.set(w, (freqs.get(w) ?? 0) + 1);
  }
  return freqs;
}

// Count every adjacent symbol pair across all words, weighted by word frequency.
function countPairs(words: { syms: string[]; freq: number }[]): Map<string, number> {
  const counts = new Map<string, number>();
  for (const { syms, freq } of words) {
    for (let i = 0; i + 1 < syms.length; i++) {
      const key = syms[i] + SEP + syms[i + 1];
      counts.set(key, (counts.get(key) ?? 0) + freq);
    }
  }
  return counts;
}

// Replace each adjacent (a, b) with a+b, greedily left-to-right.
function mergeSymbols(syms: string[], a: string, b: string): string[] {
  const out: string[] = [];
  let i = 0;
  while (i < syms.length) {
    if (i + 1 < syms.length && syms[i] === a && syms[i + 1] === b) {
      out.push(a + b);
      i += 2;
    } else {
      out.push(syms[i]);
      i += 1;
    }
  }
  return out;
}

// Learn up to numMerges merges from text. Deterministic; stops early when the
// most frequent remaining pair occurs fewer than 2 times.
export function learnBpe(text: string, numMerges: number): BpeResult {
  const words = Array.from(wordFreqs(text), ([w, freq]) => ({ syms: Array.from(w), freq }));

  const vocabSet = new Set<string>();
  for (const { syms } of words) for (const c of syms) vocabSet.add(c);
  const baseVocab = Array.from(vocabSet).sort();

  const merges: BpeMerge[] = [];
  for (let rank = 0; rank < numMerges; rank++) {
    const counts = countPairs(words);
    if (counts.size === 0) break;

    // total order: count desc, then a asc, then b asc → deterministic choice.
    const ranked = Array.from(counts, ([key, count]) => {
      const sep = key.indexOf(SEP);
      return { a: key.slice(0, sep), b: key.slice(sep + 1), count };
    }).sort(
      (x, y) =>
        y.count - x.count ||
        (x.a < y.a ? -1 : x.a > y.a ? 1 : x.b < y.b ? -1 : x.b > y.b ? 1 : 0)
    );

    const best = ranked[0];
    if (best.count < 2) break;

    const rivals = ranked.slice(1, 3).map((r) => ({ pair: r.a + r.b, count: r.count }));
    merges.push({ a: best.a, b: best.b, merged: best.a + best.b, count: best.count, rank, rivals });

    for (const word of words) word.syms = mergeSymbols(word.syms, best.a, best.b);
  }

  return { baseVocab, merges };
}

// Encode one word into subword tokens using the first k merges (greedy, in rank order).
export function encodeWord(word: string, merges: BpeMerge[], k: number): string[] {
  let syms = Array.from(word);
  const n = Math.min(k, merges.length);
  for (let i = 0; i < n; i++) syms = mergeSymbols(syms, merges[i].a, merges[i].b);
  return syms;
}

// Encode a sentence: split on whitespace, encode each word with the first k merges,
// concatenate the per-word subwords into one token list.
export function tokenizeSentence(sentence: string, merges: BpeMerge[], k: number): string[] {
  const out: string[] = [];
  for (const w of sentence.split(/\s+/)) {
    if (w.length === 0) continue;
    for (const tok of encodeWord(w, merges, k)) out.push(tok);
  }
  return out;
}
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `pnpm run test:build && node test/bpe.test.mjs`
Expected: PASS — prints `bpe: PASS`.

- [ ] **Step 6: Run the full suite to confirm nothing regressed**

Run: `pnpm test`
Expected: every test prints PASS/OK, ending with `bpe: PASS`.

- [ ] **Step 7: Commit**

```bash
git add src/lib/bpe.ts test/bpe.test.mjs package.json
git commit -m "feat(bpe): add standalone byte-pair-encoding module + tests"
```

---

### Task 2: Bilingual `bpe` UI strings

**Files:**
- Modify: `src/lib/i18n.ts` (add `bpe` to the `Strings` interface and both `bm`/`nn` bundles, after the `data` block in each)

**Interfaces:**
- Produces `Strings["bpe"]` with keys: `title, intro, mergeBtn, resetBtn, mergeCount, thisMergeHeading, noMergeYet, foundTimes, rivalsLabel, rulesHeading, noRules, sentenceHeading, vocabLine, payoff`. Functions: `mergeCount(k, n)`, `foundTimes(n)`, `sentenceHeading(now, was)`, `vocabLine(base, merges, total)`; all other keys are strings. Consumed by `BpeLab.tsx` (Task 3) and `App.tsx` (Task 4).

- [ ] **Step 1: Add the `bpe` block to the `Strings` interface**

In `src/lib/i18n.ts`, inside `export interface Strings { ... }`, add this block immediately after the `data: { ... };` member:

```ts
  bpe: {
    title: string;
    intro: string;
    mergeBtn: string;
    resetBtn: string;
    mergeCount: (k: number, n: number) => string;
    thisMergeHeading: string;
    noMergeYet: string;
    foundTimes: (n: number) => string;
    rivalsLabel: string;
    rulesHeading: string;
    noRules: string;
    sentenceHeading: (now: number, was: number) => string;
    vocabLine: (base: number, merges: number, total: number) => string;
    payoff: string;
  };
```

- [ ] **Step 2: Add the Bokmål `bpe` strings**

In `src/lib/i18n.ts`, inside `const bm: Strings = { ... }`, add immediately after the `data: { ... },` block:

```ts
  bpe: {
    title: "Fra tegn til ord-biter",
    intro:
      "Ekte språkmodeller bruker ikke enkeltbokstaver. De lærer «ord-biter» (subord) ved å slå sammen de vanligste nabopara igjen og igjen. Prøv selv på den samme teksten.",
    mergeBtn: "Slå sammen neste par",
    resetBtn: "↺ Nullstill",
    mergeCount: (k, n) => `Sammenslåinger: ${k} / ${n}`,
    thisMergeHeading: "Denne sammenslåingen",
    noMergeYet: "Trykk «Slå sammen neste par» for å starte – akkurat nå er hvert tegn sitt eget token.",
    foundTimes: (n) => `funnet ${n} ganger i teksten`,
    rivalsLabel: "konkurrenter:",
    rulesHeading: "Reglene så langt",
    noRules: "Ingen regler ennå.",
    sentenceHeading: (now, was) => `Setningen nå – ${now} token (var ${was})`,
    vocabLine: (base, merges, total) => `Vokabular: ${base} tegn + ${merges} ord-biter = ${total}`,
    payoff:
      "Når en hel ord-bit blir ett token, ser ikke modellen bokstavene inni – derfor bommer språkmodeller på å telle bokstaver i et ord.",
  },
```

- [ ] **Step 3: Add the Nynorsk `bpe` strings**

In `src/lib/i18n.ts`, inside `const nn: Strings = { ... }`, add immediately after the `data: { ... },` block:

```ts
  bpe: {
    title: "Frå teikn til ord-bitar",
    intro:
      "Ekte språkmodellar bruker ikkje enkeltbokstavar. Dei lærer «ord-bitar» (subord) ved å slå saman dei vanlegaste nabopara om att og om att. Prøv sjølv på den same teksten.",
    mergeBtn: "Slå saman neste par",
    resetBtn: "↺ Nullstill",
    mergeCount: (k, n) => `Samanslåingar: ${k} / ${n}`,
    thisMergeHeading: "Denne samanslåinga",
    noMergeYet: "Trykk «Slå saman neste par» for å starte – akkurat no er kvart teikn sitt eige token.",
    foundTimes: (n) => `funne ${n} gonger i teksten`,
    rivalsLabel: "konkurrentar:",
    rulesHeading: "Reglane så langt",
    noRules: "Ingen reglar enno.",
    sentenceHeading: (now, was) => `Setninga no – ${now} token (var ${was})`,
    vocabLine: (base, merges, total) => `Vokabular: ${base} teikn + ${merges} ord-bitar = ${total}`,
    payoff:
      "Når ein heil ord-bit blir eitt token, ser ikkje modellen bokstavane inni – difor bommar språkmodellar på å telje bokstavar i eit ord.",
  },
```

- [ ] **Step 4: Verify parity and types**

Run: `pnpm run test:build && node test/i18n-parity.test.mjs`
Expected: PASS — prints `i18n-parity: OK` (both bundles share the same key shape; the four functions are functions in both).

Run: `pnpm typecheck`
Expected: no errors. (The strings are not consumed yet; the project still type-checks.)

- [ ] **Step 5: Commit**

```bash
git add src/lib/i18n.ts
git commit -m "feat(i18n): add bilingual strings for the BPE tokenizer demo"
```

---

### Task 3: `BpeLab.tsx` component

**Files:**
- Create: `src/components/BpeLab.tsx`

**Interfaces:**
- Consumes: `learnBpe`, `tokenizeSentence` from `@/lib/bpe` (Task 1); `Strings` from `@/lib/i18n` (Task 2's `bpe` block).
- Produces: `export default function BpeLab(props: { corpus: string; sampleSentence: string; s: Strings["bpe"] })`. Consumed by `App.tsx` (Task 4).
- No unit test — consistent with the other presentational components (`LossChart`, `Architecture`, `Inspector`). Verified by `pnpm typecheck` and `pnpm build`.

- [ ] **Step 1: Create the component file**

Create `src/components/BpeLab.tsx`:

```tsx
import { useEffect, useMemo, useState } from "react";
import { learnBpe, tokenizeSentence } from "@/lib/bpe";
import type { Strings } from "@/lib/i18n";

interface Props {
  corpus: string;
  sampleSentence: string;
  s: Strings["bpe"];
}

const NUM_MERGES = 80;

export default function BpeLab({ corpus, sampleSentence, s }: Props) {
  const { baseVocab, merges } = useMemo(() => learnBpe(corpus, NUM_MERGES), [corpus]);
  const [applied, setApplied] = useState(0);

  // reset the demo when the corpus (language) changes
  useEffect(() => {
    setApplied(0);
  }, [corpus]);

  const n = merges.length;
  const k = Math.min(applied, n);

  const current = k > 0 ? merges[k - 1] : null;
  const rules = merges.slice(0, k);
  const vocabSize = baseVocab.length + k;
  const tokens = useMemo(
    () => tokenizeSentence(sampleSentence, merges, k),
    [sampleSentence, merges, k]
  );
  const baseCount = useMemo(
    () => tokenizeSentence(sampleSentence, merges, 0).length,
    [sampleSentence, merges]
  );

  return (
    <div className="space-y-6">
      {/* controls */}
      <div className="space-y-3">
        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={() => setApplied((a) => Math.min(a + 1, n))}
            disabled={k >= n}
            className="rounded-xl bg-indigo-600 px-4 py-2 text-sm font-semibold text-white shadow-md shadow-indigo-200 transition hover:bg-indigo-500 disabled:opacity-50"
          >
            {s.mergeBtn}
          </button>
          <button
            onClick={() => setApplied(0)}
            className="rounded-xl border border-slate-300 px-4 py-2 text-sm font-semibold text-slate-600 transition hover:bg-slate-50"
          >
            {s.resetBtn}
          </button>
          <span className="ml-1 text-sm tabular-nums text-slate-500">{s.mergeCount(k, n)}</span>
        </div>
        <input
          type="range"
          min={0}
          max={n}
          value={k}
          onChange={(e) => setApplied(Number(e.target.value))}
          disabled={n === 0}
          className="w-full accent-indigo-600"
        />
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* this merge */}
        <div>
          <h3 className="mb-2 text-sm font-semibold text-slate-800">{s.thisMergeHeading}</h3>
          {current === null ? (
            <p className="text-sm text-slate-400">{s.noMergeYet}</p>
          ) : (
            <div className="rounded-xl border border-indigo-200 bg-indigo-50 p-4">
              <div className="flex items-center justify-center gap-2 font-mono text-lg">
                <span className="rounded bg-white px-2 py-1 text-slate-700">{current.a}</span>
                <span className="text-slate-400">+</span>
                <span className="rounded bg-white px-2 py-1 text-slate-700">{current.b}</span>
                <span className="text-slate-400">→</span>
                <span className="rounded bg-indigo-600 px-2 py-1 font-semibold text-white">
                  {current.merged}
                </span>
              </div>
              <p className="mt-2 text-center text-sm text-indigo-700">{s.foundTimes(current.count)}</p>
              {current.rivals.length > 0 && (
                <p className="mt-1 text-center text-xs text-slate-500">
                  {s.rivalsLabel} {current.rivals.map((r) => `${r.pair} (${r.count})`).join(", ")}
                </p>
              )}
            </div>
          )}
        </div>

        {/* rules so far */}
        <div>
          <h3 className="mb-2 text-sm font-semibold text-slate-800">{s.rulesHeading}</h3>
          <div className="max-h-44 overflow-auto rounded-xl border border-slate-200 bg-white p-2">
            {rules.length === 0 ? (
              <p className="p-2 text-sm text-slate-400">{s.noRules}</p>
            ) : (
              <ol className="space-y-1">
                {rules.map((m) => (
                  <li
                    key={m.rank}
                    className="flex items-center gap-2 font-mono text-xs text-slate-600"
                  >
                    <span className="w-5 text-right text-slate-400">{m.rank + 1}.</span>
                    <span>
                      {m.a}+{m.b}
                    </span>
                    <span className="text-slate-300">→</span>
                    <span className="font-semibold text-indigo-600">{m.merged}</span>
                    <span className="ml-auto text-slate-400">{m.count}×</span>
                  </li>
                ))}
              </ol>
            )}
          </div>
        </div>
      </div>

      {/* sample sentence re-tokenized */}
      <div>
        <h3 className="mb-2 text-sm font-semibold text-slate-800">
          {s.sentenceHeading(tokens.length, baseCount)}
        </h3>
        <div className="flex flex-wrap gap-1.5">
          {tokens.map((t, i) => (
            <span
              key={i}
              className={`inline-flex items-center rounded-lg border px-2 py-1 font-mono text-sm ${
                t.length > 1
                  ? "border-indigo-300 bg-indigo-100 font-semibold text-indigo-700"
                  : "border-slate-200 bg-slate-50 text-slate-600"
              }`}
            >
              {t}
            </span>
          ))}
        </div>
      </div>

      {/* vocab tally + payoff */}
      <p className="text-sm text-slate-600">{s.vocabLine(baseVocab.length, k, vocabSize)}</p>
      <p className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800">
        💡 {s.payoff}
      </p>
    </div>
  );
}
```

- [ ] **Step 2: Verify it type-checks**

Run: `pnpm typecheck`
Expected: no errors. (If `learnBpe`/`tokenizeSentence` or `Strings["bpe"]` are reported missing, Task 1/2 were not completed.)

- [ ] **Step 3: Verify it builds (single-file)**

Run: `pnpm build`
Expected: build succeeds and writes `dist/index.html`. (The component is not yet mounted; this confirms it compiles and bundles.)

- [ ] **Step 4: Commit**

```bash
git add src/components/BpeLab.tsx
git commit -m "feat(ui): add BpeLab component (live BPE merge visualization)"
```

---

### Task 4: Mount the section in `App.tsx` and renumber

**Files:**
- Modify: `src/App.tsx` (import `BpeLab`; insert the new `<Section>` after the tokenization section; bump later `step` badges)

**Interfaces:**
- Consumes: `BpeLab` (Task 3), `s.bpe` (Task 2), and the existing `activeCorpus` and `sampleSentence` variables in `App.tsx`.

- [ ] **Step 1: Import the BpeLab component**

In `src/App.tsx`, add to the imports near `import Inspector from "@/components/Inspector";`:

```tsx
import BpeLab from "@/components/BpeLab";
```

- [ ] **Step 2: Insert the new section after the tokenization section**

In `src/App.tsx`, find the end of the data/tokenization section — the `</Section>` that is immediately followed by the `{/* Arkitektur */}` comment. Insert this block between that `</Section>` and the `{/* Arkitektur */}` comment:

```tsx
        {/* Fra tegn til ord-biter (BPE) */}
        <Section
          id="bpe"
          step={2}
          title={s.bpe.title}
          intro={s.bpe.intro}
        >
          <Card>
            <BpeLab corpus={activeCorpus} sampleSentence={sampleSentence} s={s.bpe} />
          </Card>
        </Section>

```

- [ ] **Step 3: Renumber the later section badges**

In `src/App.tsx`, change the `step` props on the existing `<Section>` elements so the badges stay sequential after the insertion:
- The architecture section: `step={2}` → `step={3}`
- The training section: `step={3}` → `step={4}`
- The inspector ("Se inni modellen") section: `step={4}` → `step={5}`
- The chat ("Prøv modellen") section: `step={5}` → `step={6}`
- The RLHF section: `step={6}` → `step={7}`
- The "Add your own text" (extra) section: `step={7}` → `step={8}`

Change ONLY the `step={N}` props on `<Section>` elements. Do NOT touch the `<input type="range">` props `step={0.0001}`, `step={0.05}`, `step={10}`. After the change, the `<Section>` badges read 0,1,2,3,4,5,6,7,8 in document order with no duplicates or gaps.

- [ ] **Step 4: Verify type-check and build**

Run: `pnpm typecheck && pnpm build`
Expected: both succeed; `dist/index.html` is regenerated.

- [ ] **Step 5: Verify the badge sequence statically**

Run: `grep -nE '^\s*step=\{[0-9]+\}' src/App.tsx`
Expected: the `<Section>` `step` values, in order, are 0,1,2,3,4,5,6,7,8 (one `step={10}` from a range input also appears — that is the `top-k`/length slider and is correct/untouched).

- [ ] **Step 6: Manual smoke test**

Run `pnpm dev`, open the printed URL, then:
1. Scroll to section **2 — Fra tegn til ord-biter**.
2. Confirm the sample sentence starts as single-character chips and the vocab line shows `… tegn + 0 ord-biter`.
3. Click **Slå sammen neste par** a few times → the "Denne sammenslåingen" panel shows a pair → merged token with a count, the rules list grows, the sentence chips start fusing into highlighted multi-character tokens, and the token count in the heading drops.
4. Drag the slider to the far right → many merges applied, sentence maximally fused.
5. Click **↺ Nullstill** → back to all characters.
6. Toggle the language switch (Bokmål ↔ Nynorsk) → labels translate, the demo resets, and the learned merges differ.

Expected: all behave as described; no console errors.

- [ ] **Step 7: Run the full test suite**

Run: `pnpm test`
Expected: every test passes (including `i18n-parity` and `bpe`).

- [ ] **Step 8: Commit**

```bash
git add src/App.tsx
git commit -m "feat(app): mount Fra tegn til ord-biter BPE section"
```

---

## Self-Review

**Spec coverage:**
- BPE learns merges, most-frequent-pair-first, deterministic, early stop, `NUM_MERGES=80` → Task 1 (`learnBpe`) + Task 3 (`NUM_MERGES`). ✓
- Step + scrubber control → Task 3 (button + slider over `applied`). ✓
- Sample sentence re-tokenizes, token count shrinks → Task 1 (`tokenizeSentence`) + Task 3 (sentence panel) + Task 1 test (monotonic). ✓
- "Denne sammenslåingen" with count + rivals → Task 1 (`rivals`) + Task 3. ✓
- Merge-rule list + vocabulary tally → Task 3. ✓
- Payoff callout → Task 2 (`payoff`) + Task 3. ✓
- New section after §1, renumber to §8 → Task 4. ✓
- Bilingual parity, learned-per-corpus subwords → Task 2 + Task 4 (`corpus={activeCorpus}`). ✓
- Standalone (no model change) → Task 1 (no `ml.ts` import) + Global Constraints. ✓
- Edge cases: empty corpus, no repeated pairs, single-char words, language switch reset, `applied=0` placeholder → Task 1 tests + Task 3 (`useEffect` reset, `Math.min` clamp, `current === null`). ✓

**Placeholder scan:** none — every code step contains complete code.

**Type consistency:** `learnBpe(text, numMerges): BpeResult`, `encodeWord(word, merges, k)`, `tokenizeSentence(sentence, merges, k)`, `BpeMerge { a, b, merged, count, rank, rivals }`, `BpeResult { baseVocab, merges }` are used identically across Tasks 1 and 3. `Strings["bpe"]` keys in Task 2 match every `s.*` reference in Task 3 (`mergeBtn`, `resetBtn`, `mergeCount`, `thisMergeHeading`, `noMergeYet`, `foundTimes`, `rivalsLabel`, `rulesHeading`, `noRules`, `sentenceHeading`, `vocabLine`, `payoff`, plus `title`/`intro` in Task 4). Props passed in Task 4 (`corpus`, `sampleSentence`, `s`) match the `BpeLab` `Props` in Task 3.
