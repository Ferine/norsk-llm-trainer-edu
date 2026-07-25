# Krittavla — usikkerheit som synleg krit — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bind visible "unsettled chalk" distortion on both `.tavle` boards to real numbers from the model — training loss drives whole-line focus in §5 Trening, per-character next-token confidence drives per-letter smudge in §7 Chat.

**Architecture:** `sampleTokens` in `ml.ts` gains an additive `conf: Float32Array` field (one extra O(V) softmax per step, V ≈ 50). A pure-math module `chalk.ts` converts loss and confidence into focus/smudge scalars. A new `<Tavle>` component wraps the existing chalkboard markup and renders per-character spans styled by those scalars (tier 1, CSS, works everywhere). Tier 2 overlays a WebGL2 canvas that samples the live DOM via the experimental html-in-canvas API and displaces it along a noise field, giving cross-character chalk bleed that CSS cannot produce. Tier 2 is strictly additive over tier 1 and never load-bearing for meaning.

**Tech Stack:** TypeScript 5.9, React 19.2, Vite 7, Tailwind 4 (`@theme` tokens in `src/index.css`), WebGL2, Chrome html-in-canvas origin trial. Tests are plain `node:assert` `.mjs` files run against `tsc`-compiled output in `test/dist`.

**Spec:** `docs/superpowers/specs/2026-07-25-krittavle-canvas-design.md`

## Global Constraints

- **Package manager is pnpm** (`pnpm@10.28.2`, Node ≥ 20.19). Never run `npm` or `yarn`.
- **No new dependencies.** The production build inlines to a single self-contained HTML file via `vite-plugin-singlefile`; there are no asset URLs, no CDN, no network calls, ever. Shader source is inlined as a TypeScript string.
- **Every effect must be a gauge, not a garnish.** Its amplitude must be bound to a real number from the model. This is the app's stated contract — see `src/components/Skruer.tsx:8`: *"Alt er ekte verdiar frå modellen, ikkje ein animasjon."*
- **`test/generate-parity.test.mjs` must pass unmodified.** It is the proof the `ml.ts` change was purely additive. Do not edit that file in any task.
- **All UI strings are bilingual.** Every new string needs a bokmål (`bm`) and nynorsk (`nn`) entry in `src/lib/i18n.ts`. `test/i18n-parity.test.mjs` enforces identical key shapes.
- **Copy rule: plain words first, technical term in parentheses if at all.** Write *"uklare bokstaver = modellen var usikker"*, never *"entropi"*. Never write "transformator" for "transformer".
- **Colour semantics are fixed** (see `src/index.css:69` `@theme`): `blekk` = content, `rettepenn` = evaluation only, `tusj` = selection/emphasis, `tavle`/`kritt` = chalkboard where the model's own output appears. This feature lives entirely inside `tavle`/`kritt`. Introduce no new colours.
- **`prefers-reduced-motion: reduce` must be honoured.** Smudge amount is information and still renders; animated dust drift is decoration and must stop. Existing precedent: `src/App.tsx:56`.
- **Existing sections other than §5 and §7 must not change.** §8 RLHF and any other `.tavle` keep their current rendering.
- Verification commands for every task: `pnpm test` and `pnpm typecheck`, both green.

## File Structure

**Create:**
- `src/lib/chalk.ts` — pure math + capability detection. No React, no DOM mutation. Compiled into `test/dist` and unit-tested directly.
- `src/components/Tavle.tsx` — the chalkboard component. Owns tier detection, per-character spans, and (from Task 5) the WebGL overlay.
- `src/components/tavle.glsl.ts` — vertex + fragment shader source as exported template strings.
- `test/conf.test.mjs` — asserts the `ml.ts` confidence output.
- `test/chalk.test.mjs` — asserts the pure math in `chalk.ts`.

**Modify:**
- `src/lib/ml.ts:841-919` — `sampleTokens` gains `conf`; new `generateDetailed`; `generate` becomes a wrapper.
- `src/lib/i18n.ts` — four new strings × two languages, plus interface entries.
- `src/App.tsx` — §5 and §7 chalkboard markup replaced by `<Tavle>`; `runGenerate` stores confidence.
- `package.json` — `test:build` compiles `chalk.ts`; `test` runs the two new test files.
- `index.html` — origin trial `<meta>` (Task 5 only).

**Rationale for the split:** `chalk.ts` holds all the tunable math so it can be unit-tested in Node without a DOM, mirroring how `ml.ts` is already tested. `Tavle.tsx` holds all the rendering. The shader lives in its own file so `Tavle.tsx` stays readable — the same reason `Skruer.tsx` is separate from `App.tsx`.

---

### Task 1: Per-character confidence in `ml.ts`

The model already computes a distribution at every sampling step and throws it away. This task keeps it.

**Files:**
- Modify: `src/lib/ml.ts:841-919`
- Modify: `package.json` (test scripts)
- Test: `test/conf.test.mjs` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `sampleTokens(...)` return type gains a field → `{ promptIds: number[]; contIds: number[]; conf: Float32Array }`
  - `generateDetailed(model, decode, encode, prompt, opts, rng) => { text: string; promptLen: number; conf: Float32Array }`
  - `generate(...) => string` — signature and behaviour unchanged.

- [ ] **Step 1: Write the failing test**

Create `test/conf.test.mjs`:

```js
import { Transformer, generate, generateDetailed, sampleTokens, mulberry32 } from "./dist/ml.js";
import { buildTokenizer, corpus } from "./dist/corpus.js";
import assert from "node:assert/strict";

const tok = buildTokenizer(corpus);
const cfg = { vocab: tok.vocab, dim: 16, nLayer: 1, nHead: 2, seqLen: 16, ffnMult: 2 };
const m = new Transformer(cfg, mulberry32(5));
const prompt = "Det var";

// sampling-vegen: conf finst, og er ein gyldig sannsyn per valt teikn
const os = { temperature: 0.9, topK: 5, length: 12 };
const st = sampleTokens(m, tok.encode, prompt, os, mulberry32(9));
assert.equal(st.conf.length, st.contIds.length, "conf must be one value per generated token");
assert.equal(st.conf.length, 12);
for (const p of st.conf) {
  assert.ok(p > 0 && p <= 1, `confidence out of range: ${p}`);
}

// grådig-vegen (temperatur 0) må òg fylle conf – slideren når 0
const og = { temperature: 0, topK: 5, length: 10 };
const gd = sampleTokens(m, tok.encode, prompt, og, mulberry32(1));
assert.equal(gd.conf.length, 10);
for (const p of gd.conf) {
  assert.ok(p > 0 && p <= 1, `greedy confidence out of range: ${p}`);
}

// grådig vel alltid det mest sannsynlege teiknet: conf må vere maksimum,
// altså minst 1/V for eit kvart ordforråd
for (const p of gd.conf) {
  assert.ok(p >= 1 / tok.vocab, "greedy pick must be the argmax of the distribution");
}

// conf er uavhengig av temperatur og top-k: same modell, same frø, same
// valde teikn => same sikkerheit. Grådig med topK 5 og topK 50 er identisk.
const a = sampleTokens(m, tok.encode, prompt, { temperature: 0, topK: 5, length: 8 }, mulberry32(3));
const b = sampleTokens(m, tok.encode, prompt, { temperature: 0, topK: 50, length: 8 }, mulberry32(3));
assert.deepEqual(Array.from(a.contIds), Array.from(b.contIds));
for (let i = 0; i < a.conf.length; i++) {
  assert.ok(Math.abs(a.conf[i] - b.conf[i]) < 1e-6, "conf must not depend on top-k");
}

// generateDetailed er nøyaktig generate, pluss tala
const d = generateDetailed(m, tok.decode, tok.encode, prompt, og, mulberry32(1));
const g = generate(m, tok.decode, tok.encode, prompt, og, mulberry32(1));
assert.equal(d.text, g);
assert.equal(d.promptLen, prompt.length);
assert.equal(d.conf.length, 10);
// teikn-nivå tokenisering: framhaldet har eitt teikn per conf-verdi
assert.equal(d.text.length - d.promptLen, d.conf.length);

console.log("conf: PASS");
```

- [ ] **Step 2: Register the test so it runs**

In `package.json`, append `test/conf.test.mjs` to the `test` script. Change:

```json
"test": "pnpm run test:build && node test/seq-logprob.test.mjs && node test/dpo-loss.test.mjs && node test/clone.test.mjs && node test/corpus-lang.test.mjs && node test/i18n-parity.test.mjs && node test/generate-parity.test.mjs && node test/dpo-smoke.test.mjs && node test/inspect.test.mjs && node test/bpe.test.mjs"
```

to:

```json
"test": "pnpm run test:build && node test/seq-logprob.test.mjs && node test/dpo-loss.test.mjs && node test/clone.test.mjs && node test/corpus-lang.test.mjs && node test/i18n-parity.test.mjs && node test/generate-parity.test.mjs && node test/dpo-smoke.test.mjs && node test/inspect.test.mjs && node test/bpe.test.mjs && node test/conf.test.mjs"
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `pnpm test`

Expected: FAIL. The `conf` test errors with something like `TypeError: Cannot read properties of undefined (reading 'length')` (on `st.conf.length`) or `SyntaxError: The requested module './dist/ml.js' does not provide an export named 'generateDetailed'`. Every test before it must still pass.

- [ ] **Step 4: Add the confidence helper to `ml.ts`**

In `src/lib/ml.ts`, immediately above `export function sampleTokens` (currently line 841, below the `SampleOpts` interface), insert:

```ts
// Kor sikker var modellen på teiknet han valde? Full softmax over heile
// ordforrådet ved temperatur 1 – modellens eigen tru, uavhengig av kva
// temperatur og top-k brukaren har skrudd på. Difor rører ikkje
// temperatur-slideren dette talet: temperatur gjev ikkje ny kunnskap, han
// gjer berre trekkinga meir vågal, og då ser du modellen plukke teikn han
// sjølv trur lite på. Kostar O(V) per steg, V ≈ 50 – forsvinnande lite mot
// eit framoversteg.
function chosenProb(logits: Float32Array, off: number, V: number, chosen: number): number {
  let mx = -Infinity;
  for (let c = 0; c < V; c++) {
    const v = logits[off + c];
    if (v > mx) mx = v;
  }
  let sum = 0;
  for (let c = 0; c < V; c++) sum += Math.exp(logits[off + c] - mx);
  // sum >= 1 alltid (maksleddet er exp(0)), så ingen deling på null
  return Math.exp(logits[off + chosen] - mx) / sum;
}
```

- [ ] **Step 5: Populate `conf` in both sampling branches**

In `src/lib/ml.ts`, change the signature and body of `sampleTokens`.

Change the return type on line 847 from:

```ts
): { promptIds: number[]; contIds: number[] } {
```

to:

```ts
): { promptIds: number[]; contIds: number[]; conf: Float32Array } {
```

After `const contIds: number[] = [];` (line 851) add:

```ts
  const conf = new Float32Array(opts.length);
```

In the greedy branch, replace:

```ts
      ctx.push(best);
      contIds.push(best);
      continue;
```

with:

```ts
      conf[step] = chosenProb(logits.d, off, V, best);
      ctx.push(best);
      contIds.push(best);
      continue;
```

At the end of the sampling branch, replace:

```ts
    ctx.push(chosen);
    contIds.push(chosen);
  }
  return { promptIds, contIds };
}
```

with:

```ts
    conf[step] = chosenProb(logits.d, off, V, chosen);
    ctx.push(chosen);
    contIds.push(chosen);
  }
  return { promptIds, contIds, conf };
}
```

- [ ] **Step 6: Add `generateDetailed` and rewrite `generate` as a wrapper**

Replace the whole of `generate` (currently `src/lib/ml.ts:908-919`) with:

```ts
// Generer tekst med tal på: teksten, kvar starteksten sluttar, og kor sikker
// modellen var på kvart teikn han sjølv skreiv. Starteksten har ingen
// sikkerheit – han vart gjeven, ikkje gjetta.
export function generateDetailed(
  model: Transformer,
  decode: (ids: number[]) => string,
  encode: (s: string) => number[],
  prompt: string,
  opts: SampleOpts,
  rng: () => number
): { text: string; promptLen: number; conf: Float32Array } {
  const { contIds, conf } = sampleTokens(model, encode, prompt, opts, rng);
  return { text: prompt + decode(contIds), promptLen: prompt.length, conf };
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
  return generateDetailed(model, decode, encode, prompt, opts, rng).text;
}
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `pnpm test`

Expected: PASS for all ten tests, including `generate-parity: PASS` (unmodified) and the new `conf: PASS`.

Then run: `pnpm typecheck`

Expected: no output, exit 0.

- [ ] **Step 8: Commit**

```bash
git add src/lib/ml.ts test/conf.test.mjs package.json
git commit -m "feat(ml): expose per-character confidence from sampleTokens"
```

---

### Task 2: `chalk.ts` — the gauge math

All tunable constants live here, unit-tested in Node, so tuning never requires a browser.

**Files:**
- Create: `src/lib/chalk.ts`
- Modify: `package.json` (test scripts)
- Test: `test/chalk.test.mjs` (create)

**Interfaces:**
- Consumes: nothing from Task 1 (pure math, no model types).
- Produces:
  - `lossToFocus(loss: number, vocab: number) => number` — 0 (random guessing) … 1 (learned)
  - `confToSmudge(conf: number) => number` — 0 (crisp) … 1 (fully smudged)
  - `meanConf(conf: Float32Array) => number`
  - `blurPx(smudge: number) => number`
  - `chalkOpacity(smudge: number) => number`
  - `supportsElementTexture() => boolean`
  - `forcedTier() => 1 | 2 | null`

- [ ] **Step 1: Write the failing test**

Create `test/chalk.test.mjs`:

```js
import {
  lossToFocus,
  confToSmudge,
  meanConf,
  blurPx,
  chalkOpacity,
  supportsElementTexture,
  forcedTier,
} from "./dist/chalk.js";
import assert from "node:assert/strict";

const V = 50;
const RANDOM = Math.log(V); // ~3.912 – tapet ved rein gjetting

// nullpunktet er rein gjetting, nøyaktig
assert.ok(Math.abs(lossToFocus(RANDOM, V)) < 1e-9, "ln(V) must map to exactly 0 focus");

// monotont fallande i tap
let prev = -1;
for (const loss of [RANDOM, 3.0, 2.5, 2.0, 1.5, 1.3]) {
  const f = lossToFocus(loss, V);
  assert.ok(f > prev, `focus must increase as loss falls (loss=${loss})`);
  prev = f;
}

// klemt til [0, 1] i begge endar
assert.equal(lossToFocus(99, V), 0, "loss worse than random clamps to 0");
assert.equal(lossToFocus(0.1, V), 1, "loss below the floor clamps to 1");
for (const loss of [0.1, 1.0, 2.0, 4.0, 99]) {
  const f = lossToFocus(loss, V);
  assert.ok(f >= 0 && f <= 1, `focus out of range: ${f}`);
}

// smudge: sikker => skarpt, usikker => uklart, monotont
assert.equal(confToSmudge(1), 0, "full confidence is perfectly crisp");
assert.ok(confToSmudge(0.01) > 0.8, "near-zero confidence is heavily smudged");
assert.ok(confToSmudge(0.9) < confToSmudge(0.3), "smudge must fall as confidence rises");
for (const p of [0, 0.25, 0.5, 0.75, 1]) {
  const s = confToSmudge(p);
  assert.ok(s >= 0 && s <= 1, `smudge out of range: ${s}`);
}
// robust mot søppel-input
assert.equal(confToSmudge(-1), 1);
assert.equal(confToSmudge(2), 0);

// snittsikkerheit
assert.ok(Math.abs(meanConf(new Float32Array([0.5, 0.5, 0.5])) - 0.5) < 1e-6);
assert.ok(Math.abs(meanConf(new Float32Array([0.2, 0.8])) - 0.5) < 1e-6);
assert.equal(meanConf(new Float32Array([])), 0, "empty conf must not be NaN");

// css-avbildingar er endelege og monotone
assert.equal(blurPx(0), 0);
assert.ok(blurPx(1) > blurPx(0.5) && blurPx(0.5) > 0);
assert.equal(chalkOpacity(0), 1);
assert.ok(chalkOpacity(1) < chalkOpacity(0.5) && chalkOpacity(1) > 0);

// nettlesar-deteksjon må ikkje krasje i Node
assert.equal(supportsElementTexture(), false, "no WebGL2 in Node");
assert.equal(forcedTier(), null, "no location in Node");

console.log("chalk: PASS");
```

- [ ] **Step 2: Register the module and test**

In `package.json`, add `src/lib/chalk.ts` to `test:build`. Change:

```json
"test:build": "tsc src/lib/ml.ts src/lib/corpus.ts src/lib/i18n.ts src/lib/bpe.ts --rootDir src/lib --outDir test/dist --target ES2020 --module ESNext --moduleResolution bundler --skipLibCheck"
```

to:

```json
"test:build": "tsc src/lib/ml.ts src/lib/corpus.ts src/lib/i18n.ts src/lib/bpe.ts src/lib/chalk.ts --rootDir src/lib --outDir test/dist --target ES2020 --module ESNext --moduleResolution bundler --skipLibCheck"
```

And append `&& node test/chalk.test.mjs` to the end of the `test` script (after `node test/conf.test.mjs`).

- [ ] **Step 3: Run the test to verify it fails**

Run: `pnpm test`

Expected: FAIL with `Cannot find module ... test/dist/chalk.js` — the module does not exist yet. All earlier tests still pass.

- [ ] **Step 4: Write `src/lib/chalk.ts`**

```ts
// Krittmatematikken. Alt som kan justerast bur her, slik at det kan
// finjusterast og testast i Node utan ein nettlesar – same mønster som
// SHARPEN i Skruer.tsx.
//
// Biletspråket er felles for begge tavlene: krit som ikkje har sett seg.
// Usikker modell => uklare, bleike bokstavar. Sikker modell => skarpt krit.

// Golvet er målt i praksis: preset «liten» på det norske korpuset legg seg
// rundt 1,25 i tap etter 3500 steg. Under golvet er biletet heilt skarpt.
const LOSS_FLOOR = 1.25;

// Maksimal uskarpleik i piksler. Over ~1,8px blir monospace-teikn uleselege
// heilt, og då er det ikkje lenger ei måling, berre grøt.
const MAX_BLUR_PX = 1.6;

// Kor bleikt det svakaste kritet blir. Under ~0,4 forsvinn teksten på tavla.
const MIN_OPACITY = 0.45;

// Tap => skarpleik. Nullpunktet er rein gjetting: eit tap på ln(V) tyder at
// modellen er like sikker på alle teikn i ordforrådet, altså at han ikkje
// veit noko som helst. Det er den ærlege botnen å måle frå.
export function lossToFocus(loss: number, vocab: number): number {
  const ceil = Math.log(Math.max(2, vocab));
  if (!Number.isFinite(loss) || loss <= 0) return 1;
  const span = ceil - LOSS_FLOOR;
  if (span <= 0) return 1;
  const t = (ceil - loss) / span;
  return Math.min(1, Math.max(0, t));
}

// Sikkerheit => uklarleik. Kvadratrota gjer at skilnaden mellom «ganske
// sikker» og «heilt sikker» ikkje får dominere biletet; det interessante
// skjer i den låge enden.
export function confToSmudge(conf: number): number {
  if (!Number.isFinite(conf)) return 1;
  const p = Math.min(1, Math.max(0, conf));
  return 1 - Math.sqrt(p);
}

export function meanConf(conf: Float32Array): number {
  if (conf.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < conf.length; i++) sum += conf[i];
  return sum / conf.length;
}

export function blurPx(smudge: number): number {
  const s = Math.min(1, Math.max(0, smudge));
  return s * MAX_BLUR_PX;
}

export function chalkOpacity(smudge: number): number {
  const s = Math.min(1, Math.max(0, smudge));
  return 1 - s * (1 - MIN_OPACITY);
}

// Tier 2 krev WebGL2 *og* den eksperimentelle html-in-canvas-utvidinga.
// Utan begge fell vi til tier 1, som er reint CSS og verkar overalt.
export function supportsElementTexture(): boolean {
  const proto = (globalThis as { WebGL2RenderingContext?: { prototype: unknown } })
    .WebGL2RenderingContext?.prototype as Record<string, unknown> | undefined;
  return typeof proto?.texElementImage2D === "function";
}

// ?tier=1 eller ?tier=2 tvingar eit nivå. Brukt til å måle kva tier 2
// faktisk kostar treninga – sjå målesteget i oppgåve 5.
export function forcedTier(): 1 | 2 | null {
  if (typeof location === "undefined") return null;
  const v = new URLSearchParams(location.search).get("tier");
  if (v === "1") return 1;
  if (v === "2") return 2;
  return null;
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pnpm test`

Expected: PASS for all eleven tests, ending with `chalk: PASS`.

Then run: `pnpm typecheck`

Expected: no output, exit 0.

- [ ] **Step 6: Commit**

```bash
git add src/lib/chalk.ts test/chalk.test.mjs package.json
git commit -m "feat(chalk): gauge math for loss-focus and confidence-smudge"
```

---

### Task 3: Bilingual strings

**Files:**
- Modify: `src/lib/i18n.ts` (interface at lines 83-121, `bm` at ~287-330, `nn` at ~510-553)
- Test: `test/i18n-parity.test.mjs` (existing, do not edit)

**Interfaces:**
- Consumes: nothing.
- Produces, on `Strings["train"]`:
  - `focusLegend: string`
  - `focusSummary: (pct: number) => string`
- Produces, on `Strings["chat"]`:
  - `confLegend: string`
  - `confSummary: (pct: number) => string`

- [ ] **Step 1: Run the parity test to confirm the current green baseline**

Run: `pnpm test`

Expected: PASS, including `i18n-parity: OK`. This is the baseline — the point of this task is that it stays green after strings are added to both languages.

- [ ] **Step 2: Add the interface entries**

In `src/lib/i18n.ts`, in the `train` block of `export interface Strings` (ends at line 106 with `livePlaceholder: string;`), add after `livePlaceholder: string;`:

```ts
    focusLegend: string;
    focusSummary: (pct: number) => string;
```

In the `chat` block (ends at line 121 with `answerLabel: string;`), add after `answerLabel: string;`:

```ts
    confLegend: string;
    confSummary: (pct: number) => string;
```

- [ ] **Step 3: Add the bokmål strings**

In the `bm` object, in `train`, after `livePlaceholder: "Trykk «Start trening» for å se eksempler underveis…",` add:

```ts
    focusLegend:
      "Teksten står uklart så lenge modellen gjetter, og blir skarpere etter hvert som den lærer. Uskarpheten er ikke pynt – den er tapet fra grafen over, tegnet om til krittstøv.",
    focusSummary: (pct) => `Skarphet: ${pct} % av veien fra ren gjetting til ferdig lært.`,
```

In the `bm` object, in `chat`, after `answerLabel: "Svar fra modellen",` add:

```ts
    confLegend:
      "Uklare bokstaver = modellen var usikker på akkurat det tegnet. Starteksten din står alltid skarpt – den ble gitt, ikke gjettet. Skru temperaturen opp, så ser du modellen velge tegn den selv tror lite på.",
    confSummary: (pct) => `I snitt var modellen ${pct} % sikker på tegnene den skrev.`,
```

- [ ] **Step 4: Add the nynorsk strings**

In the `nn` object, in `train`, after `livePlaceholder: "Trykk «Start trening» for å sjå døme undervegs…",` add:

```ts
    focusLegend:
      "Teksten står uklart så lenge modellen gjettar, og blir skarpare etter kvart som han lærer. Uskarpleiken er ikkje pynt – han er tapet frå grafen over, teikna om til krittstøv.",
    focusSummary: (pct) => `Skarpleik: ${pct} % av vegen frå rein gjetting til ferdig lært.`,
```

In the `nn` object, in `chat`, after `answerLabel: "Svar frå modellen",` add:

```ts
    confLegend:
      "Uklare bokstavar = modellen var usikker på akkurat det teiknet. Starteksten din står alltid skarpt – han vart gjeven, ikkje gjetta. Skru temperaturen opp, så ser du modellen velje teikn han sjølv trur lite på.",
    confSummary: (pct) => `I snitt var modellen ${pct} % sikker på teikna han skreiv.`,
```

- [ ] **Step 5: Run the tests to verify parity holds**

Run: `pnpm test`

Expected: PASS, `i18n-parity: OK`. If it fails with `STRINGS bm/nn key shapes must match`, a string was added to one language but not the other — the assertion message names the mismatching shape.

Then run: `pnpm typecheck`

Expected: no output, exit 0. A missing interface entry surfaces here as `Object literal may only specify known properties`.

- [ ] **Step 6: Commit**

```bash
git add src/lib/i18n.ts
git commit -m "i18n: strings for the chalkboard uncertainty gauges"
```

---

### Task 4: `<Tavle>` — tier 1, shipping everywhere

This task delivers the complete feature. Tier 2 is polish on top; if Task 5 were never done, this would still be worth shipping.

**Files:**
- Create: `src/components/Tavle.tsx`
- Modify: `src/App.tsx` (§5 board at 802-809, §7 board at 921-930, `runGenerate` at 353-378, typing effect at 380-391)

**Interfaces:**
- Consumes: `lossToFocus`, `confToSmudge`, `meanConf`, `blurPx`, `chalkOpacity` from `src/lib/chalk.ts` (Task 2); `generateDetailed` from `src/lib/ml.ts` (Task 1); `focusLegend`/`focusSummary`/`confLegend`/`confSummary` from `src/lib/i18n.ts` (Task 3).
- Produces:
  - `type Gauge = { kind: "loss"; value: number; vocab: number } | { kind: "conf"; conf: Float32Array; promptLen: number }`
  - `<Tavle label text placeholder legend summary gauge? className? children? />` — default export from `src/components/Tavle.tsx`

- [ ] **Step 1: Write `src/components/Tavle.tsx`**

```tsx
import { useMemo } from "react";
import type { ReactNode } from "react";
import { cn } from "@/utils/cn";
import { blurPx, chalkOpacity, confToSmudge, lossToFocus } from "@/lib/chalk";

// Tavla der modellen skriv. Teksten kan teiknast med «krit som ikkje har
// sett seg»: kor uklart eit teikn står, er bunde til eit ekte tal frå
// modellen – anten tapet (heile linja) eller sikkerheita per teikn.
// Utan `gauge` oppfører komponenten seg nøyaktig som tavla gjorde før.

export type Gauge =
  | { kind: "loss"; value: number; vocab: number }
  | { kind: "conf"; conf: Float32Array; promptLen: number };

interface Props {
  label: string;
  text: string;
  placeholder: string;
  legend: string;
  summary: string;
  gauge?: Gauge;
  className?: string;
  children?: ReactNode;
}

export default function Tavle({
  label,
  text,
  placeholder,
  legend,
  summary,
  gauge,
  className,
  children,
}: Props) {
  // Per-teikn-utsnitt lagar vi berre når vi faktisk måler per teikn.
  // Starteksten får ingen uskarpleik: han vart gjeven, ikkje gjetta.
  const spans = useMemo(() => {
    if (!gauge || gauge.kind !== "conf" || !text) return null;
    return Array.from(text).map((ch, i) => {
      const j = i - gauge.promptLen;
      if (j < 0) return { ch, smudge: 0 };
      // conf[j] kan mangle om teksten er kutta midt i skrivinga
      const p = gauge.conf[j];
      return { ch, smudge: p === undefined ? 0 : confToSmudge(p) };
    });
  }, [gauge, text]);

  // Tap-måleren gjeld heile linja under eitt – tapet *er* ein global skalar.
  const lineSmudge =
    gauge?.kind === "loss" ? 1 - lossToFocus(gauge.value, gauge.vocab) : 0;

  return (
    <div className={cn("space-y-2", className)}>
      <div className="tavle p-4">
        <div className="mb-2 flex items-center gap-2 font-mono text-[11px] font-semibold uppercase tracking-[0.12em] text-kritt/70">
          {children}
          {label}
        </div>
        {/* Innpakninga er posisjonert slik at lerretet i oppgåve 5 kan leggje
            seg nøyaktig oppå teksten – ikkje oppå etiketten. */}
        <div className="relative">
          <p
            className="min-h-8 whitespace-pre-wrap font-mono text-sm leading-relaxed text-kritt"
            style={
              gauge?.kind === "loss"
                ? {
                    filter: `blur(${blurPx(lineSmudge).toFixed(2)}px)`,
                    opacity: chalkOpacity(lineSmudge),
                  }
                : undefined
            }
          >
            {!text && placeholder && <span className="text-kritt/50">{placeholder}</span>}
            {spans
              ? spans.map((s, i) => (
                  // data-ch merkjer teikn-utsnitta, slik at uklarleikskartet i
                  // oppgåve 5 kan finne akkurat dei og ikkje t.d. plassholdaren
                  <span
                    key={i}
                    data-ch=""
                    style={
                      s.smudge > 0.02
                        ? {
                            filter: `blur(${blurPx(s.smudge).toFixed(2)}px)`,
                            opacity: chalkOpacity(s.smudge),
                          }
                        : undefined
                    }
                  >
                    {s.ch}
                  </span>
                ))
              : text}
          </p>
        </div>
      </div>
      {/* Måleren er reint visuell. Samandraget ber same talet i ord, slik at
          skjermlesarar – og folk som berre vil ha talet – får det same. */}
      {gauge && (
        <p className="text-xs leading-relaxed text-blyant">
          {summary} {legend}
        </p>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Store confidence alongside the generated chat text**

In `src/App.tsx`, import `generateDetailed` and the chalk helpers. Change the `ml` import to include `generateDetailed` alongside the existing `generate`, and add:

```tsx
import { meanConf } from "@/lib/chalk";
import Tavle from "@/components/Tavle";
```

Next to the existing `chatFull` state declaration, add:

```tsx
  const [chatConf, setChatConf] = useState<Float32Array>(() => new Float32Array(0));
  const [chatPromptLen, setChatPromptLen] = useState(0);
```

In `runGenerate` (line 353), replace the `generate(...)` call and its `setChatFull(out)` with:

```tsx
      const out = generateDetailed(
        eng.model,
        eng.tokenizer.decode,
        eng.tokenizer.encode,
        chatPrompt,
        { temperature: genTemp, topK: genTopK, length: genLen },
        sampleRngRef.current
      );
      setChatFull(out.text);
      setChatConf(out.conf);
      setChatPromptLen(out.promptLen);
```

- [ ] **Step 3: Replace the §5 chalkboard with `<Tavle>`**

In `src/App.tsx`, replace the block at lines 802-809 (the `<div className="tavle p-4">` under the comment `{/* live-eksempel: eleven skriv på tavla */}`) with:

```tsx
            <Tavle
              label={s.train.liveLabel}
              text={currentSample}
              placeholder={s.train.livePlaceholder}
              legend={s.train.focusLegend}
              summary={s.train.focusSummary(
                Math.round(lossToFocus(stats.last, stats.vocab) * 100)
              )}
              // måleren gjeld berre når det finst eit ekte tap å måle mot
              gauge={
                losses.length > 0
                  ? { kind: "loss", value: stats.last, vocab: stats.vocab }
                  : undefined
              }
            >
              <span
                className={cn(
                  "h-2 w-2 rounded-full",
                  running ? "animate-pulse bg-tusj" : "bg-kritt/30"
                )}
              />
            </Tavle>
```

Add `lossToFocus` to the `@/lib/chalk` import in Step 2 so it reads:

```tsx
import { lossToFocus, meanConf } from "@/lib/chalk";
```

- [ ] **Step 4: Replace the §7 chalkboard with `<Tavle>`**

In `src/App.tsx`, replace the block at lines 921-930 (the `<div className="tavle p-4">` under the comment `{/* svaret kjem på tavla */}`) with:

```tsx
            <Tavle
              label={s.chat.answerLabel}
              text={chatShown}
              placeholder=""
              legend={s.chat.confLegend}
              summary={s.chat.confSummary(Math.round(meanConf(chatConf) * 100))}
              gauge={
                chatConf.length > 0
                  ? { kind: "conf", conf: chatConf, promptLen: chatPromptLen }
                  : undefined
              }
            />
```

The blinking cursor from the old markup is dropped: with per-character smudge the line already reads as live, and a `tusj`-coloured caret next to blurred chalk fights the gauge for attention.

- [ ] **Step 5: Verify types and tests**

Run: `pnpm typecheck`

Expected: no output, exit 0. If `Property 'focusLegend' does not exist` appears, Task 3 was not completed.

Run: `pnpm test`

Expected: all eleven tests PASS.

- [ ] **Step 6: Verify in the browser**

Run: `pnpm dev`, open the printed URL.

Check, in order:
1. §7 before any training — click "Generer tekst". The output should be near-uniformly heavily smudged (an untrained model is uncertain about everything), and the summary should read a low percentage.
2. §5 — click "Start trening". The live sample starts blurred and visibly sharpens as the loss curve falls. The summary percentage climbs.
3. §7 after training — generate again. Most characters are crisp; the smudged ones cluster at word boundaries and rare letters. **Your prompt text is perfectly crisp and the smudge starts exactly where it ends.**
4. Drag temperature to 1.5 and generate. More heavily-smudged characters appear in the output — the model picking letters it does not believe in.
5. Select text on both boards and copy it. Cmd+F for a word in the output. Both must still work — these are ordinary spans.
6. Switch to nynorsk. Both legends and summaries change language.
7. Enable "Reduce motion" in macOS System Settings → Accessibility → Display. Nothing should change yet — tier 1 has no animation. This confirms the baseline before Task 5 adds drift.

- [ ] **Step 7: Commit**

```bash
git add src/components/Tavle.tsx src/App.tsx
git commit -m "feat(tavle): chalk-smudge uncertainty gauges on both boards"
```

---

### Task 5: Tier 2 — live DOM in WebGL

Everything here is an upgrade over Task 4's output. If any step proves unworkable or too slow, tier 1 remains and the feature still ships.

**Files:**
- Create: `src/components/tavle.glsl.ts`
- Modify: `src/components/Tavle.tsx`
- Modify: `index.html`

**Interfaces:**
- Consumes: `supportsElementTexture`, `forcedTier` from `src/lib/chalk.ts` (Task 2); the `Gauge` type and existing markup from `src/components/Tavle.tsx` (Task 4).
- Produces: `VERT: string`, `FRAG: string` from `src/components/tavle.glsl.ts`.

- [ ] **Step 1: Write the shader source**

Create `src/components/tavle.glsl.ts`:

```ts
// Shaderkjelde som inline strengar: bygget blir éi sjølvstendig HTML-fil,
// så ingenting kan hentast frå ein URL.

export const VERT = `#version 300 es
in vec2 a_pos;
out vec2 v_uv;
void main() {
  // DOM-teksturar er topp-ned, difor speglar vi y
  v_uv = vec2(a_pos.x * 0.5 + 0.5, 0.5 - a_pos.y * 0.5);
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

export const FRAG = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_board;   // levande DOM frå tavla
uniform sampler2D u_smudge;  // uklarleik per teikn, same geometri som tavla
uniform vec2 u_texel;        // 1.0 / oppløysing
uniform float u_time;        // sekund; frose ved prefers-reduced-motion
out vec4 outColor;

float hash(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float noise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x),
             mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y);
}

void main() {
  float s = texture(u_smudge, v_uv).r;
  if (s <= 0.004) {
    outColor = texture(u_board, v_uv);
    return;
  }

  // Retninga kjem frå eit støyfelt, ikkje frå ein fast akse: krit smiter
  // ujamnt, slik ekte krit gjer.
  float a = noise(v_uv * 90.0 + u_time * 0.15) * 6.2831853;
  vec2 dir = vec2(cos(a), sin(a));

  // Fleire prøver langs retninga => kritstøvet blør inn i nabobokstaven.
  // Dette er heile grunnen til at tier 2 finst; CSS-uskarpleik er innestengd
  // i sin eigen boks per teikn og kan ikkje gjere dette.
  vec4 acc = vec4(0.0);
  float wsum = 0.0;
  for (int k = 0; k < 6; k++) {
    float t = float(k) / 5.0;
    vec2 off = dir * s * t * u_texel * 7.0;
    float w = 1.0 - t * 0.7;
    acc += texture(u_board, v_uv + off) * w;
    wsum += w;
  }

  vec4 col = acc / wsum;
  col.a *= mix(1.0, 0.55, s);
  outColor = col;
}`;
```

- [ ] **Step 2: Add the origin trial meta tag**

Register `training.aitester.win` for the HTML-in-Canvas origin trial at <https://developer.chrome.com/origintrials>, then add the returned token to `index.html` inside `<head>`, after the `<meta name="theme-color" ...>` line:

```html
    <!-- HTML-in-Canvas origin trial. Går ut med Chrome 150; når han går ut,
         fell tavlene stille tilbake til CSS-nivået og alt verkar framleis. -->
    <meta http-equiv="origin-trial" content="REPLACE_WITH_TOKEN_FROM_CHROME_ORIGIN_TRIALS" />
```

For local development the token is not needed: run Chrome Canary 149+ with `chrome://flags/#canvas-draw-element` enabled.

If the token has not been obtained yet, skip this step and develop against the flag. The feature is complete without it — only the deployed site loses tier 2.

- [ ] **Step 3: Add the WebGL overlay to `Tavle.tsx`**

In `src/components/Tavle.tsx`, widen the existing `react` import — `useMemo` must stay — so it reads:

```tsx
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
```

and add:

```tsx
import { forcedTier, supportsElementTexture } from "@/lib/chalk";
import { FRAG, VERT } from "./tavle.glsl";
```

Add refs and tier state inside the component, above the existing `spans` memo:

```tsx
  const boardRef = useRef<HTMLParagraphElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [tier2, setTier2] = useState(false);

  useEffect(() => {
    const forced = forcedTier();
    if (forced !== null) {
      setTier2(forced === 2);
      return;
    }
    setTier2(supportsElementTexture());
  }, []);
```

Give the existing `<p>` a `ref={boardRef}`. The `relative` wrapper it already sits in (added in Task 4) is the positioning context. Add the canvas as its sibling, immediately after `</p>` and still inside that wrapper:

```tsx
          {tier2 && gauge && (
          // Lerretet ligg oppå tavla og skjuler henne. Teksten under er
          // urørt og fullt levande: markering, kopiering, Cmd+F og
          // skjermlesarar går rett gjennom (pointer-events: none).
          <canvas
            ref={canvasRef}
            /* @ts-expect-error layoutsubtree er frå html-in-canvas-forsøket */
            layoutsubtree=""
            aria-hidden
            className="pointer-events-none absolute inset-0 h-full w-full"
          />
        )}
```

When `tier2` is true, the `<p>`'s inline CSS filter must be suppressed — the shader does that work instead. Change the `style` prop on the `<p>` to:

```tsx
          style={
            !tier2 && gauge?.kind === "loss"
              ? {
                  filter: `blur(${blurPx(lineSmudge).toFixed(2)}px)`,
                  opacity: chalkOpacity(lineSmudge),
                }
              : undefined
          }
```

and the per-span `style` to:

```tsx
                  style={
                    !tier2 && s.smudge > 0.02
                      ? {
                          filter: `blur(${blurPx(s.smudge).toFixed(2)}px)`,
                          opacity: chalkOpacity(s.smudge),
                        }
                      : undefined
                  }
```

- [ ] **Step 4: Build the smudge map and drive the render loop**

Add to `src/components/Tavle.tsx`:

```tsx
Place both of these **after** the `lineSmudge` declaration from Task 4 and before the `return`. They reference `lineSmudge` and `spans` in their dependency arrays, which are evaluated on every render — putting them higher up throws a temporal-dead-zone `ReferenceError` on first paint.

```tsx
  // Uklarleikskartet: eit lite lerret med same geometri som tavla, der kvitt
  // = heilt uklart. For per-teikn-måling teiknar vi éin rute per teikn-utsnitt
  // ut frå den faktiske plasseringa, så det held sjølv når linja bryt.
  // For tap-måling er heile flata éin verdi.
  const buildSmudgeMap = useCallback(() => {
    const el = boardRef.current;
    if (!el || !gauge) return null;
    const r = el.getBoundingClientRect();
    const w = Math.max(1, Math.round(r.width / 4));
    const h = Math.max(1, Math.round(r.height / 4));
    const c = document.createElement("canvas");
    c.width = w;
    c.height = h;
    const ctx = c.getContext("2d");
    if (!ctx) return null;

    if (gauge.kind === "loss") {
      const v = Math.round(lineSmudge * 255);
      ctx.fillStyle = `rgb(${v},${v},${v})`;
      ctx.fillRect(0, 0, w, h);
      return c;
    }

    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, w, h);
    // berre teikn-utsnitta, aldri plassholdaren – elles forskyv indeksane seg
    const kids = el.querySelectorAll("span[data-ch]");
    const list = spans ?? [];
    kids.forEach((node, i) => {
      const s = list[i]?.smudge ?? 0;
      if (s <= 0.02) return;
      const b = node.getBoundingClientRect();
      const v = Math.round(s * 255);
      ctx.fillStyle = `rgb(${v},${v},${v})`;
      ctx.fillRect((b.x - r.x) / 4, (b.y - r.y) / 4, b.width / 4, b.height / 4);
    });
    return c;
  }, [gauge, lineSmudge, spans]);

  useLayoutEffect(() => {
    if (!tier2 || !gauge) return;
    const el = boardRef.current;
    const canvas = canvasRef.current;
    if (!el || !canvas) return;

    const gl = canvas.getContext("webgl2", { premultipliedAlpha: true });
    if (!gl) {
      setTier2(false); // ingen WebGL2 likevel: fall til CSS-nivået
      return;
    }

    const compile = (type: number, src: string) => {
      const sh = gl.createShader(type)!;
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        throw new Error(gl.getShaderInfoLog(sh) ?? "shader compile failed");
      }
      return sh;
    };

    let prog: WebGLProgram;
    try {
      prog = gl.createProgram()!;
      gl.attachShader(prog, compile(gl.VERTEX_SHADER, VERT));
      gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FRAG));
      gl.linkProgram(prog);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        throw new Error(gl.getProgramInfoLog(prog) ?? "link failed");
      }
    } catch {
      setTier2(false);
      return;
    }
    gl.useProgram(prog);

    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 3, -1, -1, 3]),
      gl.STATIC_DRAW
    );
    const loc = gl.getAttribLocation(prog, "a_pos");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

    const mkTex = (unit: number) => {
      const t = gl.createTexture();
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      return t;
    };
    const boardTex = mkTex(0);
    const smudgeTex = mkTex(1);
    gl.uniform1i(gl.getUniformLocation(prog, "u_board"), 0);
    gl.uniform1i(gl.getUniformLocation(prog, "u_smudge"), 1);
    const uTexel = gl.getUniformLocation(prog, "u_texel");
    const uTime = gl.getUniformLocation(prog, "u_time");

    // Rørsle er pynt; uklarleik er informasjon. Ved redusert rørsle frys tida,
    // men kartet blir teikna som før.
    const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    let raf = 0;
    let alive = true;
    const t0 = performance.now();

    const frame = () => {
      if (!alive) return;
      const r = el.getBoundingClientRect();
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const w = Math.max(1, Math.round(r.width * dpr));
      const h = Math.max(1, Math.round(r.height * dpr));
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
      gl.viewport(0, 0, w, h);
      gl.uniform2f(uTexel, 1 / w, 1 / h);
      gl.uniform1f(uTime, still ? 0 : (performance.now() - t0) / 1000);

      const map = buildSmudgeMap();
      if (map) {
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, smudgeTex);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, map);
      }

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, boardTex);
      // den eksperimentelle utvidinga: levande DOM rett inn som tekstur
      (gl as unknown as {
        texElementImage2D: (
          target: number, level: number, internalformat: number,
          format: number, type: number, element: Element
        ) => void;
      }).texElementImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, el);

      gl.drawArrays(gl.TRIANGLES, 0, 3);
      raf = requestAnimationFrame(frame);
    };

    try {
      frame();
    } catch {
      alive = false;
      setTier2(false); // utvidinga finst ikkje likevel
      return;
    }

    return () => {
      alive = false;
      cancelAnimationFrame(raf);
      gl.deleteProgram(prog);
      gl.deleteBuffer(buf);
      gl.deleteTexture(boardTex);
      gl.deleteTexture(smudgeTex);
    };
  }, [tier2, gauge, buildSmudgeMap]);
```

Add `useCallback` to the `react` import.

- [ ] **Step 5: Verify types and tests**

Run: `pnpm typecheck`

Expected: no output, exit 0.

Run: `pnpm test`

Expected: all eleven tests PASS. `chalk.test.mjs` asserts `supportsElementTexture() === false` in Node, which confirms the detection cannot throw outside a browser.

- [ ] **Step 6: Verify tier 2 in Chrome Canary**

Run `pnpm dev`. Open the URL in Chrome Canary 149+ with `chrome://flags/#canvas-draw-element` enabled.

1. Train, then generate in §7. Compare against `?tier=1` in the same browser: tier 2's smudged letters should visibly bleed **into their neighbours**; tier 1's stay boxed per character. If there is no visible difference, the shader is not running — check the console for shader compile errors.
2. Select and copy text on both boards; Cmd+F for a word in the output. All must still work — the canvas is `pointer-events: none` over untouched DOM.
3. Run Chrome DevTools → Lighthouse or the Accessibility pane and confirm the summary line is still exposed.
4. Open in Safari or Firefox. Confirm tier 1 renders and nothing errors in the console.
5. Run `pnpm build && open dist/index.html`. The `file://` build has no origin trial: confirm it falls to tier 1 and the gauges still work.
6. Enable "Reduce motion". The chalk dust must stop drifting; the smudge amount must remain.

- [ ] **Step 7: Measure the training-throughput gate**

This is the decision point the spec commits to. §5's gauge runs during the training loop, which is the app's hot path.

With DevTools closed, in Chrome Canary:

1. Open `?tier=1`, click "Start trening", wait for the ETA readout to settle (~10 s), and record the steps/sec implied by the ETA line under the step counter.
2. Hard-reload to `?tier=2`, repeat, record.
3. Do both twice more and take medians.

**Gate:** if tier 2 costs more than 10 % of median steps/sec, §5 must stay on tier 1 permanently. Implement that by passing an explicit prop — add `noCanvas?: boolean` to `Tavle`'s props, treat `tier2 && !noCanvas` as the condition everywhere `tier2` is currently checked, and set `noCanvas` on the §5 `<Tavle>` in `src/App.tsx` with a comment recording the measured numbers. §7 keeps tier 2 either way: it renders once per generation, not per training step.

Record the measured medians in the commit message regardless of outcome.

- [ ] **Step 8: Commit**

```bash
git add src/components/Tavle.tsx src/components/tavle.glsl.ts index.html
git commit -m "feat(tavle): WebGL chalk-bleed upgrade via html-in-canvas

Measured steps/sec tier1 vs tier2: <fill in from Step 7>"
```

---

## Verification

The feature is done when all of the following hold:

- `pnpm test` — eleven tests pass, including `generate-parity: PASS` from an unmodified file.
- `pnpm typecheck` — clean.
- `pnpm build` — succeeds; `dist/index.html` opens from `file://` and shows working tier 1 gauges.
- In §7 the prompt renders crisp and the smudge begins exactly where the prompt ends.
- In §5 the live sample sharpens as the loss curve falls.
- Text selection, copy, and Cmd+F work on both boards in both tiers.
- Both languages render their own legend and summary.
- `prefers-reduced-motion` stops the drift and keeps the smudge.
