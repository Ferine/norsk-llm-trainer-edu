# Krittavla — usikkerheit som synleg krit

**Date:** 2026-07-25
**Status:** Design approved, ready for planning

## Summary

Both chalkboards in the app (`.tavle` in §5 Trening and §7 Chat) gain a visual gauge
driven by real numbers from the model. The metaphor is **chalk that hasn't settled**:
text the model was unsure about renders smudged, like a pupil half-erased it.

Two gauges, one visual language:

- **§5 Trening** — whole-line focus bound to live training loss. Step 0: the words barely
  hold together. Loss drops: the text resolves. This is the app's tagline ("watch random
  numbers turn into Norwegian") made literal.
- **§7 Chat** — per-character smudge bound to the model's own next-character confidence.
  Surfaces information that is currently completely invisible in the UI.

Two rendering tiers: CSS everywhere, WebGL + the experimental html-in-canvas API as a
progressive upgrade in Chrome.

## Design principle

This app's claim is that it does not fake anything. `Skruer.tsx` states it outright:
*"Alt er ekte verdiar frå modellen, ikkje ein animasjon."*

Any visual effect must clear the same bar: **its amplitude must be bound to a real number
coming out of the model.** A shader that is a gauge earns its place. A shader that is a
garnish does not, and Kladdeboka rejects it.

This principle is why the design uses none of CanvasUI's catalogue effects (liquid, glass,
VHS, glitch) — those are a screen aesthetic and this app is a paper one. The technique
(html-in-canvas) is borrowed; the vocabulary is Kladdeboka's own `tavle`/`kritt` tokens.

## What the gauges measure

### Per-character confidence (§7)

`conf[i]` = probability of the chosen token under a **full-vocabulary softmax at
temperature 1** — the model's own belief, computed independently of the user's
temperature and top-k sliders.

Rejected alternative: the effective sampling distribution (post-temperature, post-top-k).
It matches slider intuition but conflates model uncertainty with sampler recklessness,
and goes flat at `topK = 1`.

The chosen definition yields a second-order lesson for free: crank temperature to 1.5 and
you watch the model pick heavily-smudged characters. Temperature does not add knowledge —
it only makes the sampler gamble. The smudge is the record of that.

**Prompt characters carry no confidence** and render perfectly crisp. They were given, not
predicted. The smudge therefore begins exactly where the user's prompt ends and the model's
guessing starts.

### Loss-driven focus (§5)

Focus normalizes against `Math.log(vocab)` — the measured cross-entropy of pure random
guessing — as the zero point, with a tuned floor constant. The floor carries an explanatory
comment in the style of `SHARPEN` in `Skruer.tsx:19`.

**§5 uses loss, not per-character confidence, even though confidence is free there** (the
training loop already calls `generate()` every 60 steps). Two gauges on one line is
unreadable, and the sections answer different questions: §5 asks *"is it learning?"*, a
global scalar; §7 asks *"how sure is it, letter by letter?"*, a local one.

## Architecture

### Data plumbing — `src/lib/ml.ts`

`sampleTokens` already computes a distribution at every step. It gains one O(V) pass
(V ≈ 50, character-level vocabulary — negligible).

- Additive return field: `{ promptIds, contIds, conf: Float32Array }`.
- The greedy branch (`temperature <= 0`, reachable — the slider minimum is 0) currently
  computes no probabilities at all. It gains the same softmax so the gauge does not
  silently die at temperature 0.
- New export `generateDetailed()` returns `{ text, promptLen, conf }`. `generate()` becomes
  a one-line wrapper over it and keeps its current string return type.

Compatibility: `test/generate-parity.test.mjs:12` asserts
`generate === prompt + decode(sampleTokens.contIds)`; neither side changes.
`useRlhf.ts:71` destructures `{ promptIds, contIds }` and is unaffected by an added field.

### Components

```
src/components/Tavle.tsx      — chalkboard wrapper; owns the gauge and tier detection
src/lib/chalk.ts              — focus/smudge math + capability detection, no React
src/components/tavle.glsl.ts  — inlined shader source (single-file build, no asset URLs)
```

`<Tavle>` takes a discriminated prop:

- `gauge={{ kind: "loss", value, vocab }}` → §5, one global focus for the whole line
- `gauge={{ kind: "conf", conf, promptLen }}` → §7, per-character
- no `gauge` → today's behaviour exactly, unchanged

The third case matters: §8 (RLHF) and every other `.tavle` in the app keep their current
rendering. This is an opt-in wrapper, not a global restyle.

## Rendering tiers

### Tier 1 — CSS. Everywhere, including the offline `file://` build.

Per-character `<span>` inside the existing `whitespace-pre-wrap` block. `filter: blur()`
from 0 to ~1.6px and opacity from 1 to ~0.45, both driven by `1 - conf`. Text selection,
copy, and find-in-page survive spans natively.

For the loss gauge, a single filter on the whole block rather than per-character — cheaper,
and semantically correct because loss *is* a global scalar.

### Tier 2 — canvas. Chrome with the origin trial only.

`<canvas layoutsubtree>`, `texElementImage2D` sampling the live chalkboard element, with a
fragment shader displacing along a noise gradient. Per-character amplitude is fed in as a
small 1D data texture. The transform returned by the draw call is written back to the
element's `style.transform` so hit-testing maps correctly.

What tier 2 buys over tier 1, and the only reason the experimental API is worth its risk:
**cross-character bleed.** Adjacent uncertain letters smear into each other the way real
chalk does. CSS cannot do this — every span is its own box.

Verified properties of the API (Chrome for Developers, origin trial announcement): drawn
content stays in the accessibility tree, remains find-in-page searchable, and stays
selectable. The source element must be present and laid out in the DOM. Rendering is live,
not a snapshot, redrawing on the `paint` event.

### Motion

`prefers-reduced-motion: reduce` → smudge amount still renders; dust drift does not
animate. Smudge is information, drift is decoration. This matches the existing precedent in
`LearningStrip` (`App.tsx:56`).

## Degradation ladder

Each rung falls to tier 1: no `drawElementImage` → no WebGL → no origin-trial token →
`file://` single-file build.

**The lesson never disappears; only the polish does.** This is the entire justification for
building two tiers rather than one.

## Operations

Register `training.aitester.win` for the HTML-in-Canvas origin trial and add
`<meta http-equiv="origin-trial" content="…">` to `index.html`. It survives
`vite-plugin-singlefile` inlining.

The token **expires with Chrome 150**; stable ship is only *estimated* for late 2026. When
it lapses, the site silently drops to tier 1. That is a designed property, not a maintenance
task — but it is a known, dated commitment rather than a surprise.

Local development requires Chrome Canary 149+ with `chrome://flags/#canvas-draw-element`.

## Risks

**Training throughput (accepted, with a gate).** The §5 gauge runs during the training loop,
which is the app's hot path (`CHUNK` steps per `setTimeout(0)` tick). A per-frame WebGL pass
could manufacture jank. A previous decision declined a Web Worker refactor specifically
because there is no jank today.

Mitigation: the implementation plan includes an explicit measurement step — steps/sec with
tier 2 enabled vs. disabled. **If tier 2 costs measurable training throughput, §5 stays on
tier 1 permanently and only §7 receives the canvas upgrade.** Cutting the effect is
preferred over slowing training.

**Experimental API.** Mitigated structurally by the degradation ladder: tier 2 is never
load-bearing for the feature's meaning.

## Testing

- `test/conf.test.mjs` — `conf.length === contIds.length`; every value in (0,1]; the greedy
  path (temperature 0) still populates it; deterministic under seeded `mulberry32`.
- `chalk.ts` pure math — loss→focus is monotonic decreasing, clamped to [0,1], and
  `Math.log(vocab)` maps to exactly 0.
- `test/generate-parity.test.mjs` must pass **unmodified**. That is the proof the `ml.ts`
  refactor was purely additive.
- `test/i18n-parity.test.mjs` enforces bokmål/nynorsk parity for the new strings.
- Full suite (`pnpm test`) and `pnpm typecheck` green.

## i18n and copy

New strings in both bokmål and nynorsk. Copy follows the app's plain-language rule — plain
words first, technical term in parentheses if at all. The legend reads
*"uklare bokstavar = modellen var usikker"*, never *"entropi"*.

## Accessibility

The smudge conveys meaning purely visually. The canvas preserves the accessibility tree for
the *text*, but a screen reader gets nothing from a blur.

Both boards therefore gain a one-line summary carrying the same number the gauge encodes —
not a caption describing the effect, but its non-visual equivalent. Each board summarizes
*its own* gauge:

- §7 reports mean confidence across the generated characters — *"modellen var i snitt
  62 % sikker"*.
- §5 reports the loss-derived focus against the random-guessing baseline, in the same plain
  register — never a raw cross-entropy figure.

Both are readable by assistive technology and genuinely useful to sighted users too.

## Out of scope

- CanvasUI's effect catalogue (liquid, glass, VHS, glitch, shatter) — wrong aesthetic register.
- A page-crumple effect on reset — considered and cut as decoration that gauges nothing.
- Any change to §8 RLHF or other `.tavle` instances.
- Re-litigating the Web Worker refactor.
